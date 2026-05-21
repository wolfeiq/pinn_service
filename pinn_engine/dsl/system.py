"""The user-facing :class:`System` and its compilation pipeline.

``System`` is what the user passes to the trainer. Internally it:

* validates that every symbol used in the equations is declared;
* compiles each sympy equation to a torch callable
  ``residual(input_lt, output_lt, params_) → Tensor`` compatible with PINA's
  ``Equation`` interface;
* hashes its structure for the reproducibility manifest.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib

import sympy as sp
import torch

from pinn_engine.dsl.symbols import Variable, Parameter, Unknown, Sensor


class SystemValidationError(ValueError):
    """Raised when a System fails consistency checks."""


@dataclass
class CompiledSystem:
    """Compiled form of a :class:`System` — handed to the trainer / PINA.

    Attributes:
        state_names: names of the state variables in network-output order.
        input_name: name of the *primary* independent variable (kept for
            back-compat; for PDEs this is the time variable).
        input_names: ordered tuple of *all* input variables (e.g. ``("s", "t")``).
            For ODEs this is a single-element tuple.
        unknown_names: names of unknown parameters in stable order.
        unknown_bounds: per-unknown ``(lo, hi)`` bounds.
        unknown_inits: per-unknown initial value.
        physics_residuals: list of torch callables, one per equation, with
            signature ``(input_lt, output_lt, params_dict) → Tensor[N,1]``.
        sensors: list of :class:`Sensor`.
        sensor_observation_fns: per-sensor callable
            ``(output_lt) → Tensor`` that extracts the observed quantity.
        equation_hash: SHA-256 of the canonical sympy serialization.
    """

    state_names: List[str]
    input_name: str
    unknown_names: List[str]
    unknown_bounds: Dict[str, Tuple[float, float]]
    unknown_inits: Dict[str, float]
    physics_residuals: List[Callable[..., torch.Tensor]]
    sensors: List[Sensor]
    sensor_observation_fns: Dict[str, Callable[[Any], torch.Tensor]]
    equation_hash: str
    input_names: Tuple[str, ...] = ()

    @property
    def is_pde(self) -> bool:
        return len(self.input_names) > 1


class System:
    """A user-declared inverse problem.

    Parameters:
        state: list of :class:`Variable` representing the state.
        equations: list of sympy expressions, each set to zero (residual form).
        sensors: list of :class:`Sensor`.
    """

    def __init__(
        self,
        state: List[Variable],
        equations: List[sp.Expr],
        sensors: List[Sensor],
    ):
        self.state = list(state)
        self.equations = [sp.sympify(eq) for eq in equations]
        self.sensors = list(sensors)
        self._compiled: Optional[CompiledSystem] = None

    # ------------------------------------------------------------------ helpers

    @property
    def input_variable(self) -> Variable:
        """The primary input variable (the temporal one for PDEs, the only one for ODEs).

        Convention: when multiple inputs are declared, the *last* declared dependency
        of any state variable is treated as time. The build_plan / robotics templates
        consistently declare ``depends_on=(s, t)`` so this resolves cleanly.
        """
        all_inputs = self.input_variables
        if not all_inputs:
            raise SystemValidationError(
                "No state variable has a `depends_on`; nothing to integrate."
            )
        return all_inputs[-1]

    @property
    def input_variables(self) -> List[Variable]:
        """All independent variables used across the state, in declaration order."""
        seen: List[Variable] = []
        for s in self.state:
            for v in s.depends_on_all:
                if v not in seen:
                    seen.append(v)
        return seen

    def _all_symbols(self) -> set:
        syms = set()
        for eq in self.equations:
            syms |= eq.free_symbols
            # Derivative expressions also have free_symbols, but we also need to
            # collect the base variable they differentiate:
            for d in eq.atoms(sp.Derivative):
                syms.add(d.expr)
        return syms

    def unknowns(self) -> List[Unknown]:
        return [s for s in self._all_symbols() if isinstance(s, Unknown)]

    def parameters(self) -> List[Parameter]:
        return [s for s in self._all_symbols() if isinstance(s, Parameter)]

    # ------------------------------------------------------------------ validate

    def validate(self) -> None:
        """Static checks. Raises :class:`SystemValidationError` on failure."""
        if not self.state:
            raise SystemValidationError("`state` must contain at least one Variable.")
        if not self.equations:
            raise SystemValidationError("`equations` must contain at least one residual.")
        if not self.sensors:
            raise SystemValidationError("`sensors` must contain at least one Sensor.")

        # Confirm we have at least one independent variable
        self.input_variable  # raises if none

        state_names = {s.name for s in self.state}
        all_input_vars = self.input_variables
        for eq in self.equations:
            for sym in eq.free_symbols:
                if isinstance(sym, (Variable, Parameter, Unknown)):
                    continue
                # Allow any declared independent variable
                if sym in all_input_vars:
                    continue
                raise SystemValidationError(
                    f"Equation references undeclared symbol {sym!r}; "
                    f"declare it as Variable/Parameter/Unknown."
                )

        for sens in self.sensors:
            if isinstance(sens.observes, Variable):
                if sens.observes.name not in state_names:
                    raise SystemValidationError(
                        f"Sensor {sens.name!r} observes {sens.observes.name!r} "
                        f"which is not in `state`."
                    )

        for u in self.unknowns():
            if u.bounds[0] >= u.bounds[1]:
                raise SystemValidationError(
                    f"Unknown {u.name!r} has invalid bounds {u.bounds!r}."
                )

    # ------------------------------------------------------------------ compile

    def compile(self) -> CompiledSystem:
        """Lower the sympy equations + sensors to torch callables.

        Returns the cached result if called more than once.
        """
        self.validate()
        if self._compiled is not None:
            return self._compiled

        from pinn_engine.dsl.compile import compile_equation, compile_sensor_observation

        all_inputs = self.input_variables
        primary = self.input_variable
        state_names = [s.name for s in self.state]
        unknowns = sorted(self.unknowns(), key=lambda u: u.name)
        unknown_names = [u.name for u in unknowns]
        unknown_bounds = {u.name: u.bounds for u in unknowns}
        unknown_inits = {u.name: u.init_value for u in unknowns}
        params = {p.name: p.value for p in self.parameters()}

        residuals = [
            compile_equation(
                eq,
                state=self.state,
                input_vars=all_inputs,
                parameters=params,
            )
            for eq in self.equations
        ]

        obs_fns = {
            sens.name: compile_sensor_observation(sens, state=self.state)
            for sens in self.sensors
        }

        canonical = "; ".join(sp.srepr(eq) for eq in self.equations)
        canonical += " | state=" + ",".join(state_names)
        canonical += " | input=" + ",".join(v.name for v in all_inputs)
        canonical += " | unknowns=" + ",".join(unknown_names)
        canonical += " | params=" + ",".join(f"{k}={v}" for k, v in sorted(params.items()))
        eq_hash = hashlib.sha256(canonical.encode()).hexdigest()

        self._compiled = CompiledSystem(
            state_names=state_names,
            input_name=primary.name,
            input_names=tuple(v.name for v in all_inputs),
            unknown_names=unknown_names,
            unknown_bounds=unknown_bounds,
            unknown_inits=unknown_inits,
            physics_residuals=residuals,
            sensors=self.sensors,
            sensor_observation_fns=obs_fns,
            equation_hash=eq_hash,
        )
        return self._compiled
