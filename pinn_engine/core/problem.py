"""PINA :class:`InverseProblem` adapter.

``build_problem(compiled, data, t_range)`` returns a freshly-built
PINA problem class instance (a subclass of ``TimeDependentProblem`` +
``InverseProblem``) wired up with the compiled physics residuals and the
sensor data.

This is the bridge between our DSL and PINA's runtime.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

from pina import Condition, LabelTensor
from pina.problem import TimeDependentProblem, SpatialProblem, InverseProblem
from pina.domain import CartesianDomain
from pina.equation import Equation

from pinn_engine.dsl.system import CompiledSystem


def _as_label_tensor(arr, labels: list[str]) -> LabelTensor:
    """Wrap a 1-D or 2-D array as a PINA LabelTensor with given column labels."""
    t = torch.as_tensor(arr, dtype=torch.float32)
    if t.ndim == 1:
        t = t.reshape(-1, 1)
    return LabelTensor(t, labels=list(labels))


def build_problem(
    compiled: CompiledSystem,
    data: Dict[str, Tuple[Any, Any]],
    t_range: Tuple[float, float],
    spatial_ranges: Dict[str, Tuple[float, float]] = None,
    bounds_override: Dict[str, Tuple[float, float]] = None,
    inits_override: Dict[str, float] = None,
) -> InverseProblem:
    """Construct a PINA problem instance from a compiled DSL system + data.

    Parameters:
        compiled: the result of ``System.compile()``.
        data: mapping ``sensor_name -> (input_array, observation_array)``.
            For PDE problems the input array should have shape ``(N, n_inputs)``
            with columns in the order ``compiled.input_names``.
        t_range: ``(t_min, t_max)`` for the temporal collocation domain.
        spatial_ranges: only required for PDE problems —
            ``{spatial_name: (lo, hi)}`` for each spatial input variable.

    Returns:
        an instantiated PINA problem. For PDEs the class inherits both
        ``SpatialProblem`` and ``TimeDependentProblem``.
    """
    state_names = list(compiled.state_names)
    input_name = compiled.input_name
    is_pde = compiled.is_pde
    spatial_names = [n for n in compiled.input_names if n != input_name]

    temporal_domain = CartesianDomain({input_name: [float(t_range[0]), float(t_range[1])]})
    if is_pde:
        if not spatial_ranges:
            raise ValueError(
                f"PDE problem with spatial variables {spatial_names!r} requires "
                f"`spatial_ranges` (got None)"
            )
        spatial_domain = CartesianDomain(
            {n: [float(spatial_ranges[n][0]), float(spatial_ranges[n][1])]
             for n in spatial_names}
        )
        # Joint physics-collocation domain spanning all inputs.
        physics_domain = CartesianDomain(
            {**spatial_domain.range_, **temporal_domain.range_}
        )
    else:
        spatial_domain = None
        physics_domain = temporal_domain

    # Allow per-unknown overrides (used by the iterative-bound-tightening loop:
    # after a run converges, narrow the search range around the result and
    # re-train for higher precision). Merges over the compiled defaults.
    effective_bounds = dict(compiled.unknown_bounds)
    if bounds_override:
        for name, b in bounds_override.items():
            if name in effective_bounds:
                effective_bounds[name] = (float(b[0]), float(b[1]))
    unknown_parameter_domain = CartesianDomain(
        {name: [float(lo), float(hi)] for name, (lo, hi) in effective_bounds.items()}
    )

    conditions: Dict[str, Condition] = {}

    # One physics condition per residual.
    for i, residual_fn in enumerate(compiled.physics_residuals):
        conditions[f"physics_{i}"] = Condition(
            domain=physics_domain,
            equation=Equation(residual_fn),
        )

    # One data condition per sensor that has data.
    for sens in compiled.sensors:
        if sens.name not in data:
            continue
        t_arr, obs_arr = data[sens.name]
        if is_pde:
            # Sensor input is (N, n_inputs) with columns ordered as
            # compiled.input_names.
            input_lt = _as_label_tensor(t_arr, labels=list(compiled.input_names))
        else:
            input_lt = _as_label_tensor(t_arr, labels=[input_name])
        observed_label = sens.observes.name if hasattr(sens.observes, "name") else sens.name
        target_lt = _as_label_tensor(obs_arr, labels=[observed_label])
        conditions[f"data_{sens.name}"] = Condition(input=input_lt, target=target_lt)

    attrs = {
        "output_variables": state_names,
        "temporal_domain": temporal_domain,
        "unknown_parameter_domain": unknown_parameter_domain,
        "conditions": conditions,
    }
    if is_pde:
        attrs["spatial_domain"] = spatial_domain
        base_classes = (SpatialProblem, TimeDependentProblem, InverseProblem)
    else:
        base_classes = (TimeDependentProblem, InverseProblem)
    cls = type("CompiledInverseProblem", base_classes, attrs)
    problem = cls()

    # PINA initializes unknown_parameters with U(0, range_hi) + range_lo, which
    # gives the wrong distribution unless lo == 0. Override to the requested init.
    # ``inits_override`` (from iterative refinement) takes precedence over the
    # compiled default — lets a follow-up run start from the previous result.
    effective_inits = dict(compiled.unknown_inits)
    if inits_override:
        for name, v in inits_override.items():
            if name in effective_inits:
                effective_inits[name] = float(v)
    for name, init in effective_inits.items():
        problem.unknown_parameters[name] = torch.nn.Parameter(
            torch.tensor([float(init)], requires_grad=True)
        )

    return problem
