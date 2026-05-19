"""Lower sympy expressions to torch callables.

The supported grammar covers the templates we ship in Phase 1+2:
* state variables and their derivatives (1st, 2nd order),
* :class:`Parameter` and :class:`Unknown` symbols,
* the independent variable itself,
* `+`, `-`, `*`, `/`, integer/rational powers,
* `sin`, `cos`, `exp`, `tanh`, `sqrt` (NumPy-style elementary functions).

Anything outside this grammar raises :class:`UnsupportedExpression`. That's a
deliberate design choice — silently emitting buggy torch ops for an unsupported
sympy node is worse than failing loudly.

The compiled callable has the PINA-compatible signature::

    residual(input_lt: LabelTensor,
             output_lt: LabelTensor,
             params_: Dict[str, torch.nn.Parameter]) -> torch.Tensor
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple
import sympy as sp
import torch

from pinn_engine.dsl.symbols import Variable, Parameter, Unknown, Sensor


class UnsupportedExpression(NotImplementedError):
    """Raised when the DSL grammar doesn't cover a sympy node."""


_TORCH_FUNCS = {
    sp.sin: torch.sin,
    sp.cos: torch.cos,
    sp.exp: torch.exp,
    sp.tanh: torch.tanh,
    sp.sqrt: torch.sqrt,
}


def _autograd_deriv_full(y: torch.Tensor, x_full: torch.Tensor, order: int) -> torch.Tensor:
    """Compute d^order y / d x^order via repeated autograd.

    ``y`` should be a scalar-valued (Nx1) tensor. ``x_full`` is the *original*
    multivariable input the network consumed — autograd.grad with
    ``allow_unused=True`` returns a tensor matching ``x_full``'s shape; we
    return it directly. Caller is responsible for selecting the column that
    corresponds to the differentiation variable.
    """
    out = y
    for _ in range(order):
        grad_out = torch.autograd.grad(
            out,
            x_full,
            grad_outputs=torch.ones_like(out),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if grad_out is None:
            # The output didn't depend on x_full — return zeros so the rest of
            # the computation can proceed (sensitivity to this variable is 0).
            grad_out = torch.zeros_like(x_full)
        out = grad_out
    return out


def _column(label_tensor, name: str) -> torch.Tensor:
    """Extract a named column from a PINA LabelTensor, or a plain tensor.

    Falls back to assuming the tensor has labels ``state_names`` if PINA isn't
    in the picture (e.g., during the pre-flight check which uses bare tensors).
    """
    if hasattr(label_tensor, "extract"):
        return label_tensor.extract([name])
    # Plain torch.Tensor fallback used by pre-flight + tests.
    idx = label_tensor._column_index[name]  # type: ignore[attr-defined]
    return label_tensor[..., idx : idx + 1]


def compile_equation(
    expr: sp.Expr,
    state: List[Variable],
    input_var: Variable,
    parameters: Dict[str, float],
) -> Callable[..., torch.Tensor]:
    """Return a PINA-compatible residual function for ``expr == 0``."""

    state_names = [s.name for s in state]

    def residual(input_lt, output_lt, params_=None):  # PINA signature
        params_ = params_ or {}
        ctx = _EvalContext(
            input_lt=input_lt,
            output_lt=output_lt,
            params_=params_,
            state=state,
            input_var=input_var,
            parameters=parameters,
        )
        return _eval(expr, ctx)

    residual.__name__ = f"residual_{hash(expr) & 0xFFFFFFFF:08x}"
    residual.__doc__ = f"Compiled residual for: {expr}"
    return residual


def compile_sensor_observation(
    sens: Sensor, state: List[Variable]
) -> Callable[..., torch.Tensor]:
    """Return ``output_lt -> observed_quantity`` for a sensor.

    For sensors that observe a single state variable, this is just a column
    extraction. For sensors observing an expression (future extension), we'd
    compile that expression here.
    """
    if isinstance(sens.observes, Variable):
        name = sens.observes.name

        def obs(output_lt) -> torch.Tensor:
            return _column(output_lt, name)

        obs.__name__ = f"obs_{sens.name}"
        return obs

    # Future: arbitrary sympy expression of state variables.
    raise UnsupportedExpression(
        f"Sensor {sens.name!r} observes a non-Variable expression; not yet supported."
    )


# ----------------------------------------------------------------------- eval


class _EvalContext:
    """Bag of state for the sympy → torch walker."""

    def __init__(
        self,
        input_lt,
        output_lt,
        params_: Dict[str, torch.Tensor],
        state: List[Variable],
        input_var: Variable,
        parameters: Dict[str, float],
    ):
        self.input_lt = input_lt
        self.output_lt = output_lt
        self.params_ = params_
        self.state = state
        self.state_names = {s.name for s in state}
        self.input_var = input_var
        self.parameters = parameters
        # cache state-variable values so we don't extract repeatedly
        self._state_cache: Dict[str, torch.Tensor] = {}
        self._deriv_cache: Dict[Tuple[str, int], torch.Tensor] = {}

    def state_value(self, name: str) -> torch.Tensor:
        if name not in self._state_cache:
            self._state_cache[name] = _column(self.output_lt, name)
        return self._state_cache[name]

    def input_value(self) -> torch.Tensor:
        return _column(self.input_lt, self.input_var.name)

    def derivative(self, name: str, order: int) -> torch.Tensor:
        key = (name, order)
        if key not in self._deriv_cache:
            y = self.state_value(name)
            # Differentiate w.r.t. the ORIGINAL input tensor (the one the
            # network actually consumed). The autograd graph is rooted there.
            x_full = self.input_lt
            grad_t = _autograd_deriv_full(y, x_full, order)
            # For time-only ODE problems (Phase 1+2) the input is 1-D; the
            # gradient has shape (N, 1) and we return it as-is. For multi-input
            # problems we'd slice the column corresponding to ``self.input_var``.
            if grad_t.shape[-1] > 1:
                # Future-proofing: select the column matching the input var.
                if hasattr(x_full, "labels"):
                    idx = x_full.labels.index(self.input_var.name)
                    grad_t = grad_t[..., idx : idx + 1]
            self._deriv_cache[key] = grad_t
        return self._deriv_cache[key]


def _eval(expr: sp.Expr, ctx: _EvalContext) -> torch.Tensor:
    """Recursively evaluate a sympy expression against the torch tensors."""

    # Number literals
    if expr.is_Number:
        return torch.tensor(float(expr), dtype=torch.float32)

    # The independent variable itself
    if isinstance(expr, Variable) and expr == ctx.input_var:
        return ctx.input_value()

    # State variable
    if isinstance(expr, Variable):
        if expr.name in ctx.state_names:
            return ctx.state_value(expr.name)
        raise UnsupportedExpression(f"Unknown Variable {expr.name!r} in equation.")

    # Known parameter
    if isinstance(expr, Parameter):
        return torch.tensor(ctx.parameters[expr.name], dtype=torch.float32)

    # Unknown parameter — pulled from params_ dict at call time
    if isinstance(expr, Unknown):
        if expr.name not in ctx.params_:
            raise KeyError(
                f"Unknown parameter {expr.name!r} not in params_ dict; "
                f"available keys: {list(ctx.params_)}"
            )
        return ctx.params_[expr.name]

    # Derivatives
    if isinstance(expr, sp.Derivative):
        base = expr.expr
        if not isinstance(base, Variable):
            raise UnsupportedExpression(
                f"Derivative of non-Variable {base!r} not supported."
            )
        # sympy stores variables differentiated as ((var, order), ...)
        order = 0
        for v, n in expr.variable_count:
            if v != ctx.input_var:
                raise UnsupportedExpression(
                    f"Derivative w.r.t. {v!r} not supported (only {ctx.input_var.name!r})."
                )
            order += int(n)
        return ctx.derivative(base.name, order)

    # Add
    if isinstance(expr, sp.Add):
        return sum((_eval(a, ctx) for a in expr.args), start=torch.tensor(0.0))

    # Mul
    if isinstance(expr, sp.Mul):
        out = torch.tensor(1.0)
        for a in expr.args:
            out = out * _eval(a, ctx)
        return out

    # Pow (integer or rational exponent only)
    if isinstance(expr, sp.Pow):
        base, exp = expr.args
        if not exp.is_Number:
            raise UnsupportedExpression(f"Non-numeric exponent in {expr!r}.")
        return _eval(base, ctx) ** float(exp)

    # Elementary functions
    if expr.func in _TORCH_FUNCS:
        (arg,) = expr.args
        return _TORCH_FUNCS[expr.func](_eval(arg, ctx))

    raise UnsupportedExpression(
        f"Unsupported sympy node {type(expr).__name__}: {expr!r}"
    )
