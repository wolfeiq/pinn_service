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
    sp.Abs: torch.abs,
    sp.sign: torch.sign,
    sp.log: torch.log,
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


def _autograd_deriv_full_wrt(
    y: torch.Tensor, x_full, wrt_name: str, order: int
) -> torch.Tensor:
    """Partial derivative ``∂^order y / ∂x_wrt^order`` for multi-input x.

    Each gradient step returns a tensor matching ``x_full``'s shape (one
    column per input variable). To take the partial w.r.t. ``wrt_name``,
    we select that column after every step before the next backward — so
    higher-order partials are taken consistently w.r.t. the same variable.
    """
    if not hasattr(x_full, "labels"):
        # Single-input fallback (no labels): just use the full-tensor version.
        return _autograd_deriv_full(y, x_full, order)

    try:
        idx = x_full.labels.index(wrt_name)
    except ValueError:
        # Variable isn't in the input — return zeros (output independent).
        return torch.zeros_like(y)

    out = y
    for _ in range(order):
        grad_full = torch.autograd.grad(
            out,
            x_full,
            grad_outputs=torch.ones_like(out),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if grad_full is None:
            return torch.zeros_like(y)
        # Slice the column matching wrt_name; keep the trailing singleton so
        # the next iteration's autograd has a scalar-valued tensor to grad.
        out = grad_full[..., idx : idx + 1]
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
    input_vars=None,
    parameters: Dict[str, float] = None,
    input_var: Variable = None,  # back-compat alias for single-input case
) -> Callable[..., torch.Tensor]:
    """Return a PINA-compatible residual function for ``expr == 0``.

    ``input_vars`` is the tuple of independent variables (e.g. ``(s, t)`` for
    PDEs or ``(t,)`` for ODEs). The legacy ``input_var`` kwarg is accepted
    for back-compat with the single-input case.
    """
    if input_vars is None:
        if input_var is None:
            raise ValueError("compile_equation needs `input_vars` or `input_var`")
        input_vars = (input_var,)
    state_names = [s.name for s in state]
    parameters = parameters or {}

    def residual(input_lt, output_lt, params_=None):  # PINA signature
        params_ = params_ or {}
        ctx = _EvalContext(
            input_lt=input_lt,
            output_lt=output_lt,
            params_=params_,
            state=state,
            input_vars=input_vars,
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
    """Bag of state for the sympy → torch walker.

    Supports both ODE (single input variable) and PDE (multi-input) cases.
    For PDE, partial derivatives w.r.t. any declared input variable resolve
    correctly via the autograd graph + label-based column slicing.
    """

    def __init__(
        self,
        input_lt,
        output_lt,
        params_: Dict[str, torch.Tensor],
        state: List[Variable],
        input_vars,                       # tuple of Variable
        parameters: Dict[str, float],
    ):
        self.input_lt = input_lt
        self.output_lt = output_lt
        self.params_ = params_
        self.state = state
        self.state_names = {s.name for s in state}
        self.input_vars = tuple(input_vars)
        self.input_var_names = [v.name for v in self.input_vars]
        # Back-compat: pick the "primary" input (last in tuple) for code that
        # only handled the single-input case.
        self.input_var = self.input_vars[-1]
        self.parameters = parameters
        # cache state-variable values so we don't extract repeatedly
        self._state_cache: Dict[str, torch.Tensor] = {}
        self._deriv_cache: Dict[Tuple[str, str, int], torch.Tensor] = {}

    def state_value(self, name: str) -> torch.Tensor:
        if name not in self._state_cache:
            self._state_cache[name] = _column(self.output_lt, name)
        return self._state_cache[name]

    def input_value(self) -> torch.Tensor:
        return _column(self.input_lt, self.input_var.name)

    def derivative(self, name: str, wrt: sp.Symbol, order: int) -> torch.Tensor:
        """Partial derivative ``∂^order state[name] / ∂wrt^order``.

        Differentiates w.r.t. the full input tensor (so the autograd graph is
        rooted correctly) and then slices the column corresponding to ``wrt``.
        """
        wrt_name = wrt.name if hasattr(wrt, "name") else str(wrt)
        key = (name, wrt_name, order)
        if key not in self._deriv_cache:
            y = self.state_value(name)
            x_full = self.input_lt
            grad_full = _autograd_deriv_full_wrt(y, x_full, wrt_name, order)
            self._deriv_cache[key] = grad_full
        return self._deriv_cache[key]


def _eval(expr: sp.Expr, ctx: _EvalContext) -> torch.Tensor:
    """Recursively evaluate a sympy expression against the torch tensors."""

    # Number literals
    if expr.is_Number:
        return torch.tensor(float(expr), dtype=torch.float32)

    # Any declared independent variable
    if isinstance(expr, Variable) and expr.name in ctx.input_var_names:
        return _column(ctx.input_lt, expr.name)

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
        # Multi-variable derivatives: take partials in declared order.
        # sympy stores variables differentiated as ((var, order), ...).
        # We handle one variable at a time, applying each partial in
        # sequence (∂²y/∂s∂t = ∂/∂t (∂y/∂s)).
        cur = ctx.state_value(base.name)
        x_full = ctx.input_lt
        for v, n in expr.variable_count:
            if v.name not in ctx.input_var_names:
                raise UnsupportedExpression(
                    f"Derivative w.r.t. {v!r} — not a declared input variable "
                    f"({ctx.input_var_names!r})."
                )
            # Use the cache for single-variable repeated partials (the common case).
            key = (base.name, v.name, int(n))
            if key in ctx._deriv_cache and cur is ctx.state_value(base.name):
                cur = ctx._deriv_cache[key]
            else:
                cur = _autograd_deriv_full_wrt(cur, x_full, v.name, int(n))
                if cur is ctx.state_value(base.name) is False:
                    ctx._deriv_cache[key] = cur
        return cur

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

    # Elementary functions (single-arg)
    if expr.func in _TORCH_FUNCS:
        (arg,) = expr.args
        return _TORCH_FUNCS[expr.func](_eval(arg, ctx))

    # Piecewise — torch.where chain. Each arg is (value, condition).
    # The final ``else`` branch in sympy uses condition `True`.
    if isinstance(expr, sp.Piecewise):
        # Build from the back: result = else_value; then chain torch.where for each branch.
        if not expr.args:
            raise UnsupportedExpression("Empty Piecewise expression.")
        # Find the default (condition True) branch; sympy guarantees it's last
        # if user followed convention, else we synthesise a zero default.
        branches = list(expr.args)
        default_value = torch.zeros_like(_eval(branches[-1][0], ctx))
        if branches[-1][1] == sp.true:
            default_value = _eval(branches[-1][0], ctx)
            branches = branches[:-1]
        result = default_value
        for value_expr, cond_expr in reversed(branches):
            value = _eval(value_expr, ctx)
            cond = _eval_bool(cond_expr, ctx)
            result = torch.where(cond, value, result)
        return result

    # Comparison operators used in Piecewise conditions
    if isinstance(expr, (sp.StrictGreaterThan, sp.GreaterThan,
                          sp.StrictLessThan, sp.LessThan, sp.Equality)):
        # Evaluated only via _eval_bool path; if we get here it's a bare comparison.
        return _eval_bool(expr, ctx).to(torch.float32)

    raise UnsupportedExpression(
        f"Unsupported sympy node {type(expr).__name__}: {expr!r}"
    )


def _eval_bool(expr, ctx) -> torch.Tensor:
    """Evaluate a sympy boolean / comparison expression as a torch bool tensor."""
    if expr is sp.true or expr is True:
        # Broadcast-compatible all-True; let downstream torch.where handle shape.
        return torch.tensor(True)
    if expr is sp.false or expr is False:
        return torch.tensor(False)
    if isinstance(expr, sp.StrictGreaterThan):
        a, b = expr.args
        return _eval(a, ctx) > _eval(b, ctx)
    if isinstance(expr, sp.GreaterThan):
        a, b = expr.args
        return _eval(a, ctx) >= _eval(b, ctx)
    if isinstance(expr, sp.StrictLessThan):
        a, b = expr.args
        return _eval(a, ctx) < _eval(b, ctx)
    if isinstance(expr, sp.LessThan):
        a, b = expr.args
        return _eval(a, ctx) <= _eval(b, ctx)
    if isinstance(expr, sp.Equality):
        a, b = expr.args
        return _eval(a, ctx) == _eval(b, ctx)
    raise UnsupportedExpression(
        f"Unsupported boolean expression {type(expr).__name__}: {expr!r}"
    )
