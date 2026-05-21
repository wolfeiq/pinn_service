"""Symbolic primitives for the equation DSL.

A user writes their inverse problem with five primitives:

* ``Variable(name)`` — a state variable the network will represent.
* ``Variable(name, depends_on=t)`` — a state variable that depends on the
  independent variable ``t`` (so ``x.d`` means ``dx/dt``).
* ``Parameter(name, value=v)`` — a known physical constant.
* ``Unknown(name, bounds=(lo, hi))`` — a parameter to be discovered.
* ``Sensor(name, observes=x, noise_std=σ)`` — a measurement of a state variable.

Each primitive is a thin wrapper around :class:`sympy.Symbol` so that
expressions like ``m*x.dd + c*x.d + k*x`` are valid sympy and can be
introspected, hashed, and differentiated symbolically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Any
import sympy as sp


class _SymbolBase(sp.Symbol):
    """Base class for DSL symbols; injects ``.metadata`` onto a sympy Symbol."""

    def __new__(cls, name: str, **kwargs):
        # sympy.Symbol uses __new__ extensively; we ride along.
        obj = sp.Symbol.__new__(cls, name)
        return obj


class Variable(_SymbolBase):
    """A state variable. If ``depends_on`` is set, derivatives are taken w.r.t. it.

    ``depends_on`` can be a single :class:`Variable` (ODE case — ``u`` depends
    on ``t``) or a tuple (PDE case — ``u`` depends on ``(s, t)``).

    Example::

        # ODE
        t = Variable("t")
        x = Variable("x", depends_on=t)
        x.d        # → ∂x/∂t   (first depends_on)
        x.dd       # → ∂²x/∂t²

        # PDE
        s = Variable("s"); t = Variable("t")
        u = Variable("u", depends_on=(s, t))
        u.diff(t)         # → ∂u/∂t
        u.diff(s, 2)      # → ∂²u/∂s²
        u.d               # → ∂u/∂s  (first in the tuple)
    """

    def __new__(cls, name: str, depends_on=None):
        obj = sp.Symbol.__new__(cls, name)
        # Normalize to tuple; None → empty.
        if depends_on is None:
            obj._depends_on = ()
        elif isinstance(depends_on, (list, tuple)):
            obj._depends_on = tuple(depends_on)
        else:
            obj._depends_on = (depends_on,)
        return obj

    @property
    def depends_on(self):
        """First dependency (back-compat). Use ``depends_on_all`` for the tuple."""
        deps = getattr(self, "_depends_on", ())
        return deps[0] if deps else None

    @property
    def depends_on_all(self) -> tuple:
        """All dependencies as a tuple. Empty if the variable is independent."""
        return getattr(self, "_depends_on", ())

    def _check_deps(self):
        if not self.depends_on_all:
            raise AttributeError(
                f"Variable {self.name!r} has no `depends_on`; can't take derivative."
            )

    @property
    def d(self) -> sp.Expr:
        """First derivative w.r.t. the *first* declared dependency."""
        self._check_deps()
        return sp.Derivative(self, self.depends_on)

    @property
    def dd(self) -> sp.Expr:
        """Second derivative w.r.t. the *first* declared dependency."""
        self._check_deps()
        return sp.Derivative(self, self.depends_on, 2)

    def diff(self, var: sp.Symbol, order: int = 1) -> sp.Expr:
        """Partial derivative w.r.t. ``var`` of the given ``order``.

        ``var`` must be one of this variable's declared ``depends_on``.
        """
        self._check_deps()
        if var not in self.depends_on_all:
            raise AttributeError(
                f"Variable {self.name!r} doesn't depend on {var!r}; "
                f"declared dependencies are {self.depends_on_all!r}"
            )
        return sp.Derivative(self, var, order)

    def deriv(self, order: int) -> sp.Expr:
        """Repeated derivative w.r.t. the first dependency (ODE case)."""
        self._check_deps()
        return sp.Derivative(self, self.depends_on, order)


class Parameter(_SymbolBase):
    """A *known* physical constant (its numerical value is fixed)."""

    def __new__(cls, name: str, value: float):
        obj = sp.Symbol.__new__(cls, name)
        obj._value = float(value)
        return obj

    @property
    def value(self) -> float:
        return self._value


class Unknown(_SymbolBase):
    """A parameter to be discovered. Bounds are required for AutoML and pre-flight.

    The ``bounds`` are used three ways:
    * to initialize the parameter (midpoint by default);
    * to define the PINA ``unknown_parameter_domain``;
    * to detect divergence (the ``ParamDivergenceGuard`` AutoML callback).
    """

    def __new__(
        cls,
        name: str,
        bounds: Tuple[float, float],
        init: Any = "midpoint",
    ):
        if bounds[0] >= bounds[1]:
            raise ValueError(
                f"Unknown {name!r}: bounds must be (lo, hi) with lo < hi, got {bounds!r}"
            )
        obj = sp.Symbol.__new__(cls, name)
        obj._bounds = (float(bounds[0]), float(bounds[1]))
        obj._init = init
        return obj

    @property
    def bounds(self) -> Tuple[float, float]:
        return self._bounds

    @property
    def init_value(self) -> float:
        lo, hi = self._bounds
        if self._init == "midpoint":
            return 0.5 * (lo + hi)
        if self._init == "random":
            import random
            return random.uniform(lo, hi)
        return float(self._init)


@dataclass(frozen=True)
class Sensor:
    """A measurement channel.

    Attributes:
        name: identifier used to look up data in the ``data`` dict.
        observes: a :class:`Variable` (or a sympy expression of state) being measured.
        noise_std: assumed Gaussian noise standard deviation (used for data loss weighting).
    """

    name: str
    observes: Any
    noise_std: float = 0.0

    def __post_init__(self):
        if self.noise_std < 0:
            raise ValueError(f"Sensor {self.name!r}: noise_std must be >= 0")
