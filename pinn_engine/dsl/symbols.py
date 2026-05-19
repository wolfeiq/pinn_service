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

    Example::

        t = Variable("t")
        x = Variable("x", depends_on=t)
        x.d        # → Derivative(x(t), t)
        x.dd       # → Derivative(x(t), t, 2)
    """

    def __new__(cls, name: str, depends_on: Optional[sp.Symbol] = None):
        obj = sp.Symbol.__new__(cls, name)
        obj._depends_on = depends_on
        return obj

    @property
    def depends_on(self) -> Optional[sp.Symbol]:
        return getattr(self, "_depends_on", None)

    @property
    def d(self) -> sp.Expr:
        """First derivative w.r.t. ``depends_on``."""
        if self.depends_on is None:
            raise AttributeError(
                f"Variable {self.name!r} has no `depends_on`; can't take derivative."
            )
        return sp.Derivative(self, self.depends_on)

    @property
    def dd(self) -> sp.Expr:
        """Second derivative w.r.t. ``depends_on``."""
        if self.depends_on is None:
            raise AttributeError(
                f"Variable {self.name!r} has no `depends_on`; can't take derivative."
            )
        return sp.Derivative(self, self.depends_on, 2)

    def deriv(self, order: int) -> sp.Expr:
        if self.depends_on is None:
            raise AttributeError(
                f"Variable {self.name!r} has no `depends_on`; can't take derivative."
            )
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
