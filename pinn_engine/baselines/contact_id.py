"""Proprioceptive contact estimation for a soft rod.

A soft robot interacting with its environment feels contact through its own
deformation. A point contact applies a force at one arclength — which makes the
internal **shear jump** there, so the internal moment ``m(s) = EI·κ(s)`` has a
*slope kink* at the contact. From the measured shape alone (curvature profile),
we recover **where** the rod is touching and **how hard** — whole-body tactile
sensing with no force/torque sensor.

Planar cantilever, clamped at the base, with a known tip load ``P`` (or
actuation) and an unknown normal contact force ``F_c`` at unknown arclength
``s_c``. Geometrically-exact elastica: ``m = EI·θ'``, ``m'(s) = −cosθ·n_y`` with
the transverse internal force ``n_y`` jumping by ``F_c`` across ``s_c``. Hence
``κ'(s)`` (the curvature slope) jumps by ``−cosθ(s_c)·F_c/EI`` at ``s_c``:

    s_c  = the breakpoint of the (continuous, piecewise-linear) curvature κ(s)
    F_c  = −EI · (κ'-slope jump) / cosθ(s_c)

Recovered by a changepoint fit of κ(s). Knows nothing about the obstacle — it
reads the contact straight off the shape.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def _shoot(EI, P, Fc, sc, m0, ns=201):
    def seg(s, Y, ny):
        x, y, th, m = Y
        return [np.cos(th), np.sin(th), m / EI, -np.cos(th) * ny]
    s1 = np.linspace(0, sc, max(20, int(ns * sc)))
    a = solve_ivp(lambda s, Y: seg(s, Y, P - Fc), (0, sc), [0, 0, 0, m0],
                  t_eval=s1, rtol=1e-9, atol=1e-11)
    s2 = np.linspace(sc, 1, max(20, int(ns * (1 - sc))))
    b = solve_ivp(lambda s, Y: seg(s, Y, P), (sc, 1), a.y[:, -1],
                  t_eval=s2, rtol=1e-9, atol=1e-11)
    s = np.concatenate([s1, s2[1:]])
    Y = np.concatenate([a.y, b.y[:, 1:]], axis=1)
    return s, Y


def simulate_contact(EI=1.0, P=1.0, Fc=2.0, sc=0.5, ns=201):
    """Planar cantilever with tip load ``P`` and point contact ``Fc`` at ``sc``.
    Returns ``(s, x, y, theta)``."""
    f = lambda m0: _shoot(EI, P, Fc, sc, m0, ns)[1][3, -1]   # residual m(L)=0
    m0 = brentq(f, -50, 50)
    s, Y = _shoot(EI, P, Fc, sc, m0, ns)
    return s, Y[0], Y[1], Y[2]


def generate_contact_scenario(*, EI=1.0, P=1.0, Fc=2.0, sc=0.5,
                              ang_noise_std=3e-3, ns=201, seed=0):
    """Synthetic contact measurement: tangent-angle profile of a contacting rod.
    Returns ``(data, truth)``."""
    s, x, y, th = simulate_contact(EI, P, Fc, sc, ns)
    rng = np.random.default_rng(seed)
    th = th + rng.normal(0, ang_noise_std, th.shape)
    data = {"s": s, "theta": th, "EI": EI, "P": P}
    return data, {"sc": sc, "Fc": Fc}


@dataclass
class ContactIDResult:
    sc: float       # contact arclength
    Fc: float       # contact force magnitude

    def as_dict(self):
        return {"sc": self.sc, "Fc": self.Fc}


def recover_contact(data, sc_grid=None) -> ContactIDResult:
    """Recover contact location + force from the measured curvature kink."""
    s = np.asarray(data["s"], float); theta = np.asarray(data["theta"], float)
    EI = float(data["EI"])
    kap = np.gradient(theta, s)
    if sc_grid is None:
        sc_grid = np.linspace(0.12, 0.88, 153)
    best = None
    for sc_try in sc_grid:
        # continuous piecewise-linear: κ = a + c1·min(s,sc) + c2·max(s−sc,0)
        X = np.stack([np.ones_like(s), np.minimum(s, sc_try),
                      np.maximum(s - sc_try, 0)], axis=1)
        coef, *_ = np.linalg.lstsq(X, kap, rcond=None)
        r = float(np.sum((X @ coef - kap) ** 2))
        if best is None or r < best[0]:
            best = (r, sc_try, coef)
    _, sc_hat, coef = best
    slope_jump = coef[2] - coef[1]               # κ'-slope after − before
    th_c = float(np.interp(sc_hat, s, theta))
    Fc_hat = -EI * slope_jump / np.cos(th_c)
    return ContactIDResult(sc=float(sc_hat), Fc=float(Fc_hat))
