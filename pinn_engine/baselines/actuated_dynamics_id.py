"""Actuated-dynamics self-calibration for a soft rod.

The actuation modules (`tendon_actuated_id`, `pneumatic_actuated_id`) are
quasi-static (PCC). This adds the *dynamic* response: when the actuation changes
in time the rod has inertia and damping, so its modal curvature obeys a
second-order system

    I_eff·κ̈ + c·κ̇ + EI·κ = M_act(t)

(``M_act`` = the known commanded actuation moment, ``I_eff`` effective modal
inertia, ``c`` damping, ``EI`` bending stiffness). Suddenly tensioning a tendon
and watching the rod **ring and settle** recovers all three at once — dynamic
self-calibration, no external excitation.

Because ``M_act(t)`` is known, the model is *linear* in ``(I_eff, c, EI)``:
measure the curvature response ``κ(t)``, take ``κ̇, κ̈`` (Savitzky-Golay), and
regress ``[κ̈, κ̇, κ]·(I_eff, c, EI)ᵀ = M_act`` — recovering stiffness, damping,
and inertia together. A step input is the cleanest (its transient rings at the
natural frequency ``ω_n=√(EI/I_eff)`` with damping ratio
``ζ=c/(2√(EI·I_eff))``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter


DEFAULT_AD = {"EI": 1.0, "c": 0.05, "I_eff": 0.02}


def simulate_actuated_dynamics(EI, c, I_eff, M_act, t):
    """Integrate ``I·κ̈ + c·κ̇ + EI·κ = M_act(t)`` from rest. ``M_act`` is a
    callable of time. Returns ``κ(t)``."""
    def f(tt, y):
        k, kd = y
        return [kd, (M_act(tt) - c * kd - EI * k) / I_eff]
    sol = solve_ivp(f, (t[0], t[-1]), [0.0, 0.0], t_eval=t,
                    rtol=1e-9, atol=1e-11, max_step=(t[1] - t[0]))
    return sol.y[0]


def generate_step_actuation(*, params=None, M0=0.5, t_end=6.0, n_t=1200,
                            noise_std=2e-3, seed=0):
    """Step-actuation ring-down: ``M_act = M0`` applied at t=0. Returns
    ``(data, truth)`` with the curvature response ``κ(t)`` and the input."""
    p = dict(params or DEFAULT_AD)
    t = np.linspace(0.0, t_end, n_t)
    k = simulate_actuated_dynamics(p["EI"], p["c"], p["I_eff"], lambda tt: M0, t)
    k = k + np.random.default_rng(seed).normal(0, noise_std, k.shape)
    return {"t": t, "kappa": k, "M_act": np.full_like(t, M0)}, dict(p)


@dataclass
class ActuatedDynamicsResult:
    EI: float
    c: float
    I_eff: float

    @property
    def omega_n(self):
        return float(np.sqrt(self.EI / self.I_eff))

    @property
    def zeta(self):
        return float(self.c / (2 * np.sqrt(self.EI * self.I_eff)))

    def as_dict(self):
        return {"EI": self.EI, "c": self.c, "I_eff": self.I_eff}


def recover_actuated_dynamics(data, sg_window=41, sg_poly=3) -> ActuatedDynamicsResult:
    """Recover ``(EI, c, I_eff)`` from the curvature response to a known
    actuation input, by linear regression of the equation of motion."""
    t = np.asarray(data["t"], float); kap = np.asarray(data["kappa"], float)
    M = np.asarray(data["M_act"], float)
    dt = float(t[1] - t[0])
    w = min(sg_window, len(t) - 1 if len(t) % 2 == 0 else len(t) - 2)
    w = max(w + 1 if w % 2 == 0 else w, sg_poly + 2)
    ks = savgol_filter(kap, w, sg_poly, deriv=0)
    kd = savgol_filter(kap, w, sg_poly, deriv=1, delta=dt)
    kdd = savgol_filter(kap, w, sg_poly, deriv=2, delta=dt)
    # crop the very edges (SG endpoints) and the t=0 step discontinuity
    c0 = max(w, len(t) // 50)
    sl = slice(c0, -w)
    A = np.stack([kdd[sl], kd[sl], ks[sl]], axis=1)
    (I_eff, c, EI), *_ = np.linalg.lstsq(A, M[sl], rcond=None)
    return ActuatedDynamicsResult(EI=float(EI), c=float(c), I_eff=float(I_eff))
