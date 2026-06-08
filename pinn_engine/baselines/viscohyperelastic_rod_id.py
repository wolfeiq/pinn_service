"""Combined hyper- + visco-elastic soft-rod identification.

Real silicone/rubber is **both** nonlinear (hyperelastic) **and** time-dependent
(viscoelastic). This module identifies a rod whose bending obeys a *nonlinear
viscoelastic* law — the quasi-linear-viscoelastic (QLV / nonlinear Standard
Linear Solid) form used for soft tissue and elastomers:

    M(t) = g_∞·M_e(κ) + q,   q̇ = −q/τ + (1−g_∞)·dM_e/dt
    M_e(κ) = a1·κ + a3·κ³                       (instantaneous nonlinear elastic)

``M_e`` is the instantaneous (glassy) nonlinear response; the reduced relaxation
``G(t)=g_∞+(1−g_∞)e^{−t/τ}`` scales it from the glassy modulus (``G(0)=1``) down
to the equilibrium fraction ``g_∞``. So under a step moment ``M_0`` the curvature
creeps from the instantaneous ``κ_g`` (where ``M_e(κ_g)=M_0``) to equilibrium
``κ_∞`` (where ``g_∞·M_e(κ_∞)=M_0``), with a local retardation time ``τ/g_∞``.

**Identification — a multi-level creep sweep separates the two physics:**
* the *instantaneous* curvatures ``κ_g(M_0)`` trace the **nonlinear elastic**
  curve → recover ``a1, a3`` (`M_0 = a1κ_g + a3κ_g³`);
* the *equilibrium* curvatures ``κ_∞(M_0)`` give the **relaxation strength**
  ``g_∞ = M_0 / M_e(κ_∞)``;
* the late-time decay rate gives ``τ``.

A *linear*-viscoelastic fit can't match the level-dependent instantaneous
response (nonlinearity), and a *nonlinear-elastic-only* fit can't match the
creep (time-dependence) — only the combined model fits both.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from scipy.integrate import solve_ivp


DEFAULT_VH = {"a1": 1.0, "a3": 0.6, "g_inf": 0.5, "tau": 0.8}


def _Me(k, a1, a3):
    return a1 * k + a3 * k ** 3


def _dMe(k, a1, a3):
    return a1 + 3 * a3 * k ** 2


def _kg_of(M0, a1, a3):
    roots = np.roots([a3, 0.0, a1, -M0])
    real = [r.real for r in roots if abs(r.imag) < 1e-9]
    return float(min(real, key=lambda r: abs(r - M0 / a1)))


def nonlinear_creep(a1, a3, g_inf, tau, M0, t):
    """Curvature creep ``κ(t)`` under a step moment ``M0`` for the QLV rod."""
    kg = _kg_of(M0, a1, a3)

    def f(tt, k):
        return [(M0 - g_inf * _Me(k[0], a1, a3)) / (tau * _dMe(k[0], a1, a3))]

    sol = solve_ivp(f, (t[0], t[-1]), [kg], t_eval=t, rtol=1e-9, atol=1e-11)
    return sol.y[0]


def generate_viscohyper_creep(*, params=None, M0_levels=None, t_end=10.0, n_t=160,
                              noise_std=2e-3, seed=0):
    """Multi-level creep sweep. Returns ``(data, truth)`` with one curvature
    curve per applied moment level."""
    p = dict(params or DEFAULT_VH)
    M0_levels = list(M0_levels if M0_levels is not None else [0.3, 0.6, 1.0, 1.4, 1.8])
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, t_end, n_t)
    shots = []
    for M0 in M0_levels:
        k = nonlinear_creep(p["a1"], p["a3"], p["g_inf"], p["tau"], M0, t)
        k = k + rng.normal(0, noise_std, k.shape)
        shots.append({"M0": float(M0), "t": t, "kappa": k})
    return {"shots": shots}, dict(p)


@dataclass
class ViscoHyperIDResult:
    a1: float
    a3: float
    g_inf: float
    tau: float
    linear_visco_rms: float    # RMS of a linear-elastic (a3=0) fit to κ_g(M0)

    def as_dict(self):
        return {"a1": self.a1, "a3": self.a3, "g_inf": self.g_inf, "tau": self.tau}


def recover_viscohyper(data) -> ViscoHyperIDResult:
    """Recover the nonlinear elastic coefficients + relaxation from multi-level
    creep, and report the residual of a linear-only instantaneous fit."""
    shots = data["shots"]
    M0s = np.array([sh["M0"] for sh in shots])
    kg = np.array([sh["kappa"][0] for sh in shots])      # instantaneous (t=0+)
    kinf = np.array([sh["kappa"][-1] for sh in shots])   # equilibrium (t→∞)
    # nonlinear elastic from instantaneous curvatures: M0 = a1 κg + a3 κg³
    A = np.stack([kg, kg ** 3], axis=1)
    (a1, a3), *_ = np.linalg.lstsq(A, M0s, rcond=None)
    # linear-only instantaneous fit (a3=0) residual — exposes the hyperelasticity
    a1_lin = float(np.dot(kg, M0s) / np.dot(kg, kg))
    lin_rms = float(np.sqrt(np.mean((a1_lin * kg - M0s) ** 2)))
    # relaxation strength: g_inf = M0 / M_e(κ∞), averaged over levels
    g_inf = float(np.mean(M0s / _Me(kinf, a1, a3)))
    # retardation time τ: the observable creep time constant is τ_c = τ/g_inf.
    # Estimate τ_c robustly as the 63% rise time (time to reach (1−1/e) of the
    # total creep κ_g→κ_∞), averaged over levels; then τ = g_inf·τ_c.
    tau_cs = []
    for sh in shots:
        t = np.asarray(sh["t"]); k = np.asarray(sh["kappa"])
        # lightly smooth to suppress the t=0 noise on κ_g and the tail on κ_∞
        kg_i = float(np.mean(k[:3])); kinf_i = float(np.mean(k[-5:]))
        if abs(kinf_i - kg_i) < 1e-3:
            continue
        target = kg_i + (1 - np.exp(-1.0)) * (kinf_i - kg_i)
        frac = (k - kg_i) / (kinf_i - kg_i)
        idx = np.argmax(frac >= (1 - np.exp(-1.0)))
        if idx > 0:
            # linear interpolate the crossing time
            f0, f1 = frac[idx - 1], frac[idx]; t0, t1 = t[idx - 1], t[idx]
            tc = t0 + (t1 - t0) * ((1 - np.exp(-1.0)) - f0) / max(f1 - f0, 1e-9)
            tau_cs.append(tc)
    tau_c = float(np.median(tau_cs)) if tau_cs else 1.0
    tau0 = max(1e-3, float(g_inf * tau_c))
    # refine τ with a 1-D fit of the full nonlinear creep ODE (a1,a3,g_inf fixed).
    from scipy.optimize import minimize_scalar

    def cost(tau):
        if tau <= 1e-4:
            return 1e9
        e = 0.0
        for sh in shots:
            t = np.asarray(sh["t"]); k = np.asarray(sh["kappa"])
            pred = nonlinear_creep(a1, a3, g_inf, tau, sh["M0"], t)
            e += float(np.mean((pred - k) ** 2))
        return e

    res = minimize_scalar(cost, bounds=(0.3 * tau0, 3.0 * tau0), method="bounded")
    tau = float(res.x)
    return ViscoHyperIDResult(a1=float(a1), a3=float(a3), g_inf=g_inf, tau=tau,
                              linear_visco_rms=lin_rms)
