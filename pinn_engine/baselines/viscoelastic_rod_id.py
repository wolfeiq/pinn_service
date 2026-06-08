"""Viscoelastic soft-rod identification (rate-dependent material).

Real soft-robot materials (silicone, rubber, tissue) are **viscoelastic** —
their response depends on the *rate* and *history* of deformation: they creep
under constant load, relax under constant strain, and dissipate energy
(hysteresis) under cyclic loading. Hyperelasticity captures nonlinearity;
viscoelasticity captures time-dependence. This module identifies a rod's
viscoelastic bending law from time-resolved deformation.

**Model — Standard Linear Solid (Zener).** An equilibrium spring ``E_∞`` in
parallel with a Maxwell branch (spring ``E_1``, dashpot, relaxation time ``τ``):

    M(t) = E_∞·κ(t) + q(t),     q̇ = −q/τ + E_1·κ̇

— the minimal model with a finite instantaneous modulus ``E_g = E_∞ + E_1``,
relaxation, and creep. Two independent self-experiments recover ``(E_∞, E_1, τ)``:

* **Creep** — hold a constant actuation moment ``M_0``; the curvature drifts
  ``κ(t) = M_0[J_∞ − (J_∞−J_g)e^{−t/τ_c}]`` from instantaneous ``M_0/E_g`` to
  equilibrium ``M_0/E_∞`` with retardation time ``τ_c = τ·E_g/E_∞``. Measured
  purely from shape over time.
* **DMA (oscillatory)** — drive ``M(t)=M_0 sin ωt`` over a frequency sweep; the
  curvature lags by ``δ(ω)``, giving the storage / loss moduli
  ``E'(ω)=E_∞+E_1 ω²τ²/(1+ω²τ²)``, ``E''(ω)=E_1 ωτ/(1+ω²τ²)``. The loss modulus
  **peaks at ωτ=1** — the viscoelastic fingerprint.

The two routes are cross-checked to agree.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit


DEFAULT_VE = {"E_inf": 1.0, "E1": 1.5, "tau": 0.5}   # equilibrium, relaxation strength, time


# ----------------------------------------------------------------- creep

def creep_curvature(E_inf, E1, tau, M0, t):
    """Closed-form SLS creep curvature ``κ(t)`` under a step moment ``M0``."""
    Eg = E_inf + E1
    Jinf, Jg = 1.0 / E_inf, 1.0 / Eg
    tau_c = tau * Eg / E_inf
    return M0 * (Jinf - (Jinf - Jg) * np.exp(-np.asarray(t) / tau_c))


def generate_creep_test(*, params=None, M0=0.5, t_end=8.0, n_t=120,
                        noise_std=2e-3, seed=0):
    p = dict(params or DEFAULT_VE)
    t = np.linspace(0.0, t_end, n_t)
    kap = creep_curvature(p["E_inf"], p["E1"], p["tau"], M0, t)
    kap = kap + np.random.default_rng(seed).normal(0, noise_std, kap.shape)
    return {"t": t, "kappa": kap, "M0": M0}, dict(p)


# ----------------------------------------------------------------- DMA (oscillatory)

def dma_curvature(E_inf, E1, tau, M0, omega, t):
    """Curvature time-series under ``M(t)=M0 sin(ωt)`` (transient + steady)."""
    Eg = E_inf + E1

    def f(tt, q):
        return [-q[0] * E_inf / (tau * Eg) + E1 * (M0 * omega * np.cos(omega * tt)) / Eg]

    sol = solve_ivp(f, (t[0], t[-1]), [0.0], t_eval=t, rtol=1e-9, atol=1e-11)
    q = sol.y[0]; M = M0 * np.sin(omega * t)
    return (M - q) / E_inf


def _storage_loss(omega, M0, t, kappa):
    """Extract E'(ω), E''(ω) from a steady-state curvature time-series."""
    mask = t > t[-1] - 6 * np.pi / omega
    A = np.stack([np.sin(omega * t[mask]), np.cos(omega * t[mask])], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, kappa[mask], rcond=None)
    Estar = M0 / complex(a, b)          # E* = M / κ (phasor)
    return Estar.real, abs(Estar.imag)


def generate_dma_sweep(*, params=None, M0=0.5, omegas=None, noise_std=2e-3, seed=0):
    p = dict(params or DEFAULT_VE)
    omegas = list(omegas if omegas is not None else [0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    rng = np.random.default_rng(seed)
    shots = []
    for w in omegas:
        t = np.linspace(0.0, 40 * np.pi / w, 4000)
        kap = dma_curvature(p["E_inf"], p["E1"], p["tau"], M0, w, t)
        kap = kap + rng.normal(0, noise_std, kap.shape)
        shots.append({"omega": w, "t": t, "kappa": kap})
    return {"M0": M0, "shots": shots}, dict(p)


# ----------------------------------------------------------------- recovery

@dataclass
class ViscoelasticIDResult:
    E_inf: float
    E1: float
    tau: float
    storage_loss: List = field(default_factory=list)   # [(omega, E', E'')]

    def as_dict(self):
        return {"E_inf": self.E_inf, "E1": self.E1, "tau": self.tau}


def recover_creep(data) -> ViscoelasticIDResult:
    """Recover ``(E_∞, E_1, τ)`` from a creep curvature curve."""
    t = np.asarray(data["t"], float); kap = np.asarray(data["kappa"], float); M0 = data["M0"]
    Jinf0 = kap[-1] / M0; Jg0 = max(kap[0] / M0, 1e-6 * Jinf0)
    tauc0 = max(t[-1] / 4, 1e-3)

    def model(tt, Jinf, Jg, tauc):
        return M0 * (Jinf - (Jinf - Jg) * np.exp(-tt / tauc))

    (Jinf, Jg, tauc), *_ = curve_fit(model, t, kap, p0=[Jinf0, Jg0, tauc0], maxfev=20000)
    E_inf = 1.0 / Jinf; Eg = 1.0 / Jg; E1 = Eg - E_inf
    tau = tauc * E_inf / Eg
    return ViscoelasticIDResult(E_inf=float(E_inf), E1=float(E1), tau=float(tau))


def recover_dma(data) -> ViscoelasticIDResult:
    """Recover ``(E_∞, E_1, τ)`` from a DMA storage/loss sweep."""
    M0 = data["M0"]; sl = []
    for sh in data["shots"]:
        Ep, Epp = _storage_loss(sh["omega"], M0, np.asarray(sh["t"]), np.asarray(sh["kappa"]))
        sl.append((sh["omega"], Ep, Epp))
    w = np.array([r[0] for r in sl]); Ep = np.array([r[1] for r in sl]); Epp = np.array([r[2] for r in sl])

    def model(w, E_inf, E1, tau):
        wt = w * tau
        return np.concatenate([E_inf + E1 * wt**2 / (1 + wt**2), E1 * wt / (1 + wt**2)])

    y = np.concatenate([Ep, Epp])
    (E_inf, E1, tau), *_ = curve_fit(model, w, y, p0=[Ep.min(), Ep.max() - Ep.min(), 1.0 / w[len(w)//2]],
                                     maxfev=20000)
    return ViscoelasticIDResult(E_inf=float(E_inf), E1=float(E1), tau=float(tau),
                                storage_loss=sl)
