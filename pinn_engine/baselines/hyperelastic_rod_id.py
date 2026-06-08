"""Hyperelastic soft-rod constitutive identification.

Real soft-robot materials (silicone, rubber, biological tissue) are
**hyperelastic** — their stress–strain law is nonlinear, typically
strain-stiffening at large deformation. A linear-elastic rod (constant EI, EA)
is only the small-strain limit. This module identifies the *nonlinear*
constitutive curve of a soft rod from a load sweep.

We model the symmetric leading nonlinearity (the Taylor expansion of any
symmetric hyperelastic bending / stretching response):

    bending:  M(κ) = a1·κ + a3·κ³           (a1 = small-strain EI; a3 stiffening)
    axial:    N(ε) = b1·ε + b3·ε³           (b1 = small-strain EA; b3 stiffening)

`a3 > 0` is strain-stiffening (fiber-reinforced tissue, filled rubber);
`a3 < 0` strain-softening. **Identification.** A tip moment makes the curvature
constant and statically-determinate (`M(s)=M_tip`); sweeping the load from small
to large strain and measuring the curvature at each gives `(M, κ)` pairs that fit
the cubic by linear least squares on the `[κ, κ³]` basis — recovering `a1, a3`
(and `b1, b3` from an axial-force sweep). A *linear-only* fit (`a1` alone) leaves
a large systematic residual at high load — the signature of hyperelasticity,
which this method both detects and quantifies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np


# Default true constitutive (dimensionless): linear part + strong stiffening.
DEFAULT_BEND = {"a1": 1.0, "a3": 0.6}     # M = a1 κ + a3 κ³
DEFAULT_AXIAL = {"b1": 15.0, "b3": 25.0}  # N = b1 ε + b3 ε³


def _solve_cubic_strain(c1: float, c3: float, load: float) -> float:
    """Real root of ``c1·x + c3·x³ = load`` (monotone for c1,c3 ≥ 0)."""
    if abs(c3) < 1e-14:
        return load / c1
    roots = np.roots([c3, 0.0, c1, -load])
    real = [r.real for r in roots if abs(r.imag) < 1e-9]
    # pick the root closest to the linear estimate (physical branch)
    lin = load / c1
    return float(min(real, key=lambda r: abs(r - lin)))


def simulate_hyperelastic_bending(a1, a3, M_tip, ns: int = 81):
    """Constant-curvature arc under a tip moment for a hyperelastic rod.
    Returns ``(s, theta)`` with ``theta(s) = κ·s`` (planar tangent angle)."""
    kappa = _solve_cubic_strain(a1, a3, M_tip)
    s = np.linspace(0, 1, ns)
    return s, kappa * s, kappa


def generate_hyperelastic_sweep(*, bend=None, axial=None,
                                M_loads=None, N_loads=None,
                                ang_noise_std=3e-3, strain_noise_std=2e-3,
                                ns=81, seed=0):
    """Synthetic load sweep: tangent-angle profiles under a range of tip moments
    (bending) and uniform stretches under tip axial forces. Returns
    ``(data, truth)``."""
    bend = dict(bend or DEFAULT_BEND); axial = dict(axial or DEFAULT_AXIAL)
    M_loads = list(M_loads if M_loads is not None else np.linspace(0.1, 1.6, 10))
    N_loads = list(N_loads if N_loads is not None else np.linspace(0.5, 6.0, 10))
    rng = np.random.default_rng(seed)
    bend_shots = []
    for M in M_loads:
        s, th, kap = simulate_hyperelastic_bending(bend["a1"], bend["a3"], M, ns=ns)
        th = th + rng.normal(0, ang_noise_std, th.shape)
        bend_shots.append({"M_tip": float(M), "s": s, "theta": th})
    axial_shots = []
    for N in N_loads:
        eps = _solve_cubic_strain(axial["b1"], axial["b3"], N)
        eps_meas = eps + rng.normal(0, strain_noise_std)
        axial_shots.append({"N_tip": float(N), "eps": float(eps_meas)})
    truth = {"a1": bend["a1"], "a3": bend["a3"], "b1": axial["b1"], "b3": axial["b3"]}
    return {"bend_shots": bend_shots, "axial_shots": axial_shots}, truth


@dataclass
class HyperelasticIDResult:
    a1: float
    a3: float
    b1: float
    b3: float
    linear_only_rms: float       # RMS moment residual of a linear-only fit (M=a1 κ)
    nonlinear_rms: float         # RMS residual of the cubic fit

    def as_dict(self):
        return {"a1": self.a1, "a3": self.a3, "b1": self.b1, "b3": self.b3}


def recover_hyperelastic(data) -> HyperelasticIDResult:
    """Recover the cubic constitutive coefficients from the load sweep, and report
    the linear-only residual that exposes the hyperelastic nonlinearity."""
    # bending: per shot, curvature κ = slope of measured theta vs s.
    kaps = []; Ms = []
    for shot in data["bend_shots"]:
        s = shot["s"]; th = shot["theta"]
        kap = float(np.polyfit(s, th, 1)[0])     # slope = curvature
        kaps.append(kap); Ms.append(shot["M_tip"])
    kaps = np.array(kaps); Ms = np.array(Ms)
    A = np.stack([kaps, kaps ** 3], axis=1)
    (a1, a3), *_ = np.linalg.lstsq(A, Ms, rcond=None)
    nl_rms = float(np.sqrt(np.mean((A @ np.array([a1, a3]) - Ms) ** 2)))
    a1_lin = float(np.dot(kaps, Ms) / np.dot(kaps, kaps))
    lin_rms = float(np.sqrt(np.mean((a1_lin * kaps - Ms) ** 2)))

    eps = np.array([sh["eps"] for sh in data["axial_shots"]])
    Ns = np.array([sh["N_tip"] for sh in data["axial_shots"]])
    Ab = np.stack([eps, eps ** 3], axis=1)
    (b1, b3), *_ = np.linalg.lstsq(Ab, Ns, rcond=None)
    return HyperelasticIDResult(a1=float(a1), a3=float(a3), b1=float(b1), b3=float(b3),
                                linear_only_rms=lin_rms, nonlinear_rms=nl_rms)
