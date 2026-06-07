"""3-D spatial Cosserat (Simo-Reissner) rod — forward model + stiffness ID.

The full geometrically-exact spatial rod: the centerline ``r(s) ∈ ℝ³`` plus a
cross-section orientation ``R(s) ∈ SO(3)`` (carried as a quaternion). Unlike the
planar templates, this rod **bends in two planes, shears in two directions,
extends, and twists** — six independent strains and six stiffnesses:

    n_material = diag(EA, GA1, GA2) · Γ        (axial + 2 shear forces)
    m_material = diag(GJ, EI1, EI2) · K        (torsion + 2 bending moments)

with reference tangent ``e1`` (rod along x), translational strain
``Γ = Rᵀr' − e1`` (Γ1 axial stretch, Γ2/Γ3 shear) and material curvature ``K``
from ``R' = R[K]×`` (K1 torsion, K2/K3 bending).

**Forward model** — a tip-loaded cantilever (clamped at ``s=0``, tip wrench
``(P, Mt)`` at ``s=L``, no distributed load). Lab-frame balance:

    n_spatial' = 0                  → n_spatial = P  (constant)
    m_spatial' = −r' × n_spatial
    r' = R (C_n⁻¹ Rᵀ n_spatial + e1)
    R' = R [C_m⁻¹ Rᵀ m_spatial]×

solved as a BVP (clamped root pose, tip wrench) in
:func:`simulate_spatial_cosserat`. Verified against analytic limits:
pure axial → stretch ``1+P/EA``; pure twist → tip rotation ``Mt₁/GJ``;
transverse force → planar elastica.

**Inverse (the working solver).** A cantilever is *statically determinate*: the
internal force is the (known) tip force and the internal moment is

    m_spatial(s) = Mt + (r(L) − r(s)) × P

— both computable from the measured shape + load, **independent of the
stiffnesses**. So the constitutive laws are linear in the six stiffnesses and
recovered by per-component least squares from the measured strains/curvatures.
This is the 3-D generalisation of the force-from-motion trick that closes the
dynamic-rod gap (see ``cosserat_force_id``): expose each stiffness against a
data-derived force/moment rather than fighting an under-resolved residual.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy.integrate import solve_bvp
from scipy.signal import savgol_filter


# Reference (dimensionless) stiffness numbers — anisotropic soft rod, all six
# strains observable under a general tip wrench. Truth multipliers are 1.0.
SPCOS_REFS: Dict[str, float] = {
    "EA": 15.0, "GA1": 15.0, "GA2": 12.0,   # axial + two shear (·L²/EI_ref)
    "GJ": 0.8, "EI1": 1.0, "EI2": 0.8,       # torsion + two bending (/EI_ref)
}
SPCOS_P = (2.0, -3.0, 1.5)     # tip force (dimensionless)
SPCOS_MT = (1.0, 0.3, -0.2)    # tip moment (twist + small bending)

_E1 = np.array([1.0, 0.0, 0.0])
_NAMES = ["EA", "GA1", "GA2", "GJ", "EI1", "EI2"]


def _Rmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def simulate_spatial_cosserat(stiff: Dict[str, float],
                              P=SPCOS_P, Mt=SPCOS_MT, n: int = 201):
    """Solve the spatial-Cosserat cantilever BVP. Returns ``(s, r, q)`` with
    ``r`` shape ``(n, 3)`` and unit quaternions ``q`` shape ``(n, 4)``."""
    Cn = np.array([stiff["EA"], stiff["GA1"], stiff["GA2"]])
    Cm = np.array([stiff["GJ"], stiff["EI1"], stiff["EI2"]])
    Pv = np.array(P, float); Mtv = np.array(Mt, float)

    def ode(s, Y):
        out = np.zeros_like(Y)
        for k in range(Y.shape[1]):
            q = Y[3:7, k]; q = q / np.linalg.norm(q)
            R = _Rmat(q); msp = Y[7:10, k]
            rp = R @ ((R.T @ Pv) / Cn + _E1)
            K = (R.T @ msp) / Cm
            w, x, y, z = q; a, b, c = K
            qp = 0.5 * np.array([-(x * a + y * b + z * c), w * a + y * c - z * b,
                                 w * b - x * c + z * a, w * c + x * b - y * a])
            out[0:3, k] = rp; out[3:7, k] = qp; out[7:10, k] = -np.cross(rp, Pv)
        return out

    def bc(Ya, Yb):
        return np.concatenate([Ya[0:3], Ya[3:7] - np.array([1.0, 0, 0, 0]),
                               Yb[7:10] - Mtv])

    s = np.linspace(0, 1, n)
    Y0 = np.zeros((10, n)); Y0[3] = 1.0; Y0[7:10] = Mtv[:, None]
    sol = solve_bvp(ode, bc, s, Y0, max_nodes=60000, tol=1e-8)
    if not sol.success:
        raise RuntimeError(f"spatial Cosserat BVP failed: {sol.message}")
    Y = sol.sol(s)
    q = Y[3:7].T.copy(); q /= np.linalg.norm(q, axis=1, keepdims=True)
    return s, Y[0:3].T.copy(), q


def generate_spatial_cosserat(
    *, refs: Dict[str, float] = None, P=SPCOS_P, Mt=SPCOS_MT,
    n_s: int = 121, pos_noise_std: float = 1e-3, quat_noise_std: float = 3e-3,
    seed: int = 0, **unit_overrides,
) -> Tuple[Dict, Dict[str, float]]:
    """Synthetic 3-D rod measurement: noisy centerline ``r`` + orientation ``q``.

    Stiffness truth = reference × unit-multiplier (default all 1.0); pass e.g.
    ``EA_unit=1.1`` to perturb. Returns ``(data, truth)`` where ``data`` holds
    ``s, r, q, P, Mt, refs`` and ``truth`` the six ``*_unit`` multipliers.
    """
    refs = dict(refs or SPCOS_REFS)
    truth = {f"{k}_unit": float(unit_overrides.get(f"{k}_unit", 1.0)) for k in _NAMES}
    stiff = {k: refs[k] * truth[f"{k}_unit"] for k in _NAMES}
    s, r, q = simulate_spatial_cosserat(stiff, P, Mt, n=n_s)
    rng = np.random.default_rng(seed)
    r = r + rng.normal(0, pos_noise_std, r.shape)
    q = q + rng.normal(0, quat_noise_std, q.shape)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    data = {"s": s, "r": r, "q": q, "P": np.array(P, float),
            "Mt": np.array(Mt, float), "refs": refs}
    return data, truth


@dataclass
class SpatialCosseratIDResult:
    units: Dict[str, float]            # {EA_unit, GA1_unit, ...}
    stiffness: Dict[str, float]        # absolute recovered stiffnesses

    def as_dict(self) -> Dict[str, float]:
        return dict(self.units)


def recover_spatial_stiffness(data: Dict, sg_window: int = 21, sg_poly: int = 3,
                              edge_crop: int = 4) -> SpatialCosseratIDResult:
    """Recover all six stiffnesses from measured 3-D rod shape + orientation.

    Uses the statically-determinate internal force/moment (from the measured
    shape + known tip wrench) and per-component least squares against the
    measured strains/curvatures. Savitzky-Golay derivatives for noise
    robustness.
    """
    s = np.asarray(data["s"], float); r = np.asarray(data["r"], float)
    q = np.asarray(data["q"], float); P = np.asarray(data["P"], float)
    Mt = np.asarray(data["Mt"], float); refs = data["refs"]
    n = len(s); ds = float(s[1] - s[0])
    w = min(sg_window, n - 1 if n % 2 == 0 else n - 2)   # keep window < n
    if w % 2 == 0:
        w += 1
    w = max(w, sg_poly + 2)

    rs = savgol_filter(r, w, sg_poly, axis=0)
    rp = savgol_filter(r, w, sg_poly, deriv=1, delta=ds, axis=0)
    qs = savgol_filter(q, w, sg_poly, axis=0)
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    qp = savgol_filter(q, w, sg_poly, deriv=1, delta=ds, axis=0)
    rL = rs[-1]

    G = np.zeros((n, 3)); K = np.zeros((n, 3))
    nmat = np.zeros((n, 3)); mmat = np.zeros((n, 3))
    for k in range(n):
        R = _Rmat(qs[k])
        G[k] = R.T @ rp[k] - _E1
        qc = np.array([qs[k][0], -qs[k][1], -qs[k][2], -qs[k][3]])
        K[k] = 2.0 * _qmul(qc, qp[k])[1:]
        msp = Mt + np.cross(rL - rs[k], P)        # statically-determinate moment
        nmat[k] = R.T @ P                          # internal force = tip force
        mmat[k] = R.T @ msp
    cr = slice(edge_crop, -edge_crop)

    def slope(force, strain):
        f = force[cr]; e = strain[cr]
        return float(np.dot(f, e) / np.dot(e, e))

    stiff = {
        "EA": slope(nmat[:, 0], G[:, 0]), "GA1": slope(nmat[:, 1], G[:, 1]),
        "GA2": slope(nmat[:, 2], G[:, 2]), "GJ": slope(mmat[:, 0], K[:, 0]),
        "EI1": slope(mmat[:, 1], K[:, 1]), "EI2": slope(mmat[:, 2], K[:, 2]),
    }
    units = {f"{k}_unit": stiff[k] / refs[k] for k in _NAMES}
    return SpatialCosseratIDResult(units=units, stiffness=stiff)
