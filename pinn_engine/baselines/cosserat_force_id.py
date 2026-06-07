"""Force-from-motion identification for the dynamic planar Cosserat rod.

A direct, physics-informed estimator that recovers the bending / shear / axial
stiffness of a dynamic Cosserat rod from its measured motion alone — the
companion solver to the ``dynamic_cosserat`` PINN template, and the method that
*closes* the shear/axial identifiability gap the PINN training hits.

**The idea.** The internal force ``N(s,t)`` is *kinematic*: linear-momentum
balance integrated from the free tip (where ``N=0``) gives

    Nx(s,t) = −∫ₛᴸ (x_tt + c·x_t) ds'        (ρA = 1, dimensionless)
    Ny(s,t) = −∫ₛᴸ (y_tt + g + c·y_t) ds'

so ``N`` depends only on the measured accelerations + known ``g, c`` — **not on
the unknown stiffnesses**. With ``N`` in hand, the constitutive law is *linear*
in the stiffnesses and recovers them by least squares using only first spatial
derivatives (the strains) — no 2nd-derivative "explain-away":

    [Nx; Ny] = EA·(ν−1)·[cosθ; sinθ] + GA·η·[−sinθ; cosθ]
    j·θ_tt − (x_s·Ny − y_s·Nx) = EI·θ_ss

This is why a direct least squares nails all three stiffnesses where the
collocation PINN — which must match the force *divergence* against
under-resolved translational 2nd derivatives — stalls. See
``docs/dynamic_cosserat_experiments.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.signal import savgol_filter


@dataclass
class CosseratForceIDResult:
    EI_unit: float
    GA_unit: float
    EA_unit: float

    def as_dict(self) -> Dict[str, float]:
        return {"EI_unit": self.EI_unit, "GA_unit": self.GA_unit, "EA_unit": self.EA_unit}


def _odd(w: int) -> int:
    return w + 1 if w % 2 == 0 else w


def _sg(F: np.ndarray, axis: int, delta: float, deriv: int,
        window: int, poly: int) -> np.ndarray:
    """Savitzky-Golay smoothed derivative along ``axis`` (deriv=0 → just smooth).

    Doing the smoothing and differentiation in one local-polynomial fit is far
    more noise-robust than ``np.gradient`` of a separately-smoothed field — it
    is what makes the noise-amplified spatial 2nd derivatives (θ_ss) usable.
    """
    n = F.shape[axis]
    w = _odd(min(window, n if n % 2 else n - 1))
    if w <= poly:
        # too short to fit; fall back to finite differences
        out = F
        for _ in range(deriv):
            out = np.gradient(out, delta, axis=axis)
        return out
    return savgol_filter(F, window_length=w, polyorder=poly, deriv=deriv,
                         delta=delta, axis=axis)


def recover_stiffness_from_motion(
    s: np.ndarray, t: np.ndarray,
    X: np.ndarray, Y: np.ndarray, TH: np.ndarray,
    *,
    EI0: float = 1.0, GA0: float = 15.0, EA0: float = 15.0,
    g: float = 3.0, c: float = 0.4, j: float = 0.01,
    sg_window_t: int = 25, sg_window_s: int = 9, sg_poly: int = 3,
    edge_crop: int = 3,
) -> CosseratForceIDResult:
    """Recover dimensionless ``EI_unit, GA_unit, EA_unit`` from measured rod
    motion on an ``(n_s, n_t)`` grid.

    ``X, Y, TH`` have shape ``(n_s, n_t)`` (rows = arclength nodes ``s``,
    columns = time samples ``t``). All derivatives are taken with Savitzky-Golay
    local-polynomial fits (smoothing + differentiation in one step) — a time
    window for the accelerations and a space window for the strains and
    curvature — and least squares over all interior grid points averages the
    rest. Defaults are tuned for ~1e-3 position / 5e-3 angle noise.
    """
    X = np.asarray(X, float); Y = np.asarray(Y, float); TH = np.asarray(TH, float)
    s = np.asarray(s, float); t = np.asarray(t, float)
    ds = float(s[1] - s[0]); dt = float(t[1] - t[0])
    AX = 0  # space axis
    TT = 1  # time axis

    # Time derivatives (accelerations) — SG along time.
    Xt = _sg(X, TT, dt, 1, sg_window_t, sg_poly); Xtt = _sg(X, TT, dt, 2, sg_window_t, sg_poly)
    Yt = _sg(Y, TT, dt, 1, sg_window_t, sg_poly); Ytt = _sg(Y, TT, dt, 2, sg_window_t, sg_poly)
    THtt = _sg(TH, TT, dt, 2, sg_window_t, sg_poly)
    # Spatial fields/derivatives — SG along arclength (smooth in time first).
    Xtm = _sg(X, TT, dt, 0, sg_window_t, sg_poly)
    Ytm = _sg(Y, TT, dt, 0, sg_window_t, sg_poly)
    THs_t = _sg(TH, TT, dt, 0, sg_window_t, sg_poly); THs_t = _sg(THs_t, AX, ds, 0, sg_window_s, sg_poly)
    Xsp = _sg(Xtm, AX, ds, 1, sg_window_s, sg_poly)
    Ysp = _sg(Ytm, AX, ds, 1, sg_window_s, sg_poly)
    THss = _sg(THs_t, AX, ds, 2, sg_window_s, sg_poly)

    # Internal force from inertia, integrated from the free tip (N(L)=0).
    ax = Xtt + c * Xt              # = ∂Nx/∂s
    ay = Ytt + g + c * Yt          # = ∂Ny/∂s
    def integ_from_tip(f):
        out = np.zeros_like(f)
        for i in range(f.shape[0] - 2, -1, -1):
            out[i] = out[i + 1] + 0.5 * (f[i] + f[i + 1]) * ds
        return -out
    Nx = integ_from_tip(ax); Ny = integ_from_tip(ay)

    cropf = (slice(edge_crop, -edge_crop), slice(edge_crop, -edge_crop))
    nu_1 = (Xsp * np.cos(THs_t) + Ysp * np.sin(THs_t) - 1)
    eta = (-Xsp * np.sin(THs_t) + Ysp * np.cos(THs_t))

    # Least squares for (ea, ga): [Nx; Ny] = ea·A_ea + ga·A_ga.
    A_ea = np.concatenate([(nu_1 * np.cos(THs_t))[cropf].ravel(),
                           (nu_1 * np.sin(THs_t))[cropf].ravel()])
    A_ga = np.concatenate([(-eta * np.sin(THs_t))[cropf].ravel(),
                           (eta * np.cos(THs_t))[cropf].ravel()])
    b = np.concatenate([Nx[cropf].ravel(), Ny[cropf].ravel()])
    M = np.stack([A_ea, A_ga], axis=1)
    (ea, ga), *_ = np.linalg.lstsq(M, b, rcond=None)

    # Least squares for ei: j·θ_tt − (x_s·Ny − y_s·Nx) = ei·θ_ss.
    lhs = (j * THtt - (Xsp * Ny - Ysp * Nx))[cropf].ravel()
    base = THss[cropf].ravel()
    ei = float(np.dot(base, lhs) / np.dot(base, base))

    return CosseratForceIDResult(EI_unit=ei / EI0, GA_unit=ga / GA0, EA_unit=ea / EA0)


def recover_from_template_data(data: Dict, n_s: int, n_t: int,
                               **kwargs) -> CosseratForceIDResult:
    """Convenience wrapper: recover stiffness from a ``generate_dynamic_cosserat``
    ``data`` dict (flattened ``(s,t)`` grid, row-major in ``s``)."""
    s_inp, xo = data["x_meas"]
    _, yo = data["y_meas"]
    _, to = data["theta_meas"]
    s_grid = s_inp[:, 0].reshape(n_s, n_t)[:, 0]
    t_grid = s_inp[:, 1].reshape(n_s, n_t)[0, :]
    X = xo.reshape(n_s, n_t); Y = yo.reshape(n_s, n_t); TH = to.reshape(n_s, n_t)
    return recover_stiffness_from_motion(s_grid, t_grid, X, Y, TH, **kwargs)
