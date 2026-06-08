"""Dynamic 3-D spatial Cosserat rod — forward model + stiffness identification.

The capstone of the soft-robot rod suite: the full geometrically-exact spatial
Cosserat rod **with inertia** — ``r(s,t) ∈ ℝ³`` and ``R(s,t) ∈ SO(3)`` evolving
in time, including 3-D rotational dynamics with the gyroscopic term. It unifies
the dynamic (time-domain) and 3-D (spatial) rods.

**Forward model** (`simulate_dynamic_spatial_cosserat`) — a clamped rod,
optionally pre-twisted, released under gravity; method of lines (discrete
elastic rod) in ``s`` + RK45 in ``t``. Per node: linear momentum
``ρA r_tt = ∂n_sp/∂s + ρA g − c r_t`` and angular momentum (body frame)

    J_ρ Ω_t = Rᵀ(∂m_sp/∂s + r'×n_sp) − Ω×(J_ρ Ω)

with the quaternion kinematics ``q_t = ½ q⊗(0,Ω)``. Verified: undamped energy
conserved to ~4e-4; reproduces the planar dynamic solver to ~1e-4 for an
in-plane isotropic case; off-axis / pre-twisted loading produces genuine
out-of-plane motion and dynamic torsion.

**Inverse** (`recover_dynamic_spatial_stiffness`) — in dynamics, *both* the
internal force and moment are kinematic:

    n_sp(s,t) = −∫ₛᴸ ρA (r_tt − g + c r_t) ds'               (linear momentum)
    m_sp(s,t) = −∫ₛᴸ (dH/dt − r'×n_sp) ds',  H = R J_ρ Ω      (angular momentum)

both derivable from the measured motion + known inertia, **independent of the
unknown stiffnesses**. The constitutive laws ``n_mat = C_n Γ``, ``m_mat = C_m K``
are then linear in the six stiffnesses and recovered by per-component least
squares (Savitzky-Golay derivatives in s and t). A pre-twist excites torsion so
GJ is well-conditioned. This is the force/moment-from-motion identifier extended
to the full dynamic 3-D rod.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter

from pinn_engine.baselines.spatial_cosserat_id import _Rmat, _qmul, _E1

# Defaults: anisotropic soft rod, off-axis gravity + pre-twist so all six
# strains (axial, 2 shear, torsion, 2 bending) are dynamically excited.
DYN3D_REFS: Dict[str, float] = {
    "EA": 15.0, "GA1": 15.0, "GA2": 12.0, "GJ": 0.8, "EI1": 1.0, "EI2": 0.8,
}
DYN3D_JRHO = (0.02, 0.01, 0.01)   # rotational inertia per length (polar, 2 transverse)
DYN3D_G = (0.0, -3.0, 1.2)        # off-axis gravity (excites both bending planes)
DYN3D_C = 0.2                     # translational viscous damping
DYN3D_TWIST0 = 1.5                # initial pre-twist rate about the rod axis
DYN3D_TEND = 2.5
_NAMES = ["EA", "GA1", "GA2", "GJ", "EI1", "EI2"]


def simulate_dynamic_spatial_cosserat(stiff, Jrho=DYN3D_JRHO, gvec=DYN3D_G,
                                      c=DYN3D_C, twist0=DYN3D_TWIST0, N=40,
                                      t_end=DYN3D_TEND, n_t=121, rhoA=1.0):
    """Method-of-lines forward sim. Returns ``(s, t, r, q)`` with ``r`` shape
    ``(N+1, n_t, 3)`` and unit quaternions ``q`` shape ``(N+1, n_t, 4)``."""
    Cn = np.array([stiff["EA"], stiff["GA1"], stiff["GA2"]])
    Cm = np.array([stiff["GJ"], stiff["EI1"], stiff["EI2"]])
    Jr = np.array(Jrho, float); g = np.array(gvec, float)
    ds = 1.0 / N; nn = N + 1
    r0 = np.zeros((nn, 3)); r0[:, 0] = np.linspace(0, 1, nn)
    q0 = np.zeros((nn, 4))
    for i in range(nn):
        a = 0.5 * twist0 * (i * ds)
        q0[i] = [np.cos(a), np.sin(a), 0, 0]
    Y0 = np.concatenate([r0.ravel(), q0.ravel(), np.zeros(nn * 3), np.zeros(nn * 3)])

    def deriv(t, Y):
        o = 0
        r = Y[o:o + nn * 3].reshape(nn, 3); o += nn * 3
        q = Y[o:o + nn * 4].reshape(nn, 4); o += nn * 4
        v = Y[o:o + nn * 3].reshape(nn, 3); o += nn * 3
        om = Y[o:o + nn * 3].reshape(nn, 3)
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        nsp = np.zeros((N, 3)); msp = np.zeros((N, 3)); tang = np.zeros((N, 3))
        for e in range(N):
            qm = q[e] + q[e + 1]; qm /= np.linalg.norm(qm); R = _Rmat(qm)
            dr = (r[e + 1] - r[e]) / ds; tang[e] = dr
            G = R.T @ dr - _E1
            qc = np.array([q[e][0], -q[e][1], -q[e][2], -q[e][3]])
            K = 2 * _qmul(qc, q[e + 1])[1:] / ds
            nsp[e] = R @ (Cn * G); msp[e] = R @ (Cm * K)
        dr_ = np.zeros((nn, 3)); dq = np.zeros((nn, 4))
        dv = np.zeros((nn, 3)); dom = np.zeros((nn, 3))
        for i in range(1, nn):
            nL = nsp[i - 1]; mL = msp[i - 1]; tL = tang[i - 1]
            if i < N:
                divn = (nsp[i] - nL) / ds; divm = (msp[i] - mL) / ds
                tavg = 0.5 * (tL + tang[i]); navg = 0.5 * (nL + nsp[i])
            else:  # free tip, half node, ghost force/moment = 0
                divn = (-nL) / (ds / 2); divm = (-mL) / (ds / 2); tavg = tL; navg = nL
            dv[i] = divn / rhoA + g - c * v[i]
            R = _Rmat(q[i])
            taub = R.T @ (divm + np.cross(tavg, navg))
            dom[i] = (taub - np.cross(om[i], Jr * om[i])) / Jr
            dr_[i] = v[i]
            w_, x_, y_, z_ = q[i]; a, b, cc = om[i]
            dq[i] = 0.5 * np.array([-(x_ * a + y_ * b + z_ * cc), w_ * a + y_ * cc - z_ * b,
                                    w_ * b - x_ * cc + z_ * a, w_ * cc + x_ * b - y_ * a])
        return np.concatenate([dr_.ravel(), dq.ravel(), dv.ravel(), dom.ravel()])

    te = np.linspace(0, t_end, n_t)
    sol = solve_ivp(deriv, (0, t_end), Y0, t_eval=te, method="RK45",
                    rtol=1e-7, atol=1e-9, max_step=ds / np.sqrt(max(Cn.max(), Cm.max())) * 1.5)
    if not sol.success:
        raise RuntimeError(f"dynamic 3-D Cosserat sim failed: {sol.message}")
    r = sol.y[:nn * 3].reshape(nn, 3, -1).transpose(0, 2, 1)
    q = sol.y[nn * 3:nn * 3 + nn * 4].reshape(nn, 4, -1).transpose(0, 2, 1)
    q = q / np.linalg.norm(q, axis=2, keepdims=True)
    return np.linspace(0, 1, nn), te, r, q


def generate_dynamic_spatial_cosserat(*, refs=None, Jrho=DYN3D_JRHO, gvec=DYN3D_G,
                                      c=DYN3D_C, twist0=DYN3D_TWIST0, N=40, n_t=121,
                                      pos_noise_std=1e-3, quat_noise_std=3e-3,
                                      rhoA=1.0, seed=0, **unit_overrides):
    """Synthetic dynamic 3-D rod motion (noisy ``r, q`` over space-time).
    Returns ``(data, truth)``; ``data`` holds ``s, t, r, q, refs, Jrho, gvec, rhoA``."""
    refs = dict(refs or DYN3D_REFS)
    truth = {f"{k}_unit": float(unit_overrides.get(f"{k}_unit", 1.0)) for k in _NAMES}
    stiff = {k: refs[k] * truth[f"{k}_unit"] for k in _NAMES}
    s, t, r, q = simulate_dynamic_spatial_cosserat(
        stiff, Jrho, gvec, c, twist0, N=N, t_end=DYN3D_TEND, n_t=n_t, rhoA=rhoA)
    rng = np.random.default_rng(seed)
    r = r + rng.normal(0, pos_noise_std, r.shape)
    q = q + rng.normal(0, quat_noise_std, q.shape)
    q /= np.linalg.norm(q, axis=2, keepdims=True)
    data = {"s": s, "t": t, "r": r, "q": q, "refs": refs,
            "Jrho": np.array(Jrho, float), "gvec": np.array(gvec, float),
            "rhoA": rhoA, "c": float(c)}
    return data, truth


def _sg(F, axis, delta, deriv, win, poly):
    n = F.shape[axis]; w = min(win, n - 1 if n % 2 == 0 else n - 2)
    if w % 2 == 0:
        w += 1
    if w <= poly:
        o = F
        for _ in range(deriv):
            o = np.gradient(o, delta, axis=axis)
        return o
    return savgol_filter(F, w, poly, deriv=deriv, delta=delta, axis=axis)


@dataclass
class DynamicSpatialIDResult:
    units: Dict[str, float]
    stiffness: Dict[str, float]

    def as_dict(self) -> Dict[str, float]:
        return dict(self.units)


def recover_dynamic_spatial_stiffness(data, sg_window_t=21, sg_window_s=9,
                                      sg_poly=3, edge_crop=3) -> DynamicSpatialIDResult:
    """Recover all six stiffnesses from measured dynamic 3-D rod motion via the
    kinematic internal force/moment + constitutive least squares."""
    s = np.asarray(data["s"], float); t = np.asarray(data["t"], float)
    r = np.asarray(data["r"], float); q = np.asarray(data["q"], float)
    Jr = np.asarray(data["Jrho"], float); g = np.asarray(data["gvec"], float)
    rhoA = float(data["rhoA"]); refs = data["refs"]
    c_damp = float(data.get("c", 0.0))
    nn, nt, _ = r.shape; ds = float(s[1] - s[0]); dt = float(t[1] - t[0])
    AX, TT = 0, 1
    q = q / np.linalg.norm(q, axis=2, keepdims=True)
    wt, ws, p = sg_window_t, sg_window_s, sg_poly

    r_tt = _sg(r, TT, dt, 2, wt, p)
    r_t = _sg(r, TT, dt, 1, wt, p)
    r_s = _sg(_sg(r, TT, dt, 0, wt, p), AX, ds, 1, ws, p)
    q_t = _sg(q, TT, dt, 1, wt, p)
    q_s = _sg(_sg(q, TT, dt, 0, wt, p), AX, ds, 1, ws, p)
    qsm = _sg(_sg(q, TT, dt, 0, wt, p), AX, ds, 0, ws, p)
    qsm /= np.linalg.norm(qsm, axis=2, keepdims=True)

    Gam = np.zeros((nn, nt, 3)); K = np.zeros((nn, nt, 3))
    H = np.zeros((nn, nt, 3)); Rall = np.zeros((nn, nt, 3, 3))
    for i in range(nn):
        for k in range(nt):
            qq = qsm[i, k]; R = _Rmat(qq); Rall[i, k] = R
            qc = np.array([qq[0], -qq[1], -qq[2], -qq[3]])
            Om = 2 * _qmul(qc, q_t[i, k])[1:]
            H[i, k] = R @ (Jr * Om)
            Gam[i, k] = R.T @ r_s[i, k] - _E1
            K[i, k] = 2 * _qmul(qc, q_s[i, k])[1:]
    Hdot = _sg(H, TT, dt, 1, wt, p)

    def integ_tip(F):
        out = np.zeros_like(F)
        for i in range(nn - 2, -1, -1):
            out[i] = out[i + 1] + 0.5 * (F[i] + F[i + 1]) * ds
        return -out
    # linear momentum: ρA r_tt = ∂n_sp/∂s + ρA g − ρA c r_t
    #   ⇒ ∂n_sp/∂s = ρA (r_tt − g + c r_t)
    Nsp = integ_tip(rhoA * (r_tt - g[None, None, :] + c_damp * r_t))
    Msp = integ_tip(Hdot - np.cross(r_s, Nsp))
    nmat = np.einsum("ijba,ijb->ija", Rall, Nsp)   # Rᵀ Nsp
    mmat = np.einsum("ijba,ijb->ija", Rall, Msp)

    cs = slice(edge_crop, -edge_crop)

    def slope(F, E):
        f = F[cs, cs].ravel(); e = E[cs, cs].ravel()
        return float(np.dot(f, e) / np.dot(e, e))

    stiff = {
        "EA": slope(nmat[..., 0], Gam[..., 0]), "GA1": slope(nmat[..., 1], Gam[..., 1]),
        "GA2": slope(nmat[..., 2], Gam[..., 2]), "GJ": slope(mmat[..., 0], K[..., 0]),
        "EI1": slope(mmat[..., 1], K[..., 1]), "EI2": slope(mmat[..., 2], K[..., 2]),
    }
    units = {f"{k}_unit": stiff[k] / refs[k] for k in _NAMES}
    return DynamicSpatialIDResult(units=units, stiffness=stiff)
