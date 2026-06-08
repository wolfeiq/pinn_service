"""Multi-point contact estimation for a soft rod.

Extends single-point proprioceptive sensing (`contact_id`) to **several
simultaneous contacts** — a soft finger wrapping an object, or a rod pressed
against multiple obstacles. Each point contact makes the internal shear jump, so
the rod carries several curvature kinks; we localize and size all of them from
the measured shape.

**The exact variable.** ``m'(s) = −cosθ·n_y`` and ``∫cosθ ds = x(s)`` (the
horizontal coordinate), so the bending moment ``M(s) = EI·θ'(s)`` is *exactly
piecewise-linear in x*, slope ``−n_y`` per segment, with a clean kink at each
contact. Fitting ``M`` vs ``x`` as a continuous piecewise-linear function gives
the contact positions (breakpoints) and forces (slope jumps, ``F_i`` = jump in
``n_y``) — with no within-segment curvature to confound the fit.

``recover_n_contacts`` recovers a known number of contacts (coordinate descent on
the breakpoints); ``recover_contacts`` adds a best-effort automatic count.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.signal import savgol_filter


def simulate_multi_contact(EI=1.0, P=0.5, contacts=((0.3, 1.5), (0.65, 2.0)), ns=401):
    """Planar cantilever (tip load ``P``) with point contacts ``[(s_c, F_c), …]``
    (normal reactions reducing the outboard transverse force). Returns
    ``(s, x, y, theta)``."""
    cs = sorted(contacts); locs = [c[0] for c in cs]

    def ny(s):
        return P - sum(F for sc, F in cs if sc > s)

    def rhs(s, Y):
        x, y, th, m = Y
        return [np.cos(th), np.sin(th), m / EI, -np.cos(th) * ny(s)]

    bnds = [0.0] + locs + [1.0]

    def run(m0):
        Y = [0, 0, 0, m0]; S = []; YY = []
        for i in range(len(bnds) - 1):
            a, b = bnds[i], bnds[i + 1]
            if b <= a + 1e-9:
                continue
            seg = np.linspace(a, b, max(12, int(ns * (b - a))))
            sol = solve_ivp(rhs, (a, b), Y, t_eval=seg, rtol=1e-9, atol=1e-11)
            S.append(sol.t if not S else sol.t[1:])
            YY.append(sol.y if not YY else sol.y[:, 1:])
            Y = sol.y[:, -1]
        return np.concatenate(S), np.concatenate(YY, axis=1)

    m0 = brentq(lambda m: run(m)[1][3, -1], -80, 80)
    s, Y = run(m0)
    return s, Y[0], Y[1], Y[2]


def generate_multi_contact(*, EI=1.0, P=0.5, contacts=((0.3, 1.5), (0.65, 2.0)),
                           ang_noise_std=2e-3, ns=401, seed=0):
    s, x, y, th = simulate_multi_contact(EI, P, contacts, ns)
    th = th + np.random.default_rng(seed).normal(0, ang_noise_std, th.shape)
    return {"s": s, "x": x, "theta": th, "EI": EI, "P": P}, {"contacts": list(contacts)}


@dataclass
class MultiContactResult:
    contacts: List[Tuple[float, float]]   # [(s_c, F_c), …]

    def as_list(self):
        return list(self.contacts)


def _prep(s, x, theta, EI):
    w = 13 if len(s) > 40 else 7
    w += (w % 2 == 0)
    thp = savgol_filter(theta, w, 3, deriv=1, delta=s[1] - s[0])
    xs = savgol_filter(x, w, 3)
    M = EI * thp
    o = np.argsort(xs)
    return xs[o], M[o], s[o]


def _fit(X, M, bps):
    cols = [np.ones_like(X), X] + [np.maximum(X - b, 0) for b in bps]
    A = np.stack(cols, axis=1)
    c, *_ = np.linalg.lstsq(A, M, rcond=None)
    return float(np.mean((A @ c - M) ** 2)), c


def _recover_n(s, x, theta, EI, n, min_sep=0.05):
    """Returns ``(contacts, residual)`` for exactly ``n`` contacts."""
    X, M, so = _prep(s, x, theta, EI)
    if n == 0:
        return [], _fit(X, M, [])[0]
    lo, hi = X.min() + 0.04, X.max() - 0.04
    bps = list(np.linspace(lo, hi, n + 2)[1:-1])
    grid = np.linspace(lo, hi, 121)
    for _ in range(6):
        for i in range(n):
            best = (None, bps[i])
            for g in grid:
                if any(abs(g - bps[j]) < min_sep for j in range(n) if j != i):
                    continue
                cand = sorted(bps[:i] + [g] + bps[i + 1:])
                r, _ = _fit(X, M, cand)
                if best[0] is None or r < best[0]:
                    best = (r, g)
            bps[i] = best[1]
        bps = sorted(bps)
    rr, c = _fit(X, M, bps)
    out = [(float(np.interp(b, X, so)), float(-c[2 + i])) for i, b in enumerate(bps)]
    return out, rr


def recover_n_contacts(data, n: int, min_sep=0.05) -> MultiContactResult:
    """Recover a *known* number ``n`` of contacts (positions + forces)."""
    out, _ = _recover_n(np.asarray(data["s"], float), np.asarray(data["x"], float),
                        np.asarray(data["theta"], float), float(data["EI"]), n, min_sep)
    return MultiContactResult(contacts=out)


def recover_contacts(data, n_contacts: Optional[int] = None, max_contacts=4,
                     ang_noise_std=2e-3) -> MultiContactResult:
    """Recover contacts. If ``n_contacts`` is given, use it; otherwise estimate
    the count (best-effort) by the residual elbow, then recover."""
    if n_contacts is not None:
        return recover_n_contacts(data, n_contacts)
    s = np.asarray(data["s"], float); x = np.asarray(data["x"], float)
    theta = np.asarray(data["theta"], float); EI = float(data["EI"])
    # residual vs contact count; pick the count by a BIC-style penalty (scan all
    # counts — the n-1 fit can be worse-placed than n, so don't break early).
    nobs = len(s)
    fits = [_recover_n(s, x, theta, EI, n) for n in range(0, max_contacts + 1)]
    bic = [nobs * np.log(max(f[1], 1e-12)) + 2 * n * np.log(nobs)
           for n, f in enumerate(fits)]
    n_star = int(np.argmin(bic))
    contacts = [(sc, F) for sc, F in fits[n_star][0] if abs(F) > 0.15]
    return MultiContactResult(contacts=contacts)
