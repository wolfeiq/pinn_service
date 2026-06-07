"""Tendon-actuated Cosserat rod — actuation model + self-calibration.

This is *actuation*: what turns a passive Cosserat rod into a soft robot. Cables
(tendons) routed at offsets from the centerline, when tensioned, apply a wrench
that bends and twists the rod — the dominant continuum-manipulator drive.

**Actuation model.** A tendon at material cross-section offset ``(dy, dz)`` with
tension ``τ``, routed parallel to the backbone (terminated at the tip), applies
a constant material wrench

    n_act = (−Στ, 0, 0)                              (axial compression)
    m_act = (Στ·h, −Στ·dz, Στ·dy)                   (torsion, bend₂, bend₃)

where ``h`` is a helical-routing lever that produces torsion. A constant wrench
→ **constant material strain** → the rod bends into a circular arc / helix: the
piecewise-constant-curvature (PCC) regime that is the soft-robot workhorse. A
single tendon at offset ``d`` gives curvature ``κ = τ·d/EI`` (verified).

**Self-calibration.** Because the actuation wrench is *known* (commanded
tensions × known routing), the constitutive law ``wrench = C·strain`` is linear
in the stiffnesses — so commanding several tension patterns and measuring the
resulting shapes recovers the rod's stiffness with **no external test rig**:
the robot calibrates itself by moving. Tendons excite axial, both bendings, and
(helically) torsion — recovering ``EA, EI1, EI2, GJ``. Shear (GA) is not
tendon-excitable; use ``spatial_cosserat_id`` (external load) for that.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

_E1 = np.array([1.0, 0.0, 0.0])

# Default 4-tendon routing (dy, dz, helix-lever) and a tension-pattern sweep that
# spans axial / bend2 / bend3 / torsion — enough to identify all four stiffnesses.
DEFAULT_TENDONS: List[Tuple[float, float, float]] = [
    (0.06, 0.0, 0.0),    # +y offset  → bend about ê3 (EI2)
    (0.0, 0.06, 0.0),    # +z offset  → bend about ê2 (EI1)
    (0.0, 0.0, 0.06),    # helical    → torsion (GJ)
    (0.04, 0.04, 0.0),   # diagonal   → mixed bending
]
DEFAULT_REFS: Dict[str, float] = {
    "EA": 15.0, "GA1": 15.0, "GA2": 12.0, "GJ": 0.8, "EI1": 1.0, "EI2": 0.8,
}
_ACT_NAMES = ["EA", "EI1", "EI2", "GJ"]


def _expSO3(k):
    th = np.linalg.norm(k)
    if th < 1e-12:
        return np.eye(3)
    a = k / th
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _logSO3(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    return th / (2 * np.sin(th)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def _quat_of(R):
    w = np.sqrt(max(0.0, 1 + np.trace(R))) / 2
    if w < 1e-8:
        return np.array([1.0, 0, 0, 0])
    return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w),
                     (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])


def _R_of_quat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def actuation_wrench(tensions: Sequence[float],
                     tendons: Sequence[Tuple[float, float, float]]):
    """Material-frame ``(n_act, m_act)`` from tendon tensions + routing."""
    t = np.asarray(tensions, float)
    dy = np.array([td[0] for td in tendons]); dz = np.array([td[1] for td in tendons])
    h = np.array([td[2] for td in tendons])
    n = np.array([-t.sum(), 0.0, 0.0])
    m = np.array([np.sum(t * h), -np.sum(t * dz), np.sum(t * dy)])
    return n, m


def simulate_tendon_actuated(stiff: Dict[str, float],
                             tendons: Sequence[Tuple[float, float, float]],
                             tensions: Sequence[float], ns: int = 81):
    """Constant-strain (PCC) shape of a tendon-actuated rod. Returns
    ``(s, r, q)`` with ``r`` shape ``(ns,3)`` and quaternions ``q`` ``(ns,4)``."""
    Cn = np.array([stiff["EA"], stiff["GA1"], stiff["GA2"]])
    Cm = np.array([stiff["GJ"], stiff["EI1"], stiff["EI2"]])
    n_act, m_act = actuation_wrench(tensions, tendons)
    Gam = n_act / Cn; K = m_act / Cm           # constant material strains
    s = np.linspace(0, 1, ns); ds = s[1] - s[0]
    Rs = [_expSO3(sv * K) for sv in s]
    r = np.zeros((ns, 3)); q = np.zeros((ns, 4)); rr = np.zeros(3)
    d = Gam + _E1
    for j in range(ns):
        if j > 0:
            rr = rr + 0.5 * (Rs[j - 1] @ d + Rs[j] @ d) * ds
        r[j] = rr; q[j] = _quat_of(Rs[j])
    return s, r, q


def default_tension_patterns() -> List[List[float]]:
    """A sweep of tendon-tension patterns spanning the actuation space."""
    pats = []
    for i in range(len(DEFAULT_TENDONS)):
        for tau in (1.0, 2.0, 3.0):
            p = [0.0] * len(DEFAULT_TENDONS); p[i] = tau; pats.append(p)
    pats += [[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 0], [2, 1, 0, 1]]
    return pats


def generate_tendon_calibration(*, refs=None, tendons=None, patterns=None,
                                pos_noise_std=1e-3, quat_noise_std=3e-3,
                                ns=81, seed=0, **unit_overrides):
    """Synthetic self-calibration dataset: actuated shapes for a sweep of known
    tendon-tension patterns. Returns ``(data, truth)``."""
    refs = dict(refs or DEFAULT_REFS)
    tendons = list(tendons or DEFAULT_TENDONS)
    patterns = list(patterns or default_tension_patterns())
    truth = {f"{k}_unit": float(unit_overrides.get(f"{k}_unit", 1.0)) for k in _ACT_NAMES}
    # build a full stiffness dict (shear refs unchanged; not tendon-excited)
    stiff = dict(refs)
    for k in _ACT_NAMES:
        stiff[k] = refs[k] * truth[f"{k}_unit"]
    rng = np.random.default_rng(seed)
    shots = []
    for tens in patterns:
        s, r, q = simulate_tendon_actuated(stiff, tendons, tens, ns=ns)
        r = r + rng.normal(0, pos_noise_std, r.shape)
        q = q + rng.normal(0, quat_noise_std, q.shape)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        shots.append({"tensions": list(tens), "s": s, "r": r, "q": q})
    data = {"tendons": tendons, "refs": refs, "shots": shots}
    return data, truth


@dataclass
class TendonCalibrationResult:
    units: Dict[str, float]
    stiffness: Dict[str, float]

    def as_dict(self) -> Dict[str, float]:
        return dict(self.units)


def _strains_from_shape(s, r, q):
    ns = len(s); ds = s[1] - s[0]; Gs = []; Ks = []
    for j in range(ns - 1):
        Rj = _R_of_quat(q[j]); Rj1 = _R_of_quat(q[j + 1])
        Gs.append(Rj.T @ ((r[j + 1] - r[j]) / ds) - _E1)
        Ks.append(_logSO3(Rj.T @ Rj1) / ds)
    return np.mean(Gs, axis=0), np.mean(Ks, axis=0)


def recover_tendon_stiffness(data) -> TendonCalibrationResult:
    """Self-calibrate: recover ``EA, EI1, EI2, GJ`` from actuated shapes at known
    tendon tensions (per-component least squares of wrench vs measured strain)."""
    tendons = data["tendons"]; refs = data["refs"]
    rows_n = []; rows_m = []
    for shot in data["shots"]:
        G, K = _strains_from_shape(shot["s"], shot["r"], shot["q"])
        n_act, m_act = actuation_wrench(shot["tensions"], tendons)
        rows_n.append((G, n_act)); rows_m.append((K, m_act))

    def reg(idx, rows):
        x = np.array([row[0][idx] for row in rows])
        y = np.array([row[1][idx] for row in rows])
        return float(np.dot(x, y) / np.dot(x, x))

    stiff = {"EA": reg(0, rows_n), "GJ": reg(0, rows_m),
             "EI1": reg(1, rows_m), "EI2": reg(2, rows_m)}
    units = {f"{k}_unit": stiff[k] / refs[k] for k in _ACT_NAMES}
    return TendonCalibrationResult(units=units, stiffness=stiff)
