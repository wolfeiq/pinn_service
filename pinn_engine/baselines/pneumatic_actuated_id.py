"""Pneumatic actuation + self-calibration for a soft Cosserat rod.

The other dominant soft-robot drive: pressurized chambers (PneuNets,
fiber-reinforced actuators). A chamber of effective area ``A`` at material offset
``(dy, dz)`` under pressure ``P`` pushes axially with force ``P·A`` — so unlike a
tendon (which *pulls*, compresses, and bends *toward* its offset), a pneumatic
chamber *extends* and bends the rod *away* from the chamber. With a helical
chamber lever ``h`` it can also twist.

Material-frame actuation wrench (constant → constant-strain PCC shape):

    n_act = (+Σ P·A, 0, 0)                          (axial extension — note sign)
    m_act = (Σ P·A·h, Σ P·A·dz, −Σ P·A·dy)          (torsion, two bendings)

**Self-calibration.** Pressures and chamber geometry are known, so the wrench is
known and ``wrench = C·strain`` is linear in the stiffnesses: a sweep of pressure
patterns + measured shapes recovers ``EA, EI1, EI2`` (and ``GJ`` with helical
chambers) — the pneumatic dual of the tendon calibrator. (Shear is not
pressure-excitable.) The recovered ``EA`` comes out of *extension* strain here,
vs *compression* for tendons — a nice consistency check that the sign physics is
right.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from pinn_engine.baselines.tendon_actuated_id import (
    _E1, _expSO3, _quat_of, _strains_from_shape, DEFAULT_REFS,
)

# Chambers: (area A, dy, dz, helix h). One central + offset chambers on +y / +z.
DEFAULT_CHAMBERS: List[Tuple[float, float, float, float]] = [
    (1.0, 0.06, 0.0, 0.0),    # +y chamber → bend about ê3 (EI2), away from +y
    (1.0, 0.0, 0.06, 0.0),    # +z chamber → bend about ê2 (EI1)
    (1.0, 0.0, 0.0, 0.06),    # helical chamber → torsion (GJ)
    (2.0, 0.0, 0.0, 0.0),     # central chamber → pure extension (EA)
]
_ACT_NAMES = ["EA", "EI1", "EI2", "GJ"]


def pneumatic_wrench(pressures: Sequence[float],
                     chambers: Sequence[Tuple[float, float, float, float]]):
    """Material-frame ``(n_act, m_act)`` from chamber pressures + geometry."""
    P = np.asarray(pressures, float)
    A = np.array([c[0] for c in chambers]); dy = np.array([c[1] for c in chambers])
    dz = np.array([c[2] for c in chambers]); h = np.array([c[3] for c in chambers])
    fa = P * A
    n = np.array([fa.sum(), 0.0, 0.0])                       # extension (positive)
    m = np.array([np.sum(fa * h), np.sum(fa * dz), -np.sum(fa * dy)])
    return n, m


def simulate_pneumatic_actuated(stiff, chambers, pressures, ns: int = 81):
    """Constant-strain (PCC) shape of a pressurized rod. Returns ``(s, r, q)``."""
    Cn = np.array([stiff["EA"], stiff["GA1"], stiff["GA2"]])
    Cm = np.array([stiff["GJ"], stiff["EI1"], stiff["EI2"]])
    n_act, m_act = pneumatic_wrench(pressures, chambers)
    Gam = n_act / Cn; K = m_act / Cm
    s = np.linspace(0, 1, ns); ds = s[1] - s[0]
    Rs = [_expSO3(sv * K) for sv in s]
    r = np.zeros((ns, 3)); q = np.zeros((ns, 4)); rr = np.zeros(3); d = Gam + _E1
    for j in range(ns):
        if j > 0:
            rr = rr + 0.5 * (Rs[j - 1] @ d + Rs[j] @ d) * ds
        r[j] = rr; q[j] = _quat_of(Rs[j])
    return s, r, q


def default_pressure_patterns() -> List[List[float]]:
    pats = []
    for i in range(len(DEFAULT_CHAMBERS)):
        for p in (1.0, 2.0, 3.0):
            row = [0.0] * len(DEFAULT_CHAMBERS); row[i] = p; pats.append(row)
    pats += [[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 2.0], [1, 1, 0, 1]]
    return pats


def generate_pneumatic_calibration(*, refs=None, chambers=None, patterns=None,
                                   pos_noise_std=1e-3, quat_noise_std=3e-3,
                                   ns=81, seed=0, **unit_overrides):
    """Synthetic pneumatic self-calibration dataset. Returns ``(data, truth)``."""
    refs = dict(refs or DEFAULT_REFS)
    chambers = list(chambers or DEFAULT_CHAMBERS)
    patterns = list(patterns or default_pressure_patterns())
    truth = {f"{k}_unit": float(unit_overrides.get(f"{k}_unit", 1.0)) for k in _ACT_NAMES}
    stiff = dict(refs)
    for k in _ACT_NAMES:
        stiff[k] = refs[k] * truth[f"{k}_unit"]
    rng = np.random.default_rng(seed)
    shots = []
    for P in patterns:
        s, r, q = simulate_pneumatic_actuated(stiff, chambers, P, ns=ns)
        r = r + rng.normal(0, pos_noise_std, r.shape)
        q = q + rng.normal(0, quat_noise_std, q.shape)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        shots.append({"pressures": list(P), "s": s, "r": r, "q": q})
    return {"chambers": chambers, "refs": refs, "shots": shots}, truth


@dataclass
class PneumaticCalibrationResult:
    units: Dict[str, float]
    stiffness: Dict[str, float]

    def as_dict(self):
        return dict(self.units)


def recover_pneumatic_stiffness(data) -> PneumaticCalibrationResult:
    """Recover ``EA, EI1, EI2, GJ`` from pressurized shapes at known pressures."""
    chambers = data["chambers"]; refs = data["refs"]
    rows_n = []; rows_m = []
    for shot in data["shots"]:
        G, K = _strains_from_shape(shot["s"], shot["r"], shot["q"])
        n_act, m_act = pneumatic_wrench(shot["pressures"], chambers)
        rows_n.append((G, n_act)); rows_m.append((K, m_act))

    def reg(idx, rows):
        x = np.array([row[0][idx] for row in rows]); y = np.array([row[1][idx] for row in rows])
        return float(np.dot(x, y) / np.dot(x, x))

    stiff = {"EA": reg(0, rows_n), "GJ": reg(0, rows_m),
             "EI1": reg(1, rows_m), "EI2": reg(2, rows_m)}
    units = {f"{k}_unit": stiff[k] / refs[k] for k in _ACT_NAMES}
    return PneumaticCalibrationResult(units=units, stiffness=stiff)
