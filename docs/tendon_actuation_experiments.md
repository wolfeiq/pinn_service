# Tendon actuation + self-calibration

Actuation is what turns a passive Cosserat rod into a **soft robot**. The
dominant continuum-manipulator drive is **tendons** (cables) routed at offsets
from the centerline: tensioning them applies a wrench that bends and twists the
rod. This module adds the tendon-actuation model and uses it for **stiffness
self-calibration** — the robot recovers its own material parameters just by
actuating, with no external test rig.

## Actuation model

A tendon at material cross-section offset `(dy, dz)` with tension `τ`, routed
parallel to the backbone and terminated at the tip, applies a constant material
wrench:

```
n_act = (−Στ, 0, 0)                    (axial compression)
m_act = (Στ·h, −Στ·dz, Στ·dy)         (torsion via helix lever h; two bendings)
```

A constant wrench → **constant material strain** → the rod bends into a circular
arc / helix. This is the **piecewise-constant-curvature (PCC)** regime, the
workhorse model of soft robotics. A single tendon at offset `d` gives curvature
`κ = τ·d/EI` — verified exactly in `test_templates.py` (single tendon `d=0.05,
τ=2`, `EI₂=0.8` → `κ = 0.125`, planar arc).

Multiple tendons at different angular positions steer the bending in any
direction; a helically-routed tendon (`h≠0`) produces torsion.

## Self-calibration (the inverse)

The actuation wrench is **known** (commanded tensions × known routing), so the
constitutive law `wrench = C · strain` is linear in the stiffnesses. Commanding
a sweep of tension patterns and measuring the resulting shapes (extract the
constant strains `Γ, K` from each) gives a per-component least-squares estimate
of the stiffness — **no external load or test fixture needed**. This is exactly
how a real tendon-driven continuum robot would be calibrated: sweep actuation,
watch it move, fit stiffness.

Tendons excite axial, both bendings, and (helically) torsion → recover
`EA, EI1, EI2, GJ`. Shear (`GA`) is not tendon-excitable; use
`spatial_cosserat_id` (external transverse load) for that — and for slender soft
rods shear is usually negligible anyway.

## Results (`scripts/exp_tendon_actuation.py`)

4 tendons (+y, +z, helical, diagonal), 16 tension patterns:

| noise (pos, quat) | EA | EI1 | EI2 | GJ |
|---|---|---|---|---|
| clean | 0.0% | 0.0% | 0.0% | 0.0% |
| 1e-3, 3e-3 | **0.3%** | **2.3%** | **1.0%** | **3.2%** |
| 2e-3, 6e-3 | 0.5% | 5.6% | 2.7% | 7.7% |

Clean recovers exactly; standard noise ≤3.2%. Also recovers arbitrary non-unit
stiffness (e.g. EA=1.2, EI1=1.4, GJ=0.6 → exact), so it's a genuine estimator.

## Where this sits

This is the first **actuation** capability — the rod is no longer passive. It
composes with the rest of the suite: the actuation model gives the forward
control map (tensions → shape), and self-calibration gives the stiffness the
controller needs. Still open for full soft-robotics coverage:
**pneumatic actuation** (pressure-driven), **hyperelastic / viscoelastic**
materials, and **contact**.

## Reproduce

```
python3 scripts/exp_tendon_actuation.py
```

Module: `pinn_engine/baselines/tendon_actuated_id.py`
(`simulate_tendon_actuated`, `generate_tendon_calibration`,
`recover_tendon_stiffness`).
