# Pneumatic actuation + self-calibration

The second dominant soft-robot drive: **pressurized chambers** (PneuNets,
fiber-reinforced actuators). Companion to the tendon module.

## Model

A chamber of effective area `A` at material offset `(dy, dz)` under pressure `P`
pushes axially with force `P·A`. So unlike a tendon (pull → compress → bend
*toward* the offset), a pneumatic chamber **extends** the rod and bends it
*away* from the chamber. Material-frame wrench (constant → constant-strain PCC):

```
n_act = (+Σ P·A, 0, 0)                    (axial extension — note the + sign)
m_act = (Σ P·A·h, Σ P·A·dz, −Σ P·A·dy)    (torsion via helix h; two bendings)
```

Verified (`test_templates.py`): a +y chamber extends (tip x>1) and bends to −y
(away), where a +y tendon bends to +y; a central chamber gives pure extension.

## Self-calibration

Pressures and chamber geometry are known → the wrench is known → `wrench =
C·strain` is linear in the stiffnesses. A sweep of pressure patterns + measured
shapes recovers `EA, EI1, EI2` (and `GJ` with a helical chamber). The recovered
`EA` comes from *extension* strain here (vs *compression* for tendons) — a clean
sign-consistency check.

## Results (`scripts/exp_pneumatic_actuation.py`)

4 chambers (+y, +z, helical, central), 16 pressure patterns:

| noise (pos, quat) | EA | EI1 | EI2 | GJ |
|---|---|---|---|---|
| clean | 0.0% | 0.0% | 0.0% | 0.0% |
| 1e-3, 3e-3 | **0.2%** | **3.6%** | **2.6%** | **3.3%** |
| 2e-3, 6e-3 | 0.4% | 7.5% | 7.0% | 8.4% |

Clean exact; standard noise ≤3.6%. Recovers arbitrary non-unit stiffness too.

## Where this sits

Both major soft-robot drives are now covered — **tendon** (pull) and
**pneumatic** (push) — each with self-calibration. Shear (GA) is not
actuation-excitable (use `spatial_cosserat_id`). Still open for full coverage:
**hyperelastic/viscoelastic** materials and **contact**.

## Reproduce

```
python3 scripts/exp_pneumatic_actuation.py
```

Module: `pinn_engine/baselines/pneumatic_actuated_id.py`.
