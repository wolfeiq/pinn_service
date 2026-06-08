# Actuated-dynamics self-calibration

The actuation modules (tendon, pneumatic) are quasi-static (PCC). This adds the
**dynamic** response: when the actuation changes in time, the rod has inertia and
damping. Its modal curvature obeys a second-order system

```
I_eff·κ̈ + c·κ̇ + EI·κ = M_act(t)
```

(`M_act` = the known commanded actuation moment, `I_eff` effective modal inertia,
`c` damping, `EI` stiffness). Suddenly tensioning a tendon and watching the rod
**ring and settle** recovers all three at once — dynamic self-calibration, no
external excitation rig.

## Identification

Because `M_act(t)` is known, the model is *linear in the parameters*: measure the
curvature response `κ(t)`, take `κ̇, κ̈` (Savitzky-Golay), and regress

```
[κ̈, κ̇, κ] · (I_eff, c, EI)ᵀ = M_act
```

A step input is cleanest — its transient rings at the natural frequency
`ω_n=√(EI/I_eff)` with damping ratio `ζ=c/(2√(EI·I_eff))`.

## Results (`scripts/exp_actuated_dynamics.py`)

| rod (EI/c/I) | noise | EI | c | I_eff |
|---|---|---|---|---|
| 1.0/0.05/0.02 | clean | 1.000 | 0.048 | 0.021 |
| 1.0/0.05/0.02 | 5e-3 | 1.000 | 0.048 | 0.020 |
| 2.0/0.12/0.03 | 2e-3 | 2.000 | 0.114 | 0.031 |

From a single ring-down: `EI` recovered to ~1% (the most robust — it comes from
`κ` directly), and `c`, `I_eff` to ~5–6% (they come from `κ̇, κ̈`, so they need
slightly cleaner data but recover well). The natural frequency and damping ratio
follow consistently.

## Where this sits

This merges actuation with dynamics — the rod is actuated *and* inertial, the
regime a real soft robot operates in during fast motion. It complements the
quasi-static self-calibration (which gives stiffness only) by also recovering
damping and modal inertia, the parameters a dynamic controller needs.

## Reproduce

```
python3 scripts/exp_actuated_dynamics.py
```

Module: `pinn_engine/baselines/actuated_dynamics_id.py`.
