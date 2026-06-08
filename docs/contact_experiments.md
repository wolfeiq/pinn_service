# Proprioceptive contact estimation

A soft robot interacting with its environment feels contact through its own
deformation. This module recovers **where** a soft rod is touching an obstacle
and **how hard**, from the measured shape alone — whole-body tactile sensing
with no force/torque sensor.

## Principle

A point contact applies a force at one arclength → the internal **shear jumps**
there → the internal moment `m(s) = EI·κ(s)` has a *slope kink*. For a planar
cantilever (clamped base, known tip load `P`, unknown normal contact `F_c` at
unknown `s_c`), the geometrically-exact balance `m'(s) = −cosθ·n_y` with `n_y`
jumping by `F_c` across `s_c` gives a jump in the curvature slope:

```
κ'(s)  jumps by  −cosθ(s_c)·F_c/EI   at  s_c
```

so:

```
s_c = breakpoint of the (continuous, piecewise-linear) curvature κ(s)
F_c = −EI · (κ'-slope jump) / cosθ(s_c)
```

Recovered by a changepoint fit of `κ(s)` — the estimator knows nothing about the
obstacle; it reads the contact straight off the shape.

## Results (`scripts/exp_contact.py`)

| true s_c | true F_c | clean s_c / F_c | noisy s_c / F_c |
|---|---|---|---|
| 0.30 | 1.50 | 0.295 / 1.468 | 0.310 / 1.454 |
| 0.50 | 2.00 | 0.500 / 1.991 | 0.505 / 2.020 |
| 0.70 | 3.00 | 0.700 / 3.034 | 0.700 / 3.074 |
| 0.60 | 4.00 | 0.600 / 4.044 | 0.585 / 4.012 |

Contact location to ~0.01 (1% of length) and force to ~3%, robust to shape
noise — the rod "feels" contact through its own bending.

## Where this sits

This adds **environment interaction** — the rod can now sense contact, the basis
for safe interaction / grasping in soft robotics. Together with the actuation
and hyperelastic modules, the soft-rod stack now spans drive (tendon + pneumatic),
nonlinear material, and contact sensing. Natural extensions: multi-point /
distributed contact, and contact + simultaneous stiffness estimation.

## Reproduce

```
python3 scripts/exp_contact.py
```

Module: `pinn_engine/baselines/contact_id.py`.
