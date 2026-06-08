# Hyperelastic soft-rod constitutive identification

Real soft-robot materials (silicone, rubber, tissue) are **hyperelastic**: their
stress–strain law is nonlinear, usually strain-stiffening at large deformation.
A linear-elastic rod (constant EI, EA) is only the small-strain limit. This
module recovers the *nonlinear* constitutive curve from a load sweep.

## Model

The symmetric leading nonlinearity (Taylor expansion of any symmetric
hyperelastic bending / stretching response):

```
bending:  M(κ) = a1·κ + a3·κ³      (a1 = small-strain EI;  a3 = stiffening)
axial:    N(ε) = b1·ε + b3·ε³      (b1 = small-strain EA;  b3 = stiffening)
```

`a3 > 0` strain-stiffening (filled rubber, fiber-reinforced tissue); `a3 < 0`
strain-softening. A tip moment makes the curvature constant and
statically-determinate (`M(s)=M_tip`), so a load sweep gives clean `(M, κ)`
pairs.

## Identification

Fit the cubic by linear least squares on the `[κ, κ³]` basis → recover `a1, a3`
(and `b1, b3` from an axial sweep). A **linear-only** fit (`a1` alone) leaves a
large systematic residual at high load — the signature of hyperelasticity, which
this both detects and quantifies.

## Results (`scripts/exp_hyperelastic.py`)

True `a1=1.0, a3=0.6, b1=15, b3=25`:

| noise | a1 (EI) | a3 (stiffen) | b1 (EA) | b3 | linear / cubic residual |
|---|---|---|---|---|---|
| clean | 1.000 | 0.600 | 15.00 | 25.0 | 0.105 / 0.0000 |
| 3e-3 | 1.004 | 0.595 | 14.97 | 25.0 | 0.104 / 0.0010 |
| 6e-3 | 1.009 | 0.589 | 14.94 | 24.9 | 0.104 / 0.0021 |

All four coefficients recovered to ~1%. The **linear-fit residual is ~100× the
cubic-fit residual**, so the hyperelastic nonlinearity is decisively detected,
not assumed. Strain-softening (`a3<0`) is recovered with the correct sign too.

## Where this sits

This closes the **material-nonlinearity** gap — the rod is no longer assumed
linear-elastic. It composes with the actuation modules (sweep actuation instead
of external load to get the `(M,κ)` pairs). Remaining for full coverage:
**viscoelasticity** (rate-dependence / relaxation) and **contact**.

## Reproduce

```
python3 scripts/exp_hyperelastic.py
```

Module: `pinn_engine/baselines/hyperelastic_rod_id.py`.
