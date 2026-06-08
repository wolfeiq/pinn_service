# Viscoelastic soft-rod identification

Real soft-robot materials are **viscoelastic** — their response depends on the
*rate* and *history* of deformation: they creep under constant load, relax under
constant strain, and dissipate energy (hysteresis) under cyclic loading.
Hyperelasticity captures nonlinearity; viscoelasticity captures time-dependence.
This module identifies a rod's viscoelastic bending law from time-resolved
deformation, two independent ways.

## Model — Standard Linear Solid (Zener)

An equilibrium spring `E_∞` in parallel with a Maxwell branch (spring `E_1`,
dashpot, relaxation time `τ`):

```
M(t) = E_∞·κ(t) + q(t),     q̇ = −q/τ + E_1·κ̇
```

the minimal model with a finite instantaneous (glassy) modulus `E_g = E_∞ + E_1`,
relaxation, and creep.

## Two self-experiments

**Creep** — hold a constant actuation moment `M_0`; the curvature drifts

```
κ(t) = M_0[J_∞ − (J_∞ − J_g)·e^{−t/τ_c}],   τ_c = τ·E_g/E_∞
```

from instantaneous `M_0/E_g` to equilibrium `M_0/E_∞`. Measured purely from shape
over time; fit the three parameters.

**DMA (oscillatory)** — drive `M(t) = M_0 sin ωt` over a frequency sweep; the
curvature lags by `δ(ω)`, giving the storage / loss moduli

```
E'(ω) = E_∞ + E_1·ω²τ²/(1+ω²τ²)      E''(ω) = E_1·ωτ/(1+ω²τ²)
```

The loss modulus **peaks at ωτ = 1** — the viscoelastic fingerprint.

## Results (`scripts/exp_viscoelastic.py`)

True `E_∞=1.0, E_1=1.5, τ=0.5` (glassy `E_g=2.5`, loss peak at ω=2):

| method | noise | E_∞ | E_1 | τ |
|---|---|---|---|---|
| creep | clean | 1.000 | 1.500 | 0.500 |
| DMA | clean | 1.000 | 1.500 | 0.500 |
| creep | 4e-3 | 0.999 | 1.507 | 0.499 |
| DMA | 4e-3 | 1.000 | 1.501 | 0.499 |

Storage / loss sweep (clean): `E'` rises from `E_∞=1.0` toward `E_g=2.5`; `E''`
peaks at **ω=2 = 1/τ** (`E''=0.750`). The **two independent methods agree** to
<1% and recover other materials equally well — a strong internal consistency
check.

## Where this sits

This closes the **rate-dependent material** gap, completing the material story
(linear → hyperelastic → viscoelastic). The soft-rod stack now spans: full
Cosserat kinematics (planar/3-D, static/dynamic), stiffness ID, two drives
(tendon + pneumatic), nonlinear + rate-dependent material, and contact sensing.

## Reproduce

```
python3 scripts/exp_viscoelastic.py
```

Module: `pinn_engine/baselines/viscoelastic_rod_id.py`.
