# 3-D spatial Cosserat rod: recovering all six stiffnesses

The full geometrically-exact spatial Cosserat (Simo-Reissner) rod — the 3-D
generalisation of the planar templates and the model real continuum
manipulators need. The rod **bends in two planes, shears in two directions,
extends, and twists**, so it has six independent strains and six stiffnesses:

```
n_material = diag(EA, GA1, GA2) · Γ        (axial force + 2 shear forces)
m_material = diag(GJ, EI1, EI2) · K        (torsion + 2 bending moments)
```

State: centerline `r(s) ∈ ℝ³` and cross-section orientation `R(s) ∈ SO(3)`
(carried as a unit quaternion). Reference tangent `e1` (rod along x);
translational strain `Γ = Rᵀr' − e1` (Γ1 axial, Γ2/Γ3 shear); material curvature
`K` from `R' = R[K]×` (K1 torsion, K2/K3 bending).

## Forward model (verified)

A tip-loaded cantilever (clamped root, tip wrench `(P, Mt)`, no distributed
load), solved as a BVP with quaternion kinematics
(`simulate_spatial_cosserat`):

```
n_spatial' = 0            → n_spatial = P (constant)
m_spatial' = −r' × n_spatial
r' = R (C_n⁻¹ Rᵀ n_spatial + e1)
R' = R [C_m⁻¹ Rᵀ m_spatial]×
```

Verified against closed-form limits (in `test_templates.py`):

- **Pure axial force** → uniform stretch, tip `x = 1 + P/EA`, no lateral motion.
- **Pure twist moment** → straight rod, tip rotation `Mt₁/GJ` about the axis.
- **Transverse force** → planar elastica (motion stays in-plane, `z = 0`).

The quaternion norm holds to 1.0 along the rod (no drift). A general tip wrench
`P = (2, −3, 1.5)`, `Mt = (1, 0.3, −0.2)` excites all six strains (axial 0.13–0.24,
shears up to 0.20, torsion ~0.55, bendings up to ~1.9) and produces genuine
out-of-plane 3-D deformation.

## Inverse: statically-determinate force/moment + constitutive regression

A cantilever is **statically determinate**: the internal force is the known tip
force, and the internal moment follows from the measured shape + tip wrench,

```
n_spatial(s) = P            m_spatial(s) = Mt + (r(L) − r(s)) × P
```

— both **independent of the unknown stiffnesses**. So the constitutive laws are
linear in the six stiffnesses and recovered by per-component least squares from
the measured strains (`Γ` from `r', R`) and curvatures (`K` from `q'`). This is
the 3-D generalisation of the force-from-motion identifier that closes the
dynamic-rod gap: expose each stiffness against a data-derived force/moment
rather than fighting an under-resolved residual. Savitzky-Golay derivatives
give noise robustness.

## Results (`scripts/exp_spatial_cosserat.py`, n_s=121)

| noise (pos, quat) | EA | GA1 | GA2 | EI1 | EI2 | GJ |
|---|---|---|---|---|---|---|
| clean | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 1e-3, 3e-3 | **0.2%** | **0.7%** | **0.5%** | **3.0%** | **0.5%** | **1.4%** |
| 2e-3, 6e-3 | 0.7% | 3.3% | 1.8% | 10.8% | 1.5% | 4.8% |

All six stiffnesses of a fully **anisotropic** 3-D rod recovered (clean: exact;
standard noise: ≤3%). It also recovers arbitrary non-unit stiffness (e.g.
`EA=1.3, GJ=0.7, EI1=1.2` → exact), confirming it's a genuine estimator, not a
fit to truth=1. `EI1` is the hardest under noise because its bending curvature
changes sign along the rod (small-denominator slope); a larger smoothing window
mitigates it.

## Scope

This covers the **static** 3-D rod (the geometry/constitutive core). Still open
for "all soft robotics": actuation inputs (tendon/pneumatic), hyperelastic /
viscoelastic materials, contact, and the dynamic 3-D case (the planar dynamic
solver + this 3-D static solver are the two halves to merge).

## Reproduce

```
python3 scripts/exp_spatial_cosserat.py        # full table
```

Module: `pinn_engine/baselines/spatial_cosserat_id.py`
(`simulate_spatial_cosserat`, `generate_spatial_cosserat`,
`recover_spatial_stiffness`).
