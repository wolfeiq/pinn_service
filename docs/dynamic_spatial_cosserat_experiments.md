# Dynamic 3-D spatial Cosserat rod: stiffness ID from 3-D motion

The capstone of the soft-robot rod suite: the full geometrically-exact **spatial
Cosserat rod with inertia** — `r(s,t) ∈ ℝ³` and `R(s,t) ∈ SO(3)` evolving in
time, with 3-D rotational dynamics (gyroscopic term included). It unifies the
dynamic (time-domain) and 3-D (spatial) rods, and recovers all **six**
stiffnesses (EA, GA₁, GA₂, GJ, EI₁, EI₂) from the measured motion.

## Forward model (verified)

A clamped rod, optionally **pre-twisted**, released under off-axis gravity;
method of lines (discrete elastic rod) in `s` + RK45 in `t`. Per node:

```
ρA r_tt = ∂n_sp/∂s + ρA g − c r_t                       (linear momentum)
J_ρ Ω_t = Rᵀ(∂m_sp/∂s + r'×n_sp) − Ω×(J_ρ Ω)            (angular momentum, body)
q_t = ½ q ⊗ (0, Ω)                                       (quaternion kinematics)
```

Verified (`test_templates.py`):

- **Energy conservation** (undamped): total energy varies < 0.05 while ~2.4 of
  PE↔KE is exchanged (conserved to ~2e-2 of the swing; ~4e-4 in the prototype at
  finer resolution).
- **Planar reduction**: an in-plane isotropic load keeps the motion at `z = 0`.
- **Cross-check**: reproduces the independent planar dynamic solver
  (`_simulate_dynamic_cosserat`) tip trajectory to ~1e-4.
- **Genuine 3-D**: off-axis gravity + pre-twist produce out-of-plane motion and
  dynamic torsion.

## Inverse: kinematic force *and* moment + constitutive regression

In dynamics, both internal resultants are kinematic — derivable from the
measured motion and known inertia, **independent of the stiffnesses**:

```
n_sp(s,t) = −∫ₛᴸ ρA (r_tt − g) ds'                       (linear momentum)
H = R J_ρ Ω,   m_sp(s,t) = −∫ₛᴸ (dH/dt − r'×n_sp) ds'    (angular momentum)
```

The constitutive laws `n_mat = Rᵀn_sp = C_n Γ` and `m_mat = Rᵀm_sp = C_m K` are
then linear in the six stiffnesses and recovered by per-component least squares.
Savitzky-Golay derivatives (in `s` and `t`) give noise robustness; the
**pre-twist** is what makes torsion (GJ) well-conditioned — a gravity swing
alone barely excites it.

## Results (`scripts/exp_dynamic_spatial_cosserat.py`, N=50, n_t=161)

| noise (pos, quat) | EA | GA₁ | GA₂ | EI₁ | EI₂ | GJ |
|---|---|---|---|---|---|---|
| clean | 4.1% | 1.9% | 0.1% | 2.7% | 1.6% | 3.2% |
| 1e-3, 3e-3 | **4.3%** | **2.3%** | **1.0%** | **1.3%** | **2.0%** | **3.3%** |

All six stiffnesses of a fully **anisotropic, dynamic, 3-D** rod recovered to
≤4.3% from noisy motion — and barely degraded by noise (the residual is mostly a
systematic smoothing bias on the two *axial-direction* modes, EA and GJ, whose
strains are small relative to the frame). Recovers arbitrary non-unit stiffness
too (e.g. EI₁=1.2, GJ=0.7 → 1.21, 0.66).

## The rod suite is complete (geometry × time)

|  | static | dynamic |
|---|---|---|
| **planar** | `planar_cosserat` (0.30%) | `dynamic_cosserat` + force-from-motion |
| **3-D** | `spatial_cosserat_id` (≤3%) | **this** (≤4.3%) |

The same idea carried through all four: a statically-determinate or
inertia-derived **internal force/moment exposes each stiffness against a
data-derived quantity**, turning the inverse into a well-conditioned linear
regression — instead of fighting an under-resolved high-order derivative in a
collocation PINN.

Still open for "all soft robotics" (physics richness, not rod kinematics):
**actuation** (tendon/pneumatic inputs), **hyperelastic/viscoelastic** materials,
and **contact**.

## Reproduce

```
python3 scripts/exp_dynamic_spatial_cosserat.py        # full table (N=50)
```

Module: `pinn_engine/baselines/dynamic_spatial_cosserat_id.py`.
