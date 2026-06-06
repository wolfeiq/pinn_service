# Soft-robotics planar elastica inverse: recovering bending stiffness

Recover the dimensionless bending stiffness `EI_unit` (truth 1.0) of a soft
continuum rod from its measured tangent-angle profile `θ(s̃)`. This is the
**geometrically-exact** Cosserat/Kirchhoff planar rod — the large-deflection
step up from the linear `euler_bernoulli_beam`, and the regime a soft-robotic
finger / continuum manipulator actually operates in.

## The problem

A slender rod clamped horizontal at `s=0` carries a dead tip load `P0`
(downward) at the free end `s=L`. With the centerline written by its tangent
angle `θ(s)` (`x' = cos θ`, `y' = sin θ`), the bending moment `M = EI·θ'`
balances the tip load:

```
EI · θ''(s)  =  −P0 · cos(θ(s)),   θ(0) = 0,   θ'(L) = 0
```

The `cos(θ)` is the geometric nonlinearity. At the default load parameter
`α = P0·L²/EI = 2.5` the tip rotates **51°** and droops **0.56·L** — far
outside the small-slope regime where `euler_bernoulli_beam`'s linear
`w'''' = q` would hold (linear theory mispredicts the tip by tens of percent
here). So this template genuinely exercises a nonlinear residual, not just a
relabelled beam.

Non-dimensionalised (`s̃ = s/L`, `EI = EI_unit·EI_ref`), the compiled residual
is `EI_unit·θ''(s̃) + α_ref·cos(θ) = 0` with `α_ref = 2.5`. The unknown sits
multiplicatively on the highest derivative, O(1) — same well-conditioned family
as the beam and bar templates.

**Measurement model.** `θ(s̃)` is what flexible curvature sensors (fiber-Bragg
gratings, IMU arrays, stretch sensors) report directly along a soft rod, so the
angle formulation *is* the physical sensing model — no need to numerically
differentiate a measured shape into curvature.

Ground truth comes from `scipy.integrate.solve_bvp` on the nonlinear BVP
(`pinn_engine/data/synthetic.py::generate_planar_elastica`), sampled at 31
interior angle sensors with `noise_std = 1e-2` rad.

## Identifiability (CRLB preflight)

```
   Unknown       truth     CRLB SE  SE/|truth|   95% CI half
   EI_unit                0.004594       0.46%      0.009003
```

The data-theoretic floor is **0.46%** — a well-identified single-unknown
problem (a single angle sensor already pins curvature; 31 of them average down
the noise). Any engine result in the 0.5–1% band is essentially at the floor.

## Results

Sweep on CPU, seed 0 (`scripts/exp_planar_elastica.py all`):

| config | epochs | wall | final EI_unit | rel_err |
|---|---|---|---|---|
| **rar** (residual-adaptive collocation) | 3000 | 69 s | 1.00565 | **0.565%** |
| baseline (param_lr_scale=20) | 3000 | 68 s | 1.00648 | 0.648% |
| lbfgs (Adam + L-BFGS-100) | 3000 | 69 s | 1.00655 | 0.655% |
| long | 8000 | 187 s | 1.00724 | 0.724% |
| adaptive (controller, scale=20 base) | 3000 | 68 s | 1.00866 | 0.866% |
| ~~adaptive (scale reset to 1.0)~~ | 3000 | 66 s | 4.74971 | **375%** ✗ |

**The template converges near the CRLB floor (0.46%) out of the box.** The
nonlinear `cos(θ)` residual is *not* a problem for the engine — a plain tanh
MLP with `param_lr_scale=20` lands at 0.648% in ~70 s, and RAR shaves it to
**0.565%** (~1.2× the data floor) at negligible extra wall-time. As a smoke
check, even 500 Adam epochs already reaches 1.57%.

### Findings

1. **Geometric nonlinearity is cheap here.** Unlike the `cosserat_rod` wave
   equation (non-convex `u(s, t/√2)` basin at E=2), the static elastica is
   convex in `EI_unit` given the measured `θ(s̃)` — `EI_unit = −α_ref·cos θ /
   θ''` is a direct read-off. No Fourier features, no two-phase LR, no causal
   weighting needed. `θ(s̃)` is smooth and monotone, so a tanh MLP's smooth
   inductive bias is exactly right.

2. **More budget doesn't help a converged minimum.** 8000 epochs is no better
   than 3000 (0.72% vs 0.65%, within noise) — the residual data-fit, not Adam
   budget, sets the floor. L-BFGS likewise leaves it unchanged: the Adam
   minimum is already the L-BFGS minimum (contrast `cosserat_rod`/beam, where
   L-BFGS diverges or the problem is under-budgeted).

3. **RAR is the right (small) lever.** Even on this smooth-solution problem RAR
   gives a modest, reliable edge (0.648% → 0.565%) by concentrating
   collocation where `|EI_unit·θ'' + α_ref·cos θ|` is largest — mirroring its
   diffusion_1d win (0.64% → 0.14%). It composes cleanly with the nonlinear
   residual.

4. **Adaptive-controller pitfall (reproduced, then explained).** Resetting
   `param_lr_scale` to 1.0 when enabling the controller is a trap: the
   controller's `max_mult=4` cap then bounds the unknown's LR at 4e-3, which
   cannot move `EI_unit` off its midpoint init (5.05) in 3000 epochs — it
   stalls at 4.75 (**375% error**). Keeping the template's `param_lr_scale=20`
   as the controller's base LR fixes it (0.866%). This is the concrete failure
   mode behind the standing guidance *"start the controller from each
   template's own `param_lr_scale` default."*

**Bottom line:** the geometrically-exact soft-rod bending inverse is *solved*
to ~1.2× the CRLB floor with the default recipe; RAR is the recommended
production setting.

## Reproduce

```
python3 scripts/exp_planar_elastica.py all        # full sweep
python3 scripts/exp_planar_elastica.py baseline   # single config
```

Template: `pinn_engine/dsl/templates_lib/planar_elastica.py`.
Data: `generate_planar_elastica` in `pinn_engine/data/synthetic.py`.
