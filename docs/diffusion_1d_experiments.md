# Diffusion-1D inverse: does the Cosserat recipe generalize?

Recover diffusivity `D` (truth 0.1) in `∂u/∂t = D·∂²u/∂x²` from a dense
noisy `u(x, t)` grid. This is the second PDE inverse template, added to test
whether the two-phase-LR recipe that converged the Cosserat wave equation
(`docs/cosserat_causal_experiments.md`) transfers to a parabolic problem.

## Result: converges in one shot — the recipe is NOT needed here

| config | epochs | wall | final D | rel_err |
|---|---|---|---|---|
| plain: lr_scale=100 + PINA warmup, **no two-phase**, vanilla PINN | 50 | **33 s** | **0.0983** | **1.7%** |

Trajectory: brief overshoot during the warmup (0.50 → 0.22 → 0.05 → ~0 at
ep2), then smooth monotonic convergence once full LR kicks in at ep5:
0.065 (ep5) → 0.088 (ep15) → 0.093 (ep20) → 0.098 (ep50).

## Why this is the expected (and informative) outcome

The Cosserat 1.98 plateau was a **wave-equation** non-convexity (a spurious
`u(s, t/√2)` solution at E=2). Diffusion is **parabolic and well-posed** — no
such basin. So `D` descends freely toward truth; the only transient is a mild
overshoot/oscillation (opposite failure mode to Cosserat's trap), which the
warmup + a moderate `lr_scale` damp out on their own.

**Conclusion on generalization:** the engine's inverse machinery transfers
cleanly. The *universal* pieces are `param_lr_scale` (amplify the unknown's
gradient vs Adam normalization) + PINA's `ConstantLR` warmup (let the network
fit before the unknown moves fast). The **two-phase trigger/taper is an opt-in
tool for non-convex basins** (wave eq), not an always-on requirement. Diffusion
needs only the universal pieces.

Note: `D` briefly dipped to −0.002 (ep2) despite bounds `(1e-3, 1.0)` — PINA's
unknown bounds are init hints, not hard clamps. It recovered without issue
here, but a hard projection would be a cheap robustness add for tighter-bounded
problems.

## Reproduce

```
python3 scripts/run_diffusion.py        # ~33 s on MPS
```
Template: `pinn_engine/dsl/templates_lib/diffusion_1d.py`. Driver: `scripts/run_diffusion.py`.
