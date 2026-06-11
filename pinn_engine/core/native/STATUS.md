# Native PINN core — de-PINA migration status

Goal: make `pinn-engine` standalone by replacing the runtime pieces currently
borrowed from PINA with the engine's own. **The engine's distinctive parts (the
DSL→torch residual compiler with autograd, networks, adaptive controller, CRLB,
RAR) were never PINA's** — only the PINN training scaffolding is.

## Scope (what actually touches PINA)

Only the **PINN training path** for the 14 templates:
`core/trainer.py`, `core/problem.py`, the `LabelTensor` users
(`core/rar_sampler.py`, `diagnostics/residual_heatmap.py`,
`diagnostics/sensor_residuals.py`, `preflight/wellposedness.py`),
`core/weightings.py`. **All soft-robot baselines are already pure NumPy/SciPy —
PINA-free.**

## Done

- `native/labeltensor.py` — `LabelTensor` (named-column `torch.Tensor` subclass
  with `.labels` + `.extract()`), drop-in for `pina.LabelTensor`. The DSL
  residual compiler works against it unchanged.
- `native/solver.py` — `NativeProblem` (collocation problem container),
  `build_native_problem`, and `train_native` (Adam loop: per-condition data +
  physics losses, separate unknown-LR group, warmup, bounds projection).
- **Verified PINA-free recovery:**
  - `damped_oscillator` (ODE, 2 unknowns): c 0.7%, k 1.0%.
  - `lorenz` (multi-output chaotic ODE, 3 unknowns): σ 0.2%, ρ 0.0%, β 1.0%.

  i.e. the DSL residual + autograd + multi-unknown / multi-output inverse all
  train natively, on par with the PINA path.

## Remaining (the parity work)

- **PDE parity.** `diffusion_1d` trains the field fine but the diffusion
  coefficient drifts to its bound — a degenerate (over-smooth u + large D)
  minimum the minimal `train_native` doesn't yet suppress. PINA's path avoids it
  via its tuned collocation/loss handling; the native trainer needs the same
  (collocation strategy, possibly residual-point weighting, the full warmup/LR
  schedule). This gates Burgers / Fisher-KPP / the rod PDE templates.
- Port **CausalPINN** (time-bucketed causal weighting) for the wave-equation /
  cosserat templates.
- **Callback compatibility**: the adaptive controller, RAR sampler, and
  param-LR scheduler read PINA-solver internals + Lightning metrics; they need a
  native solver exposing the same surface (or light adapters).
- Swap `trainer.py` / `problem.py` to the native path and **re-validate every
  template** against the recorded PINA results before removing the PINA import.

The PINA path remains the default until parity is proven end-to-end.
