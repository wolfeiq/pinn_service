# Auto-adaptive LR controller for inverse unknowns

`pinn_engine/core/adaptive_controller.py` — `AdaptiveUnknownsController`.

Goal: make the engine converge inverse problems **without per-problem LR
tuning**. Replaces hand-picked `param_lr_scale` + the two-phase
trigger/taper (`docs/cosserat_causal_experiments.md`) with one runtime control
law. Enable with `TrainConfig.adaptive_unknowns_lr = True`.

## The control law

Every epoch (after PINA's warmup, which the controller leaves alone), it
adapts the unknowns' optimizer-group LR from two signals:

1. **Velocity** of each unknown, relative to its bound width.
   - too fast / sign-flipping (oscillating) → **brake** (`lr *= lr_down`)
   - in the healthy band → **hold**
   - too slow (frozen/plateaued) → see loss signal
2. **Loss response** (the key to not breaking easy problems):
   - frozen **and loss still improving** → **hold** (the current LR is working;
     the network is still fitting the field and the unknown will follow —
     ramping here destabilises a converging solution)
   - frozen **and loss stagnant** → **probe**: ramp LR to test whether a
     lower-loss solution is reachable (a shallow basin). If the probe reduces
     loss, it keeps going; if it never does after `patience` epochs → **latch
     converged** and rest the LR low.
   - a probe that makes loss **worse** → **brake immediately** (overshoot).

This unifies the three failure modes characterised on Cosserat/diffusion:
frozen (ramp), basin trap (probe escapes → loss drops → keep going), and
overshoot/oscillation (brake). The loss signal is what distinguishes a shallow
*trap* (escapable, loss drops on probe) from a *true optimum* (loss can't drop
→ converged) — velocity alone cannot, since an unknown sitting at truth also
has ~0 velocity.

## Validation

| problem | class | result (no per-problem tuning) | notes |
|---|---|---|---|
| `diffusion_1d` | parabolic PDE | D=0.0984, **rel_err 1.6%** | matches the hand-tuned baseline; controller braked the early bounce then held |
| `damped_oscillator` | ODE (2 unknowns) | c=0.501 (0.25%), k=9.995 (0.05%) | both unknowns; latched converged |
| `cosserat_rod` | wave-eq PDE (basin) | **pending** (run #adaptive) | the crown test: must auto-escape the 1.98 basin and brake near truth, reproducing run #16 (~4.5%) with no hand-tuned trigger/taper |

## Known rough edges

- On temporary (non-converged) loss plateaus the probe can briefly ramp the LR
  and bump the loss before the loss-worse brake catches it. Self-corrects, but
  wastes a few epochs. Gentler probing / longer stagnation window would smooth
  it.
- `param_lr_scale` is still used as the controller's *starting* scale (default
  ~50 in validation runs); the controller adapts from there. Not per-problem
  tuned, but a fixed sane starting point.
