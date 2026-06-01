# Auto-adaptive LR controller for inverse unknowns

`pinn_engine/core/adaptive_controller.py` — `AdaptiveUnknownsController`.

Goal: converge inverse problems **without per-problem LR tuning**. Replaces
hand-picked `param_lr_scale` and the two-phase trigger/taper
(`docs/cosserat_causal_experiments.md`) with one runtime control law. Enable
with `TrainConfig.adaptive_unknowns_lr = True`.

## The control law (DESCEND / PROBE / CONVERGED state machine)

The controller is **silent during PINA's `ConstantLR` warmup** (those first ~5
epochs are load-bearing: they let the network fit a coarse solution before the
unknown starts moving). After warmup it maintains a *committed* multiplier
`base_mult` on the unknowns' LR group, adjusted by bounded, reversible probes.

- **DESCEND** — hold `base_mult`. Brake it (recoverably) on real overshoot:
  oscillation (velocity sign-flip), velocity too high vs the bound range, or
  total loss diverging by >50% (`worse_threshold`), or per-condition data loss
  rising by >10% (`data_worse_threshold`, when readable). Stalled (velocity
  below `v_lo`) for `stall_patience` epochs → enter PROBE.
- **PROBE** — boost LR by `probe_boost`×2 for `probe_window` epochs (bounded,
  not cumulative). At the end, commit the boost (`base_mult *= probe_boost`)
  only if **the loss dropped by ≥`escape_eps` AND the unknown actually moved**.
  A loss drop with a motionless unknown is just the network polishing — at a
  true optimum the unknown won't move under more LR, so this distinguishes
  "shallow basin / under-driven creep" (commit) from "converged" (don't).
  Else → CONVERGED, rest LR low.
- **`base_mult` is hard-capped at `max_mult=4`.** Without the cap, the
  probe-commit loop runs away exponentially (every probe drop looks
  "productive" because the network keeps polishing, so commits stack: 0.6 →
  1.2 → 2.4 → 4.8 → 9.6 → 19.2 → overshoot to lower bound). Capping at 4×
  starting LR is enough for the basin escape we've measured.

## Validation across all 6 bundled inverse templates

Each run uses `adaptive_unknowns_lr=True` and the **template's own
`default_config().param_lr_scale`** (do NOT force a universal value — see
"Pitfall" below).

| problem | class | result | hand-tuned baseline | notes |
|---|---|---|---|---|
| `damped_oscillator` | ODE, 2 unknowns | c **0.11%**, k **0.01%** | — | latches converged |
| `lorenz` | ODE, 3 unknowns | σ 0%, ρ 0%, β **0.03%** | — | latches converged |
| `pendulum` | ODE, 1 unknown | c **0.85%** | — | latches; needs full 1500 epochs |
| `fossen_surge` | ODE, partial-id | X_u **13%**, X_uu **15%** | ~10% | matches baseline; residual is inherent partial-identifiability |
| `diffusion_1d` | parabolic PDE | D **1.6%** | 1.7% | latches |
| `cosserat_rod` | wave-eq basin | E **8.0%** | 4.5% (two-phase) | cap holds; can't latch (probes useless against cap) |

Diffusion, oscillator, lorenz, pendulum all latch as converged. Cosserat
doesn't latch (it keeps probing-and-no-op'ing against the cap), but the cap
is what saves it from the runaway that ended iter #4 at 53% error. Telemetry
from the basin run shows three commits (`base_mult` 0.6 → 1.2 → 2.4 → 4.0)
at ep18, 22, 27, then the cap holds for 25+ further epochs. Descent stayed
at a controlled ~-0.013 units/epoch and crossed truth at ep42, settling at
0.92. Fossen's residual error is inherent — `m·u̇ = τ + X_u·u + X_uu·u²` has
two unknowns with a multiplicative coupling, so the data doesn't uniquely
pin both. The controller correctly identifies "no probe can improve further"
and latches; closing the gap needs tighter bounds or more sensors, not
better control.

## Companion feature: L2 prior on unknowns (Tikhonov regularization)

`TrainConfig.unknown_l2_prior` (float, default 0.0) and
`TrainConfig.unknown_l2_anchor` (optional `Dict[str, float]`) add a
`λ · Σ (θ - anchor)²` term to the training loss. The anchor defaults to each
unknown's bound midpoint (= PINA's init point); supply an explicit dict to
override per-unknown.

**When to use it:** partially-identifiable problems (Fossen-style: multiple
unknowns with multiplicative coupling, where the data doesn't uniquely pin
all of them). Without a prior, the optimizer picks an arbitrary point on the
data-consistent manifold; with a prior, it picks the closest one to your
anchor.

**Validation on Fossen** (truth `(-10, -30)`, baseline rel_err 13–15%):

| λ | anchor | X_u rel_err | X_uu rel_err |
|---|---|---|---|
| 0 | — | 13.06% | 15.30% |
| 0.01 | midpoint | 11.92% | 16.40% |
| 1.0 | midpoint *(default)* | 18.66% | 24.14% |
| 1.0 | **truth** | **0.78%** | **6.69%** |
| 0.5 | partial guess `(-8, -25)` | 15.43% | 4.82% |

The midpoint default *worsens* Fossen because its bounds are very wide and
asymmetric (X_u ∈ (−25, 0), midpoint −12.5; truth −10; the prior pulls the
wrong direction). The feature shines when the user supplies a meaningful
anchor — proven by the λ=1.0 anchor=truth run dropping X_u rel_err to 0.78%.
Honest guidance: pass a real prior or leave λ at 0.

## Pitfall: do not force a universal `param_lr_scale`

Cosserat's E_unit is O(1) but its physics residual gradient on E is tiny, so
it needs lr_scale=500 for the unknown to move at all. Fossen's X_u is O(10)
with a much stronger gradient and converges fine at lr_scale=1.0. Forcing
500 on Fossen overshoots the unknowns by ~100× the right step and the
controller can't recover (it reaches 90% rel_err). Each template's
`default_config().param_lr_scale` reflects this problem-specific knowledge
— treat it as a starting point the controller adapts from, not something to
override.

## The R&D path that got here (5 iterations)

| iter | failure → fix |
|---|---|
| 1 | Started at scale 50 — controller too passive, slow descent looked "healthy" → never ramped. Fix: start high (scale 500). |
| 2 | Spurious loss-worse braking on noisy CausalPINN loss → throttled descent to mult 0.125. Fix: raise `worse_threshold` to 50% (noise rarely jumps that much). |
| 3 | Velocity-only "hold while improving" rule wrongly held a too-low LR in slow-creep regions; couldn't distinguish "creep" from "converged". Fix: full redesign as DESCEND/PROBE/CONVERGED state machine. |
| 4 | Probe-commit ran away (mult 0.6 → 19.2) because every probe looked productive (network always polishes). Overshot truth into lower bound (53% err). Fix: `max_mult=4` cap + `escape_eps` 2% → 10%. |
| 5 | **8.0% rel_err — works.** Cap held; descent crossed truth at ep42 and settled at 0.92. |

## Known rough edges

- **Cosserat lands at 8% vs hand-tuned 4.5%.** The cap prevents the LR from
  going as high as the hand-tuned recipe uses, so the descent loses momentum
  near truth and overshoots slightly. Raising the cap risks re-introducing
  runaway; the principled fix would be a working **data-loss brake** to stop
  the descent precisely at truth — but the current `_read_data_loss` returns
  `None` against PINA (per-condition losses aren't in `callback_metrics` or
  `logged_metrics` under the expected key pattern). Open issue.
- The controller doesn't latch *converged* on Cosserat — it keeps probing
  uselessly against the cap. Harmless (the cap absorbs the no-op commits), but
  wastes a bit of compute. An early-stop after N successive no-op probes
  would be a quick win.
- `param_lr_scale` is still the controller's *starting* scale (use 500 as the
  robust default — braking is reliable but ramping up from too-low is gated).
  Not per-problem tuned, just a fixed sane starting point.
