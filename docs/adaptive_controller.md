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

## Validation (all with zero per-problem config)

| problem | class | result | hand-tuned baseline |
|---|---|---|---|
| `diffusion_1d` | parabolic PDE | **D = 0.0984, rel_err 1.6%** | 1.7% |
| `damped_oscillator` | ODE, 2 unknowns | **c=0.501 (0.11%), k=10.001 (0.01%)** | — |
| `cosserat_rod` | wave-eq PDE (basin) | **E = 0.9201, rel_err 8.0%** | 4.5% (two-phase #16) |

Diffusion and oscillator latch as converged. Cosserat does not latch (it keeps
probing-and-no-op'ing against the cap), but the cap is what saves it from the
runaway that ended iter #4 at 53% error. Telemetry from the basin run shows
three commits (`base_mult` 0.6 → 1.2 → 2.4 → 4.0) at ep18, 22, 27, then the
cap holds for 25+ further epochs. The descent stayed at a controlled ~-0.013
units/epoch and crossed truth at ep42, settling at 0.92.

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
