# Cosserat-rod CausalPINN inverse-problem experiments

Tracking the May 24–25, 2026 run series. Truth: `E_unit = 1.0`. All runs use
`template=cosserat_rod`, `seed=42`, `adam_epochs=50` (unless noted), with
`UnknownsDumper` writing E_unit per epoch (run #4 onward).

## The bug

PINA's `CausalPINN.__init__` defaults to `eps=100`. The temporal causal weight
`ω_i = exp(-ε · Σ L_r(t_k))` collapses to ~0 on epoch 1 for any non-trivial
residual, silently muting the physics term. v466 (May 23) ran 235 epochs with
`physics_0_loss ≡ 0` because of this.

**Fix** (committed in trainer.py + causal_eps_scheduler.py + unknowns_dumper.py):

1. Plumb `causal_eps` through `TrainConfig` (default 1.0, not 100).
2. Override `loss_phys` to log `causal/{max_bucket_loss, min_weight, active_buckets, eps}`.
3. `CausalEpsAnnealer` callback: bidirectional ε adjustment (shrinks on collapse,
   grows when `max_bucket_loss < threshold` per Wang 2022 §3.2).
4. `UnknownsDumper` callback: writes `problem.unknown_parameters` to
   `logs/<run_id>_live.json` every epoch — survives OOM/SIGKILL.

## Runs

| # | run_id | lr_scale | lam_data | balancer | causal_eps | epochs (done/budget) | E_unit final | rel_err | wall (s) | status | summary |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | (v466, May 23) | 1 | 5000 | none | **100** (broken) | 235/1500 | n/a (PINA stores unknowns outside state_dict; no dumper) | n/a | ~3h | physics_loss flat at 0 | (lost — /tmp/ wiped) |
| 2 | (PID 7616) | 1 | 5000 | none | 1e-2 → shrink | 4/200 | n/a (killed during shrink phase) | n/a | ~16 min | killed: shrink phase too slow; reset eps_init | — |
| 3 | (PID 7880) | 1 | 5000 | none | 1e-8 | 68/100 | n/a (no dumper, PINA loses unknowns on SIGKILL) | n/a | ~11h | OOM-killed (~62MB free, macOS reclaimed) | — |
| 4 | d8f486e2 | 1 | 5000 | none | 1e-8 | **50/50** | **5.0403** | **4.04** | 19 917 | completed | [summary](../logs/d8f486e2f858413c9ebda5c3f3b281a2_summary.json) |
| 5 | (PID 16400, lr_scale=10, original) | 10 | 5000 | none | 1e-8 | killed @ ep13 | (was 4.96 trending to ~4.93) | (~3.93 est.) | ~1.5h | killed: prior to user choice — re-launched with lam_data=100 | — |
| 6 | f95c053b | 10 | **100** | none | 1e-8 | **50/50** | **4.9533** | **3.95** | 19 611 | completed | [summary](../logs/f95c053b4d2e44f2929327bab72c2157_summary.json) |
| 7 | 40b1b194 | 10 | 100 | **sapinn** | 1e-8 | killed @ ep30 | 4.9529 | 3.95 | ~3.5h | killed: trajectory identical to #6 to within 0.001 — balancer doesn't move the needle | (no summary, killed) |
| 8 | cbebd61f | **100** | 100 | none | 1e-8 | **50/50** | **4.0908** | **3.09** | ~5.6h | completed | [summary](../logs/cbebd61f030940d39bba1764300601be_summary.json) |
| 9 | ea06e951 | **500** + cosine→0.05 | 100 | none | 1e-8 | **50/50** | **1.9833** | **0.98** | 19 420 | completed — but flatlines at 1.98 (likely wave-eq non-uniqueness basin) | [summary](../logs/ea06e9514ca840adbb3b329eb2ab4a37_summary.json) |
| 10 | (PID 45787) | **500** + **no cosine** (min=1.0) | 100 | none | 1e-8 | running | tbd | tbd | running | — | — |

### Trajectory comparison at matched epochs (E_unit)

| Epoch | #4 lr=1 | #6 lr=10 | #7 lr=10 + sapinn | #8 lr=100 |
|---|---|---|---|---|
| 0 | 5.0487 | 5.0367 | 5.0367 | 4.917 |
| 1 | 5.0475 | 5.0246 | 5.0246 | 4.796 |
| 2 | 5.0465 | 5.0149 | 5.0148 | 4.700 |
| 5 | 5.0440 | 4.9897 | 4.9893 | (interpolated ~4.34) |
| 10 | 5.0411 | 4.9614 | 4.9606 | tbd |
| 20 | 5.0408* | 4.9545 | 4.9536 | tbd |
| 30 | — | 4.9539 | 4.9529 | tbd |
| 50 (final) | 5.0403 | 4.9533 | — | **4.091** |

**Linear scaling of total E_unit drift with `param_lr_scale`:**

| `param_lr_scale` | Final E_unit | Δ from 5.05 midpoint | Predicted at lr=500 (5× more) |
|---|---|---|---|
| 1 | 5.040 | -0.010 | -0.050 (→ 5.000) |
| 10 | 4.953 | -0.097 | -0.485 (→ 4.565) |
| 100 | 4.091 | -0.959 | -4.795 (→ 0.255) — **overshoots truth=1.0** |
| 500 + cosine | 1.983 | -3.067 | actual; linear predicted -4.795 → cosine cut it short |

**Run #9 trajectory exposes a basin at E_unit ≈ 1.98:**

| Epoch | E_unit | Δ per epoch |
|---|---|---|
| 0 | 4.385 | — |
| 1 | 3.788 | -0.597 |
| 2 | 3.315 | -0.473 |
| 5 | 2.508 | ~-0.27 avg |
| 10 | 2.094 | -0.083 |
| 15 | 2.014 | -0.016 |
| 20 | 1.997 | -0.003 |
| 30 | 1.987 | -0.001 |
| 50 | 1.983 | ~-0.0001 |

E_unit reaches 2.0 by epoch 15 and then stalls — but the cosine LR has only decayed to ~80% of peak at that point, so this isn't the LR cutoff. The basin at E ≈ 2 is locally stable: the network has likely fit `u(s, t/√2)` (scaled time) which makes `ρ·u_tt = E·u_ss` balance at E = 2 instead of E = 1.

Run #10 tests whether removing cosine annealing lets the unknown escape that basin (oscillation around 1.98 → trapped; drift continues below 1.98 → escape possible).

\* #4 history beyond epoch 13 wasn't dumped with the same cadence (callback added in #6).

### physics_0_loss trajectory (epoch 0 → 50)

All non-broken runs follow nearly identical paths because the seed and (until #6)
loss weights only enter the train_loss aggregate, not the per-condition gradients:

- Epoch 0: ~91 700 (collapsed in #1 because ε=100; meaningful in #2 onward)
- Epoch 5: ~2 800
- Epoch 10: ~290
- Epoch 25: ~20
- Epoch 50: ~8.4

### Diagnostic findings

* **Bug fix verified.** Going from #1 to #4: `physics_0_loss` went from flat-zero
  to a 4-orders-of-magnitude decrease over the run. The PINA `eps=100` default
  was the silent muter.
* **`causal_eps` needs scale-matching.** Init residuals on this problem are ~1.4e7.
  Even ε=1e-2 collapses ω → 0. ε=1e-8 keeps ω ≈ 0.87 at epoch 0; the annealer
  was supposed to grow ε once `max_bucket_loss < threshold=1e-2`, but on this
  problem `max_bucket_loss` plateaus around 100–200, so **ε never grows** —
  the causal weighting stays effectively flat (ω ≈ 1) the whole run. The mechanism
  works; it just doesn't *activate* on this Cosserat config.
* **`param_lr_scale=1` (the cosserat template default) makes E_unit nearly stationary.**
  Drift was ~0.001/epoch decaying. 50 epochs moved E_unit by 0.008.
* **`param_lr_scale=10` gave ~10× drift but same plateau.** E_unit moves 5.05 → 4.95
  (~0.1 total) instead of 5.05 → 5.04 — confirms scaling holds but the plateau
  is a deeper issue.
* **`lam_data=5000 → 100` did nothing.** Hypothesis was data-overweighting; falsified:
  while physics_loss is large it dominates train_loss either way, and once
  physics_loss is small the network has already locked into a local basin.
* **`sapinn` balancer did nothing.** Per-condition learnable λ produces trajectory
  identical to static `balancer="none"` within 0.001. The bottleneck is not loss
  weighting at all.
* **Next test: `param_lr_scale=100` (run #8).** Effective Adam LR on the unknown is
  1e-1, aggressive for a (0.1, 10)-bounded parameter — overshoot risk noted.

### Open hypotheses for the plateau

1. **Local minimum / lazy regime.** The network finds an (E_unit ≈ 5, u(s,t))
   pair that fits the wave equation. Once there, gradient flow is genuinely small
   regardless of LR scaling. Would need to test a hot-start from E_unit ≈ 1
   (truth) to see if the basin is stable or escape-able.
2. **Identifiability with this sensor set.** A single noisy strain-gauge + exact
   BC/IC + 50 epochs may not be enough information to pin E down. Published
   inverse-Cosserat runs use 10k–100k epochs.
3. **Adam normalization.** Adam's `1/sqrt(exp_avg_sq)` denominator stays elevated
   from early high gradients, so even with raised LR the *effective* step is
   damped. Switching to plain SGD or LBFGS on the unknown might help.

## Files

- Code changes:
  - `pinn_engine/core/trainer.py` — `causal_eps`, `causal_eps_anneal`,
    `causal_eps_max`, `causal_eps_threshold` in `TrainConfig`; overridden
    `loss_phys` on `CausalLabeledDataPINN`.
  - `pinn_engine/core/causal_eps_scheduler.py` — `CausalEpsAnnealer` callback.
  - `pinn_engine/core/unknowns_dumper.py` — `UnknownsDumper` callback.
- Driver: `scripts/run_cosserat_causal.py`.
- Logs: `logs/cosserat_causal_<timestamp>.out`, `logs/<run_id>_live.json`,
  `logs/<run_id>_summary.json`.
- Lightning metrics: `lightning_logs/version_<N>/metrics.csv` (468–472 for this series).
