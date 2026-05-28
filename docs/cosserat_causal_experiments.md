# Cosserat-rod CausalPINN inverse-problem experiments

Tracking the May 24–25, 2026 run series. Truth: `E_unit = 1.0`. All runs use
`template=cosserat_rod`, `seed=42`, `adam_epochs=50` (unless noted), with
`UnknownsDumper` writing E_unit per epoch (run #4 onward).

## SOLVED (run #16, May 28): E_unit = 0.9546, rel_err 4.5%

First converged run after a 16-run series that was stuck at ≥35% error. The
working recipe:

- `param_lr_scale = 500` — the unknown needs this much LR amplification to move
  against Adam's per-parameter normalization (lr=1/100 leave it ~frozen).
- **Keep** PINA's `ConstantLR(factor=1/3, 5-epoch)` warmup — load-bearing: it
  lets the network fit a coarse solution before the unknown moves fast. Removing
  it (run #15) overshot to the lower bound.
- `UnknownsParamLRScheduler` **silent pre-trigger** — the warmup governs the
  gradual descent through the basin; the callback does nothing until the trigger.
- `param_lr_trigger_below = 2.0` — fire the brake just past the basin floor (~2),
  while E is still high enough to leave deceleration room.
- `param_lr_taper_epochs = 5`, `min_scale = 0.02` — brake *fast* (a full-budget
  cosine brakes too late). 5-epoch cosine from full LR → 2%.

Trajectory: warmup+descent to 2.10 (ep5) → trigger at ep6 → decelerate
1.34→1.13→1.02 (ep7-9) → cross truth ep10 → settle at 0.955 and hold for 40
epochs. Predicted landing (hand-calc off #10's per-epoch steps) was 0.93-1.0.

Open: it settles slightly *below* truth (0.955). Triggering a hair higher or
braking a touch harder would center it on 1.0; 4.5% is already a converged result.

## The LR-capture bug (May 28) — invalidates the "cosine traps the basin" conclusion

**TL;DR: runs #9, #11, #12, #14 never actually ran at lr_scale=500. They ran at
~167. The "cosine annealing traps the unknown in the 1.98 basin" finding was an
artifact of this bug, not a real property of the loss landscape.**

PINA wraps every optimizer in `torch.optim.lr_scheduler.ConstantLR(factor=1/3,
total_iters=5)` — a 5-epoch warmup that runs the LR at ⅓ of target, then jumps to
full at epoch 5. `UnknownsParamLRScheduler` captured its `base_lr` by reading
`param_groups[idx]["lr"]` at `on_train_start` — i.e. the **warmup-discounted**
value (0.1667 = 0.5 × ⅓), not the true 0.5. It then *pinned* the LR there every
epoch, both running the unknown 3× too slow and defeating ConstantLR's jump to
full LR at epoch 5.

Consequence: any run with a scheduler attached (cosine or two-phase) trained the
unknown at effective lr_scale ≈ 167. Run #10 escaped the basin *only* because it
had **no** scheduler attached, so ConstantLR was free to restore lr=0.5 at epoch
5. So the real discriminator across #9–#14 was "scheduler attached (LR pinned 3×
low) vs not", not the cosine shape.

Verified by an instrumented run (`/tmp/debug_lr.py`-style probe): pre-fix the
unknowns LR read 0.1667 and stuck; post-fix it holds 0.5 every epoch.

**Fix** (trainer.py + param_lr_scheduler.py):

1. `UnknownsParamLRScheduler.on_train_start` now reads the true target LR from the
   scheduler's `base_lrs[idx]`, falling back to the param-group LR only if absent.
2. When the engine drives the unknowns' LR (cosine/two-phase), `train()` swaps
   PINA's warmup `ConstantLR` for a true-constant `ConstantLR(factor=1.0,
   total_iters=1)` — otherwise the warmup's epoch-5 milestone multiplies the
   current LR by 3, turning our 0.5 into a 1.5 spike for one epoch (the two
   schedulers fight over the same param group).

Re-running the two-phase experiment at the true lr=500 as run #15.

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
| 10 | 940de904 | **500** + **no cosine** (min=1.0) | 100 | none | 1e-8 | **36/50** | **0.3757** | **0.62** | ~16 500 | jetsam-killed at ep36 (system memory pressure, ran in parallel with #11) — **transited the 1.98 basin, crossed truth=1.0 around ep9, kept descending** | (no summary, killed) |
| 11 | 47b2886b | **500** + cosine→0.30 (min=0.3) | 100 | none | 1e-8 | **22/50** | **1.9786** | **0.98** | ~10 700 | jetsam-killed at ep22 (same memory pressure event) — re-trapped at 1.98; min_scale=0.30 still over-damps | (no summary, killed) |
| 12 | 9a9c5932 | **500** + cosine→0.70 (min=0.7) | 100 | none | 1e-8 | killed @ ep20 | (trending ~1.95) | ~0.95 | ~2.8h | killed: trajectory trapping like #9/#11 — **CONFOUNDED, see LR-capture bug below** | — |
| 13 | 8cc08c1b | 400 + no cosine | 100 | none | 1e-8 | killed @ ep14 | (asymptote ~1.35 est.) | ~0.35 | ~1.75h | killed: escaped basin but stalled ~1.35. No scheduler attached → **not** confounded; genuine lr=400 result | — |
| 14 | 1ff4955a | **500** two-phase (trigger<1.5, min=0.05) | 100 | none | 1e-8 | killed @ ep36 | (trending ~1.91) | ~0.91 | ~4.2h | killed: trapped — **CONFOUNDED.** Surfaced the LR-capture bug: trigger never fired because LR was pinned 3× low so E never reached 1.5 | — |
| 15 | b5a0e180 | **500** two-phase, **warmup removed** | 100 | none | 1e-8 | killed @ ep24 | 0.099 (lower bound) | 0.90 | ~2.8h | killed: OVERSHOT. Removing PINA's warmup → full lr=500 vs cold network → 1.5/epoch → blew past truth to lower bound in 3 epochs. Warmup is load-bearing | — |
| **16** | **50900a86** | **500** two-phase, **warmup KEPT**, trigger<2.0, taper 5ep, min 0.02 | 100 | none | 1e-8 | **50/50** | **0.9546** | **0.045** | 19 823 | **✓ CONVERGED** — first run to land near truth. Warmup + silent-pre-trigger callback + fast 5-epoch brake | [summary](../logs/50900a868cd449d892e11a935bf4d606_summary.json) |

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

**Result: drift continued — basin is escapable.** Run #10 trajectory:

| Epoch | E_unit | Δ per epoch | note |
|---|---|---|---|
| 0 | 4.385 | — | identical to #9 (seed=42) |
| 2 | 3.313 | -0.473 | identical to #9 |
| 5 | 2.103 | ~-0.27 | already below #9's 2.508 — diverging |
| 7 | 1.344 | -0.32 | below the basin |
| 8 | 1.110 | -0.235 | approaching truth |
| **9** | **0.939** | **-0.171** | **crossed truth=1.0 between ep8 and ep9** |
| 10 | 0.815 | -0.124 | overshoot continues |
| 15 | 0.542 | ~-0.05 | decelerating |
| 20 | 0.466 | -0.011 | still descending |
| 30 | 0.402 | -0.005 | heading toward lower bound 0.1 |
| 36 | 0.376 | -0.004 | jetsam-killed before convergence |

Run #11 (cosine→0.30 instead of cosine→0.05) was the obvious follow-up — keep enough late-LR to finish drift, enough decay to brake near truth. **It re-trapped at 1.98.** Trajectory matches #9 nearly exactly: ep10=2.084, ep15=2.001, ep20=1.982, ep22=1.978 (vs #9 at ep20=1.997, ep22≈1.99). A 30% LR floor is still effectively zero in this basin — the gradient near E=1.98 is too weak for that LR to escape.

**Diagnosis (PARTIALLY SUPERSEDED — see "The LR-capture bug" section above):** The
1.98 plateau is *not* identifiability and *not* a stationary point — that part holds:
run #10 transited it cleanly at full LR. But the claim that "cosine schedules trap
because they decay LR below ~70-80%" was wrong. #11's re-trap (and #12's, #14's) was
the LR-capture bug pinning the LR at 3× too low, *not* the cosine shape. The correct
statement: the basin is shallow and escapable at the true lr=500; the open question
(braking near truth without overshooting) is what run #15 tests with the bug fixed.

Both #10 and #11 died at the same time (06:16–06:19 May 27) from a jetsam memory-pressure event (confirmed by imagent SIGABRT crashes at 06:20 and 06:41). **Do not run two MPS Lightning training jobs in parallel on this machine.**

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

1. **Local minimum / lazy regime.** ~~The network finds an (E_unit ≈ 5, u(s,t))
   pair that fits the wave equation. Once there, gradient flow is genuinely small
   regardless of LR scaling. Would need to test a hot-start from E_unit ≈ 1
   (truth) to see if the basin is stable or escape-able.~~ **Resolved by run #10:**
   the 1.98 basin is shallow, not a stationary point — full LR transits it from
   init in <10 epochs. The convergence problem reduces to LR-schedule braking
   near truth, not identifiability or lazy regime.
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
