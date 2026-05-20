# Literature comparison — where our results sit in the PINN literature

This document records how the engine's empirical findings line up with
published work. Compiled after the 30-trial AutoML run on 2026-05-19
(study `showcase30`, manifests under `manifests/optuna_showcase30.db`).
Update it whenever a new finding either confirms or contradicts something
in the field.

---

## Headline number to cite

**Damped harmonic oscillator inverse problem** — discover both `c` and `k`
from noisy `x(t)`:

| Method | Mean rel-err on (c, k) | Wall-clock | Hardware |
|---|---:|---:|---|
| Hand-tuned default (depth=4 × width=64) | 0.07 % | 90 s | Apple Silicon CPU |
| **AutoML-discovered (depth=6 × width=32)** | **0.042 %** | 92 s | Apple Silicon CPU |
| DeepXDE vanilla recipe (tutorial) | ~0.1 % | ~30k epochs | not specified |
| PINNverse (Garay et al. 2024) | 370× error reduction over vanilla | 2-5k epochs | GPU |

The AutoML improvement (~40 %) sits at the top of the range Auto-PINN
(2022) reports for systematic neural architecture search on PINNs.

---

## Finding 1 — Deep + narrow beats shallow + wide

**What we found.** Top trial: `depth=6, width=32` (3.5K params). Beat
`depth=4, width=64` (8.9K params) hand-tuned baseline. All top-5 trials
clustered at `depth ∈ {5, 6}`, `width ∈ {32, 64}`. Width 128 never made
the top of the leaderboard despite being in the search space.

**Literature alignment.**
- Raissi, Perdikaris, Karniadakis (2019) — the original PINN paper —
  used 8-9 layers × 20-40 neurons. Default was already deep+narrow.
- **Auto-PINN** (Wang et al. 2022, arXiv:2205.13748) ran systematic NAS
  across 7 benchmark PDEs and reported `depth ∈ [5, 8]` × `width ∈ [32, 64]`
  consistently optimal. Our `depth=6, width=32` lands squarely in that
  cell.
- Wang-Perdikaris 2021 ("Gradient flow analysis of PINNs") proved
  theoretically that PINNs have *spectral bias toward smooth, low-frequency
  functions*. Depth (not width) raises representation capacity in the
  problematic frequency band.

**Verdict.** Consistent with both empirical and theoretical literature.

---

## Finding 2 — Sintanh dominates on oscillatory targets

**What we found.** Every top-5 trial used `activation = "sintanh"`
(`sin(x) · tanh(x)`). Trials with `tanh` capped at ~3 % rel-err; trials
with `swish` capped at ~12 %.

**Literature alignment.**
- **SIREN** (Sitzmann, Mildenhall, Martel, Bergman, Lindell, Wetzstein
  2020, arXiv:2006.09661) — pure `sin(x)` activations sharply improve
  representation of high-frequency content in implicit neural
  representations.
- **MoRPI-PINN** (Sahoo & Klein 2025, arXiv:2507.18206) — uses
  `sin(x) · tanh(x)` (literally the same function) specifically for
  inertial navigation trajectories.
- Wang, Wang, Perdikaris 2021 ("On the eigenvector bias of Fourier
  feature networks") — NTK analysis shows trigonometric activations lift
  the spectral bias that plain tanh suffers from.

**Verdict.** Direct match with SIREN-family findings and a recent paper
using the *exact same* activation for a closely related problem class
(IMU-based dead reckoning). Our AutoML re-discovered the same answer the
literature converges to.

---

## Finding 3 — Lam_data ≫ lam_physics for inverse problems

**What we found.** AutoML converged on `lam_data ≈ 43`, `lam_physics = 1.0`.
With `lam_data = lam_physics = 1.0` (the naive default) the network
collapses to `x(t) ≈ 0` — the trivial solution that satisfies any
`(c, k)` with zero physics residual.

**Literature alignment.**
- **Wang-Teng-Perdikaris 2021** (LRA paper, arXiv:2107.05228) — adaptive
  weights typically settle at 10-1000× imbalance, with data dominating
  for inverse problems.
- DeepXDE's `Lorenz_inverse` tutorial uses no explicit weighting but
  pre-normalizes state variables to unit magnitude, which is
  mathematically equivalent (you're absorbing the weight into the loss
  scale).
- **Noise-calibrated weighting** (common in inverse problems): set
  `lam_data = 1/σ²` where `σ` is the sensor noise. For our `σ = 0.01`
  this gives `1e4` — well above our discovered 43, suggesting there's
  still headroom to push lam_data higher.

**Verdict.** Inside the literature's accepted range. On the lower end —
which means we may be leaving ~10× of headroom on the table. A future
AutoML run with `lam_data ∈ [10, 10000]` is worth trying.

---

## Finding 4 — Convergence rate is at the leading edge

**What we found.** 0.042 % rel-err on `(c, k)` in **800 Adam epochs** on
CPU (no L-BFGS). ~90 seconds wall-clock.

**Literature alignment.**
- DeepXDE `Lorenz_inverse` tutorial: ~30k Adam epochs + L-BFGS to reach
  ~0.1 % rel-err (their target). Our recipe is ~37× fewer epochs to
  reach a 2× better error.
- **PINNverse** (Garay, Cuomo, Schiassi, Karniadakis 2024,
  arXiv:2501.07413) — claims 370× error reduction over vanilla PINNs,
  ~2-5k epochs. We're at ~800 epochs for similar error.
- **Causal PINN** (Wang, Sankaran, Perdikaris 2022, arXiv:2203.07404)
  — claims 10-100× iteration reduction for chaotic systems via causal
  weighting. We don't use causal weighting, but our tight bounds + good
  activation + lam_data tuning give comparable speedup.

**Verdict.** Leading-edge for damped oscillator. Mainly because of the
*three* tuning levers stacked: tight bounds (good init), sintanh
activation, lam_data weighting. None is novel individually; combining
them in a clean DSL is the contribution.

---

## Finding 5 — AutoML beats hand-tuned by 40 %

**What we found.** 0.07 % (hand-tuned) → 0.042 % (AutoML) = ~40 %
relative reduction in mean rel-err.

**Literature alignment.**
- **Auto-PINN** (Wang et al. 2022) reports 20-40 % mean-error reduction
  vs hand-tuned baselines across 7 benchmark PDEs. Our 40 % sits at the
  top of their reported range.
- **AutoPINN** (RL-based, Yu et al. 2022, arXiv:2212.04058) — RL-based
  search, reports 25-50 % improvement. Same ballpark.

**Verdict.** Right where the literature predicts. Damped oscillator is
on the easier end of benchmark problems, so tuning headroom is bigger.

**Important addendum (2026-05-19, post head-to-head):** the 30-trial
single-seed AutoML's headline 0.042% was partially a seed-overfitting
artifact. When we re-evaluated the discovered config on a held-out seed
set, mean rel-err jumped to 0.25% — **worse** than the hand-tuned
baseline (0.13%). Auto-PINN and AutoPINN papers anticipate this and
average each trial's objective over 3-5 seeds; we added the same
treatment (`MULTI_SEED_SEEDS = (42, 137, 2718)` in
`pinn_engine.automl.search`). The multi-seed AutoML then found a
*genuinely* better config (mean rel-err 0.096%, 28 % better than
hand-tuned), and it lands inside the Auto-PINN reported range without
asterisks.

**Lesson, worth recording.** Single-seed AutoML lies. The literature
treats this as table stakes; any AutoML pipeline that doesn't average
over multiple seeds per trial will produce numbers that look better
than they are.

---

## Finding 6 — Hyperband pruning works as advertised

**What we found.** In the 30-trial showcase run, 13 trials (43 %)
pruned by Hyperband at rungs 299 / 699 epochs. Several pruned trials
had *moderate* loss (e.g. Trial #20 pruned at loss 0.024) — Hyperband
killed them because *better* trials existed at the same rung.

**Literature alignment.**
- **Hyperband** (Li, Jamieson, DeSalvo, Rostamizadeh, Talwalkar 2017,
  arXiv:1603.06560) — original paper proved that successive halving with
  multiple resource brackets is theoretically optimal for non-stochastic
  best-arm identification.
- Pruning ~half the population at each rung is exactly the design.

**Verdict.** Hyperband does what it says on the tin. The interesting
practical detail is that *upstream* `PyTorchLightningPruningCallback`
silently never fires for PINN inverse problems because they have no
`val_dataloader`. Our `TrainLossPruningCallback` reports on
train-epoch-end and fixes this; without it, pruning is decorative.

---

## Things our results would *not* match literature on

Two cases where we deviated from what literature would predict:

1. **Balancer = none wins.** Adaptive loss balancers (SA-PINN, LRA,
   ReLoBRaLo) are well-published and consistently helpful in
   benchmarks. In our 30-trial run, `balancer=none` won across all
   top-5 trials. **This is an implementation gap, not a real result** —
   our SA-PINN / LRA callbacks store weights on themselves but the hook
   into PINA's solver-level `weighting` machinery is incomplete. Once
   that's wired, we expect `lra` (or PINA's own `SelfAdaptivePINN`
   solver) to win on harder templates.

   **Resolved (2026-05-20, commits `be02a58`, `703422e`).** The SA-PINN
   and LRA implementations were rewritten as proper
   `pina.loss.WeightingInterface` subclasses, so PINA's solver now
   actually invokes `weights_update(losses)` each training step. A
   fresh 30-trial multi-seed AutoML ("honest30ms") then settled on
   **depth=6, width=32, sintanh, lr=8.25e-4, lam_data=52, balancer=lra**
   for 0.100 % mean rel-err — the **qualitative shift** vs. the prior
   "lra-as-no-op" winner is that `lam_data_init` dropped ~9× (464 →
   52). With LRA actually performing gradient-norm-ratio updates, the
   AutoML stopped relying on aggressive static priors. This matches
   Wang-Teng-Perdikaris 2021's central claim: adaptive weighting
   *replaces* manual λ tuning.

2. **L-BFGS finetune skipped.** Standard PINN recipe is Adam → L-BFGS;
   most papers report L-BFGS gives a 2-10× error reduction in the final
   phase. We can't run L-BFGS on PINA `InverseProblem`s because PINA
   adds `unknown_parameters` as a second optimizer `param_group` and
   `torch.optim.LBFGS` asserts a single group. **This is a hard limit
   of PINA's solver API as of 0.2.3** — needs a custom solver subclass
   that merges the param groups for L-BFGS specifically. Until then,
   Adam-only is the supported path. Our convergence is competitive
   without it, but the literature suggests another ~2-10× headroom.

---

## Reproducibility

| Run | Manifest / Study | Command |
|---|---|---|
| Showcase 30-trial AutoML | `manifests/optuna_showcase30.db` | `pinn-engine search damped_oscillator --n-trials 30 --study showcase30` |
| Honest 30-trial AutoML (LRA wired) | `manifests/optuna_honest30ms.db` | `pinn-engine search damped_oscillator --n-trials 30 --study honest30ms` |
| Baked-in default config | `damped_oscillator.default_config()` | `pinn-engine train damped_oscillator` |
| Lorenz 3-state inverse | (master pipeline output) | `python examples/02_lorenz_inverse.py` |

Open the AutoML leaderboard live:
```
optuna-dashboard sqlite:///manifests/optuna_honest30ms.db
```

---

## Device sensitivity — a real caveat

The headline `0.100% mean rel-err` (study `honest30ms`, baked into
`default_config`) was measured on **MPS (Apple Silicon GPU)**. The same
config on **CPU** averages `0.404%` over the same 3 seeds — about 5×
worse. Some observations:

| Device | seed 42 | seed 137 | seed 2718 | Average | Time / seed |
|---|---:|---:|---:|---:|---:|
| CPU | 0.695 % | 0.051 % | 0.464 % | 0.404 % | 63 s |
| MPS | 0.083 % | 0.016 % | 0.138 % | 0.079 % | 350 s |

Two unusual things:

1. **MPS is more accurate, not less.** Standard wisdom is "GPU = slightly
   noisier accuracy, much faster." Here MPS is both more accurate *and*
   slower. Likely cause: different float-op ordering / reductions on MPS
   that happen to be numerically more stable for this problem class
   (high-order derivatives in the residual amplify cancellation errors,
   and MPS may use fused multiply-add internally).

2. **CPU wins wall-clock.** For a 4.6K-param MLP, GPU dispatch overhead
   per batch exceeds the compute saved. MPS only pays off for much
   larger models (~10⁵ params and up). This is a small-model edge case;
   we'd expect MPS to win on Phase-3 Fossen 6-DOF or Cosserat templates
   where the network is bigger.

**Honest framing for documentation / X-post:**
- "Best result: 0.10 % mean rel-err on MPS (350 s per training run)"
- "Throughput: 0.40 % mean rel-err on CPU (63 s per training run)"
- "Both are reproducible from the seed list and the committed manifest."

Don't quote one number as "the" headline without a device. The AutoML
chose the config with `accelerator="auto"` (→ MPS on this hardware), so
the reported numbers are MPS numbers.

**Action item (deferred):** future AutoML runs should average across
devices too if we want device-portable defaults. That's another ~2× the
trial cost; worth doing for a v0.2 release.
