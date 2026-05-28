"""Cosserat CausalPINN run — replaces the lost heredoc from /tmp.

What this exercises:
  * `causal_eps` plumbed through TrainerConfig (default 1.0, not PINA's 100)
  * Per-bucket residual + weight logging in CausalLabeledDataPINN.loss_phys
  * Wang 2022 §3.2 ε-annealing via CausalEpsAnnealer

Compare against v466 baseline (lightning_logs/version_466): physics_0_loss
was ~0 throughout because eps=100 collapsed ω_i to ~0.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

from pinn_engine.core.trainer import TrainConfig, train
from pinn_engine.core.unknowns_dumper import UnknownsDumper
from pinn_engine.dsl.templates_lib.cosserat_rod import CosseratRod


def main():
    system = CosseratRod.system()
    data = CosseratRod.synthetic_data()
    cfg = CosseratRod.default_config()

    cfg.solver_type = "causal"
    # Empirically measured init residual on this problem is ~1.4e7, so the
    # annealer's shrink phase from a "small" ε=1e-2 would burn ~6 epochs
    # of useless training before weights become non-trivial. Start at
    # ε=1e-8 (chosen so exp(-1e-8 × 1.4e7) ≈ 0.87 — well above the
    # collapse floor), then let the annealer grow ε up as residuals drop.
    cfg.causal_eps = 1e-8
    cfg.causal_eps_anneal = True
    cfg.causal_eps_max = 100.0
    cfg.causal_eps_threshold = 1e-2
    cfg.adam_epochs = 50
    cfg.lbfgs_iters = 0
    # Previous run (run_id d8f486e2...) converged physics_loss → 8.46 but
    # E_unit barely budged: 5.05 → 5.04 (rel_err 4.04). With default
    # param_lr_scale=1.0, Adam's per-parameter normalization throttles the
    # unknown. 10× LR on the unknowns' param-group only — network weights
    # untouched.
    # Run ea06e951 (lr=500 + cosine to 0.05): E_unit landed at 1.9833 —
    # rel_err 0.98, dramatic improvement over lr=100's 4.09. BUT
    # trajectory shows it converged by epoch 15 and stayed flat at 1.98:
    # a stable local minimum (likely the wave-eq non-uniqueness at
    # E_unit=2 where u(s, t/√2) also fits the residual). Cosine decay
    # cut off the LR before it could escape. Same lr=500, no cosine
    # this time, to test whether the 1.98 basin is escapable.
    # Run 940de904 (lr=500, no cosine): broke through the 1.98 basin
    # at ep5-7, crossed truth=1.0 around ep9, then kept descending —
    # ended at 0.376 (ep36, jetsam-killed). Basin IS escapable, but
    # no-cosine overshoots.
    # Run 47b2886b (lr=500, min_scale=0.3): re-trapped at 1.978 (ep22,
    # jetsam-killed). Run 9a9c5932 (lr=500, min_scale=0.7): also
    # trapping, asymptoting at ~1.95 (geometric decay confirmed by ep20).
    # Run 8cc08c1b (lr=400, no cosine): escaped basin but stalled at
    # ~1.35 (geometric asymptote) — pure lr_scale tuning is brittle.
    # *** LR-CAPTURE BUG (found & fixed): PINA wraps the optimizer in
    # ConstantLR(factor=1/3, 5-ep warmup). UnknownsParamLRScheduler was
    # capturing the warmup-discounted LR (0.1667) as "base" and pinning it,
    # so every scheduler-attached run (#9,#11,#12,#14) secretly trained the
    # unknown at lr_scale~167, not 500 — that, not "cosine", is why they
    # trapped at ~1.95. Fix: read ConstantLR.base_lrs (true 0.5) + pass a
    # constant scheduler so our callback owns the LR. ***
    # Run #15: two-phase at the TRUE lr=500. Hold full LR until E_unit < 1.5
    # (past the basin floor at ~2), then cosine taper to 5% over remaining
    # epochs to brake near truth=1.0.
    cfg.param_lr_scale = 500.0
    cfg.param_lr_min_scale = 0.05
    cfg.param_lr_trigger_below = 1.5
    cfg.lam_data_init = 100.0
    cfg.balancer = "none"

    Path("/Users/mary/pinn_service/logs").mkdir(exist_ok=True)
    live_path = f"/Users/mary/pinn_service/logs/{cfg.run_id}_live.json"
    dumper = UnknownsDumper(live_path)

    t0 = time.time()
    result = train(system, data, cfg, callbacks=[dumper])
    elapsed = time.time() - t0

    truth = CosseratRod.truth
    out = {
        "run_id": result.run_id,
        "elapsed_sec": round(elapsed, 1),
        "final_loss": result.final_loss,
        "final_params": result.final_params,
        "truth": truth,
        "rel_err": {
            k: abs(result.final_params[k] - v) / max(abs(v), 1e-6)
            for k, v in truth.items()
        },
        "config": {
            "solver_type": cfg.solver_type,
            "causal_eps_init": cfg.causal_eps,
            "causal_eps_anneal": cfg.causal_eps_anneal,
            "causal_eps_max": cfg.causal_eps_max,
            "adam_epochs": cfg.adam_epochs,
        },
    }
    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print(json.dumps(out, indent=2))
    Path("/Users/mary/pinn_service/logs").mkdir(exist_ok=True)
    summary_path = f"/Users/mary/pinn_service/logs/{result.run_id}_summary.json"
    Path(summary_path).write_text(json.dumps(out, indent=2))
    print(f"\nSummary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
