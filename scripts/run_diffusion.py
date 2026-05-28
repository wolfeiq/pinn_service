"""Diffusion-1D inverse run — testing whether the Cosserat two-phase-LR
recipe transfers to a parabolic PDE.

Recover D (truth 0.1) in u_t = D·u_xx. Unlike Cosserat's wave equation,
diffusion is well-posed and the smoke test showed D moves freely toward
truth (no 1.98-style basin) — if anything it overshoots and oscillates.
So this baseline runs the PLAIN config (moderate lr_scale + PINA warmup,
NO two-phase trigger) to see whether it converges on its own.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

from pinn_engine.dsl.templates_lib.diffusion_1d import Diffusion1D
from pinn_engine.core.trainer import train
from pinn_engine.core.unknowns_dumper import UnknownsDumper


def main():
    system = Diffusion1D.system()
    data, truth = Diffusion1D.synthetic_data(seed=0)
    cfg = Diffusion1D.default_config()

    # Baseline: plain single-phase. lr_scale=100 + PINA warmup, vanilla PINN.
    cfg.adam_epochs = 50
    cfg.lbfgs_iters = 0
    cfg.param_lr_scale = 100.0
    cfg.balancer = "none"

    Path("/Users/mary/pinn_service/logs").mkdir(exist_ok=True)
    live_path = f"/Users/mary/pinn_service/logs/{cfg.run_id}_live.json"
    dumper = UnknownsDumper(live_path)

    t0 = time.time()
    result = train(system, data, cfg, callbacks=[dumper])
    elapsed = time.time() - t0

    D = result.final_params["D"]
    rel_err = abs(D - truth["D"]) / abs(truth["D"])
    out = {
        "run_id": cfg.run_id,
        "elapsed_sec": elapsed,
        "final_params": result.final_params,
        "truth": truth,
        "rel_err": {"D": rel_err},
        "config": {
            "param_lr_scale": cfg.param_lr_scale,
            "adam_epochs": cfg.adam_epochs,
            "two_phase": cfg.param_lr_trigger_below is not None,
        },
    }
    print(json.dumps(out, indent=2))
    summary_path = f"/Users/mary/pinn_service/logs/{cfg.run_id}_summary.json"
    Path(summary_path).write_text(json.dumps(out, indent=2))
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
