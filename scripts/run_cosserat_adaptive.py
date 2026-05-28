"""Cosserat CausalPINN inverse run driven by the AUTO-ADAPTIVE controller.

The crown test for AdaptiveUnknownsController: can it auto-escape the 1.98
wave-equation basin and brake near truth=1.0, reproducing the hand-tuned
two-phase result (run #16, ~4.5%) with NO manual trigger/taper config?

Mirrors scripts/run_cosserat_causal.py's working config, but swaps the manual
two-phase scheduler for the adaptive controller (passed as an instance so we
can dump its telemetry). param_lr_scale=50 is only the controller's starting
scale — it ramps/brakes from there.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

from pinn_engine.dsl.templates_lib.cosserat_rod import CosseratRod
from pinn_engine.core.trainer import train
from pinn_engine.core.unknowns_dumper import UnknownsDumper
from pinn_engine.core.adaptive_controller import AdaptiveUnknownsController


def main():
    system = CosseratRod.system()
    data = CosseratRod.synthetic_data()
    cfg = CosseratRod.default_config()

    cfg.solver_type = "causal"
    cfg.causal_eps = 1e-8
    cfg.causal_eps_anneal = True
    cfg.causal_eps_max = 100.0
    cfg.causal_eps_threshold = 1e-2
    cfg.adam_epochs = 50
    cfg.lbfgs_iters = 0
    cfg.balancer = "none"
    cfg.lam_data_init = 100.0
    # Adaptive controller instead of the hand-tuned two-phase recipe. 50 is the
    # controller's STARTING scale; it ramps to escape the basin and brakes near
    # truth on its own.
    cfg.param_lr_scale = 50.0

    ctrl = AdaptiveUnknownsController()

    Path("/Users/mary/pinn_service/logs").mkdir(exist_ok=True)
    live_path = f"/Users/mary/pinn_service/logs/{cfg.run_id}_live.json"
    dumper = UnknownsDumper(live_path)

    t0 = time.time()
    result = train(system, data, cfg, callbacks=[dumper, ctrl])
    elapsed = time.time() - t0

    E = result.final_params["E_unit"]
    out = {
        "run_id": cfg.run_id,
        "elapsed_sec": elapsed,
        "final_params": result.final_params,
        "truth": {"E_unit": 1.0},
        "rel_err": {"E_unit": abs(E - 1.0) / 1.0},
        "controller": {
            "converged": ctrl.converged,
            "history": ctrl.history,
        },
    }
    print(json.dumps({k: v for k, v in out.items() if k != "controller"}, indent=2))
    print("controller converged:", ctrl.converged)
    summary_path = f"/Users/mary/pinn_service/logs/{cfg.run_id}_adaptive_summary.json"
    Path(summary_path).write_text(json.dumps(out, indent=2))
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
