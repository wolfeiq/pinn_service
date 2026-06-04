"""Cosserat at its intended 10k-epoch budget — does the CRLB headroom show up?

CRLB on E_unit was computed at ~0.06% (machine-tight). All previous Cosserat
runs were 50-epoch debug runs that converged to ~5–10% via the controller.
The question: at the production budget the template was *designed* for
(default_config().adam_epochs = 10000), does E_unit close the gap?

We also try two layered improvements:
  * Adaptive controller (the wave-eq R&D crown jewel)
  * +L-BFGS finetune (1000 iters) after Adam
  * +RAR resampling (so collocation chases the moving wave front)

Configurations:
  A. Full budget Adam only (baseline at design budget)
  B. + L-BFGS post-Adam
  C. + RAR
  D. + L-BFGS + RAR (kitchen sink)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

from pinn_engine.dsl.templates_lib.cosserat_rod import CosseratRod
from pinn_engine.core.trainer import train
from pinn_engine.core.adaptive_controller import AdaptiveUnknownsController


CONFIGS = {
    "A_adam_only":     dict(rar=False, lbfgs=0),
    "B_adam_lbfgs":    dict(rar=False, lbfgs=1000),
    "C_adam_rar":      dict(rar=True,  lbfgs=0),
    "D_adam_rar_lbfgs":dict(rar=True,  lbfgs=1000),
}


def run(label, cfg_opts, epochs):
    system = CosseratRod.system()
    data = CosseratRod.synthetic_data()
    cfg = CosseratRod.default_config()
    cfg.solver_type = "causal"
    cfg.causal_eps = 1e-8
    cfg.causal_eps_anneal = True
    cfg.adam_epochs = epochs
    cfg.lbfgs_iters = cfg_opts["lbfgs"]
    cfg.param_lr_scale = 500.0
    cfg.adaptive_unknowns_lr = True
    if cfg_opts["rar"]:
        cfg.rar_enable = True
        cfg.rar_refresh_every = max(200, epochs // 20)
        cfg.rar_candidate_pool = 40_000
        cfg.rar_keep_old_fraction = 0.5
        cfg.rar_warmup_epochs = 200

    print(f"=== [{label}] starting (Adam={epochs} lbfgs={cfg_opts['lbfgs']} rar={cfg_opts['rar']})", flush=True)
    t0 = time.time()
    result = train(system, data, cfg)
    dt = time.time() - t0
    E = result.final_params["E_unit"]
    err = abs(E - 1.0)
    out = {
        "label": label,
        "epochs": epochs,
        "lbfgs": cfg_opts["lbfgs"],
        "rar": cfg_opts["rar"],
        "wall_sec": round(dt, 1),
        "E_unit": E,
        "rel_err": err,
    }
    print(f"=== [{label}] DONE: E_unit={E:.6f} rel_err={err:.2%} wall={dt:.1f}s", flush=True)
    return out


if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    only = sys.argv[2] if len(sys.argv) > 2 else None
    Path("/Users/mary/pinn_service/logs").mkdir(exist_ok=True)
    results = []
    for label, opts in CONFIGS.items():
        if only and only not in label:
            continue
        try:
            r = run(label, opts, epochs)
            results.append(r)
        except Exception as e:
            print(f"=== [{label}] FAILED: {type(e).__name__}: {e}", flush=True)
            results.append({"label": label, "error": str(e)})
    out_path = Path(f"/Users/mary/pinn_service/logs/cosserat_full_budget_{epochs}.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSummary written to {out_path}", flush=True)
    print(json.dumps(results, indent=2))
