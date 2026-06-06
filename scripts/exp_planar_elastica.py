"""R&D sweep for the planar-elastica (large-deflection soft-rod) inverse.

Goal: drive EI_unit rel_err toward the CRLB floor (0.46% at 31 angle
sensors, noise_std=1e-2 rad) and find the right recipe for the nonlinear
``cos(θ)`` bending residual.

Configs probed:
  baseline   — template default (3000 Adam, param_lr_scale=20)
  adaptive   — runtime LR controller instead of fixed param_lr_scale
  lbfgs      — baseline Adam + L-BFGS finetune
  rar        — residual-adaptive collocation refinement
  long       — 8000 Adam epochs (budget headroom check)

Run a single config:  python3 scripts/exp_planar_elastica.py <name>
Run all:              python3 scripts/exp_planar_elastica.py all
"""
from __future__ import annotations
import sys, time
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

from pinn_engine.dsl.templates import get_template
from pinn_engine.core.trainer import train
from pinn_engine.core.adaptive_controller import AdaptiveUnknownsController

TRUTH = 1.0
CRLB = 0.0046  # SE/|truth| from compute_template_crlb("planar_elastica")


def _base_cfg():
    tpl = get_template("planar_elastica")
    cfg = tpl.default_config()
    cfg.seed = 0
    cfg.skip_preflight = True
    cfg.accelerator = "cpu"
    return tpl, cfg


def run(name: str):
    tpl, cfg = _base_cfg()
    callbacks = None
    if name == "baseline":
        pass
    elif name == "adaptive":
        # Keep the template's param_lr_scale=20 as the controller's BASE LR and
        # let it modulate from there. Resetting to 1.0 caps the unknown's LR at
        # ~4e-3 (max_mult=4), too weak to move EI_unit off its midpoint init —
        # that diverges to ~375%. (See memory / engine guidance: start the
        # controller from each template's own param_lr_scale default.)
        cfg.adaptive_unknowns_lr = True
        callbacks = [AdaptiveUnknownsController()]
    elif name == "lbfgs":
        cfg.lbfgs_iters = 100
    elif name == "rar":
        cfg.rar_enable = True
        cfg.rar_refresh_every = 300
        cfg.rar_warmup_epochs = 300
    elif name == "long":
        cfg.adam_epochs = 8000
    else:
        raise SystemExit(f"unknown config {name!r}")

    data, truth = tpl.synthetic_data(seed=0)
    t0 = time.time()
    res = train(system=tpl.system(), data=data, config=cfg, callbacks=callbacks)
    dt = time.time() - t0
    ei = res.final_params["EI_unit"]
    err = abs(ei - TRUTH) / TRUTH
    print(f"[{name:9s}] EI_unit={ei:.5f}  rel_err={err:7.3%}  "
          f"(CRLB {CRLB:.2%})  wall={dt:5.1f}s", flush=True)
    return name, ei, err, dt


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    names = ["baseline", "adaptive", "lbfgs", "rar", "long"] if cmd == "all" else [cmd]
    rows = [run(n) for n in names]
    if len(rows) > 1:
        print("\n=== summary ===")
        for name, ei, err, dt in sorted(rows, key=lambda r: r[2]):
            print(f"  {name:9s} rel_err={err:7.3%}  wall={dt:5.1f}s")
