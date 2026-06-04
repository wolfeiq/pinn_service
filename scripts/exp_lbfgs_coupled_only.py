"""Just the LBFGS arm on coupled_drag_3d (baseline already measured)."""
from __future__ import annotations
import sys
import time

import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

from pinn_engine.dsl.templates import get_template
from pinn_engine.core.trainer import train


def run_one(label, lbfgs_iters):
    tpl = get_template("coupled_drag_3d")
    cfg = tpl.default_config()
    cfg.lbfgs_iters = lbfgs_iters
    cfg.seed = 0
    cfg.skip_preflight = True
    data, truth = tpl.synthetic_data(seed=0)
    t0 = time.time()
    print(f"[{label}] starting Adam={cfg.adam_epochs} + LBFGS={lbfgs_iters}...", flush=True)
    result = train(system=tpl.system(), data=data, config=cfg)
    dt = time.time() - t0
    rel_errs = {n: abs(result.final_params[n] - truth[n]) / abs(truth[n]) for n in truth}
    print(f"\n=== {label} | wall={dt:.1f}s ===", flush=True)
    print(f"  recovered: {result.final_params}", flush=True)
    print(f"  rel_err: c_x={rel_errs['c_x']:.2%}  c_y={rel_errs['c_y']:.2%}  c_n={rel_errs['c_n']:.2%}", flush=True)


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    label = f"Adam2000 + LBFGS{iters}"
    run_one(label, iters)
