"""Shorter beam L-BFGS probe: figure out where the time is going.

Prior run hung 19+ min without printing baseline result, suggesting a
hang somewhere. Run baseline (no LBFGS) first to confirm Adam timing,
then LBFGS=50 small enough to bound wall time.
"""
from __future__ import annotations
import sys, time
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

from pinn_engine.dsl.templates import get_template
from pinn_engine.core.trainer import train


def run_one(label, lbfgs_iters, adam_epochs=1500, accelerator="cpu"):
    tpl = get_template("euler_bernoulli_beam")
    cfg = tpl.default_config()
    cfg.lbfgs_iters = lbfgs_iters
    cfg.adam_epochs = adam_epochs
    cfg.seed = 0
    cfg.skip_preflight = True
    cfg.accelerator = accelerator
    data, truth = tpl.synthetic_data(seed=0)
    print(f"[{label}] Adam={adam_epochs}, LBFGS={lbfgs_iters}, accel={accelerator}", flush=True)
    t0 = time.time()
    result = train(system=tpl.system(), data=data, config=cfg)
    dt = time.time() - t0
    err = abs(result.final_params["EI_unit"] - truth["EI_unit"]) / truth["EI_unit"]
    print(f"[{label}] EI_unit={result.final_params['EI_unit']:.4f} rel_err={err:.2%} wall={dt:.1f}s", flush=True)
    return dt, err


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    if cmd == "baseline":
        run_one("baseline_cpu", lbfgs_iters=0, accelerator="cpu")
    elif cmd == "lbfgs50":
        run_one("lbfgs50_cpu", lbfgs_iters=50, accelerator="cpu")
    elif cmd == "lbfgs100":
        run_one("lbfgs100_cpu", lbfgs_iters=100, accelerator="cpu")
    elif cmd == "mps":
        run_one("baseline_mps", lbfgs_iters=0, accelerator="mps")
    else:
        print(f"unknown cmd: {cmd}")
