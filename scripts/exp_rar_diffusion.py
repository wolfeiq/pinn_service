"""RAR ablation on diffusion_1d.

Diffusion has a smooth solution, so RAR isn't expected to help much
here — the goal is the *null result* baseline: RAR shouldn't HURT on
problems where it's not needed. Anything within ~10% relative-error
band of the baseline is a pass; a regression of >2× would mean RAR is
adding noise we can't justify.
"""
from __future__ import annotations
import time
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

from pinn_engine.dsl.templates import get_template
from pinn_engine.core.trainer import train


def run_one(label, rar):
    tpl = get_template("diffusion_1d")
    cfg = tpl.default_config()
    cfg.seed = 0
    cfg.skip_preflight = True
    if rar:
        cfg.rar_enable = True
        cfg.rar_refresh_every = 200
        cfg.rar_candidate_pool = 10_000
        cfg.rar_keep_old_fraction = 0.5
        cfg.rar_warmup_epochs = 100
    data, truth = tpl.synthetic_data(seed=0)
    t0 = time.time()
    print(f"[{label}] rar={rar} starting...", flush=True)
    result = train(system=tpl.system(), data=data, config=cfg)
    dt = time.time() - t0
    err = abs(result.final_params["D"] - truth["D"]) / truth["D"]
    print(f"[{label}] D rel_err={err:.2%} wall={dt:.1f}s recovered={result.final_params}", flush=True)


if __name__ == "__main__":
    run_one("baseline", rar=False)
    run_one("RAR",      rar=True)
