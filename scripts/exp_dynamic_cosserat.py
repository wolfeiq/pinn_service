"""R&D driver for the dynamic planar Cosserat rod inverse (time-domain).

The engine's hardest template: 5 fields (x,y,θ,Nx,Ny) over a 2-D space-time
domain, 5 residuals, 3 unknowns (EI/GA/EA). CRLB floors are tiny (EI 0.02%,
GA 0.05%, EA 0.02%) — the motion is enormously informative — so this is purely
a PINN-training problem. We have dense data over the whole (s,t) grid, so the
fields are well anchored; the network mainly needs enough capacity + epochs to
fit them, after which the unknowns sharpen.

Prints the unknowns every `report_every` epochs so a long run can be watched.

  python3 scripts/exp_dynamic_cosserat.py [epochs] [width] [ncol] [fourier]
"""
from __future__ import annotations
import sys, time
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

import pytorch_lightning as pl
from pinn_engine.dsl.templates import get_template
from pinn_engine.core.trainer import train

NAMES = ["EI_unit", "GA_unit", "EA_unit"]
CRLB = {"EI_unit": 0.0002, "GA_unit": 0.0005, "EA_unit": 0.0002}


class Report(pl.Callback):
    def __init__(self, every=1000):
        self.every = every; self.t0 = time.time()
    def on_train_epoch_end(self, trainer, pl_module):
        e = trainer.current_epoch
        if e % self.every == 0 or e == trainer.max_epochs - 1:
            up = getattr(pl_module.problem, "unknown_parameters", {}) or {}
            vals = {}
            for k, v in up.items():
                try: vals[k] = float(v)
                except Exception:
                    try: vals[k] = float(v.detach().cpu().item())
                    except Exception: vals[k] = None
            cells = "  ".join(
                f"{k.split('_')[0]} {vals.get(k):.4f}" if vals.get(k) is not None else f"{k} ?"
                for k in NAMES)
            print(f"  ep {e:6d}  {cells}  ({time.time()-self.t0:.0f}s)", flush=True)


def main():
    ep = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
    width = int(sys.argv[2]) if len(sys.argv) > 2 else 96
    ncol = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    fourier = int(sys.argv[4]) if len(sys.argv) > 4 else 32

    tpl = get_template("dynamic_cosserat")
    cfg = tpl.default_config()
    cfg.adam_epochs = ep
    cfg.width = width
    cfg.n_collocation = ncol
    cfg.batch_size = ncol
    cfg.fourier_features = fourier
    cfg.param_lr_scale = 120.0
    cfg.lam_data_init = 200.0
    # The unknowns live only in the physics loss; with the auxiliary-force
    # formulation their gradient path is indirect, so weight physics up (data is
    # dense enough at lam=200 that the fields stay well anchored).
    cfg.lam_physics_init = 10.0
    cfg.seed = 0
    cfg.skip_preflight = True
    cfg.accelerator = "cpu"

    data, truth = tpl.synthetic_data(seed=0)
    print(f"[dynamic_cosserat] ep={ep} width={width} ncol={ncol} fourier={fourier}", flush=True)
    t0 = time.time()
    res = train(system=tpl.system(), data=data, config=cfg, callbacks=[Report(1000)])
    dt = time.time() - t0
    errs = {k: abs(res.final_params[k] - 1.0) for k in NAMES}
    print("\n=== final ===")
    for k in NAMES:
        print(f"  {k} = {res.final_params[k]:.4f}  rel_err={errs[k]:.3%}  (CRLB {CRLB[k]:.2%})")
    print(f"  MEAN rel_err={sum(errs.values())/3:.3%}  wall={dt:.0f}s")


if __name__ == "__main__":
    main()
