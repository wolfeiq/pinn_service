"""R&D sweep for the full planar Cosserat (Simo-Reissner) rod inverse —
recover bending / shear / axial stiffness (EI_unit, GA_unit, EA_unit) from the
deformed shape of a tip-loaded soft rod.

The engine's first multi-output, multi-unknown *PDE* inverse. CRLB floors
(41 sensors, soft-rod design): EI 0.29%, GA 0.53%, EA 0.11% — all identifiable.
The three unknowns converge at very different rates: EI gets a strong signal
from moment balance; GA and EA depend on the shear / axial strains and only
sharpen once the shape fit is accurate. The adaptive LR controller handles the
spread far better than a single fixed param_lr_scale.

Recipe note: the rod is deliberately *soft* (GA0=EA0=15, strong combined tip
load) so shear (~7-27%) and axial (~17-31%) strains are large. An earlier
stiff-axial design (EA0=40, ~5% axial strain) left EA stuck near its init — the
network explained the tiny axial residual away within the position-noise
latitude. Enlarging the strain signal fixed it (EA 379% -> 0.20%).

Configs (all ncol=512 — solution is smooth, ~2.7x faster/epoch than 1500):
  fast       — fixed param_lr_scale=100, 8000 epochs   (recommended; mean 0.30%)
  adaptive   — runtime LR controller, 8000 epochs      (2.06%)
  long       — runtime LR controller, 12000 epochs     (unstable: 72.9% — the
               controller wanders off EA's minimum at higher budgets)
  baseline   — template default (slow ncol=1500 reference), 4000 epochs

Run one:  python3 scripts/exp_planar_cosserat.py <name>
Run all:  python3 scripts/exp_planar_cosserat.py all
"""
from __future__ import annotations
import sys, time
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

from pinn_engine.dsl.templates import get_template
from pinn_engine.core.trainer import train
from pinn_engine.core.adaptive_controller import AdaptiveUnknownsController

NAMES = ["EI_unit", "GA_unit", "EA_unit"]
CRLB = {"EI_unit": 0.0029, "GA_unit": 0.0053, "EA_unit": 0.0011}


def run(name: str):
    tpl = get_template("planar_cosserat")
    cfg = tpl.default_config()
    cfg.seed = 0
    cfg.skip_preflight = True
    cfg.accelerator = "cpu"
    callbacks = None
    if name == "baseline":
        pass
    elif name == "fast":
        cfg.n_collocation = 512; cfg.batch_size = 512
        cfg.param_lr_scale = 100.0; cfg.adam_epochs = 8000
    elif name == "adaptive":
        cfg.n_collocation = 512; cfg.batch_size = 512
        cfg.adam_epochs = 8000
        cfg.adaptive_unknowns_lr = True
        callbacks = [AdaptiveUnknownsController()]
    elif name == "long":
        cfg.n_collocation = 512; cfg.batch_size = 512
        cfg.adam_epochs = 12000
        cfg.adaptive_unknowns_lr = True
        callbacks = [AdaptiveUnknownsController()]
    else:
        raise SystemExit(f"unknown config {name!r}")

    data, truth = tpl.synthetic_data(seed=0)
    t0 = time.time()
    res = train(system=tpl.system(), data=data, config=cfg, callbacks=callbacks)
    dt = time.time() - t0
    errs = {k: abs(res.final_params[k] - 1.0) for k in NAMES}
    mean = sum(errs.values()) / 3
    cells = "  ".join(f"{k.split('_')[0]} {res.final_params[k]:.4f}({errs[k]:.2%})" for k in NAMES)
    print(f"[{name:9s}] {cells}  MEAN={mean:.3%}  wall={dt:5.1f}s", flush=True)
    return name, errs, mean, dt


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fast"
    names = ["baseline", "fast", "physics", "adaptive", "long"] if cmd == "all" else [cmd]
    rows = [run(n) for n in names]
    if len(rows) > 1:
        print("\n=== summary (sorted by mean rel_err) ===")
        for name, errs, mean, dt in sorted(rows, key=lambda r: r[2]):
            print(f"  {name:9s} MEAN={mean:.3%}  wall={dt:5.1f}s")
    print(f"\nCRLB floors: " + ", ".join(f"{k.split('_')[0]} {v:.2%}" for k, v in CRLB.items()))
