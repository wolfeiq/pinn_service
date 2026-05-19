"""AutoML vs hand-tuned: the Phase-2 headline benchmark.

For each of {damped_oscillator, lorenz}:
  * run three "manual" configs (small / medium / large hand-picked architectures)
  * run AutoML with a budget of N trials
  * record final parameter error and wall-clock time
  * write the comparison table to ``benchmarks/results.md``

Caveats:
  * Single seed per condition by default — set ``N_REPEATS > 1`` for more robust
    numbers if you have the compute.
  * Wall-clock is sensitive to background load. Use the report qualitatively.
"""
from __future__ import annotations

import time
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

from pathlib import Path
import statistics

from pinn_engine.dsl.templates import get_template
from pinn_engine.dsl.templates_lib import damped_oscillator, lorenz  # registers
from pinn_engine.core.trainer import train, TrainConfig
from pinn_engine.automl import run_search


TEMPLATES = ["damped_oscillator", "lorenz"]
N_AUTOML_TRIALS = 15
N_REPEATS = 1


def manual_configs(tpl):
    """Three hand-picked baselines: small, medium, large."""
    base = tpl.default_config().model_dump()
    return {
        "manual_small": {**base, "depth": 3, "width": 32, "adam_epochs": 1000},
        "manual_medium": {**base, "depth": 4, "width": 64, "adam_epochs": 1500},
        "manual_large": {**base, "depth": 6, "width": 128, "adam_epochs": 2000},
    }


def run_one(tpl, cfg_dict, seed):
    cfg = TrainConfig(**{**cfg_dict, "seed": seed, "accelerator": "cpu"})
    system = tpl.system()
    data, truth = tpl.synthetic_data(seed=seed)
    t0 = time.time()
    result = train(system=system, data=data, config=cfg)
    elapsed = time.time() - t0
    err = tpl.objective(result)
    return err, elapsed


def main():
    rows = []
    for template_name in TEMPLATES:
        tpl = get_template(template_name)

        # Manual baselines.
        for label, cfg_dict in manual_configs(tpl).items():
            errs, ts = [], []
            for r in range(N_REPEATS):
                err, t = run_one(tpl, cfg_dict, seed=r)
                errs.append(err); ts.append(t)
            rows.append({
                "template": template_name,
                "method": label,
                "median_err": statistics.median(errs),
                "median_time_s": statistics.median(ts),
            })
            print(f"{template_name:>20} | {label:>15} | err={statistics.median(errs):.3g} "
                  f"time={statistics.median(ts):.1f}s")

        # AutoML.
        t0 = time.time()
        study = run_search(template_name=template_name, n_trials=N_AUTOML_TRIALS,
                           study_name=f"bench_{template_name}", seed=0)
        elapsed = time.time() - t0
        rows.append({
            "template": template_name,
            "method": f"automl_{N_AUTOML_TRIALS}trials",
            "median_err": study.best_value,
            "median_time_s": elapsed,
        })
        print(f"{template_name:>20} | automl_{N_AUTOML_TRIALS:>2}trial | "
              f"err={study.best_value:.3g} time={elapsed:.1f}s")

    # Write markdown report.
    out = Path(__file__).parent / "results.md"
    with out.open("w") as f:
        f.write("# AutoML vs manual benchmark\n\n")
        f.write("| Template | Method | Median rel-err | Median wall-clock (s) |\n")
        f.write("|---|---|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['template']} | {r['method']} | {r['median_err']:.3g} "
                    f"| {r['median_time_s']:.1f} |\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
