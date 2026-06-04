"""Test L-BFGS post-Adam on coupled_drag_3d.

c_y has been stuck at ~22% rel-err vs CRLB 0.62% (36x headroom).
The current adaptive_controller R&D path was a dead end. This
checks the proper lever: a second-order optimizer pass.

Three configurations:
  A. Adam 2000 (baseline)              - current default
  B. Adam 2000 + L-BFGS 100            - small LBFGS chaser
  C. Adam 2000 + L-BFGS 500            - bigger LBFGS chaser
"""
from __future__ import annotations
import time
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
    result = train(system=tpl.system(), data=data, config=cfg)
    dt = time.time() - t0
    rel_errs = {
        n: abs(result.final_params[n] - truth[n]) / abs(truth[n])
        for n in truth
    }
    print(f"\n=== {label} | lbfgs_iters={lbfgs_iters} | wall={dt:.1f}s ===")
    print(f"  truth: {truth}")
    print(f"  recovered: {result.final_params}")
    print(f"  rel_err: c_x={rel_errs['c_x']:.2%}  c_y={rel_errs['c_y']:.2%}  c_n={rel_errs['c_n']:.2%}")
    print(f"  mean rel_err: {sum(rel_errs.values()) / 3:.2%}")
    return rel_errs


if __name__ == "__main__":
    A = run_one("A: Adam-only baseline",      lbfgs_iters=0)
    B = run_one("B: Adam + L-BFGS 100 iters", lbfgs_iters=100)
    C = run_one("C: Adam + L-BFGS 500 iters", lbfgs_iters=500)

    print("\n=== SUMMARY ===")
    print(f"  baseline c_y={A['c_y']:.2%}, +LBFGS100 c_y={B['c_y']:.2%}, +LBFGS500 c_y={C['c_y']:.2%}")
