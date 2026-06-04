"""Test L-BFGS post-Adam on euler_bernoulli_beam.

The 4th-order static beam template was training-limited (high
relative error on EI_unit despite tight CRLB bound). A second-order
optimizer pass after Adam is the canonical fix for this kind of
plateau.

Three configs at the same wall budget:
  A. Adam 1500 (baseline)
  B. Adam 1500 + L-BFGS 100
  C. Adam 1500 + L-BFGS 500
"""
from __future__ import annotations
import time
from pinn_engine.dsl.templates import get_template
from pinn_engine.core.trainer import train


def run_one(label, lbfgs_iters):
    tpl = get_template("euler_bernoulli_beam")
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
    print(f"  rel_err: {rel_errs}")
    return rel_errs


if __name__ == "__main__":
    A = run_one("A: Adam-only baseline", lbfgs_iters=0)
    B = run_one("B: Adam + L-BFGS 100",  lbfgs_iters=100)
    C = run_one("C: Adam + L-BFGS 500",  lbfgs_iters=500)

    print("\n=== SUMMARY ===")
    for label, r in [("baseline", A), ("+LBFGS100", B), ("+LBFGS500", C)]:
        print(f"  {label}: EI_unit rel_err = {r['EI_unit']:.2%}")
