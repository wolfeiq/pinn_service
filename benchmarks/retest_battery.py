"""End-to-end retest battery.

Sweeps seeds × devices × epoch budgets on the damped oscillator after the
'tighten bounds' fix, plus a smoke pass on the bounds-too-wide warning.
Prints a single table so regressions are easy to spot.
"""
from __future__ import annotations

import warnings, logging, time, sys
warnings.filterwarnings("ignore")
for n in ("pytorch_lightning", "lightning.pytorch", "pina"):
    logging.getLogger(n).setLevel(logging.ERROR)

import torch
from pinn_engine.dsl.templates import get_template
from pinn_engine.dsl.templates_lib import damped_oscillator  # registers
from pinn_engine.core.trainer import train, TrainConfig
from pinn_engine.dsl import Variable, Parameter, Unknown, Sensor, System
from pinn_engine.preflight import check_wellposedness, BoundsTooWideWarning


def run_one(seed: int, accelerator: str, adam_epochs: int):
    tpl = get_template("damped_oscillator")
    system = tpl.system()
    data, truth = tpl.synthetic_data(seed=seed)
    config = tpl.default_config().model_copy(update={
        "seed": seed,
        "adam_epochs": adam_epochs,
        "accelerator": accelerator,
        "lbfgs_iters": 0,
    })
    t0 = time.time()
    result = train(system=system, data=data, config=config)
    elapsed = time.time() - t0
    c_err = abs(result.final_params["c"] - truth["c"])
    k_err = abs(result.final_params["k"] - truth["k"]) / abs(truth["k"])
    return {
        "seed": seed, "device": accelerator, "epochs": adam_epochs,
        "c": result.final_params["c"], "k": result.final_params["k"],
        "c_err": c_err, "k_rel": k_err, "time_s": elapsed,
    }


def check_warning_fires() -> bool:
    """Build a system with loose bounds → BoundsTooWideWarning should fire."""
    t = Variable("t"); x = Variable("x", depends_on=t)
    m = Parameter("m", value=1.0)
    c = Unknown("c", bounds=(0.0, 100.0))   # 100× wider than needed
    k = Unknown("k", bounds=(0.0, 1000.0))
    sys = System(
        state=[x],
        equations=[m * x.dd + c * x.d + k * x],
        sensors=[Sensor("x_meas", observes=x, noise_std=0.01)],
    )
    comp = sys.compile()
    from pinn_engine.core.problem import build_problem
    from pinn_engine.core.networks import build_network
    problem = build_problem(comp, data={}, t_range=(0.0, 5.0))
    net = build_network(input_dim=1, output_dim=1, depth=3, width=16, activation="tanh")
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        check_wellposedness(problem, net, comp, n=64)
        fired = any(issubclass(w.category, BoundsTooWideWarning) for w in w_list)
    return fired


def main():
    print("=== bounds-too-wide warning check ===")
    fired = check_warning_fires()
    print(f"  fires on (0,100) and (0,1000) bounds: {fired}\n")

    print("=== convergence sweep ===")
    print(f"{'seed':>5} {'device':>7} {'epochs':>6}  {'c':>8} {'k':>9}  "
          f"{'c_err':>7} {'k_rel':>8}  {'time':>6}")
    print("-" * 70)
    rows = []
    mps_avail = torch.backends.mps.is_available()
    sweep = []
    # 3 seeds × CPU at 800 epochs
    for s in [0, 7, 42]:
        sweep.append((s, "cpu", 800))
    # MPS if available
    if mps_avail:
        sweep.append((42, "mps", 800))
    # smaller budget — does 400 epochs still converge?
    sweep.append((42, "cpu", 400))
    # tiny budget — does 200 epochs at least move things?
    sweep.append((42, "cpu", 200))

    for seed, dev, ep in sweep:
        try:
            r = run_one(seed, dev, ep)
            rows.append(r)
            ok = "✓" if (r["c_err"] < 0.05 and r["k_rel"] < 0.05) else "·"
            print(f"{r['seed']:>5} {r['device']:>7} {r['epochs']:>6}  "
                  f"{r['c']:>8.4f} {r['k']:>9.4f}  "
                  f"{r['c_err']:>7.4f} {r['k_rel']:>8.2%}  "
                  f"{r['time_s']:>5.1f}s {ok}")
        except Exception as e:
            print(f"{seed:>5} {dev:>7} {ep:>6}  ERROR: {str(e)[:60]}")

    # Summary
    ok_count = sum(1 for r in rows if r["c_err"] < 0.05 and r["k_rel"] < 0.05)
    print(f"\n=== {ok_count}/{len(rows)} runs converged to <5% on both params ===")


if __name__ == "__main__":
    main()
