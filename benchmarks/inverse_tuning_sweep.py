"""Quick parameter sweep to find a damped-oscillator config that actually converges.

Trial-and-error guided by Auto-PINN / inverse-PINN folklore:
* sin / sintanh activations win on oscillatory targets
* width > depth (128-wide × 4-deep beats 32-wide × 8-deep)
* data-loss weight 10-1000× physics for inverse problems
* tighter unknown bounds → faster convergence (less to search)
* Adam lr in 1e-3 to 1e-2 range
"""
from __future__ import annotations

import warnings, logging, time, json
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("pina").setLevel(logging.ERROR)
import os; os.environ["PYTHONWARNINGS"] = "ignore"

from pinn_engine.dsl import Variable, Parameter, Unknown, Sensor, System
from pinn_engine.data import generate_damped_oscillator
from pinn_engine.core.trainer import train, TrainConfig


def build_system(c_bounds=(0.0, 5.0), k_bounds=(0.0, 100.0)):
    t = Variable("t")
    x = Variable("x", depends_on=t)
    m = Parameter("m", value=1.0)
    c = Unknown("c", bounds=c_bounds)
    k = Unknown("k", bounds=k_bounds)
    return System(
        state=[x],
        equations=[m * x.dd + c * x.d + k * x],
        sensors=[Sensor("x_meas", observes=x, noise_std=0.01)],
    )


def run_one(label, sys_kwargs, config_kwargs, data, truth, epochs):
    config = TrainConfig(
        depth=4, width=64, activation="sintanh",
        adam_epochs=epochs, lbfgs_iters=0, balancer="none",
        t_range=(0.0, 5.0), n_collocation=1500, batch_size=512,
        seed=42, accelerator="cpu", deterministic=False,
        log_every_n_steps=200,
        **config_kwargs,
    )
    system = build_system(**sys_kwargs)
    t0 = time.time()
    result = train(system=system, data=data, config=config)
    elapsed = time.time() - t0
    c_err = abs(result.final_params["c"] - truth["c"])
    k_err = abs(result.final_params["k"] - truth["k"]) / abs(truth["k"])
    return {
        "label": label,
        "c": result.final_params["c"],
        "k": result.final_params["k"],
        "c_abs_err": c_err,
        "k_rel_err": k_err,
        "loss": result.final_loss,
        "time_s": elapsed,
    }


def main():
    data, truth = generate_damped_oscillator(c=0.5, k=10.0, t_end=5.0, noise_std=0.01, seed=42)
    print(f"Truth: c={truth['c']}, k={truth['k']}\n")

    EPOCHS = 800  # quick screen — ~30s each on CPU

    trials = [
        # baseline (current default)
        ("baseline_lam100", {}, {"lr": 2e-3, "lam_data_init": 100.0, "lam_physics_init": 1.0}),
        # crank data weight
        ("lam_data_1000", {}, {"lr": 2e-3, "lam_data_init": 1000.0, "lam_physics_init": 1.0}),
        ("lam_data_10000", {}, {"lr": 2e-3, "lam_data_init": 10000.0, "lam_physics_init": 1.0}),
        # higher lr
        ("lr_5e-3_lam100", {}, {"lr": 5e-3, "lam_data_init": 100.0}),
        ("lr_1e-2_lam100", {}, {"lr": 1e-2, "lam_data_init": 100.0}),
        # tight bounds — easier search
        ("tight_bounds_lam100",
         {"c_bounds": (0.0, 1.5), "k_bounds": (0.0, 20.0)},
         {"lr": 2e-3, "lam_data_init": 100.0}),
        # different activations
        ("sin_act_lam1000",
         {}, {"lr": 2e-3, "lam_data_init": 1000.0, "activation": "sin"}),
        ("tanh_act_lam1000",
         {}, {"lr": 2e-3, "lam_data_init": 1000.0, "activation": "tanh"}),
        # wider net
        ("wide128_lam1000",
         {}, {"lr": 2e-3, "lam_data_init": 1000.0, "width": 128}),
    ]

    rows = []
    for label, sk, ck in trials:
        try:
            r = run_one(label, sk, ck, data, truth, EPOCHS)
        except Exception as e:
            r = {"label": label, "error": str(e)[:80]}
        rows.append(r)
        if "error" in r:
            print(f"  {label:>22}  ERROR: {r['error']}")
        else:
            print(f"  {label:>22}  c={r['c']:7.3f}  k={r['k']:7.3f}  "
                  f"c_err={r['c_abs_err']:.3f}  k_rel={r['k_rel_err']:.2%}  "
                  f"t={r['time_s']:.1f}s")

    # Rank by combined error
    scored = [r for r in rows if "error" not in r]
    scored.sort(key=lambda r: r["c_abs_err"] / 0.5 + r["k_rel_err"])
    print("\n=== Ranked (best first) ===")
    for r in scored[:5]:
        print(f"  {r['label']:>22}  c={r['c']:7.3f}  k={r['k']:7.3f}  "
              f"c_err={r['c_abs_err']:.3f}  k_rel={r['k_rel_err']:.2%}")


if __name__ == "__main__":
    main()
