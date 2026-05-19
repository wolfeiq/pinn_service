"""Discover the Lorenz system parameters (σ, ρ, β) from noisy (x, y, z) trajectories.

Chaos makes this a real stress test — small parameter errors compound rapidly.
The PINN has to find σ ≈ 10, ρ ≈ 28, β ≈ 8/3 from observations alone.

Expected runtime: 5-10 minutes on CPU at the default config.
"""
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

from pinn_engine.dsl.templates_lib import lorenz  # registers template
from pinn_engine.dsl.templates import get_template
from pinn_engine.core.trainer import train
from pinn_engine.diagnostics import default_bundle
from pinn_engine.repro import write_manifest


def main():
    tpl = get_template("lorenz")
    system = tpl.system()
    data, truth = tpl.synthetic_data(seed=42)
    config = tpl.default_config().model_copy(update={"seed": 42, "accelerator": "cpu"})

    print(f"Truth: σ={truth['sigma']}, ρ={truth['rho']}, β={truth['beta']:.4f}")
    print(f"Data: 3 noisy sensors × {data['x_meas'][0].shape[0]} samples each\n")

    result = train(system=system, data=data, config=config, callbacks=default_bundle())

    print(f"\nDiscovered:")
    for name in ["sigma", "rho", "beta"]:
        truth_val = truth[name]
        dis = result.final_params[name]
        rel = abs(dis - truth_val) / abs(truth_val)
        print(f"  {name}: {dis:.4f}  (truth={truth_val:.4f}, rel_err={rel:.2%})")

    path = write_manifest(template="lorenz_example", result=result, data=data)
    print(f"\nManifest: {path}")


if __name__ == "__main__":
    main()
