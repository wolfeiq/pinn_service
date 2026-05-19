"""Discover the damping and stiffness of a damped harmonic oscillator from noisy x(t).

Run::

    python examples/01_damped_oscillator.py

What happens:
1. Synthetic ``x(t)`` is generated from ``ẍ + 0.5·ẋ + 10·x = 0`` with Gaussian noise.
2. The engine compiles the equation, runs the well-posedness pre-flight, and trains.
3. The discovered ``c`` and ``k`` are printed alongside their truth values.
4. A reproducibility manifest is written under ``manifests/``.

Expected output: ``c ≈ 0.5``, ``k ≈ 10.0`` after a few thousand Adam epochs.
"""
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

from pinn_engine.dsl import Variable, Parameter, Unknown, Sensor, System
from pinn_engine.data import generate_damped_oscillator
from pinn_engine.core.trainer import train, TrainConfig
from pinn_engine.diagnostics import default_bundle
from pinn_engine.repro import write_manifest


def main():
    # 1. Declare the inverse problem with the DSL.
    #    Bounds matter: PINA initializes each Unknown to the midpoint of its
    #    bounds, so loose bounds give a bad init that may never converge.
    #    See README "Why your bounds matter".
    t = Variable("t")
    x = Variable("x", depends_on=t)
    m = Parameter("m", value=1.0)
    c = Unknown("c", bounds=(0.0, 1.5))
    k = Unknown("k", bounds=(0.0, 20.0))
    system = System(
        state=[x],
        equations=[m * x.dd + c * x.d + k * x],
        sensors=[Sensor("x_meas", observes=x, noise_std=0.01)],
    )

    # 2. Generate synthetic data with known truth.
    data, truth = generate_damped_oscillator(c=0.5, k=10.0, t_end=5.0, noise_std=0.01, seed=42)
    print(f"Truth: c={truth['c']}, k={truth['k']}")
    print(f"Data: {data['x_meas'][0].shape[0]} noisy samples of x(t)")

    # 3. Train. The pre-flight check runs first; diagnostics are attached.
    config = TrainConfig(
        depth=4, width=64, activation="sintanh",
        lr=2e-3, adam_epochs=3000, lbfgs_iters=50,
        t_range=(0.0, 5.0), n_collocation=1500, batch_size=512,
        seed=42, accelerator="cpu",
    )
    result = train(system=system, data=data, config=config, callbacks=default_bundle())

    # 4. Report.
    print(f"\nDiscovered: c={result.final_params['c']:.4f}, k={result.final_params['k']:.4f}")
    print(f"Errors:     c={abs(result.final_params['c']-truth['c']):.4f}, "
          f"k={abs(result.final_params['k']-truth['k']):.4f}")
    print(f"Final loss: {result.final_loss:.4g}")

    # 5. Write manifest.
    manifest_path = write_manifest(template="damped_oscillator_example",
                                    result=result, data=data)
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
