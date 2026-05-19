"""Lorenz inverse problem: discover ``σ``, ``ρ``, ``β`` from noisy ``(x, y, z)``.

ODE system:

* ``ẋ = σ(y - x)``
* ``ẏ = x(ρ - z) - y``
* ``ż = x·y - β·z``
"""
from __future__ import annotations

from pinn_engine.dsl.symbols import Variable, Unknown, Sensor
from pinn_engine.dsl.system import System
from pinn_engine.dsl.templates import register_template
from pinn_engine.core.trainer import TrainConfig


def build_system() -> System:
    t = Variable("t")
    x = Variable("x", depends_on=t)
    y = Variable("y", depends_on=t)
    z = Variable("z", depends_on=t)
    sigma = Unknown("sigma", bounds=(0.0, 30.0))
    rho = Unknown("rho", bounds=(0.0, 50.0))
    beta = Unknown("beta", bounds=(0.0, 10.0))
    eqs = [
        x.d - sigma * (y - x),
        y.d - (x * (rho - z) - y),
        z.d - (x * y - beta * z),
    ]
    sensors = [
        Sensor("x_meas", observes=x, noise_std=0.02),
        Sensor("y_meas", observes=y, noise_std=0.02),
        Sensor("z_meas", observes=z, noise_std=0.02),
    ]
    return System(state=[x, y, z], equations=eqs, sensors=sensors)


@register_template("lorenz")
class Lorenz:
    """Lorenz system. Chaotic — a stress test for inverse identifiability."""

    truth = {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0}
    unknown_bounds = {"sigma": (0.0, 30.0), "rho": (0.0, 50.0), "beta": (0.0, 10.0)}

    @staticmethod
    def system() -> System:
        return build_system()

    @staticmethod
    def default_config() -> TrainConfig:
        return TrainConfig(
            depth=6,
            width=128,
            activation="sintanh",
            lr=1e-3,
            adam_epochs=8000,
            lbfgs_iters=100,
            balancer="lra",
            t_range=(0.0, 3.0),
            n_collocation=4000,
            batch_size=512,
        )

    @staticmethod
    def synthetic_data(seed: int = 0):
        from pinn_engine.data.synthetic import generate_lorenz
        return generate_lorenz(seed=seed)

    @staticmethod
    def automl_space(trial):
        return TrainConfig(
            depth=trial.suggest_int("depth", 4, 8),
            width=trial.suggest_categorical("width", [64, 128, 256]),
            activation=trial.suggest_categorical(
                "activation", ["tanh", "sintanh", "swish"]
            ),
            lr=trial.suggest_float("lr", 5e-5, 5e-3, log=True),
            adam_epochs=5000,
            lbfgs_iters=0,
            balancer=trial.suggest_categorical("balancer", ["lra", "sapinn", "none"]),
            t_range=(0.0, 3.0),
            n_collocation=3000,
            batch_size=512,
            seed=trial.suggest_int("seed", 0, 9999),
        )

    @staticmethod
    def objective(result) -> float:
        truth = Lorenz.truth
        errs = [
            abs(result.final_params[name] - val) / max(abs(val), 1e-6)
            for name, val in truth.items()
        ]
        return float(sum(errs) / len(errs))
