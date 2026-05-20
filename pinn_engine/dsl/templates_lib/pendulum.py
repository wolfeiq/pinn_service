"""1-DOF pendulum with friction: discover the damping coefficient.

A classical robotics demo. A pendulum swings under gravity with an
unknown viscous damping torque; an angle sensor (encoder / IMU) gives
noisy θ(t). The engine recovers the damping coefficient.

Equation::

    I · θ̈  +  c · θ̇  +  m · g · L · sin(θ)  =  0

Knowns:
    I      — moment of inertia about the pivot
    m·g·L  — effective torque constant (treat as a known group; ``g`` and
             geometry are measurable, only damping is uncertain)
Unknown:
    c      — viscous damping coefficient (truth around 0.3 for a typical
             classroom pendulum)

This template exercises a couple of corners the oscillator templates
miss: nonlinear restoring force via ``sin(θ)``, and only *one* unknown
parameter (so identifiability is unambiguous — the data must constrain
``c`` directly).
"""
from __future__ import annotations

import numpy as np
import sympy as sp

from pinn_engine.dsl.symbols import Variable, Parameter, Unknown, Sensor
from pinn_engine.dsl.system import System
from pinn_engine.dsl.templates import register_template
from pinn_engine.core.trainer import TrainConfig


# Defaults — typical classroom / lab pendulum.
I_PEND = 1.0   # moment of inertia [kg·m²]
MGL = 10.0     # m·g·L torque constant [N·m]


def build_system() -> System:
    t = Variable("t")
    theta = Variable("theta", depends_on=t)
    I = Parameter("I", value=I_PEND)
    mgL = Parameter("mgL", value=MGL)
    c = Unknown("c", bounds=(0.0, 1.0))  # midpoint 0.5; truth 0.3
    return System(
        state=[theta],
        equations=[I * theta.dd + c * theta.d + mgL * sp.sin(theta)],
        sensors=[Sensor("theta_meas", observes=theta, noise_std=0.01)],
    )


@register_template("pendulum")
class Pendulum:
    """1-DOF pendulum with viscous damping; recover ``c`` from noisy ``θ(t)``."""

    truth = {"c": 0.3}
    unknown_bounds = {"c": (0.0, 1.0)}

    @staticmethod
    def system() -> System:
        return build_system()

    @staticmethod
    def default_config() -> TrainConfig:
        return TrainConfig(
            depth=5,
            width=32,
            activation="sintanh",
            lr=1.5e-3,
            adam_epochs=1500,
            lbfgs_iters=50,
            balancer="lra",
            t_range=(0.0, 10.0),
            n_collocation=2000,
            batch_size=512,
            lam_data_init=100.0,
            lam_physics_init=1.0,
        )

    @staticmethod
    def synthetic_data(seed: int = 0):
        from pinn_engine.data.synthetic import generate_pendulum
        return generate_pendulum(seed=seed)

    @staticmethod
    def automl_space(trial):
        return TrainConfig(
            depth=trial.suggest_int("depth", 4, 7),
            width=trial.suggest_categorical("width", [32, 64, 128]),
            activation=trial.suggest_categorical(
                "activation", ["tanh", "sintanh", "swish"]
            ),
            lr=trial.suggest_float("lr", 5e-4, 5e-3, log=True),
            lam_data_init=trial.suggest_float("lam_data_init", 10.0, 1000.0, log=True),
            lam_physics_init=1.0,
            balancer=trial.suggest_categorical("balancer", ["none", "lra", "sapinn"]),
            adam_epochs=1500,
            lbfgs_iters=0,
            t_range=(0.0, 10.0),
            n_collocation=2000,
            batch_size=512,
        )

    @staticmethod
    def objective(result) -> float:
        truth = Pendulum.truth
        errs = [
            abs(result.final_params[name] - val) / max(abs(val), 1e-6)
            for name, val in truth.items()
        ]
        return float(sum(errs) / len(errs))
