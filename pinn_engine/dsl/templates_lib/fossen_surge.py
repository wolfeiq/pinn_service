"""Fossen 1-DOF surge inverse: discover drag coefficients from DVL data.

The simplest meaningful AUV inverse problem. The vehicle moves forward
under constant body-frame thrust ``τ_u`` and is observed via a DVL
giving noisy body-frame surge velocity ``u(t)``.

Equation (Fossen 2021, "Handbook of Marine Craft Hydrodynamics and
Motion Control", §6.3.2):

    (m − X_u̇) · u̇  −  X_u · u  −  X_u|u| · |u| · u  =  τ_u

For forward motion (``u > 0``) the absolute-value collapses and the
quadratic term simplifies to ``X_uu · u²``. We assume ``u > 0`` over the
mission window; that's true any time the vehicle moves forward, which
covers most operational AUV scenarios.

Knowns:
    m        — vehicle mass (e.g. 40 kg for the Snapir test platform)
    X_u̇      — surge added mass (e.g. −5 kg)
    τ_u      — constant control thrust (e.g. 10 N)
Unknowns:
    X_u      — linear drag coefficient (Fossen sign convention: negative;
               truth around −10 for Snapir-class AUVs)
    X_uu     — quadratic drag coefficient (also negative; truth around −30)

Reference values are from Mary Koryakina's existing AUV PINN work
(synthetic mission specification) — see
``/Users/mary/.claude/projects/-Users-mary/memory/project_auv_pinn.md``.
"""
from __future__ import annotations

import numpy as np

from pinn_engine.dsl.symbols import Variable, Parameter, Unknown, Sensor
from pinn_engine.dsl.system import System
from pinn_engine.dsl.templates import register_template
from pinn_engine.core.trainer import TrainConfig


# Vehicle / control constants (Snapir-class AUV defaults).
M_VEHICLE = 40.0          # mass [kg]
X_UDOT = -5.0             # surge added mass [kg]; M_EFF = M - X_UDOT
M_EFF = M_VEHICLE - X_UDOT  # = 45 kg
TAU_U = 10.0              # constant control thrust [N]


def build_system() -> System:
    t = Variable("t")
    u = Variable("u", depends_on=t)
    m_eff = Parameter("m_eff", value=M_EFF)
    tau_u = Parameter("tau_u", value=TAU_U)
    # Fossen-convention drag coefficients (negative). Bounds span the
    # literature range for AUVs of this class but deliberately *off-centre*
    # so the midpoint init (X_u ≈ -12.5, X_uu ≈ -37.5) doesn't accidentally
    # land at truth (-10, -30) — the inverse problem must actually move
    # the parameters to converge.
    X_u = Unknown("X_u", bounds=(-25.0, 0.0))
    X_uu = Unknown("X_uu", bounds=(-75.0, 0.0))
    # Equation: M_eff · u̇ − X_u · u − X_uu · u² − τ_u = 0
    return System(
        state=[u],
        equations=[m_eff * u.d - X_u * u - X_uu * u ** 2 - tau_u],
        sensors=[Sensor("u_meas", observes=u, noise_std=0.02)],  # DVL surge noise σ
    )


@register_template("fossen_surge")
class FossenSurge:
    """1-DOF AUV surge inverse: discover drag coefficients X_u, X_uu."""

    truth = {"X_u": -10.0, "X_uu": -30.0}
    unknown_bounds = {"X_u": (-25.0, 0.0), "X_uu": (-75.0, 0.0)}

    @staticmethod
    def system() -> System:
        return build_system()

    @staticmethod
    def default_config() -> TrainConfig:
        # Untuned starting point — AutoML will refine. The damped-oscillator
        # default-config recipe (depth=6, sintanh, lra balancer) is a
        # reasonable prior for this oscillator-free template too; lam_data
        # higher because the DVL data signal is small (u ≤ 0.5 m/s).
        return TrainConfig(
            depth=5,
            width=32,
            activation="sintanh",
            lr=1.5e-3,
            adam_epochs=1500,
            lbfgs_iters=50,
            balancer="lra",
            t_range=(0.0, 10.0),
            n_collocation=1500,
            batch_size=512,
            lam_data_init=200.0,
            lam_physics_init=1.0,
        )

    @staticmethod
    def synthetic_data(seed: int = 0):
        from pinn_engine.data.synthetic import generate_fossen_surge
        return generate_fossen_surge(seed=seed)

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
            n_collocation=1500,
            batch_size=512,
        )

    @staticmethod
    def objective(result) -> float:
        truth = FossenSurge.truth
        errs = [
            abs(result.final_params[name] - val) / max(abs(val), 1e-6)
            for name, val in truth.items()
        ]
        return float(sum(errs) / len(errs))
