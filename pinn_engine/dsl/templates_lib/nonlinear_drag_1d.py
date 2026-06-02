"""1-DOF nonlinear-drag inverse: discover linear + quadratic damping
coefficients from a noisy 1-D velocity measurement.

Equation of motion (a point mass under constant force, with linear and
quadratic damping):

    m · u̇  =  τ  +  c_lin · u  +  c_quad · u²

With negative damping coefficients ``c_lin``, ``c_quad`` and positive
``τ``, the body accelerates from rest and asymptotes to a steady-state
velocity ``u_ss`` set by ``τ + c_lin · u_ss + c_quad · u_ss² = 0``.

This is the simplest meaningful inverse problem with two unknowns that
trade off (linear vs quadratic damping): the data can pin them together
but only weakly separates them in a monotone-acceleration regime.
"""
from __future__ import annotations

from pinn_engine.dsl.symbols import Variable, Parameter, Unknown, Sensor
from pinn_engine.dsl.system import System
from pinn_engine.dsl.templates import register_template
from pinn_engine.core.trainer import TrainConfig


# Reference constants for the synthetic problem.
M_EFF = 45.0              # effective mass [kg]
TAU = 10.0                # constant control force [N]


def build_system() -> System:
    t = Variable("t")
    u = Variable("u", depends_on=t)
    m_eff = Parameter("m_eff", value=M_EFF)
    tau = Parameter("tau", value=TAU)
    # Negative-convention damping coefficients. Bounds span a wide
    # plausible range but deliberately *off-centre* so the midpoint init
    # (-12.5, -37.5) doesn't accidentally land at truth (-10, -30) —
    # the inverse problem must actually move the parameters to converge.
    c_lin = Unknown("c_lin", bounds=(-25.0, 0.0))
    c_quad = Unknown("c_quad", bounds=(-75.0, 0.0))
    # Equation: M_eff · u̇ − c_lin · u − c_quad · u² − τ = 0
    return System(
        state=[u],
        equations=[m_eff * u.d - c_lin * u - c_quad * u ** 2 - tau],
        sensors=[Sensor("u_meas", observes=u, noise_std=0.02)],
    )


@register_template("nonlinear_drag_1d")
class NonlinearDrag1D:
    """1-DOF inverse problem: discover linear + quadratic damping
    coefficients (``c_lin``, ``c_quad``) from a noisy velocity sensor."""

    truth = {"c_lin": -10.0, "c_quad": -30.0}
    unknown_bounds = {"c_lin": (-25.0, 0.0), "c_quad": (-75.0, 0.0)}

    @staticmethod
    def system() -> System:
        return build_system()

    @staticmethod
    def default_config() -> TrainConfig:
        # Discovered by a 12-trial multi-seed AutoML run; best trial reached
        # mean rel-err ~10% across seeds {42, 137, 2718}. The remaining ~10%
        # is genuine partial identifiability: ``c_lin`` and ``c_quad`` both
        # contribute drag, but the monotone-acceleration profile doesn't
        # crisply separate the linear-in-u and quadratic-in-u regimes.
        #
        # Notable choices for this template:
        #   * activation: swish — dynamics are non-oscillatory.
        #   * lam_data_init: 13.5 — for partial-identifiability problems
        #     heavy data weighting doesn't help; the bottleneck is
        #     observation-side, not convergence-side.
        return TrainConfig(
            depth=5,
            width=32,
            activation="swish",
            lr=7.4e-4,
            adam_epochs=1500,
            lbfgs_iters=0,
            balancer="lra",
            t_range=(0.0, 10.0),
            n_collocation=1500,
            batch_size=512,
            lam_data_init=13.5,
            lam_physics_init=1.0,
        )

    @staticmethod
    def synthetic_data(seed: int = 0):
        from pinn_engine.data.synthetic import generate_nonlinear_drag_1d
        return generate_nonlinear_drag_1d(seed=seed)

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
        truth = NonlinearDrag1D.truth
        errs = [
            abs(result.final_params[name] - val) / max(abs(val), 1e-6)
            for name, val in truth.items()
        ]
        return float(sum(errs) / len(errs))
