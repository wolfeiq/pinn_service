"""1-D advection-diffusion inverse: discover velocity ``v`` and diffusivity ``D``.

PDE: ``u_t + v·u_x = D·u_xx`` on ``x∈[−2,2]``, ``t∈[0,1]`` — the canonical linear
transport equation (a contaminant carried by a flow while spreading, heat in a
moving medium, a tracer in groundwater). A Gaussian pulse **advects** at speed
``v`` and **broadens** with diffusivity ``D``.

Two unknowns that are cleanly **separable**: the pulse's mean motion fixes ``v``,
its broadening fixes ``D``. Ground truth is the closed-form advected heat kernel
(see ``generate_advection_diffusion``), so there is no solver error. Inverse:
recover ``v`` and ``D`` from a dense noisy ``u(x,t)`` grid.
"""
from __future__ import annotations

from pinn_engine.dsl.symbols import Variable, Unknown, Sensor
from pinn_engine.dsl.system import System
from pinn_engine.dsl.templates import register_template
from pinn_engine.core.trainer import TrainConfig


X_MIN, X_MAX = -2.0, 2.0
T_MIN, T_MAX = 0.0, 1.0


def build_system() -> System:
    x = Variable("x")
    t = Variable("t")
    u = Variable("u", depends_on=(x, t))   # space first, time last
    v = Unknown("v", bounds=(-2.0, 2.0))   # advection velocity; truth 0.5
    D = Unknown("D", bounds=(1e-3, 1.0))   # diffusivity; truth 0.1
    return System(
        state=[u],
        equations=[u.diff(t, 1) + v * u.diff(x, 1) - D * u.diff(x, 2)],
        sensors=[Sensor("u_meas", observes=u, noise_std=1e-2)],
    )


@register_template("advection_diffusion_1d")
class AdvectionDiffusion1D:
    """1-D advection-diffusion; recover velocity ``v`` and diffusivity ``D`` from
    a dense ``u(x,t)`` grid."""

    truth = {"v": 0.5, "D": 0.1}
    unknown_bounds = {"v": (-2.0, 2.0), "D": (1e-3, 1.0)}

    @staticmethod
    def system() -> System:
        return build_system()

    @staticmethod
    def default_config() -> TrainConfig:
        # Smooth Gaussian pulse — modest Fourier capacity. v and D are O(0.1-1);
        # a generous param-LR moves them from the midpoint init.
        return TrainConfig(
            depth=5,
            width=64,
            activation="tanh",
            lr=1e-3,
            adam_epochs=12000,
            lbfgs_iters=0,
            balancer="none",
            t_range=(T_MIN, T_MAX),
            spatial_ranges={"x": (X_MIN, X_MAX)},
            n_collocation=4000,
            batch_size=2048,
            lam_data_init=100.0,
            lam_physics_init=1.0,
            param_lr_scale=50.0,
            fourier_features=32,
            fourier_sigma=3.0,
        )

    @staticmethod
    def synthetic_data(seed: int = 0):
        from pinn_engine.data.synthetic import generate_advection_diffusion
        return generate_advection_diffusion(seed=seed)

    @staticmethod
    def automl_space(trial):
        return TrainConfig(
            depth=trial.suggest_int("depth", 4, 6),
            width=trial.suggest_categorical("width", [64, 128]),
            activation=trial.suggest_categorical("activation", ["tanh", "sintanh", "swish"]),
            lr=trial.suggest_float("lr", 5e-4, 5e-3, log=True),
            lam_data_init=trial.suggest_float("lam_data_init", 10.0, 1000.0, log=True),
            lam_physics_init=1.0,
            param_lr_scale=trial.suggest_float("param_lr_scale", 10.0, 200.0, log=True),
            balancer=trial.suggest_categorical("balancer", ["none", "lra", "sapinn"]),
            adam_epochs=3000,
            lbfgs_iters=0,
            t_range=(T_MIN, T_MAX),
            spatial_ranges={"x": (X_MIN, X_MAX)},
            n_collocation=4000,
            batch_size=2048,
            fourier_features=trial.suggest_categorical("fourier_features", [16, 32, 64]),
            fourier_sigma=trial.suggest_float("fourier_sigma", 1.0, 8.0, log=True),
        )

    @staticmethod
    def objective(result) -> float:
        truth = AdvectionDiffusion1D.truth
        errs = [abs(result.final_params[k] - val) / max(abs(val), 1e-6)
                for k, val in truth.items()]
        return float(sum(errs) / len(errs))
