"""1D diffusion inverse problem: discover diffusivity ``D`` from ``u(x, t)``.

PDE: ``∂u/∂t = D · ∂²u/∂x²`` on ``x ∈ [-1, 1]``, ``t ∈ [1e-3, 1]``.

The synthetic data is the closed-form Gaussian solution spreading from a
narrow initial bump; a single dense ``u_meas`` grid covers the whole domain
(so the initial/boundary behaviour is implied by the data — no separate
BC/IC pseudo-sensors are needed, unlike the Cosserat wave problem).

This is the second PDE inverse template (after ``cosserat_rod``) and exists
to test whether the two-phase-LR recipe that converged the wave equation
also transfers to a parabolic problem. ``D`` is already O(0.1–1), so unlike
Cosserat's Young's modulus it needs no non-dimensionalisation.
"""
from __future__ import annotations

from pinn_engine.dsl.symbols import Variable, Unknown, Sensor
from pinn_engine.dsl.system import System
from pinn_engine.dsl.templates import register_template
from pinn_engine.core.trainer import TrainConfig


X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX = 1e-3, 1.0


def build_system() -> System:
    x = Variable("x")
    t = Variable("t")
    # depends_on=(x, t): space first, time LAST — the engine treats the last
    # declared dependency as the temporal variable (see System.input_variable).
    u = Variable("u", depends_on=(x, t))
    # Diffusivity, truth 0.1. Bounds midpoint (~0.5) is the init; the unknown
    # must descend to 0.1.
    D = Unknown("D", bounds=(1e-3, 1.0))
    return System(
        state=[u],
        equations=[u.diff(t, 1) - D * u.diff(x, 2)],
        sensors=[
            # Dense interior+boundary measurement grid (noisy). The closed-form
            # Gaussian is sampled over the full (x, t) grid, so IC and BC are
            # implied by the data — no separate pseudo-sensors.
            Sensor("u_meas", observes=u, noise_std=1e-2),
        ],
    )


@register_template("diffusion_1d")
class Diffusion1D:
    """1D diffusion; recover diffusivity ``D`` from a dense ``u(x, t)`` grid."""

    truth = {"D": 0.1}
    unknown_bounds = {"D": (1e-3, 1.0)}

    @staticmethod
    def system() -> System:
        return build_system()

    @staticmethod
    def default_config() -> TrainConfig:
        # Diffusion is parabolic and well-posed — smoother than the wave
        # equation, so it needs less spectral capacity (modest Fourier
        # features) than Cosserat. D is well-scaled, so a moderate param-LR
        # multiplier suffices. Single-phase by default; the two-phase trigger
        # is opt-in for the wave-eq basin and not expected to be needed here.
        return TrainConfig(
            depth=5,
            width=64,
            activation="tanh",
            lr=1e-3,
            adam_epochs=10000,
            lbfgs_iters=0,
            balancer="none",
            t_range=(T_MIN, T_MAX),
            spatial_ranges={"x": (X_MIN, X_MAX)},
            n_collocation=4000,
            batch_size=1024,
            lam_data_init=100.0,
            lam_physics_init=1.0,
            param_lr_scale=100.0,
            fourier_features=16,
            fourier_sigma=2.0,
        )

    @staticmethod
    def synthetic_data(seed: int = 0):
        from pinn_engine.data.synthetic import generate_diffusion_1d
        data, truth = generate_diffusion_1d(seed=seed)
        # The generator emits u_meas input columns as [t, x]; the engine's
        # convention (and this template's depends_on) is space-first [x, t].
        # Swap columns so the data matches compiled.input_names = (x, t).
        meas_input, meas_target = data["u_meas"]
        data["u_meas"] = (meas_input[:, [1, 0]], meas_target)
        return data, truth

    @staticmethod
    def automl_space(trial):
        return TrainConfig(
            depth=trial.suggest_int("depth", 3, 6),
            width=trial.suggest_categorical("width", [32, 64, 128]),
            activation=trial.suggest_categorical("activation", ["tanh", "sintanh", "swish"]),
            lr=trial.suggest_float("lr", 5e-4, 5e-3, log=True),
            lam_data_init=trial.suggest_float("lam_data_init", 10.0, 1000.0, log=True),
            lam_physics_init=1.0,
            balancer=trial.suggest_categorical("balancer", ["none", "lra", "sapinn"]),
            param_lr_scale=trial.suggest_float("param_lr_scale", 10.0, 500.0, log=True),
            adam_epochs=2000,
            lbfgs_iters=0,
            t_range=(T_MIN, T_MAX),
            spatial_ranges={"x": (X_MIN, X_MAX)},
            n_collocation=3000,
            batch_size=1024,
            fourier_features=trial.suggest_categorical("fourier_features", [0, 16, 32, 64]),
            fourier_sigma=trial.suggest_float("fourier_sigma", 1.0, 10.0, log=True),
        )

    @staticmethod
    def objective(result) -> float:
        return abs(result.final_params["D"] - Diffusion1D.truth["D"]) / 0.1
