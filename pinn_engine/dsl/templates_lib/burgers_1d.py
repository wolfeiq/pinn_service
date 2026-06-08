"""1-D viscous Burgers' equation inverse: discover the viscosity ``ν``.

PDE: ``u_t + u·u_x = ν·u_xx`` on ``x ∈ [-1, 1]``, ``t ∈ [0, 1]``, with
``u(x,0) = −sin(πx)`` and ``u(±1,t) = 0``. Burgers is *the* canonical nonlinear
PDE — the advective term ``u·u_x`` steepens the profile into a sharp internal
layer while viscosity ``ν·u_xx`` smooths it; the balance sets the layer width.
It is the classic PINN benchmark (Raissi et al. 2019).

This is the engine's first **nonlinear-advection** PDE template (the unknown
multiplies the diffusion term, but the residual also contains the nonlinear
``u·u_x`` self-advection). The inverse problem: recover ``ν`` from a dense noisy
``u(x,t)`` grid. Ground truth comes from a stable conservative method-of-lines
solver (see ``generate_burgers_1d``) — the steep layer that blows up naive
finite differences is handled by the flux form + a stiff integrator.
"""
from __future__ import annotations

from pinn_engine.dsl.symbols import Variable, Unknown, Sensor
from pinn_engine.dsl.system import System
from pinn_engine.dsl.templates import register_template
from pinn_engine.core.trainer import TrainConfig


X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 1.0


def build_system() -> System:
    x = Variable("x")
    t = Variable("t")
    u = Variable("u", depends_on=(x, t))   # space first, time last
    # Viscosity, truth 0.05. Bounds span two decades; midpoint init must descend.
    nu = Unknown("nu", bounds=(1e-3, 0.5))
    return System(
        state=[u],
        # u_t + u·u_x − ν·u_xx = 0   (nonlinear self-advection + diffusion)
        equations=[u.diff(t, 1) + u * u.diff(x, 1) - nu * u.diff(x, 2)],
        sensors=[
            Sensor("u_meas", observes=u, noise_std=1e-2),
        ],
    )


@register_template("burgers_1d")
class Burgers1D:
    """1-D viscous Burgers; recover viscosity ``ν`` from a dense ``u(x,t)`` grid."""

    truth = {"nu": 0.05}
    unknown_bounds = {"nu": (1e-3, 0.5)}

    @staticmethod
    def system() -> System:
        return build_system()

    @staticmethod
    def default_config() -> TrainConfig:
        # Burgers develops a steep internal layer near x=0, so the network needs
        # spectral capacity (Fourier features) to represent the sharp gradient —
        # more than smooth diffusion. ν is well-scaled (O(0.01-0.1)); a generous
        # param-LR multiplier moves it from the midpoint init.
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
            param_lr_scale=100.0,
            fourier_features=64,
            fourier_sigma=4.0,
        )

    @staticmethod
    def synthetic_data(seed: int = 0):
        from pinn_engine.data.synthetic import generate_burgers_1d
        return generate_burgers_1d(seed=seed)

    @staticmethod
    def automl_space(trial):
        return TrainConfig(
            depth=trial.suggest_int("depth", 4, 7),
            width=trial.suggest_categorical("width", [64, 128]),
            activation=trial.suggest_categorical("activation", ["tanh", "sintanh", "swish"]),
            lr=trial.suggest_float("lr", 5e-4, 5e-3, log=True),
            lam_data_init=trial.suggest_float("lam_data_init", 10.0, 1000.0, log=True),
            lam_physics_init=1.0,
            param_lr_scale=trial.suggest_float("param_lr_scale", 10.0, 500.0, log=True),
            balancer=trial.suggest_categorical("balancer", ["none", "lra", "sapinn"]),
            adam_epochs=3000,
            lbfgs_iters=0,
            t_range=(T_MIN, T_MAX),
            spatial_ranges={"x": (X_MIN, X_MAX)},
            n_collocation=4000,
            batch_size=2048,
            fourier_features=trial.suggest_categorical("fourier_features", [32, 64, 128]),
            fourier_sigma=trial.suggest_float("fourier_sigma", 1.0, 10.0, log=True),
        )

    @staticmethod
    def objective(result) -> float:
        return abs(result.final_params["nu"] - Burgers1D.truth["nu"]) / Burgers1D.truth["nu"]
