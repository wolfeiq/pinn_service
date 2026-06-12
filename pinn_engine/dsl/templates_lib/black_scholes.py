"""Black-Scholes inverse: discover the implied volatility ``σ`` from option
prices — a non-physics inverse PDE (quantitative finance).

In log-price ``x = ln(S)`` the Black-Scholes equation is constant-coefficient::

    V_t + (r − σ²/2)·V_x + (σ²/2)·V_xx − r·V = 0,   x∈[ln 0.4, ln 2.5], t∈[0, 0.9]

where ``V(x,t)`` is a European-call price, ``r`` the (known) risk-free rate,
``K`` the strike, ``T`` the expiry. Ground truth is the exact Black-Scholes call
formula (`generate_black_scholes`); the inverse recovers the **implied
volatility ``σ``** — precisely the quantity options traders back out of market
prices every day.

This template exists to show the engine is a *general* inverse-PDE solver, not a
physics-only tool: same DSL, same CRLB preflight, same training stack, an
entirely different field. ``σ`` enters as the variance ``σ²`` in both the drift
and diffusion terms.
"""
from __future__ import annotations

import numpy as np

from pinn_engine.dsl.symbols import Variable, Parameter, Unknown, Sensor
from pinn_engine.dsl.system import System
from pinn_engine.dsl.templates import register_template
from pinn_engine.core.trainer import TrainConfig


R_RATE = 0.05                       # risk-free rate (known)
K_STRIKE = 1.0
X_MIN, X_MAX = float(np.log(0.4)), float(np.log(2.5))
T_MIN, T_MAX = 0.0, 0.9


def build_system() -> System:
    x = Variable("x")
    t = Variable("t")
    V = Variable("V", depends_on=(x, t))   # log-price first, time last
    r = Parameter("r", value=R_RATE)
    sigma = Unknown("sigma", bounds=(0.05, 1.0))   # implied volatility; truth 0.3
    return System(
        state=[V],
        equations=[V.diff(t, 1) + (r - 0.5 * sigma ** 2) * V.diff(x, 1)
                   + 0.5 * sigma ** 2 * V.diff(x, 2) - r * V],
        sensors=[Sensor("V_meas", observes=V, noise_std=1e-3)],
    )


@register_template("black_scholes")
class BlackScholes:
    """Black-Scholes (log-price); recover the implied volatility ``σ`` from a
    dense European-call price grid ``V(x,t)``."""

    truth = {"sigma": 0.3}
    unknown_bounds = {"sigma": (0.05, 1.0)}

    @staticmethod
    def system() -> System:
        return build_system()

    @staticmethod
    def default_config() -> TrainConfig:
        # The call price is a smooth, monotone surface — no Fourier features
        # needed. σ enters as σ², a sub-decade O(0.1-1) unknown; a generous
        # param-LR moves it from the midpoint init.
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
            param_lr_scale=30.0,
            fourier_features=0,
        )

    @staticmethod
    def synthetic_data(seed: int = 0):
        from pinn_engine.data.synthetic import generate_black_scholes
        return generate_black_scholes(seed=seed)

    @staticmethod
    def automl_space(trial):
        return TrainConfig(
            depth=trial.suggest_int("depth", 4, 6),
            width=trial.suggest_categorical("width", [32, 64, 128]),
            activation=trial.suggest_categorical("activation", ["tanh", "sintanh", "swish"]),
            lr=trial.suggest_float("lr", 5e-4, 5e-3, log=True),
            lam_data_init=trial.suggest_float("lam_data_init", 10.0, 1000.0, log=True),
            lam_physics_init=1.0,
            param_lr_scale=trial.suggest_float("param_lr_scale", 5.0, 100.0, log=True),
            balancer=trial.suggest_categorical("balancer", ["none", "lra", "sapinn"]),
            adam_epochs=3000,
            lbfgs_iters=0,
            t_range=(T_MIN, T_MAX),
            spatial_ranges={"x": (X_MIN, X_MAX)},
            n_collocation=4000,
            batch_size=2048,
            fourier_features=trial.suggest_categorical("fourier_features", [0, 16, 32]),
            fourier_sigma=trial.suggest_float("fourier_sigma", 1.0, 8.0, log=True),
        )

    @staticmethod
    def objective(result) -> float:
        return abs(result.final_params["sigma"] - BlackScholes.truth["sigma"]) / 0.3
