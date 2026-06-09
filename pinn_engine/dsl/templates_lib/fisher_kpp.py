"""Fisher-KPP reaction-diffusion inverse: discover diffusion ``D`` and growth
rate ``r`` from a travelling-front ``u(x,t)``.

PDE: ``u_t = D·u_xx + r·u(1−u)`` on ``x∈[0,L]``, ``u(0,t)=1``, ``u(L,t)=0``.
This is the archetypal **reaction-diffusion** equation — population genetics
(spread of an advantageous gene, Fisher 1937), invasive-species ecology,
combustion fronts, and tumour growth. It blends linear diffusion with a
logistic reaction ``r·u(1−u)`` and admits a travelling front advancing at the
KPP speed ``~2√(rD)`` with width ``~√(D/r)``.

A different physical domain from the engine's mechanics/transport templates, and
its first **logistic (quadratic) reaction** term. Two unknowns: the front *speed*
constrains the combination ``rD`` and the front *width* the ratio ``D/r``, so the
pair is separately identifiable. Inverse: recover ``D, r`` from a dense noisy
``u(x,t)`` grid.
"""
from __future__ import annotations

from pinn_engine.dsl.symbols import Variable, Unknown, Sensor
from pinn_engine.dsl.system import System
from pinn_engine.dsl.templates import register_template
from pinn_engine.core.trainer import TrainConfig


X_MIN, X_MAX = 0.0, 20.0
T_MIN, T_MAX = 0.0, 8.0


def build_system() -> System:
    x = Variable("x")
    t = Variable("t")
    u = Variable("u", depends_on=(x, t))   # space first, time last
    D = Unknown("D", bounds=(0.05, 5.0))   # diffusion; truth 0.5
    r = Unknown("r", bounds=(0.1, 5.0))    # growth rate; truth 1.0
    return System(
        state=[u],
        # u_t − D·u_xx − r·u(1−u) = 0   (diffusion + logistic reaction)
        equations=[u.diff(t, 1) - D * u.diff(x, 2) - r * u * (1 - u)],
        sensors=[
            Sensor("u_meas", observes=u, noise_std=1e-2),
        ],
    )


@register_template("fisher_kpp")
class FisherKPP:
    """Fisher-KPP reaction-diffusion; recover diffusion ``D`` and growth rate
    ``r`` from a travelling-front ``u(x,t)`` grid."""

    truth = {"D": 0.5, "r": 1.0}
    unknown_bounds = {"D": (0.05, 5.0), "r": (0.1, 5.0)}

    @staticmethod
    def system() -> System:
        return build_system()

    @staticmethod
    def default_config() -> TrainConfig:
        # The front is a smooth tanh-like profile (not as steep as Burgers), so
        # modest Fourier capacity suffices. Both D and r are O(1). NOTE: the
        # growth rate r recovers robustly (~1%), but the diffusion D is
        # sub-dominant to the reaction and prone to "explain-away" (the network
        # represents the front shape and drives D→0). D is identifiable on clean
        # data (direct FD regression → 0.501); see docs/fisher_kpp_experiments.md.
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
            fourier_features=32,
            fourier_sigma=3.0,
        )

    @staticmethod
    def synthetic_data(seed: int = 0):
        from pinn_engine.data.synthetic import generate_fisher_kpp
        return generate_fisher_kpp(seed=seed)

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
        truth = FisherKPP.truth
        errs = [abs(result.final_params[k] - v) / abs(v) for k, v in truth.items()]
        return float(sum(errs) / len(errs))
