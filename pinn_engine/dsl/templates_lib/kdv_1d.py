"""Korteweg-de Vries (KdV) inverse: discover the dispersion coefficient ``δ``.

PDE: ``u_t + 6·u·u_x + δ·u_xxx = 0`` on ``x∈[−8,8]``, ``t∈[0,2]`` — the canonical
**dispersive, nonlinear, third-order** PDE (shallow-water waves, ion-acoustic
plasma waves, optical fibres). It is the birthplace of the **soliton**: a
localized travelling wave where nonlinear steepening (``6·u·u_x``) exactly
balances dispersion (``δ·u_xxx``).

This is the engine's first **third-order** (``u_xxx``) and first **dispersive**
PDE template — it exercises the autograd path to one order higher than the
wave/diffusion templates. Ground truth is the exact single-soliton solution
(`generate_kdv_soliton`), so there is no solver error. The inverse recovers the
dispersion ``δ`` from a dense noisy ``u(x,t)`` grid; because dispersion is a
leading-order term in a soliton, ``δ`` is cleanly identifiable.
"""
from __future__ import annotations

from pinn_engine.dsl.symbols import Variable, Unknown, Sensor
from pinn_engine.dsl.system import System
from pinn_engine.dsl.templates import register_template
from pinn_engine.core.trainer import TrainConfig


X_MIN, X_MAX = -8.0, 8.0
T_MIN, T_MAX = 0.0, 2.0


def build_system() -> System:
    x = Variable("x")
    t = Variable("t")
    u = Variable("u", depends_on=(x, t))   # space first, time last
    delta = Unknown("delta", bounds=(0.05, 5.0))   # dispersion; truth 0.5
    return System(
        state=[u],
        # u_t + 6·u·u_x + δ·u_xxx = 0   (nonlinear advection + 3rd-order dispersion)
        equations=[u.diff(t, 1) + 6.0 * u * u.diff(x, 1) + delta * u.diff(x, 3)],
        sensors=[Sensor("u_meas", observes=u, noise_std=5e-3)],
    )


@register_template("kdv_1d")
class KdV1D:
    """KdV soliton; recover the dispersion coefficient ``δ`` from a dense
    ``u(x,t)`` grid."""

    truth = {"delta": 0.5}
    unknown_bounds = {"delta": (0.05, 5.0)}

    @staticmethod
    def system() -> System:
        return build_system()

    @staticmethod
    def default_config() -> TrainConfig:
        # The soliton is a smooth localized pulse; Fourier features help the
        # network represent it and (crucially) its 3rd derivative. NOTE: δ is the
        # coefficient of the highest-order (3rd) derivative and is prone to PINN
        # "explain-away" — the network represents the soliton shape and drives δ
        # to its bound. δ is identifiable (direct FD regression → 0.504); see
        # docs/pde_zoo_experiments.md. Use the field-regression route for a robust
        # δ, or treat the PINN result as a lower-confidence estimate.
        return TrainConfig(
            depth=5,
            width=96,
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
            fourier_features=64,
            fourier_sigma=3.0,
        )

    @staticmethod
    def synthetic_data(seed: int = 0):
        from pinn_engine.data.synthetic import generate_kdv_soliton
        return generate_kdv_soliton(seed=seed)

    @staticmethod
    def automl_space(trial):
        return TrainConfig(
            depth=trial.suggest_int("depth", 4, 7),
            width=trial.suggest_categorical("width", [64, 96, 128]),
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
            fourier_features=trial.suggest_categorical("fourier_features", [32, 64, 128]),
            fourier_sigma=trial.suggest_float("fourier_sigma", 1.0, 8.0, log=True),
        )

    @staticmethod
    def objective(result) -> float:
        return abs(result.final_params["delta"] - KdV1D.truth["delta"]) / KdV1D.truth["delta"]
