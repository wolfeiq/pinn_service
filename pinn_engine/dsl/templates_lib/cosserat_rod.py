"""Cosserat rod 1D longitudinal vibration: discover Young's modulus E.

The simplest meaningful Cosserat-style problem: a 1D elastic rod
(soft-robotics building block) undergoing axial vibration. Pure
linear-elastic continuum mechanics, governed by the wave equation::

    ρ · ∂²u/∂t²  =  E · ∂²u/∂s²

where ``u(s, t)`` is axial displacement, ``ρ`` is density (known),
``E`` is Young's modulus (UNKNOWN).

Boundary conditions: ``u(0, t) = 0`` (fixed), ``∂u/∂s|_{s=L} = 0`` (free).
Initial conditions: ``u(s, 0) = u₀(s)`` (small Gaussian bump),
``∂u/∂t|_{t=0} = 0``.

We don't enforce BCs/ICs as separate PINN conditions in this MVP — the
sensor data over the (s, t) domain implicitly constrains them. Future
work: explicit BC / IC conditions for sharper convergence.

Cosserat-rod families that build on this: include shear and curvature
strains to recover the full PyElastica / soft-robotics formulation
(arxiv 2312.09165 DD-PINN, 44 000× FEM speedup). This template is the
"linear elasticity" entry point.
"""
from __future__ import annotations

from pinn_engine.dsl.symbols import Variable, Parameter, Unknown, Sensor
from pinn_engine.dsl.system import System
from pinn_engine.dsl.templates import register_template
from pinn_engine.core.trainer import TrainConfig


# Physical defaults — typical for a soft-rubber rod (truth ~ 1 MPa).
RHO = 1000.0   # density [kg/m³]
L = 1.0        # rod length [m]
T_END = 0.01   # simulation horizon [s] — short, captures the first wave reflection


def build_system() -> System:
    s = Variable("s")
    t = Variable("t")
    u = Variable("u", depends_on=(s, t))
    rho = Parameter("rho", value=RHO)
    # Bounds span 0.1× to 10× the truth — wide enough to be interesting
    # without overflowing the well-posedness check.
    E = Unknown("E", bounds=(1e5, 1e7))
    return System(
        state=[u],
        equations=[rho * u.diff(t, 2) - E * u.diff(s, 2)],
        sensors=[Sensor("u_meas", observes=u, noise_std=1e-4)],
    )


@register_template("cosserat_rod")
class CosseratRod:
    """1D axial vibration of a soft rod; recover ``E`` from a strain-gauge array."""

    truth = {"E": 1.0e6}
    unknown_bounds = {"E": (1e5, 1e7)}

    @staticmethod
    def system() -> System:
        return build_system()

    @staticmethod
    def default_config() -> TrainConfig:
        return TrainConfig(
            depth=6,
            width=64,
            activation="sintanh",
            lr=1e-3,
            adam_epochs=3000,
            lbfgs_iters=0,
            balancer="lra",
            t_range=(0.0, T_END),
            spatial_ranges={"s": (0.0, L)},
            n_collocation=4000,
            batch_size=1024,
            lam_data_init=1000.0,
            lam_physics_init=1.0,
        )

    @staticmethod
    def synthetic_data(seed: int = 0):
        from pinn_engine.data.synthetic import generate_cosserat_rod
        return generate_cosserat_rod(seed=seed)

    @staticmethod
    def automl_space(trial):
        return TrainConfig(
            depth=trial.suggest_int("depth", 4, 8),
            width=trial.suggest_categorical("width", [32, 64, 128]),
            activation=trial.suggest_categorical(
                "activation", ["tanh", "sintanh", "swish"]
            ),
            lr=trial.suggest_float("lr", 5e-4, 5e-3, log=True),
            lam_data_init=trial.suggest_float("lam_data_init", 100.0, 10000.0, log=True),
            lam_physics_init=1.0,
            balancer=trial.suggest_categorical("balancer", ["none", "lra", "sapinn"]),
            adam_epochs=2000,
            lbfgs_iters=0,
            t_range=(0.0, T_END),
            spatial_ranges={"s": (0.0, L)},
            n_collocation=3000,
            batch_size=1024,
        )

    @staticmethod
    def objective(result) -> float:
        truth = CosseratRod.truth
        errs = [
            abs(result.final_params[name] - val) / max(abs(val), 1e-6)
            for name, val in truth.items()
        ]
        return float(sum(errs) / len(errs))
