"""Cosserat rod 1D longitudinal vibration: discover Young's modulus E.

The simplest meaningful Cosserat-style problem: a 1D elastic rod
(soft-robotics building block) undergoing axial vibration. Pure
linear-elastic continuum mechanics, governed by the wave equation::

    ρ · ∂²u/∂t²  =  E · ∂²u/∂s²

where ``u(s, t)`` is axial displacement, ``ρ`` is density (known),
``E`` is Young's modulus (UNKNOWN).

Boundary conditions: ``u(0, t) = 0`` (fixed), ``∂u/∂s|_{s=L} = 0`` (free).
Initial conditions: ``u(s, 0) = u₀(s)`` (Gaussian bump, amplitude 1.0
after non-dimensionalisation), ``∂u/∂t|_{t=0} = 0``.

**Non-dimensionalisation.** Real ``E`` for a soft rubber is ~ 1 MPa
(10⁶ Pa) — a number that makes the loss landscape ill-conditioned
because the gradient ``∂L/∂E ∝ u_ss`` is small relative to the
distance E needs to travel from any reasonable init. To fix this, we
factor out a reference scale ``E_ref = 10⁶`` and let the DSL discover
a dimensionless multiplier ``E_unit ∈ [0.1, 10]`` (truth = 1.0). The
compiled residual becomes ``ρ·u_tt - E_ref·E_unit·u_ss``; everything
the optimizer sees is O(1).

This is standard PDE-inverse-problem practice and matches the
treatment in Wang-Wang-Perdikaris 2021 and the Auto-PINN paper.

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
RHO = 1000.0       # density [kg/m³]
E_REF = 1.0e6      # reference Young's modulus [Pa] (non-dimensionalisation scale)
L = 1.0            # rod length [m]
T_END = 0.01       # simulation horizon [s]


def build_system() -> System:
    s = Variable("s")
    t = Variable("t")
    u = Variable("u", depends_on=(s, t))
    # Divide the equation by E_ref so the residual is O(1) when u and E_unit
    # are both O(1). Original PDE: ρ·u_tt = E_ref·E_unit·u_ss
    # Divided by E_ref: (ρ/E_ref)·u_tt = E_unit·u_ss
    # At truth (E_unit=1, u_amp=1, ρ/E_ref=1e-3, u_tt≈1e5, u_ss≈100):
    #   1e-3 · 1e5 − 1.0 · 100 = 0  ✓
    rho_eref = Parameter("rho_over_E_ref", value=RHO / E_REF)
    # Dimensionless multiplier in (0.1, 10). Midpoint 5.05 vs truth 1.0;
    # AutoML must move it ~5× to converge.
    E_unit = Unknown("E_unit", bounds=(0.1, 10.0))
    return System(
        state=[u],
        equations=[rho_eref * u.diff(t, 2) - E_unit * u.diff(s, 2)],
        sensors=[Sensor("u_meas", observes=u, noise_std=1e-2)],
    )


@register_template("cosserat_rod")
class CosseratRod:
    """1D axial vibration of a soft rod; recover ``E`` from a strain-gauge array.

    The discovered unknown is the *dimensionless* multiplier ``E_unit``;
    physical Young's modulus is ``E = E_unit · E_ref`` with
    ``E_ref = 1 MPa``.
    """

    truth = {"E_unit": 1.0}
    unknown_bounds = {"E_unit": (0.1, 10.0)}

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
        # Fourier features are critical here. Without them the network's
        # u(s, t) prediction is too smooth → u_ss ≈ 0 → ∂L/∂E ≈ 0 → E
        # never updates from its midpoint init. With Fourier features the
        # network can represent high-frequency content and u_ss carries
        # the gradient signal that lets the optimizer move E.
        return TrainConfig(
            depth=trial.suggest_int("depth", 4, 8),
            width=trial.suggest_categorical("width", [32, 64, 128]),
            activation=trial.suggest_categorical(
                "activation", ["tanh", "sintanh", "swish", "sin"]
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
            # Mandatory Fourier embedding — the spectral-bias fix.
            fourier_features=trial.suggest_categorical("fourier_features", [16, 32, 64, 128]),
            fourier_sigma=trial.suggest_float("fourier_sigma", 1.0, 20.0, log=True),
        )

    @staticmethod
    def objective(result) -> float:
        truth = CosseratRod.truth
        errs = [
            abs(result.final_params[name] - val) / max(abs(val), 1e-6)
            for name, val in truth.items()
        ]
        return float(sum(errs) / len(errs))
