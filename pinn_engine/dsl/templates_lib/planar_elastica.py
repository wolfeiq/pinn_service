"""Large-deflection planar elastica inverse: discover bending stiffness
``EI`` of a soft continuum rod from its measured tangent-angle profile.

This is the **geometrically-exact** soft-robotics rod — the step up from
the linear ``euler_bernoulli_beam`` template that the ``cosserat_rod``
docstring points to ("include shear and curvature strains to recover the
full PyElastica / soft-robotics formulation"). Here we add the curvature
nonlinearity: the Kirchhoff/Cosserat planar rod valid at *arbitrarily
large* deflection, which is the regime a soft-robotic finger or continuum
manipulator actually operates in.

A slender rod of length ``L`` is clamped horizontal at ``s=0`` and carries
a dead tip load ``P0`` (downward) at the free end. Writing the centerline
by its tangent angle ``θ(s)`` (``x' = cos θ``, ``y' = sin θ``), the bending
moment is ``M(s) = EI · θ'(s)`` and static balance against the tip load
gives the **elastica equation**::

    EI · θ''(s)  =  −P0 · cos(θ(s))

with ``θ(0) = 0`` (clamped horizontal) and ``θ'(L) = 0`` (moment-free tip).
The ``cos(θ)`` is what makes this geometrically nonlinear — at the default
load parameter ``α = P0·L²/EI ≈ 2.5`` the tip rotates ~51° and droops
~0.56·L, where linear (small-slope) beam theory errs by tens of percent.

**Non-dimensionalisation.** With ``s̃ = s/L`` and ``EI = EI_unit · EI_ref``,
the compiled residual is

    EI_unit · θ''(s̃)  +  α_ref · cos(θ)  =  0,     α_ref = P0·L²/EI_ref

so the unknown ``EI_unit`` sits multiplicatively on the highest derivative
and is O(1) — the same well-conditioned family as ``euler_bernoulli_beam``
and ``axial_elastic_bar``. Truth ``EI_unit = 1``; bounds span a decade
either side.

**Measurement model.** ``θ(s̃)`` is exactly what flexible curvature sensors
(fiber-Bragg gratings, IMU arrays, stretch sensors) report along a soft
rod, so the angle formulation *is* the physical sensing model — no need to
numerically differentiate a measured shape. Noise is in radians.

Reference: geometrically-exact elastica / Cosserat rod theory (Antman,
*Nonlinear Problems of Elasticity*); soft-robotics PINN rod identification
(arxiv 2312.09165, DD-PINN). This template is the nonlinear-bending entry
point complementing the linear ``euler_bernoulli_beam``.
"""
from __future__ import annotations

import sympy as sp

from pinn_engine.dsl.symbols import Variable, Parameter, Unknown, Sensor
from pinn_engine.dsl.system import System
from pinn_engine.dsl.templates import register_template
from pinn_engine.core.trainer import TrainConfig


# Reference physical scales.
L_DEFAULT = 1.0        # rod length [m]
P0_DEFAULT = 2.5       # dead tip load [N]
EI_REF = 1.0           # reference bending stiffness [N·m²] (EI_unit = 1 at truth)
ALPHA_REF = P0_DEFAULT * L_DEFAULT * L_DEFAULT / EI_REF   # 2.5 — elastica load parameter


def build_system() -> System:
    # Single spatial variable s̃ ∈ [0, 1]; the engine treats the lone input
    # as its "time" axis internally (see euler_bernoulli_beam for the same
    # static-problem convention).
    s = Variable("s")
    theta = Variable("theta", depends_on=(s,))
    alpha_ref = Parameter("alpha_ref", value=ALPHA_REF)
    # Dimensionless bending stiffness. Truth EI_unit = 1.0; midpoint init 5.05
    # is well off centre, spanning a decade either side.
    EI_unit = Unknown("EI_unit", bounds=(0.1, 10.0))
    # Geometrically-exact residual (dimensionless):
    #   EI_unit · θ''(s̃) + α_ref · cos(θ) = 0
    return System(
        state=[theta],
        equations=[EI_unit * theta.diff(s, 2) + alpha_ref * sp.cos(theta)],
        sensors=[
            # Noisy interior tangent-angle measurements (radians).
            Sensor("theta_meas", observes=theta, noise_std=1e-2),
            # Clamped-root BC θ(0) = 0 as a noise-free pseudo-sensor.
            Sensor("theta_bc", observes=theta, noise_std=0.0),
        ],
    )


@register_template("planar_elastica")
class PlanarElastica:
    """Large-deflection soft-rod cantilever; recover dimensionless bending
    stiffness ``EI_unit`` from a noisy tangent-angle profile. Physical
    ``EI = EI_unit × EI_ref`` with ``EI_ref = 1 N·m²``."""

    truth = {"EI_unit": 1.0}
    unknown_bounds = {"EI_unit": (0.1, 10.0)}

    @staticmethod
    def system() -> System:
        return build_system()

    @staticmethod
    def default_config() -> TrainConfig:
        # θ(s̃) is smooth and monotone (no high-frequency content), so a plain
        # tanh MLP is the right inductive bias — no Fourier features. The
        # nonlinearity is in cos(θ), not in the spatial spectrum. A moderate
        # param_lr_scale moves EI_unit briskly through the well-conditioned
        # dimensionless landscape; the adaptive controller can replace it.
        return TrainConfig(
            depth=4,
            width=64,
            activation="tanh",
            lr=1e-3,
            adam_epochs=3000,
            lbfgs_iters=0,
            balancer="none",
            t_range=(0.0, 1.0),         # s̃ ∈ [0, 1] (non-dimensional arclength)
            n_collocation=1000,
            batch_size=512,
            lam_data_init=100.0,
            lam_physics_init=1.0,
            param_lr_scale=20.0,
            fourier_features=0,
        )

    @staticmethod
    def synthetic_data(seed: int = 0):
        from pinn_engine.data.synthetic import generate_planar_elastica
        return generate_planar_elastica(seed=seed)

    @staticmethod
    def automl_space(trial):
        return TrainConfig(
            depth=trial.suggest_int("depth", 3, 6),
            width=trial.suggest_categorical("width", [32, 64, 128]),
            activation=trial.suggest_categorical("activation", ["tanh", "sintanh", "swish"]),
            lr=trial.suggest_float("lr", 5e-4, 5e-3, log=True),
            lam_data_init=trial.suggest_float("lam_data_init", 10.0, 1000.0, log=True),
            lam_physics_init=1.0,
            param_lr_scale=trial.suggest_float("param_lr_scale", 1.0, 50.0, log=True),
            balancer=trial.suggest_categorical("balancer", ["none", "lra", "sapinn"]),
            adam_epochs=2000,
            lbfgs_iters=0,
            t_range=(0.0, 1.0),
            n_collocation=1000,
            batch_size=512,
            fourier_features=0,
        )

    @staticmethod
    def objective(result) -> float:
        truth = PlanarElastica.truth
        errs = [
            abs(result.final_params[name] - val) / max(abs(val), 1e-6)
            for name, val in truth.items()
        ]
        return float(sum(errs) / len(errs))
