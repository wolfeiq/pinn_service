"""Full planar Cosserat (Simo-Reissner) rod inverse: discover bending,
shear *and* axial stiffness of a soft continuum rod from its deformed shape.

This is the geometrically-exact planar rod **with shear and extension** — the
step beyond ``planar_elastica`` (which is inextensible and unshearable, i.e.
bending only). Here the cross-section can shear (its normal ``θ`` is no longer
the centerline tangent) and the centerline can stretch, so the rod carries
three independent strains and three independent stiffnesses:

    bending   EI · κ   = M            κ = θ'        (curvature)
    shear     GA · η   = n₂           η = shear strain
    axial     EA · (ν−1) = n₁         ν = stretch

For a **tip-loaded cantilever with no distributed load**, the internal force is
constant along the rod and equal to the applied tip load ``(Px, Py)``. That
collapses the Simo-Reissner balance laws to three residuals in the measured
fields ``x(s), y(s), θ(s)`` (centerline position + cross-section angle), with
each unknown isolated in one equation::

    axial    EA · (x'cosθ + y'sinθ − 1)  =  Px cosθ + Py sinθ
    shear    GA · (−x'sinθ + y'cosθ)     =  −Px sinθ + Py cosθ
    moment   EI · θ''  +  (Py x' − Px y') =  0

(``x' = cosθ·ν − sinθ·η`` etc.; the first two lines are the constitutive laws
read directly off the measured shape, the third is moment balance.)

**Non-dimensionalisation.** The unknowns are dimensionless multipliers
(truth = 1.0 each) on reference stiffness *numbers* ``EI0, GA0, EA0``
(= stiffness·L²/EI_ref). At the default thick/soft setup (``GA0=EA0=15``,
tip load ``Px=2.5, Py=−4``) the rod develops ~7-27% shear and ~17-31% axial
strain alongside a ~45° tip rotation — large enough that all three stiffnesses
leave a strong, identifiable signature and the network can't explain the
strains away within the sensor-noise latitude (verify with the CRLB preflight).

**Measurement model.** ``x, y`` are centerline position markers (motion
capture / vision) and ``θ`` is the cross-section orientation (IMU array) — the
standard instrumentation of a soft-robotic continuum arm.

Reference: Simo & Vu-Quoc (1986), geometrically-exact rod; Antman, *Nonlinear
Problems of Elasticity*; soft-robotics PINN rod ID (arxiv 2312.09165). This is
the engine's first **multi-output, multi-unknown PDE** inverse — the
structural-mechanics analogue of ``coupled_drag_3d``.
"""
from __future__ import annotations

import sympy as sp

from pinn_engine.dsl.symbols import Variable, Parameter, Unknown, Sensor
from pinn_engine.dsl.system import System
from pinn_engine.dsl.templates import register_template
from pinn_engine.core.trainer import TrainConfig


# Reference dimensionless stiffness numbers and constant tip load (kept in sync
# with pinn_engine.data.synthetic.generate_planar_cosserat).
EI0 = 1.0      # bending number (reference scale)
GA0 = 15.0     # shear number   GA_ref·L²/EI_ref
EA0 = 15.0     # axial number   EA_ref·L²/EI_ref
PX = 2.5       # tip force, axial component (dimensionless)
PY = -4.0      # tip force, transverse component (downward)


def build_system() -> System:
    s = Variable("s")
    x = Variable("x", depends_on=(s,))
    y = Variable("y", depends_on=(s,))
    theta = Variable("theta", depends_on=(s,))

    ei0 = Parameter("EI0", value=EI0)
    ga0 = Parameter("GA0", value=GA0)
    ea0 = Parameter("EA0", value=EA0)
    px = Parameter("Px", value=PX)
    py = Parameter("Py", value=PY)

    # Three dimensionless stiffness multipliers; truth = 1.0, midpoint init 5.05.
    EI_unit = Unknown("EI_unit", bounds=(0.1, 10.0))
    GA_unit = Unknown("GA_unit", bounds=(0.1, 10.0))
    EA_unit = Unknown("EA_unit", bounds=(0.1, 10.0))

    xp = x.diff(s)
    yp = y.diff(s)
    cos_t, sin_t = sp.cos(theta), sp.sin(theta)

    # R1 axial constitutive — isolates EA_unit.
    r_axial = ea0 * EA_unit * (xp * cos_t + yp * sin_t - 1) - (px * cos_t + py * sin_t)
    # R2 shear constitutive — isolates GA_unit.
    r_shear = ga0 * GA_unit * (-xp * sin_t + yp * cos_t) - (-px * sin_t + py * cos_t)
    # R3 moment balance — isolates EI_unit.
    r_moment = ei0 * EI_unit * theta.diff(s, 2) + (py * xp - px * yp)

    return System(
        state=[x, y, theta],
        equations=[r_axial, r_shear, r_moment],
        sensors=[
            Sensor("x_meas", observes=x, noise_std=1e-3),
            Sensor("y_meas", observes=y, noise_std=1e-3),
            Sensor("theta_meas", observes=theta, noise_std=1e-2),
            # Clamped-root BCs as noise-free pseudo-sensors.
            Sensor("x_bc", observes=x, noise_std=0.0),
            Sensor("y_bc", observes=y, noise_std=0.0),
            Sensor("theta_bc", observes=theta, noise_std=0.0),
        ],
    )


@register_template("planar_cosserat")
class PlanarCosserat:
    """Full planar Cosserat rod (shear + extension); recover dimensionless
    bending / shear / axial stiffness ``EI_unit, GA_unit, EA_unit`` from the
    measured deformed shape ``(x, y, θ)`` of a tip-loaded soft rod."""

    truth = {"EI_unit": 1.0, "GA_unit": 1.0, "EA_unit": 1.0}
    unknown_bounds = {
        "EI_unit": (0.1, 10.0),
        "GA_unit": (0.1, 10.0),
        "EA_unit": (0.1, 10.0),
    }

    @staticmethod
    def system() -> System:
        return build_system()

    @staticmethod
    def default_config() -> TrainConfig:
        # x, y, θ are smooth and monotone over the rod — a plain tanh MLP is the
        # right inductive bias (no Fourier features). Three outputs share one
        # trunk. Positions are O(1) with tight noise so data weighting is heavy;
        # param_lr_scale moves the three well-conditioned multipliers briskly.
        return TrainConfig(
            depth=5,
            width=96,
            activation="tanh",
            lr=1e-3,
            adam_epochs=4000,
            lbfgs_iters=0,
            balancer="none",
            t_range=(0.0, 1.0),         # s̃ ∈ [0, 1] (non-dimensional arclength)
            n_collocation=1500,
            batch_size=512,
            lam_data_init=100.0,
            lam_physics_init=1.0,
            param_lr_scale=20.0,
            fourier_features=0,
        )

    @staticmethod
    def synthetic_data(seed: int = 0):
        from pinn_engine.data.synthetic import generate_planar_cosserat
        return generate_planar_cosserat(seed=seed)

    @staticmethod
    def automl_space(trial):
        return TrainConfig(
            depth=trial.suggest_int("depth", 4, 7),
            width=trial.suggest_categorical("width", [64, 96, 128]),
            activation=trial.suggest_categorical("activation", ["tanh", "sintanh", "swish"]),
            lr=trial.suggest_float("lr", 5e-4, 5e-3, log=True),
            lam_data_init=trial.suggest_float("lam_data_init", 10.0, 1000.0, log=True),
            lam_physics_init=1.0,
            param_lr_scale=trial.suggest_float("param_lr_scale", 1.0, 50.0, log=True),
            balancer=trial.suggest_categorical("balancer", ["none", "lra", "sapinn"]),
            adam_epochs=3000,
            lbfgs_iters=0,
            t_range=(0.0, 1.0),
            n_collocation=1500,
            batch_size=512,
            fourier_features=0,
        )

    @staticmethod
    def objective(result) -> float:
        truth = PlanarCosserat.truth
        errs = [
            abs(result.final_params[name] - val) / max(abs(val), 1e-6)
            for name, val in truth.items()
        ]
        return float(sum(errs) / len(errs))
