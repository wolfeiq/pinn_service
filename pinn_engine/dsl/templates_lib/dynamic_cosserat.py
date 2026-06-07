"""Dynamic planar Cosserat rod inverse: recover bending / shear / axial
stiffness from the **time-resolved motion** of a soft rod.

The time-domain (inertial) Simo-Reissner rod — the dynamic extension of
``planar_cosserat`` and the engine's most ambitious template: a multi-output,
multi-unknown, **nonlinear PDE inverse over a 2-D space-time domain**. A soft
rod clamped at ``s=0`` is released from the straight horizontal configuration
under a distributed gravity load; it swings down (~50° tip) and oscillates under
light viscous damping, settling toward the static droop. The stiffnesses are
encoded in *how* it moves.

Fields (functions of arclength ``s`` and time ``t``): centerline position
``x, y`` and cross-section angle ``θ`` — three outputs.

Dimensionless equations of motion (time scale ``T = L²√(ρA/EI_ref)``)::

    x_tt      =  ∂Nx/∂s − c·x_t                          (linear momentum, x)
    y_tt      =  ∂Ny/∂s − g − c·y_t                      (linear momentum, y)
    j·θ_tt    =  EI·θ_ss + (x_s·Ny − y_s·Nx)             (angular momentum)

with internal forces ``Nx = n1·cosθ − n2·sinθ``, ``Ny = n1·sinθ + n2·cosθ``,
material forces ``n1 = EA·(ν−1)``, ``n2 = GA·η`` and strains
``ν = x_s cosθ + y_s sinθ`` (stretch), ``η = −x_s sinθ + y_s cosθ`` (shear).
``g`` is the distributed gravity load, ``c`` the (known) viscous damping and
``j = ρI/(ρA·L²)`` the (known) rotary inertia. The rod is loaded by gravity
(not a tip point force — that would shock-excite fast axial/shear waves at the
near-massless free-end node). The unknowns are dimensionless multipliers
``EI_unit, GA_unit, EA_unit`` (truth = 1.0) on reference numbers ``EI0,GA0,EA0``.

The force divergence ``∂Nx/∂s`` is expanded **directly** into second spatial
derivatives of ``x, y, θ`` (programmatically via sympy, see ``build_system``) so
that ``EA, GA`` appear *in the momentum residuals* — which are tied to the
data-anchored accelerations ``x_tt, y_tt``. (Carrying ``Nx, Ny`` as free
auxiliary outputs instead makes the constitutive residual ``Nx = EA·(…)``
trivially satisfiable by ``Nx`` tracking ``EA`` — giving the stiffnesses *no*
gradient. The direct form is what makes the inverse identifiable in training.)
``EI`` appears in the angular-momentum residual.

Reference: Simo & Vu-Quoc (1986) geometrically-exact dynamic rod; PyElastica.
Ground truth from a verified method-of-lines solver (energy conserved to ~1e-7;
damped steady state reproduces the static BVP to ~1e-4).
"""
from __future__ import annotations

import sympy as sp

from pinn_engine.dsl.symbols import Variable, Parameter, Unknown, Sensor
from pinn_engine.dsl.system import System
from pinn_engine.dsl.templates import register_template
from pinn_engine.core.trainer import TrainConfig


# Reference numbers / loads / material constants (kept in sync with
# pinn_engine.data.synthetic.generate_dynamic_cosserat).
EI0 = 1.0
GA0 = 15.0
EA0 = 15.0
G_GRAV = 3.0       # distributed gravity load (force per unit length, downward)
C_DAMP = 0.4       # translational viscous damping (known)
J_ROT = 0.01       # rotary inertia ρI/(ρA·L²) (known)
T_END = 2.5


def build_system() -> System:
    s = Variable("s")
    t = Variable("t")
    x = Variable("x", depends_on=(s, t))
    y = Variable("y", depends_on=(s, t))
    theta = Variable("theta", depends_on=(s, t))

    ei0 = Parameter("EI0", value=EI0)
    ga0 = Parameter("GA0", value=GA0)
    ea0 = Parameter("EA0", value=EA0)
    g = Parameter("g", value=G_GRAV)
    c = Parameter("c", value=C_DAMP)
    j = Parameter("j", value=J_ROT)

    EI_unit = Unknown("EI_unit", bounds=(0.1, 10.0))
    GA_unit = Unknown("GA_unit", bounds=(0.1, 10.0))
    EA_unit = Unknown("EA_unit", bounds=(0.1, 10.0))

    ea = ea0 * EA_unit
    ga = ga0 * GA_unit
    ei = ei0 * EI_unit

    xs = x.diff(s); ys = y.diff(s)
    cos_t, sin_t = sp.cos(theta), sp.sin(theta)
    nu = xs * cos_t + ys * sin_t          # stretch
    eta = -xs * sin_t + ys * cos_t        # shear strain
    Nx = ea * (nu - 1) * cos_t - ga * eta * sin_t   # spatial internal force, x
    Ny = ea * (nu - 1) * sin_t + ga * eta * cos_t   # spatial internal force, y

    # Expand ∂Nx/∂s and ∂Ny/∂s into second spatial derivatives of x, y, θ via the
    # chain rule, programmatically (no hand algebra). Nx, Ny are functions of
    # (x_s, y_s, θ); their s-derivative is
    #   ∂N/∂s = ∂N/∂x_s·x_ss + ∂N/∂y_s·y_ss + ∂N/∂θ·θ_s.
    # We compute the partials on dummy symbols, then substitute the DSL nodes.
    _xs, _ys, _th = sp.symbols("_xs _ys _th")
    _nu = _xs * sp.cos(_th) + _ys * sp.sin(_th)
    _eta = -_xs * sp.sin(_th) + _ys * sp.cos(_th)
    _Nx = ea * (_nu - 1) * sp.cos(_th) - ga * _eta * sp.sin(_th)
    _Ny = ea * (_nu - 1) * sp.sin(_th) + ga * _eta * sp.cos(_th)
    subs = {_xs: xs, _ys: ys, _th: theta}
    xss = x.diff(s, 2); yss = y.diff(s, 2); ths = theta.diff(s)

    def dds(expr):
        return (expr.diff(_xs).subs(subs) * xss
                + expr.diff(_ys).subs(subs) * yss
                + expr.diff(_th).subs(subs) * ths)

    dNx_ds = dds(_Nx)
    dNy_ds = dds(_Ny)

    # Linear-momentum residuals — EA, GA enter here against data-anchored accel.
    r_momx = x.diff(t, 2) - dNx_ds + c * x.diff(t)
    r_momy = y.diff(t, 2) - dNy_ds + g + c * y.diff(t)
    # Angular-momentum residual (isolates EI).
    r_ang = j * theta.diff(t, 2) - ei * theta.diff(s, 2) - (xs * Ny - ys * Nx)

    return System(
        state=[x, y, theta],
        equations=[r_momx, r_momy, r_ang],
        sensors=[
            Sensor("x_meas", observes=x, noise_std=1e-3),
            Sensor("y_meas", observes=y, noise_std=1e-3),
            Sensor("theta_meas", observes=theta, noise_std=5e-3),
            # Clamped-root BCs (noise-free pseudo-sensors).
            Sensor("x_bc", observes=x, noise_std=0.0),
            Sensor("y_bc", observes=y, noise_std=0.0),
            Sensor("theta_bc", observes=theta, noise_std=0.0),
            # Initial straight-horizontal shape at t=0 (noise-free).
            Sensor("x_ic", observes=x, noise_std=0.0),
            Sensor("y_ic", observes=y, noise_std=0.0),
            Sensor("theta_ic", observes=theta, noise_std=0.0),
        ],
    )


@register_template("dynamic_cosserat")
class DynamicCosserat:
    """Dynamic planar Cosserat rod (inertial, shear + extension); recover
    ``EI_unit, GA_unit, EA_unit`` from the time-resolved motion ``x(s,t),
    y(s,t), θ(s,t)`` of a soft rod swinging under a distributed gravity load
    with light damping."""

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
        # 2-D space-time, 3 outputs. The motion is oscillatory in time, so
        # Fourier features help the network represent it (as in cosserat_rod).
        # Heavy data weighting (positions precise) keeps the fields anchored;
        # lam_physics raised so the stiffnesses (which live only in the physics
        # loss) get a usable gradient. Best-known settings — EI converges; the
        # full 3-unknown problem is training-limited on GA/EA (see
        # docs/dynamic_cosserat_experiments.md).
        return TrainConfig(
            depth=5,
            width=96,
            activation="tanh",
            lr=1e-3,
            adam_epochs=12000,
            lbfgs_iters=0,
            balancer="none",
            t_range=(0.0, T_END),
            spatial_ranges={"s": (0.0, 1.0)},
            n_collocation=2000,
            batch_size=2000,
            lam_data_init=200.0,
            lam_physics_init=10.0,
            param_lr_scale=80.0,
            fourier_features=32,
            fourier_sigma=3.0,
        )

    @staticmethod
    def synthetic_data(seed: int = 0):
        from pinn_engine.data.synthetic import generate_dynamic_cosserat
        return generate_dynamic_cosserat(seed=seed)

    @staticmethod
    def objective(result) -> float:
        truth = DynamicCosserat.truth
        errs = [
            abs(result.final_params[name] - val) / max(abs(val), 1e-6)
            for name, val in truth.items()
        ]
        return float(sum(errs) / len(errs))
