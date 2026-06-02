"""Static axial elastic bar inverse: discover axial stiffness ``EA`` from
noisy displacement measurements.

A clamped-free bar of length ``L`` carries a uniform distributed axial
load ``p_0`` per unit length; axial displacement ``u(x)`` is observed
along the bar. The 2nd-order static equation

    EA · u''(x)  =  −p_0

with boundary conditions ``u(0) = 0`` (clamped) and ``EA · u'(L) = 0``
(traction-free) admits the closed form ``u(x) = p_0 / (2·EA) · x(2L − x)``.

Like ``euler_bernoulli_beam``, this template is non-dimensionalised so
both the dimensionless displacement and the unknown ``EA_unit`` sit at
O(1). It's the simplest mechanical-elasticity inverse problem in the
engine — a useful canonical reference for any user testing a structural
template.
"""
from __future__ import annotations

from pinn_engine.dsl.symbols import Variable, Parameter, Unknown, Sensor
from pinn_engine.dsl.system import System
from pinn_engine.dsl.templates import register_template
from pinn_engine.core.trainer import TrainConfig


L_DEFAULT = 1.0       # bar length [m]
P0_DEFAULT = 100.0    # uniform distributed axial load [N/m]
EA_REF = 1000.0       # reference axial stiffness [N] (EA_unit = 1 at truth)


def build_system() -> System:
    x = Variable("x")
    u = Variable("u", depends_on=(x,))
    # Dimensionless axial stiffness. Truth EA_unit = 1.0. Bounds (0.1, 10)
    # span one decade either side; midpoint init 5.05 is well off centre.
    EA_unit = Unknown("EA_unit", bounds=(0.1, 10.0))
    # Residual (dimensionless): EA_unit · û''(x) + 2 = 0
    return System(
        state=[u],
        equations=[EA_unit * u.diff(x, 2) + 2.0],
        sensors=[
            Sensor("u_meas", observes=u, noise_std=1e-3),
            Sensor("u_bc",   observes=u, noise_std=0.0),
        ],
    )


@register_template("axial_elastic_bar")
class AxialElasticBar:
    """Static clamped-free elastic bar; recover ``EA_unit``. Physical
    ``EA = EA_unit × EA_ref`` with ``EA_ref = 1000 N``."""

    truth = {"EA_unit": 1.0}
    unknown_bounds = {"EA_unit": (0.1, 10.0)}

    @staticmethod
    def system() -> System:
        return build_system()

    @staticmethod
    def default_config() -> TrainConfig:
        # 2nd-order with a smooth quadratic û — basic tanh MLP, no Fourier
        # features needed. lr_scale moderate since the gradient on EA_unit
        # is well-conditioned through the dimensionless formulation.
        return TrainConfig(
            depth=4,
            width=32,
            activation="tanh",
            lr=1e-3,
            adam_epochs=1500,
            lbfgs_iters=0,
            balancer="none",
            t_range=(0.0, L_DEFAULT),
            n_collocation=500,
            batch_size=256,
            lam_data_init=100.0,
            lam_physics_init=1.0,
            param_lr_scale=10.0,
            fourier_features=0,
        )

    @staticmethod
    def synthetic_data(seed: int = 0):
        from pinn_engine.data.synthetic import generate_axial_elastic_bar
        return generate_axial_elastic_bar(seed=seed)

    @staticmethod
    def objective(result) -> float:
        truth = AxialElasticBar.truth
        errs = [
            abs(result.final_params[name] - val) / max(abs(val), 1e-6)
            for name, val in truth.items()
        ]
        return float(sum(errs) / len(errs))
