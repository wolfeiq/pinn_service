"""Static Euler-Bernoulli beam inverse: discover flexural rigidity ``EI``
from noisy deflection measurements.

Canonical structural-engineering inverse problem. A simply-supported beam
of length ``L`` carries a uniform distributed load ``q_0``; deflection
``w(x)`` is observed at sensors along the beam. The 4th-order static
equation

    EI · w''''(x)  =  q_0

with boundary conditions ``w(0) = w(L) = 0`` admits a closed-form
solution.

**Non-dimensionalisation.** Real-world beam deflections are tiny (~mm)
and ``EI`` is large (~kN·m²); raw quantities make the loss landscape
ill-conditioned. We non-dimensionalise:

    ŵ      = w / W_ref,           W_ref = q_0 / (24 · EI_ref)
    EI_unit = EI / EI_ref

The compiled residual is then ``EI_unit · ŵ''''(x) − 24 = 0``; both
``ŵ`` and ``EI_unit`` are O(1), and the unknown sits in ``(0.1, 10)``
spanning an order of magnitude on either side of truth.

(This mirrors the Cosserat-rod template's wave-equation
non-dimensionalisation — same motivation: stop the optimiser fighting
with floating-point scale.)
"""
from __future__ import annotations

from pinn_engine.dsl.symbols import Variable, Parameter, Unknown, Sensor
from pinn_engine.dsl.system import System
from pinn_engine.dsl.templates import register_template
from pinn_engine.core.trainer import TrainConfig


# Reference physical scales — real-world ranges.
L_DEFAULT = 1.0       # beam length [m]
Q0_DEFAULT = 100.0    # uniform distributed load [N/m]
EI_REF = 1000.0       # reference flexural rigidity [N·m²] — truth multiplier sits at 1


def build_system() -> System:
    # The engine treats the LAST declared dependency of a state variable as
    # its temporal axis (see System.input_variable). For a static problem
    # we have a single spatial variable; the engine's internal naming will
    # treat ``x`` as time but the user-facing physics is purely spatial.
    x = Variable("x")
    w = Variable("w", depends_on=(x,))
    # Dimensionless multiplier on EI_ref. Truth EI_unit = 1.0. Midpoint init
    # 5.05 — well off centre.
    EI_unit = Unknown("EI_unit", bounds=(0.1, 10.0))
    # Residual (dimensionless): EI_unit · ŵ''''(x) − 24 = 0
    return System(
        state=[w],
        equations=[EI_unit * w.diff(x, 4) - 24.0],
        sensors=[
            # Noisy interior-deflection measurements (already dimensionless;
            # the data generator divides by W_ref before returning).
            Sensor("w_meas", observes=w, noise_std=1e-3),
            # Boundary deflection at x=0 and x=L (simply supported) as a
            # noise-free pseudo-sensor.
            Sensor("w_bc", observes=w, noise_std=0.0),
        ],
    )


@register_template("euler_bernoulli_beam")
class EulerBernoulliBeam:
    """Static simply-supported beam under uniform load; recover dimensionless
    flexural rigidity ``EI_unit``. Physical ``EI = EI_unit × EI_ref`` with
    ``EI_ref = 1000 N·m²``."""

    truth = {"EI_unit": 1.0}
    unknown_bounds = {"EI_unit": (0.1, 10.0)}

    @staticmethod
    def system() -> System:
        return build_system()

    @staticmethod
    def default_config() -> TrainConfig:
        # The deflection ŵ is a smooth degree-4 polynomial — *no Fourier
        # features needed*. Smooth tanh MLP is the right inductive bias for
        # a polynomial. The 4th derivative is a constant (24/EI_unit), so
        # the network has to learn the right curvature profile, not
        # high-frequency content.
        #
        # The "temporal" range here is the spatial range [0, L] (the engine
        # treats the single-input variable as t internally).
        return TrainConfig(
            depth=5,
            width=64,
            activation="tanh",
            lr=1e-3,
            adam_epochs=3000,
            lbfgs_iters=0,
            balancer="none",
            t_range=(0.0, L_DEFAULT),
            n_collocation=1000,
            batch_size=512,
            lam_data_init=100.0,
            lam_physics_init=1.0,
            param_lr_scale=50.0,
            fourier_features=0,
        )

    @staticmethod
    def synthetic_data(seed: int = 0):
        from pinn_engine.data.synthetic import generate_euler_bernoulli_beam
        return generate_euler_bernoulli_beam(seed=seed)

    @staticmethod
    def objective(result) -> float:
        truth = EulerBernoulliBeam.truth
        errs = [
            abs(result.final_params[name] - val) / max(abs(val), 1e-6)
            for name, val in truth.items()
        ]
        return float(sum(errs) / len(errs))
