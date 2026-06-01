"""Fossen 3-DOF planar inverse: surge-sway-yaw drag discovery.

The next step up from `fossen_surge` (1-DOF): a marine vessel moving in the
horizontal plane under constant body-frame thrust + side force + yaw moment,
observed via DVL (u, v) + gyro (r). The inverse problem recovers the three
*linear* drag coefficients X_u, Y_v, N_r — six interacting state-control
channels but with the cleanest possible drag model (no quadratic terms in v1).

Equations (Fossen 2021, "Handbook of Marine Craft Hydrodynamics and Motion
Control", §6.5; simplified to constant inertia and no added-mass coupling):

    m11 · u̇  −  m22 · v · r  −  X_u · u  =  τ_x
    m22 · v̇  +  m11 · u · r  −  Y_v · v  =  τ_y
    m33 · ṙ  +  (m22 − m11) · u · v  −  N_r · r  =  τ_n

The Coriolis terms ``−m22·v·r``, ``+m11·u·r``, ``(m22−m11)·u·v`` couple the
three channels — that's what makes this a *coupled* multi-unknown ODE inverse
and not three independent 1-DOF problems. Sign convention is Fossen's:
drag coefficients are NEGATIVE (e.g., X_u ≈ −10).

This is the template the L2 prior + iterative refinement features were built
to stress-test — six interacting state-control channels with potential
partial-identifiability between (X_u, Y_v) when surge and sway are both
nontrivial.
"""
from __future__ import annotations

from pinn_engine.dsl.symbols import Variable, Parameter, Unknown, Sensor
from pinn_engine.dsl.system import System
from pinn_engine.dsl.templates import register_template
from pinn_engine.core.trainer import TrainConfig


# Snapir-class vessel parameters (matches fossen_surge defaults where they overlap).
M11 = 45.0     # surge effective mass (m + |X_udot|) [kg]
M22 = 60.0     # sway effective mass (m + |Y_vdot|) [kg]
M33 = 8.0      # yaw effective inertia (Iz + |N_rdot|) [kg m²]
TAU_X = 10.0   # surge thrust [N]
TAU_Y = 3.0    # sway side-force [N]   (e.g., from a lateral thruster)
TAU_N = 1.0    # yaw moment [N m]

T_END = 10.0


def build_system() -> System:
    t = Variable("t")
    u = Variable("u", depends_on=t)
    v = Variable("v", depends_on=t)
    r = Variable("r", depends_on=t)
    m11 = Parameter("m11", value=M11)
    m22 = Parameter("m22", value=M22)
    m33 = Parameter("m33", value=M33)
    tx = Parameter("tau_x", value=TAU_X)
    ty = Parameter("tau_y", value=TAU_Y)
    tn = Parameter("tau_n", value=TAU_N)
    # Fossen-convention linear drag coefficients (negative). Bounds span a wide
    # but plausible literature range for AUV-class vessels.
    X_u = Unknown("X_u", bounds=(-25.0, 0.0))
    Y_v = Unknown("Y_v", bounds=(-75.0, 0.0))
    N_r = Unknown("N_r", bounds=(-15.0, 0.0))
    return System(
        state=[u, v, r],
        equations=[
            m11 * u.d - m22 * v * r - X_u * u - tx,    # surge
            m22 * v.d + m11 * u * r - Y_v * v - ty,    # sway
            m33 * r.d + (m22 - m11) * u * v - N_r * r - tn,  # yaw
        ],
        sensors=[
            Sensor("u_meas", observes=u, noise_std=0.02),   # DVL surge
            Sensor("v_meas", observes=v, noise_std=0.02),   # DVL sway
            Sensor("r_meas", observes=r, noise_std=0.01),   # gyro yaw rate
        ],
    )


@register_template("fossen_3dof")
class Fossen3DOF:
    """3-DOF planar AUV inverse: recover X_u, Y_v, N_r from (u, v, r) sensors."""

    truth = {"X_u": -10.0, "Y_v": -30.0, "N_r": -5.0}
    unknown_bounds = {
        "X_u": (-25.0, 0.0),
        "Y_v": (-75.0, 0.0),
        "N_r": (-15.0, 0.0),
    }

    @staticmethod
    def system() -> System:
        return build_system()

    @staticmethod
    def default_config() -> TrainConfig:
        # Conservative starting point; mirrors fossen_surge's hand-tuned choices.
        # param_lr_scale=1.0 because the drag coefficients are O(10) — strong
        # gradients already, no Adam-normalization throttling.
        return TrainConfig(
            depth=5,
            width=64,
            activation="swish",
            lr=1e-3,
            adam_epochs=2000,
            lbfgs_iters=0,
            balancer="none",
            t_range=(0.0, T_END),
            n_collocation=2000,
            batch_size=512,
            lam_data_init=20.0,
            lam_physics_init=1.0,
            param_lr_scale=1.0,
        )

    @staticmethod
    def synthetic_data(seed: int = 0):
        from pinn_engine.data.synthetic import generate_fossen_3dof
        return generate_fossen_3dof(seed=seed)

    @staticmethod
    def objective(result) -> float:
        truth = Fossen3DOF.truth
        errs = [
            abs(result.final_params[name] - val) / max(abs(val), 1e-6)
            for name, val in truth.items()
        ]
        return float(sum(errs) / len(errs))
