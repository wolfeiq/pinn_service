"""3-DOF planar coupled-drag inverse: discover three linear damping
coefficients in a rigid body moving in the plane with translational +
rotational coupling.

State: ``(u, v, r)`` — two body-frame translational velocities and a yaw
rate. Constant body-frame forces ``(τ_x, τ_y, τ_n)`` drive the system;
observations are noisy ``(u, v, r)`` over time.

Equations of motion (simplified to constant inertias, no added-mass
coupling — pure rigid-body planar dynamics + per-axis linear drag):

    m11 · u̇  −  m22 · v · r  −  c_x · u  =  τ_x
    m22 · v̇  +  m11 · u · r  −  c_y · v  =  τ_y
    m33 · ṙ  +  (m22 − m11) · u · v  −  c_n · r  =  τ_n

The cross terms ``−m22·v·r``, ``+m11·u·r``, ``(m22−m11)·u·v`` are
Coriolis-type couplings that prevent the three channels from being
treated as independent 1-DOF problems — that's what makes this a
genuine *coupled* multi-unknown ODE inverse problem.

Convention: drag coefficients are negative (drag opposes motion).
"""
from __future__ import annotations

from pinn_engine.dsl.symbols import Variable, Parameter, Unknown, Sensor
from pinn_engine.dsl.system import System
from pinn_engine.dsl.templates import register_template
from pinn_engine.core.trainer import TrainConfig


# Reference inertias and control inputs for the synthetic problem.
M11 = 45.0     # effective inertia, x-axis [kg]
M22 = 60.0     # effective inertia, y-axis [kg]
M33 = 8.0      # effective rotational inertia, z-axis [kg m²]
TAU_X = 10.0   # x-axis force [N]
TAU_Y = 3.0    # y-axis force [N]
TAU_N = 1.0    # z-axis moment [N m]

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
    # Negative-convention linear drag coefficients per axis. Bounds span a
    # wide plausible range.
    c_x = Unknown("c_x", bounds=(-25.0, 0.0))
    c_y = Unknown("c_y", bounds=(-75.0, 0.0))
    c_n = Unknown("c_n", bounds=(-15.0, 0.0))
    return System(
        state=[u, v, r],
        equations=[
            m11 * u.d - m22 * v * r - c_x * u - tx,
            m22 * v.d + m11 * u * r - c_y * v - ty,
            m33 * r.d + (m22 - m11) * u * v - c_n * r - tn,
        ],
        sensors=[
            Sensor("u_meas", observes=u, noise_std=0.02),
            Sensor("v_meas", observes=v, noise_std=0.02),
            Sensor("r_meas", observes=r, noise_std=0.01),
        ],
    )


@register_template("coupled_drag_3d")
class CoupledDrag3D:
    """3-DOF planar coupled-drag inverse: recover ``c_x``, ``c_y``, ``c_n``
    from noisy ``(u, v, r)`` sensors."""

    truth = {"c_x": -10.0, "c_y": -30.0, "c_n": -5.0}
    unknown_bounds = {
        "c_x": (-25.0, 0.0),
        "c_y": (-75.0, 0.0),
        "c_n": (-15.0, 0.0),
    }

    @staticmethod
    def system() -> System:
        return build_system()

    @staticmethod
    def default_config() -> TrainConfig:
        # Conservative starting point. param_lr_scale=1.0 because the drag
        # coefficients are O(10) — strong gradients already, no
        # Adam-normalization throttling.
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
        from pinn_engine.data.synthetic import generate_coupled_drag_3d
        return generate_coupled_drag_3d(seed=seed)

    @staticmethod
    def objective(result) -> float:
        truth = CoupledDrag3D.truth
        errs = [
            abs(result.final_params[name] - val) / max(abs(val), 1e-6)
            for name, val in truth.items()
        ]
        return float(sum(errs) / len(errs))
