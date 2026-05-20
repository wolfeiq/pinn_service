"""Augmented-state Extended Kalman Filter for inverse-parameter discovery.

The PINN engine discovers parameters by fitting a neural network to the
governing equations + sensor data. The classical approach is an
*augmented-state EKF* — treat the unknown parameters as additional
state variables (with zero process noise, since they're constant), and
let the filter estimate them alongside the dynamic state.

This is the apples-to-apples baseline the build plan calls for: same
data, same noise model, same problem statement, different algorithm.

Implementation is the damped-oscillator-specific augmented EKF; we keep
it simple and focused. The pattern generalizes — any template with a
known equation form can produce its own EKF baseline by linearising the
augmented dynamics around the current estimate.

State (augmented): ``[x, ẋ, c, k]``
Dynamics: ``m·ẍ + c·ẋ + k·x = 0``  →  ``ẍ = (-c·ẋ - k·x)/m``
Observation: ``y = x`` (noisy measurement of position)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class EKFResult:
    """What :func:`run_ekf_baseline` returns."""

    final_state: np.ndarray             # [x, ẋ, c, k] at last step
    final_covariance: np.ndarray        # 4×4 covariance at last step
    parameter_history: Dict[str, np.ndarray]
    final_params: Dict[str, float]
    final_param_stds: Dict[str, float]


class EKFInverseDampedOscillator:
    """Augmented-state EKF for ``m·ẍ + c·ẋ + k·x = 0`` with unknown ``c, k``.

    Parameters:
        m: known mass (default 1.0).
        x0_state: initial state guess ``[x, ẋ, c, k]``.
        P0_diag: initial covariance diagonal — large for unknowns.
        Q_diag: process-noise diagonal — small for parameters
            (random-walk model with very low rate).
        R: measurement-noise variance (matches the sensor σ²).
    """

    def __init__(
        self,
        m: float = 1.0,
        x0_state: np.ndarray | None = None,
        P0_diag: np.ndarray | None = None,
        Q_diag: np.ndarray | None = None,
        R: float = 0.0001,
    ):
        self.m = float(m)
        self.x0_state = (
            np.asarray(x0_state, dtype=np.float64).copy()
            if x0_state is not None
            else np.array([0.0, 0.0, 0.5, 10.0], dtype=np.float64)
        )
        self.P0 = np.diag(
            P0_diag if P0_diag is not None
            else np.array([1.0, 1.0, 1.0, 50.0], dtype=np.float64)
        )
        self.Q = np.diag(
            Q_diag if Q_diag is not None
            else np.array([1e-5, 1e-4, 1e-7, 1e-6], dtype=np.float64)
        )
        self.R = float(R)
        self.H = np.array([[1.0, 0.0, 0.0, 0.0]])   # observe x only

    def _jacobian_F(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Linearise the discrete-time augmented dynamics around ``state``."""
        x, xdot, c, k = state
        F = np.eye(4)
        F[0, 1] = dt                         # ∂x_new/∂xdot
        F[1, 0] = -k / self.m * dt           # ∂xdot_new/∂x
        F[1, 1] = 1.0 - c / self.m * dt      # ∂xdot_new/∂xdot
        F[1, 2] = -xdot / self.m * dt        # ∂xdot_new/∂c
        F[1, 3] = -x / self.m * dt           # ∂xdot_new/∂k
        return F

    def _step(self, state: np.ndarray, P: np.ndarray, dt: float, y: float):
        # Predict (Euler integration of the continuous-time dynamics)
        x, xdot, c, k = state
        new_state = np.array(
            [x + xdot * dt, xdot + (-c * xdot - k * x) / self.m * dt, c, k],
            dtype=np.float64,
        )
        F = self._jacobian_F(state, dt)
        new_P = F @ P @ F.T + self.Q

        # Update
        innov = y - (self.H @ new_state).item()
        S = (self.H @ new_P @ self.H.T).item() + self.R
        K = (new_P @ self.H.T / S).reshape(-1)
        new_state = new_state + K * innov
        new_P = (np.eye(4) - np.outer(K, self.H[0])) @ new_P
        return new_state, new_P

    def run(self, t: np.ndarray, y_obs: np.ndarray) -> EKFResult:
        """Run the filter on a noisy ``y_obs`` trajectory sampled at ``t``."""
        state = self.x0_state.copy()
        P = self.P0.copy()
        history = {"c": np.zeros_like(t), "k": np.zeros_like(t)}
        history["c"][0] = state[2]
        history["k"][0] = state[3]

        for i in range(1, len(t)):
            dt = float(t[i] - t[i - 1])
            state, P = self._step(state, P, dt, float(y_obs[i]))
            history["c"][i] = state[2]
            history["k"][i] = state[3]

        return EKFResult(
            final_state=state,
            final_covariance=P,
            parameter_history=history,
            final_params={"c": float(state[2]), "k": float(state[3])},
            final_param_stds={
                "c": float(np.sqrt(max(P[2, 2], 0.0))),
                "k": float(np.sqrt(max(P[3, 3], 0.0))),
            },
        )


def run_ekf_baseline(
    data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    m: float = 1.0,
    sensor_name: str = "x_meas",
) -> EKFResult:
    """Convenience wrapper: pull (t, x) from a sensor dict and run the EKF."""
    if sensor_name not in data:
        raise KeyError(
            f"Sensor {sensor_name!r} not in data; have {list(data.keys())}"
        )
    t_arr, x_obs = data[sensor_name]
    t_arr = np.asarray(t_arr, dtype=np.float64)
    x_obs = np.asarray(x_obs, dtype=np.float64)
    ekf = EKFInverseDampedOscillator(m=m)
    return ekf.run(t_arr, x_obs)
