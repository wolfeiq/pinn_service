"""Synthetic data generators for the bundled templates.

Each generator returns ``dict[str, (t_array, observation_array)]`` keyed by
sensor name — the format :func:`pinn_engine.core.trainer.train` expects.

We use :func:`scipy.integrate.solve_ivp` for ground-truth dynamics so the
synthetic data is exactly what the underlying ODE produces, plus i.i.d.
Gaussian noise per sensor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.integrate import solve_ivp


# -------------------------------------------------------------- damped oscillator


def generate_damped_oscillator(
    c: float = 0.5,
    k: float = 10.0,
    m: float = 1.0,
    x0: float = 1.0,
    v0: float = 0.0,
    t_end: float = 5.0,
    n_samples: int = 1000,
    noise_std: float = 0.01,
    seed: int = 0,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, float]]:
    """Forward-simulate ``m·ẍ + c·ẋ + k·x = 0`` with i.i.d. Gaussian noise on x.

    Returns ``(data, truth)`` where ``truth = {'c': c, 'k': k}``.
    """
    rng = np.random.default_rng(seed)

    def rhs(t, y):
        x, v = y
        return [v, -(c / m) * v - (k / m) * x]

    t = np.linspace(0.0, t_end, n_samples)
    sol = solve_ivp(rhs, (0.0, t_end), [x0, v0], t_eval=t, rtol=1e-8, atol=1e-10)
    x_clean = sol.y[0]
    x_noisy = x_clean + rng.normal(0.0, noise_std, size=x_clean.shape)
    return {"x_meas": (t.astype(np.float32), x_noisy.astype(np.float32))}, {"c": c, "k": k}


# -------------------------------------------------------------- lorenz


def generate_lorenz(
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    x0: float = 1.0,
    y0: float = 1.0,
    z0: float = 1.0,
    t_end: float = 3.0,
    n_samples: int = 3000,
    noise_std: float = 0.02,
    seed: int = 0,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, float]]:
    """Forward-simulate the Lorenz system and add i.i.d. noise to each component."""
    rng = np.random.default_rng(seed)

    def rhs(t, y):
        x, yy, z = y
        return [sigma * (yy - x), x * (rho - z) - yy, x * yy - beta * z]

    t = np.linspace(0.0, t_end, n_samples)
    sol = solve_ivp(rhs, (0.0, t_end), [x0, y0, z0], t_eval=t, rtol=1e-8, atol=1e-10)
    x_clean, y_clean, z_clean = sol.y
    return {
        "x_meas": (t.astype(np.float32), (x_clean + rng.normal(0, noise_std, x_clean.shape)).astype(np.float32)),
        "y_meas": (t.astype(np.float32), (y_clean + rng.normal(0, noise_std, y_clean.shape)).astype(np.float32)),
        "z_meas": (t.astype(np.float32), (z_clean + rng.normal(0, noise_std, z_clean.shape)).astype(np.float32)),
    }, {"sigma": sigma, "rho": rho, "beta": beta}


# -------------------------------------------------------------- cosserat rod (1D wave eq)


def generate_cosserat_rod(
    E: float = 1.0e6,
    rho: float = 1000.0,
    L: float = 1.0,
    t_end: float = 0.01,
    n_s: int = 41,
    n_t: int = 201,
    n_sensors: int = 10,
    noise_std: float = 1e-2,
    ic_amplitude: float = 1.0,
    seed: int = 0,
):
    """Simulate 1-D longitudinal vibration of a Cosserat rod.

    PDE: ``ρ · u_tt = E · u_ss`` on ``s ∈ [0, L]``, ``t ∈ [0, T]``.
    BCs: ``u(0, t) = 0`` (fixed end), ``u_s(L, t) = 0`` (free end).
    IC: ``u(s, 0) = u₀(s)`` (initial Gaussian deflection), ``u_t(s, 0) = 0``.

    Solved with an explicit finite-difference scheme (central in space,
    leapfrog in time) — accurate enough for synthetic ground truth.

    The output sensor data places ``n_sensors`` virtual gauges uniformly
    along the rod, each reporting ``u(s_i, t_k)`` for every time step.
    Sensor data is flattened: ``(t, s)`` pairs in column-major order.
    Returns ``(data, truth)`` with ``data['u_meas'] = ((N, 2), (N,))``.
    """
    rng = np.random.default_rng(seed)

    c_wave = (E / rho) ** 0.5
    ds = L / (n_s - 1)
    dt_max = ds / c_wave * 0.9   # CFL safety factor
    n_t_safe = max(n_t, int(np.ceil(t_end / dt_max)) + 1)
    dt = t_end / (n_t_safe - 1)
    t_grid = np.linspace(0.0, t_end, n_t_safe)
    s_grid = np.linspace(0.0, L, n_s)

    # Initial Gaussian centred at L/2. Amplitude is O(1) so the optimizer
    # has a well-conditioned loss landscape (see template docstring on
    # non-dimensionalisation).
    s_mid = 0.5 * L
    width = 0.1 * L
    u0 = ic_amplitude * np.exp(-((s_grid - s_mid) ** 2) / (2 * width ** 2))
    # Zero at boundaries (BCs).
    u0[0] = 0.0

    u_now = u0.copy()
    u_prev = u0.copy()   # zero initial velocity → u_prev = u0
    u_grid = np.zeros((n_t_safe, n_s), dtype=np.float64)
    u_grid[0] = u_now

    coeff = (c_wave * dt / ds) ** 2
    for k in range(1, n_t_safe):
        u_new = np.zeros_like(u_now)
        u_new[1:-1] = (
            2 * u_now[1:-1] - u_prev[1:-1]
            + coeff * (u_now[2:] - 2 * u_now[1:-1] + u_now[:-2])
        )
        # BCs
        u_new[0] = 0.0
        u_new[-1] = u_new[-2]  # u_s(L, t) = 0 (Neumann via ghost-point mirror)
        u_prev = u_now
        u_now = u_new
        u_grid[k] = u_now

    # Down-sample interior sensors uniformly.
    sensor_idx = np.linspace(1, n_s - 1, n_sensors, dtype=int)
    sensor_s = s_grid[sensor_idx]
    T, S = np.meshgrid(t_grid, sensor_s, indexing="ij")
    u_obs = u_grid[:, sensor_idx]
    u_obs_noisy = u_obs + rng.normal(0.0, noise_std, size=u_obs.shape)
    meas_input = np.stack([S.flatten(), T.flatten()], axis=1).astype(np.float32)
    meas_target = u_obs_noisy.flatten().astype(np.float32)

    # Boundary condition: u(0, t) = 0 for every t in the grid (noise-free).
    bc_input = np.stack(
        [np.zeros_like(t_grid), t_grid], axis=1
    ).astype(np.float32)
    bc_target = np.zeros(t_grid.shape, dtype=np.float32)

    # Initial condition: u(s, 0) = u0(s) for every s in the grid.
    ic_input = np.stack(
        [s_grid, np.zeros_like(s_grid)], axis=1
    ).astype(np.float32)
    ic_target = u0.astype(np.float32)

    return (
        {
            "u_meas": (meas_input, meas_target),   # interior sensors w/ noise
            "u_bc":   (bc_input, bc_target),       # Dirichlet at s=0
            "u_ic":   (ic_input, ic_target),       # initial condition at t=0
        },
        {"E": E},
    )


# -------------------------------------------------------------- pendulum with friction


def generate_pendulum(
    c: float = 0.3,
    I: float = 1.0,
    mgL: float = 10.0,
    theta0: float = 1.0,
    omega0: float = 0.0,
    t_end: float = 10.0,
    n_samples: int = 1000,
    noise_std: float = 0.01,
    seed: int = 0,
):
    """Forward-simulate ``I·θ̈ + c·θ̇ + m·g·L·sin(θ) = 0`` with i.i.d. noise on θ."""
    rng = np.random.default_rng(seed)

    def rhs(t, y):
        theta, omega = y
        return [omega, -(c / I) * omega - (mgL / I) * np.sin(theta)]

    t = np.linspace(0.0, t_end, n_samples)
    sol = solve_ivp(rhs, (0.0, t_end), [theta0, omega0], t_eval=t, rtol=1e-9, atol=1e-11)
    theta_clean = sol.y[0]
    theta_noisy = theta_clean + rng.normal(0.0, noise_std, size=theta_clean.shape)
    return (
        {"theta_meas": (t.astype(np.float32), theta_noisy.astype(np.float32))},
        {"c": c},
    )


# -------------------------------------------------------------- fossen surge (AUV 1-DOF)


def generate_fossen_surge(
    X_u: float = -10.0,
    X_uu: float = -30.0,
    m_eff: float = 45.0,
    tau_u: float = 10.0,
    t_end: float = 10.0,
    n_samples: int = 1000,
    noise_std: float = 0.02,
    seed: int = 0,
):
    """Forward-simulate Fossen 1-DOF surge dynamics.

    ODE: ``m_eff · u̇ − X_u · u − X_uu · u² − τ_u = 0`` with ``u(0) = 0``.
    With Fossen-convention negative drag coefficients, the vehicle
    accelerates from rest and asymptotes to a steady-state forward
    velocity ``u_ss`` set by ``τ_u + X_u · u + X_uu · u² = 0``.

    Returns ``(data, truth)`` where data has one sensor ``u_meas``.
    """
    rng = np.random.default_rng(seed)

    def rhs(t, y):
        (u,) = y
        return [(tau_u + X_u * u + X_uu * u * u) / m_eff]

    t = np.linspace(0.0, t_end, n_samples)
    sol = solve_ivp(rhs, (0.0, t_end), [0.0], t_eval=t, rtol=1e-9, atol=1e-11)
    u_clean = sol.y[0]
    u_noisy = u_clean + rng.normal(0.0, noise_std, size=u_clean.shape)
    return (
        {"u_meas": (t.astype(np.float32), u_noisy.astype(np.float32))},
        {"X_u": X_u, "X_uu": X_uu},
    )


# -------------------------------------------------------------- fossen 3-DOF (planar surge-sway-yaw)


def generate_fossen_3dof(
    X_u: float = -10.0,
    Y_v: float = -30.0,
    N_r: float = -5.0,
    m11: float = 45.0,
    m22: float = 60.0,
    m33: float = 8.0,
    tau_x: float = 10.0,
    tau_y: float = 3.0,
    tau_n: float = 1.0,
    t_end: float = 10.0,
    n_samples: int = 1000,
    noise_std_uv: float = 0.02,
    noise_std_r: float = 0.01,
    seed: int = 0,
):
    """Forward-simulate Fossen 3-DOF planar dynamics (surge, sway, yaw).

    ODE system (Fossen 2021 §6.5, simplified — no added-mass coupling):

        m11·u̇ − m22·v·r − X_u·u = τ_x
        m22·v̇ + m11·u·r − Y_v·v = τ_y
        m33·ṙ + (m22−m11)·u·v − N_r·r = τ_n

    Initial conditions ``(u, v, r) = (0, 0, 0)`` (vehicle at rest). With
    constant body-frame thrust + side force + yaw moment, the vehicle
    accelerates and the three channels reach a steady state coupled through
    the Coriolis terms.

    Returns ``(data, truth)`` with three sensors: ``u_meas``, ``v_meas``,
    ``r_meas``.
    """
    rng = np.random.default_rng(seed)

    def rhs(t, y):
        u, v, r = y
        du = (tau_x + m22 * v * r + X_u * u) / m11
        dv = (tau_y - m11 * u * r + Y_v * v) / m22
        dr = (tau_n - (m22 - m11) * u * v + N_r * r) / m33
        return [du, dv, dr]

    t = np.linspace(0.0, t_end, n_samples)
    sol = solve_ivp(rhs, (0.0, t_end), [0.0, 0.0, 0.0], t_eval=t, rtol=1e-9, atol=1e-11)
    u_c, v_c, r_c = sol.y[0], sol.y[1], sol.y[2]
    u_obs = u_c + rng.normal(0.0, noise_std_uv, size=u_c.shape)
    v_obs = v_c + rng.normal(0.0, noise_std_uv, size=v_c.shape)
    r_obs = r_c + rng.normal(0.0, noise_std_r, size=r_c.shape)
    return (
        {
            "u_meas": (t.astype(np.float32), u_obs.astype(np.float32)),
            "v_meas": (t.astype(np.float32), v_obs.astype(np.float32)),
            "r_meas": (t.astype(np.float32), r_obs.astype(np.float32)),
        },
        {"X_u": X_u, "Y_v": Y_v, "N_r": N_r},
    )


# -------------------------------------------------------------- 1d diffusion (placeholder)


def generate_diffusion_1d(
    D: float = 0.1,
    t_end: float = 1.0,
    n_t: int = 200,
    n_x: int = 51,
    noise_std: float = 0.01,
    seed: int = 0,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, float]]:
    """Closed-form Gaussian solution to ``∂u/∂t = D·∂²u/∂x²`` with a Gaussian IC.

    Phase-1 placeholder — true 2-D collocation isn't wired through the
    time-only Problem adapter yet; the diffusion template will require
    extending build_problem to SpatialProblem in Phase 3. We still emit
    synthetic data so the test suite can use it.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(1e-3, t_end, n_t)
    x = np.linspace(-1.0, 1.0, n_x)
    T, X = np.meshgrid(t, x, indexing="ij")
    sigma2 = 0.05 + 2 * D * T
    u_clean = np.exp(-X ** 2 / (2 * sigma2)) / np.sqrt(2 * np.pi * sigma2)
    u_noisy = u_clean + rng.normal(0, noise_std, u_clean.shape)
    return {
        "u_meas": (
            np.stack([T.flatten(), X.flatten()], axis=1).astype(np.float32),
            u_noisy.flatten().astype(np.float32),
        )
    }, {"D": D}
