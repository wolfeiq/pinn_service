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
