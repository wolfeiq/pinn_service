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
from scipy.integrate import solve_ivp, solve_bvp


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


# -------------------------------------------------------------- nonlinear drag 1-DOF


def generate_nonlinear_drag_1d(
    c_lin: float = -10.0,
    c_quad: float = -30.0,
    m_eff: float = 45.0,
    tau: float = 10.0,
    t_end: float = 10.0,
    n_samples: int = 1000,
    noise_std: float = 0.02,
    seed: int = 0,
):
    """Forward-simulate 1-DOF nonlinear-drag dynamics.

    ODE: ``m_eff · u̇ − c_lin · u − c_quad · u² − τ = 0`` with ``u(0) = 0``.
    With negative drag coefficients the body accelerates from rest and
    asymptotes to a steady-state velocity ``u_ss`` set by
    ``τ + c_lin · u_ss + c_quad · u_ss² = 0``.

    Returns ``(data, truth)`` where data has one sensor ``u_meas``.
    """
    rng = np.random.default_rng(seed)

    def rhs(t, y):
        (u,) = y
        return [(tau + c_lin * u + c_quad * u * u) / m_eff]

    t = np.linspace(0.0, t_end, n_samples)
    sol = solve_ivp(rhs, (0.0, t_end), [0.0], t_eval=t, rtol=1e-9, atol=1e-11)
    u_clean = sol.y[0]
    u_noisy = u_clean + rng.normal(0.0, noise_std, size=u_clean.shape)
    return (
        {"u_meas": (t.astype(np.float32), u_noisy.astype(np.float32))},
        {"c_lin": c_lin, "c_quad": c_quad},
    )


# -------------------------------------------------------------- coupled drag 3-DOF (planar)


def generate_coupled_drag_3d(
    c_x: float = -10.0,
    c_y: float = -30.0,
    c_n: float = -5.0,
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
    """Forward-simulate 3-DOF planar coupled-drag dynamics.

    ODE system (rigid-body planar motion with per-axis linear drag, no
    added-mass coupling):

        m11·u̇ − m22·v·r − c_x·u = τ_x
        m22·v̇ + m11·u·r − c_y·v = τ_y
        m33·ṙ + (m22−m11)·u·v − c_n·r = τ_n

    Initial conditions ``(u, v, r) = (0, 0, 0)`` (body at rest). With
    constant body-frame forces + moment, the three channels accelerate
    and reach a steady state coupled through the Coriolis-type terms.

    Returns ``(data, truth)`` with three sensors: ``u_meas``, ``v_meas``,
    ``r_meas``.
    """
    rng = np.random.default_rng(seed)

    def rhs(t, y):
        u, v, r = y
        du = (tau_x + m22 * v * r + c_x * u) / m11
        dv = (tau_y - m11 * u * r + c_y * v) / m22
        dr = (tau_n - (m22 - m11) * u * v + c_n * r) / m33
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
        {"c_x": c_x, "c_y": c_y, "c_n": c_n},
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


# -------------------------------------------------------------- Burgers' equation (nonlinear advection-diffusion)


def generate_burgers_1d(
    nu: float = 0.05,
    t_end: float = 1.0,
    n_x: int = 64,
    n_t: int = 50,
    solver_nx: int = 256,
    noise_std: float = 1e-2,
    seed: int = 0,
):
    """Viscous Burgers' equation ``u_t + u·u_x = ν·u_xx`` on ``x∈[-1,1]``,
    ``u(x,0) = −sin(πx)``, ``u(±1,t)=0`` — the canonical nonlinear PDE
    (advection steepens, diffusion smooths) and the classic PINN benchmark.

    Ground truth from a **stable** method-of-lines solver: conservative flux form
    ``u_t = −½(u²)_x + ν·u_xx`` with central differences on a fine grid and a
    stiff BDF integrator. The conservative flux + implicit integrator are what
    keep it from blowing up at the steep internal layer (naive non-conservative
    FD does). Returns a dense noisy ``u(x,t)`` grid; the inverse problem is to
    recover ``ν``.
    """
    rng = np.random.default_rng(seed)
    xf = np.linspace(-1.0, 1.0, solver_nx); dx = xf[1] - xf[0]
    u0 = -np.sin(np.pi * xf)

    def rhs(t, u):
        uu = u.copy(); uu[0] = 0.0; uu[-1] = 0.0
        f = 0.5 * uu * uu
        dfdx = np.zeros_like(uu); dfdx[1:-1] = (f[2:] - f[:-2]) / (2 * dx)
        uxx = np.zeros_like(uu); uxx[1:-1] = (uu[2:] - 2 * uu[1:-1] + uu[:-2]) / dx ** 2
        du = -dfdx + nu * uxx; du[0] = 0.0; du[-1] = 0.0
        return du

    te = np.linspace(0.0, t_end, n_t)
    sol = solve_ivp(rhs, (0.0, t_end), u0, t_eval=te, method="BDF", rtol=1e-8, atol=1e-10)
    if not sol.success:
        raise RuntimeError(f"Burgers solve failed at nu={nu}: {sol.message}")
    # subsample the fine spatial grid to the measurement grid
    idx = np.linspace(0, solver_nx - 1, n_x, dtype=int)
    x = xf[idx]
    Xg, Tg = np.meshgrid(x, te, indexing="ij")  # (n_x, n_t)
    Uxt = sol.y[idx, :]            # (n_x, n_t) — matches (x, t) ordering
    u_noisy = Uxt + rng.normal(0, noise_std, Uxt.shape)
    inp = np.stack([Xg.ravel(), Tg.ravel()], axis=1).astype(np.float32)  # (x, t)
    return {"u_meas": (inp, u_noisy.ravel().astype(np.float32))}, {"nu": float(nu)}


# -------------------------------------------------------------- Euler-Bernoulli beam (4th-order static)


def generate_euler_bernoulli_beam(
    EI_unit: float = 1.0,
    q0: float = 100.0,
    L: float = 1.0,
    EI_ref: float = 1000.0,
    n_sensors: int = 21,
    noise_std: float = 1e-3,
    seed: int = 0,
):
    """Static deflection of a simply-supported Euler-Bernoulli beam under a
    uniform distributed load — returned **non-dimensional**.

    PDE (static, 4th-order in one spatial variable):
        EI · w''''(x)  =  q_0
    Boundary conditions: ``w(0) = w(L) = 0`` (supports), and implicitly
    ``w''(0) = w''(L) = 0`` (moment-free ends).

    Closed-form solution for constant ``q_0`` and homogeneous BCs:

        w(x) = q_0 / (24·EI) · x · (L − x) · (L² + L·x − x²)

    We non-dimensionalise: ``ŵ = w / W_ref`` with ``W_ref = q_0·L⁴ /
    (24·EI_ref)``, and ``EI_unit = EI / EI_ref``. The compiled residual
    used by the template is ``EI_unit · ŵ''''(x) − 24 = 0``, so the
    generator returns ``ŵ`` (not ``w``) directly — both ``ŵ`` and
    ``EI_unit`` are O(1) and the loss landscape is well conditioned.

    The inverse problem: recover ``EI_unit`` from noisy ``ŵ(x_k) + ε_k``.

    Returns ``(data, truth)`` with two sensors:
      * ``w_meas`` — ``n_sensors`` noisy interior dimensionless-deflection
        samples.
      * ``w_bc``  — the two boundary conditions ``ŵ(0) = ŵ(L) = 0`` as
        noise-free pseudo-sensors.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, L, n_sensors)
    # Dimensionless closed-form: ŵ(x) = x · (L−x) · (L² + L·x − x²) / EI_unit
    # (factor of 1 from cancelling q_0·L⁴/(24·EI_ref·W_ref) = 1).
    w_unit_clean = x * (L - x) * (L * L + L * x - x * x) / float(EI_unit)
    w_unit_noisy = w_unit_clean + rng.normal(0.0, noise_std, size=w_unit_clean.shape)
    # BC at x=0 and x=L: ŵ = 0 (simply-supported).
    x_bc = np.array([0.0, L], dtype=np.float64)
    w_bc = np.array([0.0, 0.0], dtype=np.float64)
    return (
        {
            "w_meas": (x.astype(np.float32), w_unit_noisy.astype(np.float32)),
            "w_bc":   (x_bc.astype(np.float32), w_bc.astype(np.float32)),
        },
        {"EI_unit": float(EI_unit)},
    )



# -------------------------------------------------------------- axial elastic bar (2nd-order static)


def generate_axial_elastic_bar(
    EA_unit: float = 1.0,
    p0: float = 100.0,
    L: float = 1.0,
    EA_ref: float = 1000.0,
    n_sensors: int = 21,
    noise_std: float = 1e-3,
    seed: int = 0,
):
    """Static axial displacement of an elastic bar fixed at ``x=0``, free at
    ``x=L``, under a uniform distributed axial load ``p_0`` per unit length.

    ODE (static, 2nd-order in one spatial variable):
        EA · u''(x)  =  −p_0
    Boundary conditions: ``u(0) = 0`` (clamped) and ``EA · u'(L) = 0``
    (traction-free at the free end).

    Closed-form solution:
        u(x) = (p_0 / (2 · EA)) · x · (2L − x)

    We non-dimensionalise as in the Euler-Bernoulli template:
    ``û = u / U_ref`` with ``U_ref = p_0 · L² / (2 · EA_ref)``, and
    ``EA_unit = EA / EA_ref``. The compiled residual used by the
    template is ``EA_unit · û''(x) + 2 = 0`` so the dimensionless
    deflection peaks at ``û(L) = 1 / EA_unit`` — same scale family as
    the beam template.

    Returns ``(data, truth)`` with two sensors:
      * ``u_meas`` — ``n_sensors`` noisy interior displacement samples.
      * ``u_bc``  — the clamped boundary ``û(0) = 0`` as a noise-free
        pseudo-sensor.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, L, n_sensors)
    # Dimensionless closed-form: û(x) = x · (2L − x) / EA_unit
    # (factor of 1 from cancelling p_0·L²/(2·EA_ref·U_ref) = 1).
    u_unit_clean = x * (2.0 * L - x) / float(EA_unit)
    u_unit_noisy = u_unit_clean + rng.normal(0.0, noise_std, size=u_unit_clean.shape)
    # BC at x=0: clamped.
    x_bc = np.array([0.0], dtype=np.float64)
    u_bc = np.array([0.0], dtype=np.float64)
    return (
        {
            "u_meas": (x.astype(np.float32), u_unit_noisy.astype(np.float32)),
            "u_bc":   (x_bc.astype(np.float32), u_bc.astype(np.float32)),
        },
        {"EA_unit": float(EA_unit)},
    )


# -------------------------------------------------------------- planar elastica (large-deflection Cosserat rod)


def _solve_elastica_bvp(alpha: float, n_nodes: int = 201):
    """Solve the dimensionless planar-elastica cantilever BVP for the tangent
    angle ``θ(s̃)`` on ``s̃ ∈ [0, 1]``.

        θ''(s̃)  =  −α · cos(θ(s̃)),   θ(0) = 0,   θ'(1) = 0

    ``α = P·L²/EI`` is the elastica load parameter. Returns a callable
    ``sol.sol`` (continuous interpolant) from :func:`scipy.integrate.solve_bvp`.
    The initial guess uses the small-deflection parabola so the Newton solve
    converges for the whole α-range we use (≤ 3).
    """
    def ode(s, y):
        return np.vstack([y[1], -alpha * np.cos(y[0])])

    def bc(ya, yb):
        # Clamped angle at the root, moment-free (zero curvature) at the tip.
        return np.array([ya[0], yb[1]])

    s = np.linspace(0.0, 1.0, n_nodes)
    y0 = np.zeros((2, s.size))
    y0[0] = alpha * s * (1.0 - 0.5 * s)   # small-deflection seed
    y0[1] = alpha * (1.0 - s)
    sol = solve_bvp(ode, bc, s, y0, max_nodes=50000, tol=1e-9)
    if not sol.success:
        raise RuntimeError(f"elastica BVP failed to converge at alpha={alpha}: {sol.message}")
    return sol


def generate_planar_elastica(
    EI_unit: float = 1.0,
    P0: float = 2.5,
    L: float = 1.0,
    EI_ref: float = 1.0,
    n_sensors: int = 31,
    noise_std: float = 1e-2,
    seed: int = 0,
):
    """Large-deflection planar elastica cantilever — the geometrically-exact
    soft-robot continuum-rod inverse. Returns the tangent-angle profile.

    A slender soft rod is clamped horizontal at ``s=0`` and carries a dead
    tip load ``P0`` (downward) at the free end ``s=L``. Unlike the linear
    ``euler_bernoulli_beam`` template (small-deflection, ``w'''' = q``), this
    is the **geometrically-exact** Kirchhoff/Cosserat planar rod valid at
    arbitrarily large deflection. In terms of the tangent angle ``θ(s)`` the
    static balance of the bending moment ``M = EI·θ'`` against a tip load is

        EI · θ''(s)  =  −P0 · cos(θ(s))

    with ``θ(0) = 0`` (clamped horizontal) and ``θ'(L) = 0`` (moment-free
    tip). The ``cos(θ)`` makes the problem nonlinear — at ``α = P0·L²/EI ≈
    2.5`` the tip rotates ~51° and droops ~0.56·L, a regime where linear beam
    theory is off by tens of percent. This is exactly the operating point of
    a soft-robotic finger / continuum manipulator under its own payload.

    **Non-dimensionalisation.** With ``s̃ = s/L`` and ``EI = EI_unit·EI_ref``,
    the residual the template compiles is ``EI_unit·θ''(s̃) + α_ref·cos(θ) =
    0`` with ``α_ref = P0·L²/EI_ref`` (= 2.5 at the defaults). The unknown
    ``EI_unit`` sits multiplicatively on the highest derivative, O(1), in the
    same well-conditioned family as the beam and bar templates.

    **Sensors.** ``θ(s̃)`` is what flexible curvature sensors (fiber-Bragg
    gratings, IMU arrays, stretch sensors) actually report along a soft rod,
    so the angle formulation is the physical measurement model — no need to
    differentiate a measured shape. ``noise_std`` is in radians.

    Returns ``(data, truth)`` with two sensors:
      * ``theta_meas`` — ``n_sensors`` noisy interior tangent-angle samples.
      * ``theta_bc``  — the clamped root ``θ(0) = 0`` as a noise-free
        pseudo-sensor.
    """
    rng = np.random.default_rng(seed)
    alpha_ref = P0 * L * L / EI_ref
    # The physical load parameter scales inversely with the true stiffness.
    alpha = alpha_ref / float(EI_unit)
    sol = _solve_elastica_bvp(alpha)

    s = np.linspace(0.0, 1.0, n_sensors)
    theta_clean = sol.sol(s)[0]
    theta_noisy = theta_clean + rng.normal(0.0, noise_std, size=theta_clean.shape)

    # Clamped-root BC θ(0) = 0 as a noise-free pseudo-sensor.
    s_bc = np.array([0.0], dtype=np.float64)
    theta_bc = np.array([0.0], dtype=np.float64)
    return (
        {
            "theta_meas": (s.astype(np.float32), theta_noisy.astype(np.float32)),
            "theta_bc":   (s_bc.astype(np.float32), theta_bc.astype(np.float32)),
        },
        {"EI_unit": float(EI_unit)},
    )


# -------------------------------------------------------------- full planar Cosserat rod (shear + extension)


# Dimensionless reference stiffness numbers (= stiffness · L² / EI_ref) and the
# constant tip load. Chosen so a *thick / soft* rod develops sizeable shear
# (~6-15%) and axial (~5-9%) strain — i.e. all three stiffnesses are excited
# and identifiable, not just bending. See docs/cosserat_planar_experiments.md.
COSSERAT_EI0 = 1.0     # bending number (the reference scale itself)
COSSERAT_GA0 = 15.0    # shear number  GA_ref·L²/EI_ref
COSSERAT_EA0 = 15.0    # axial number  EA_ref·L²/EI_ref
COSSERAT_PX = 2.5      # tip force, axial component  (P·L²/EI_ref, dimensionless)
COSSERAT_PY = -4.0     # tip force, transverse component (downward)


def _solve_cosserat_planar_bvp(ei: float, ga: float, ea: float,
                               Px: float, Py: float, n_nodes: int = 201):
    """Solve the geometrically-exact planar Cosserat (Simo-Reissner) rod BVP.

    Tip-loaded cantilever, no distributed load → the internal force is constant
    and equal to the applied tip load ``(Px, Py)``. First-order state
    ``(x, y, θ, M)`` on ``s̃ ∈ [0, 1]``::

        x' = ν cosθ − η sinθ,     ν = 1 + (Px cosθ + Py sinθ)/ea   (axial stretch)
        y' = ν sinθ + η cosθ,     η = (−Px sinθ + Py cosθ)/ga      (shear)
        θ' = M/ei                                                  (curvature)
        M' = −(x' Py − y' Px)                                      (moment balance)

    BCs: clamped root ``x(0)=y(0)=θ(0)=0`` and moment-free tip ``M(1)=0``.
    Returns the ``solve_bvp`` interpolant ``sol.sol``.
    """
    def ode(s, Y):
        x, y, th, M = Y
        nu = 1.0 + (Px * np.cos(th) + Py * np.sin(th)) / ea
        eta = (-Px * np.sin(th) + Py * np.cos(th)) / ga
        xp = nu * np.cos(th) - eta * np.sin(th)
        yp = nu * np.sin(th) + eta * np.cos(th)
        thp = M / ei
        Mp = -(xp * Py - yp * Px)
        return np.vstack([xp, yp, thp, Mp])

    def bc(Ya, Yb):
        return np.array([Ya[0], Ya[1], Ya[2], Yb[3]])

    s = np.linspace(0.0, 1.0, n_nodes)
    Y0 = np.zeros((4, s.size))
    Y0[2] = 0.3 * s   # small linear seed for θ
    sol = solve_bvp(ode, bc, s, Y0, max_nodes=80000, tol=1e-9)
    if not sol.success:
        raise RuntimeError(
            f"Cosserat BVP failed (ei={ei}, ga={ga}, ea={ea}): {sol.message}")
    return sol


def generate_planar_cosserat(
    EI_unit: float = 1.0,
    GA_unit: float = 1.0,
    EA_unit: float = 1.0,
    EI0: float = COSSERAT_EI0,
    GA0: float = COSSERAT_GA0,
    EA0: float = COSSERAT_EA0,
    Px: float = COSSERAT_PX,
    Py: float = COSSERAT_PY,
    n_sensors: int = 41,
    pos_noise_std: float = 5e-4,
    ang_noise_std: float = 5e-3,
    seed: int = 0,
):
    """Full planar Cosserat rod inverse: recover **three** dimensionless
    stiffnesses — bending ``EI_unit``, shear ``GA_unit``, axial ``EA_unit`` —
    from the measured deformed shape ``(x, y)`` and cross-section orientation
    ``θ`` of a tip-loaded soft rod.

    This is the geometrically-exact Simo-Reissner planar rod: unlike
    ``planar_elastica`` (inextensible, unshearable — bending only), the
    cross-section here can **shear** (θ ≠ tangent angle) and the centerline can
    **extend**. At the default thick/soft setup the rod develops ~7-27% shear
    strain and ~17-31% axial strain (plus a ~45° tip rotation), so all three
    stiffnesses leave a strong signature in the shape — large enough that the
    network can't "explain them away" within the sensor-noise latitude.

    The unknowns are dimensionless multipliers (truth = 1.0 each) on reference
    stiffness numbers ``EI0, GA0, EA0`` (= stiffness·L²/EI_ref). The template
    residuals isolate one unknown apiece:

      * axial constitutive  ``EA0·EA_unit·(x'cosθ + y'sinθ − 1) = Px cosθ + Py sinθ``
      * shear constitutive   ``GA0·GA_unit·(−x'sinθ + y'cosθ) = −Px sinθ + Py cosθ``
      * moment balance       ``EI0·EI_unit·θ'' + (Py x' − Px y') = 0``

    Ground truth from :func:`_solve_cosserat_planar_bvp`. Sensors measure
    ``x(s̃), y(s̃), θ(s̃)`` (position markers + IMU/orientation), plus the
    clamped-root BCs as noise-free pseudo-sensors.

    Returns ``(data, truth)`` with sensors ``x_meas, y_meas, theta_meas`` and
    the clamped-root pseudo-sensors ``x_bc, y_bc, theta_bc``.
    """
    rng = np.random.default_rng(seed)
    ei = EI0 * float(EI_unit)
    ga = GA0 * float(GA_unit)
    ea = EA0 * float(EA_unit)
    sol = _solve_cosserat_planar_bvp(ei, ga, ea, Px, Py)

    s = np.linspace(0.0, 1.0, n_sensors)
    x, y, th, _M = sol.sol(s)
    x_n = x + rng.normal(0.0, pos_noise_std, size=x.shape)
    y_n = y + rng.normal(0.0, pos_noise_std, size=y.shape)
    th_n = th + rng.normal(0.0, ang_noise_std, size=th.shape)

    s_bc = np.array([0.0], dtype=np.float64)
    zero = np.array([0.0], dtype=np.float64)
    return (
        {
            "x_meas":     (s.astype(np.float32), x_n.astype(np.float32)),
            "y_meas":     (s.astype(np.float32), y_n.astype(np.float32)),
            "theta_meas": (s.astype(np.float32), th_n.astype(np.float32)),
            "x_bc":       (s_bc.astype(np.float32), zero.astype(np.float32)),
            "y_bc":       (s_bc.astype(np.float32), zero.astype(np.float32)),
            "theta_bc":   (s_bc.astype(np.float32), zero.astype(np.float32)),
        },
        {"EI_unit": float(EI_unit), "GA_unit": float(GA_unit), "EA_unit": float(EA_unit)},
    )


# -------------------------------------------------------------- dynamic planar Cosserat rod (time-domain)


# Dynamic-rod reference numbers, gravity load, damping, rotary inertia. Soft rod
# released from horizontal under a DISTRIBUTED gravity load (no concentrated tip
# force — that would shock-excite fast axial/shear waves at the near-massless end
# node). Light damping keeps the large-deflection swing's strains moderate.
DYNCOS_EI0 = 1.0       # bending number
DYNCOS_GA0 = 15.0      # shear number
DYNCOS_EA0 = 15.0      # axial number
DYNCOS_G = 3.0         # distributed gravity (force per unit length, downward)
DYNCOS_C = 0.4         # translational viscous damping coefficient
DYNCOS_J = 0.01        # dimensionless rotary inertia ρI/(ρA·L²)
DYNCOS_TEND = 2.5      # time horizon (~1.3 bending periods; settling tail trimmed)


def _simulate_dynamic_cosserat(ei, ga, ea, g, c, j, N=60, t_end=DYNCOS_TEND,
                               n_t=81):
    """Method-of-lines forward simulation of the dynamic planar Cosserat rod.

    Clamped at ``s=0``, free at the tip, under a distributed gravity load
    ``(0, −g)`` per unit length; released from the straight horizontal
    configuration at rest. Staggered finite differences in ``s`` (forces/strains
    at element midpoints, divergence at nodes) + adaptive RK45 in time.
    Translational viscous damping ``c``; rotary inertia ``j``.

    Verified (see test): undamped energy conserved to ~1e-7, and the damped
    steady state reproduces the static gravity-loaded rod shape to ~1e-5.
    Gravity (not a tip point load) keeps the max acceleration O(g) — no
    boundary-node spike. Returns ``(s_nodes, t_grid, X, Y, TH)`` with arrays
    shaped ``(n_t, N+1)``.
    """
    ds = 1.0 / N
    nn = N + 1
    s = np.linspace(0.0, 1.0, nn)
    Y0 = np.concatenate([s.copy(), np.zeros(nn), np.zeros(nn), np.zeros(3 * nn)])
    mass = np.full(nn, ds); mass[N] = ds / 2; massj = mass * j

    def deriv(t, Y):
        x = Y[0:nn].copy(); y = Y[nn:2*nn].copy(); th = Y[2*nn:3*nn].copy()
        vx = Y[3*nn:4*nn].copy(); vy = Y[4*nn:5*nn].copy(); om = Y[5*nn:6*nn].copy()
        x[0] = y[0] = th[0] = vx[0] = vy[0] = om[0] = 0.0
        dxs = (x[1:]-x[:-1])/ds; dys = (y[1:]-y[:-1])/ds
        the = (th[:-1]+th[1:])/2; dth = (th[1:]-th[:-1])/ds
        nu = dxs*np.cos(the)+dys*np.sin(the); eta = -dxs*np.sin(the)+dys*np.cos(the)
        n1 = ea*(nu-1); n2 = ga*eta
        Nx = n1*np.cos(the)-n2*np.sin(the); Ny = n1*np.sin(the)+n2*np.cos(the); m = ei*dth
        ax = np.zeros(nn); ay = np.zeros(nn); al = np.zeros(nn)
        for i in range(1, N):
            ax[i] = (Nx[i]-Nx[i-1])/mass[i] - c*vx[i]
            ay[i] = (Ny[i]-Ny[i-1])/mass[i] - g - c*vy[i]
            Tq = (m[i]-m[i-1]) + ds*0.5*((dxs[i]*Ny[i]-dys[i]*Nx[i])
                                         + (dxs[i-1]*Ny[i-1]-dys[i-1]*Nx[i-1]))
            al[i] = Tq/massj[i]
        # Free tip: ghost element force = 0 (no tip load).
        ax[N] = (0.0-Nx[N-1])/mass[N] - c*vx[N]
        ay[N] = (0.0-Ny[N-1])/mass[N] - g - c*vy[N]
        al[N] = ((0.0-m[N-1]) + ds*0.5*(dxs[N-1]*Ny[N-1]-dys[N-1]*Nx[N-1]))/massj[N]
        ax[0] = ay[0] = al[0] = 0.0
        return np.concatenate([vx, vy, om, ax, ay, al])

    te = np.linspace(0.0, t_end, n_t)
    sol = solve_ivp(deriv, (0.0, t_end), Y0, t_eval=te, method="RK45",
                    rtol=1e-8, atol=1e-10, max_step=ds/np.sqrt(max(ea, ga))*2)
    if not sol.success:
        raise RuntimeError(f"dynamic Cosserat sim failed: {sol.message}")
    X = sol.y[0:nn, :].T; Yc = sol.y[nn:2*nn, :].T; TH = sol.y[2*nn:3*nn, :].T
    return s, te, X, Yc, TH


def generate_dynamic_cosserat(
    EI_unit: float = 1.0,
    GA_unit: float = 1.0,
    EA_unit: float = 1.0,
    EI0: float = DYNCOS_EI0,
    GA0: float = DYNCOS_GA0,
    EA0: float = DYNCOS_EA0,
    g: float = DYNCOS_G,
    c: float = DYNCOS_C,
    j: float = DYNCOS_J,
    t_end: float = DYNCOS_TEND,
    n_s: int = 41,
    n_t: int = 81,
    pos_noise_std: float = 1e-3,
    ang_noise_std: float = 5e-3,
    seed: int = 0,
):
    """Dynamic planar Cosserat rod inverse: recover bending / shear / axial
    stiffness from the **time-resolved** motion of a soft rod.

    The dynamic (inertial) Simo-Reissner rod — the time-domain extension of
    ``planar_cosserat``. A soft rod clamped at ``s=0`` is released from the
    straight horizontal configuration under a distributed gravity load; it swings
    down (~50° tip) and oscillates under light viscous damping, settling toward
    the static droop. The measured space-time fields ``x(s,t), y(s,t), θ(s,t)``
    encode the stiffnesses through the equations of motion (dimensionless)::

        x_tt = ∂Nx/∂s − c·x_t
        y_tt = ∂Ny/∂s − g − c·y_t
        j·θ_tt = EI·θ_ss + (x_s·Ny − y_s·Nx)

    with the constitutive forces ``Nx, Ny`` from the same axial/shear laws as the
    static template. Ground truth from a verified method-of-lines solver
    (:func:`_simulate_dynamic_cosserat`). The motion keeps axial strain ≲0.16 and
    shear ≲0.34 with max acceleration O(g) — large enough to identify GA, EA,
    gentle enough for the linear constitutive law and tractable for a PINN.

    Sensors: noisy ``x, y, θ`` on an ``n_s × n_t`` space-time grid, plus
    clamped-root and initial-shape pseudo-sensors (noise-free). The auxiliary
    force fields ``Nx, Ny`` are not measured — the template pins them via the
    constitutive residuals.

    Returns ``(data, truth)``. Interior/BC/IC inputs are ``(s, t)`` pairs.
    """
    rng = np.random.default_rng(seed)
    ei = EI0*float(EI_unit); ga = GA0*float(GA_unit); ea = EA0*float(EA_unit)
    # Solver resolution N must be a multiple of (n_s-1) so sensors sit on nodes.
    N = (n_s - 1) * 3
    s_nodes, t_grid, X, Yc, TH = _simulate_dynamic_cosserat(
        ei, ga, ea, g, c, j, N=N, t_end=t_end, n_t=n_t)
    idx = np.linspace(0, N, n_s, dtype=int)
    s_sel = s_nodes[idx]

    SS, TT = np.meshgrid(s_sel, t_grid, indexing="ij")   # (n_s, n_t)
    inp = np.stack([SS.ravel(), TT.ravel()], axis=1)
    # X is (n_t, nn); X[:, idx] -> (n_t, n_s); .T -> (n_s, n_t) to match SS layout.
    xo = X[:, idx].T.ravel(); yo = Yc[:, idx].T.ravel(); to = TH[:, idx].T.ravel()
    x_n = xo + rng.normal(0, pos_noise_std, xo.shape)
    y_n = yo + rng.normal(0, pos_noise_std, yo.shape)
    t_n = to + rng.normal(0, ang_noise_std, to.shape)

    # Clamped-root BC: x=y=θ=0 at s=0 for every time (noise-free).
    s0 = np.zeros_like(t_grid)
    bc_in = np.stack([s0, t_grid], axis=1)
    bc_zero = np.zeros_like(t_grid)
    # Initial shape at t=0: straight horizontal x=s, y=0, θ=0 (noise-free).
    ic_in = np.stack([s_sel, np.zeros_like(s_sel)], axis=1)
    ic_x = s_sel.copy(); ic_y = np.zeros_like(s_sel); ic_th = np.zeros_like(s_sel)

    f32 = lambda a: a.astype(np.float32)
    return (
        {
            "x_meas":     (f32(inp), f32(x_n)),
            "y_meas":     (f32(inp), f32(y_n)),
            "theta_meas": (f32(inp), f32(t_n)),
            "x_bc":       (f32(bc_in), f32(bc_zero)),
            "y_bc":       (f32(bc_in), f32(bc_zero)),
            "theta_bc":   (f32(bc_in), f32(bc_zero)),
            "x_ic":       (f32(ic_in), f32(ic_x)),
            "y_ic":       (f32(ic_in), f32(ic_y)),
            "theta_ic":   (f32(ic_in), f32(ic_th)),
        },
        {"EI_unit": float(EI_unit), "GA_unit": float(GA_unit), "EA_unit": float(EA_unit)},
    )
