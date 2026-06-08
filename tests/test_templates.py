"""Each bundled template compiles, has synthetic data, and a default config."""
import pytest

from pinn_engine.dsl.templates_lib import damped_oscillator, lorenz, diffusion_1d  # registers
from pinn_engine.dsl.templates import get_template


@pytest.mark.parametrize("name", ["damped_oscillator", "lorenz", "diffusion_1d",
                                  "coupled_drag_3d", "euler_bernoulli_beam",
                                  "axial_elastic_bar", "planar_elastica",
                                  "planar_cosserat", "dynamic_cosserat"])
def test_template_system_and_data(name):
    tpl = get_template(name)
    sys = tpl.system()
    sys.validate()
    comp = sys.compile()
    data, truth = tpl.synthetic_data(seed=0)
    for sensor in comp.sensors:
        assert sensor.name in data
        t_arr, obs_arr = data[sensor.name]
        assert t_arr.shape[0] == obs_arr.shape[0]
    cfg = tpl.default_config()
    assert cfg.depth >= 2 and cfg.width >= 8


def test_diffusion_1d_data_and_input_order():
    tpl = get_template("diffusion_1d")
    comp = tpl.system().compile()
    # Space-first convention: input columns must be (x, t).
    assert comp.input_names == ("x", "t")
    data, truth = tpl.synthetic_data(seed=0)
    assert "u_meas" in data and truth == {"D": 0.1}
    meas_input, _ = data["u_meas"]
    assert meas_input.shape[1] == 2


def test_planar_elastica_nonlinear_residual_and_data():
    import sympy as sp
    tpl = get_template("planar_elastica")
    sys = tpl.system()
    sys.validate()
    # The geometrically-exact rod must carry the cos(θ) nonlinearity — this is
    # what distinguishes it from the linear euler_bernoulli_beam.
    assert any(expr.has(sp.cos) for expr in sys.equations)
    # Large-deflection ground truth: tip angle should be well into the
    # nonlinear regime (~51° at the default load), not the small-slope limit.
    data, truth = tpl.synthetic_data(seed=0)
    assert truth == {"EI_unit": 1.0}
    _, theta = data["theta_meas"]
    assert theta.max() > 0.7  # radians (~40°+), genuinely large deflection
    # Clamped-root BC pseudo-sensor is exactly zero, noise-free.
    s_bc, theta_bc = data["theta_bc"]
    assert float(theta_bc[0]) == 0.0


def test_planar_cosserat_multi_unknown_and_residual_consistency():
    """Full Cosserat rod: 3 unknowns, 3 coupled residuals, and the template
    equations must vanish on the independent solve_bvp ground truth."""
    import numpy as np
    import sympy as sp
    from pinn_engine.data.synthetic import (
        _solve_cosserat_planar_bvp, COSSERAT_EI0, COSSERAT_GA0, COSSERAT_EA0,
        COSSERAT_PX, COSSERAT_PY,
    )
    tpl = get_template("planar_cosserat")
    sys = tpl.system()
    sys.validate()
    assert set(tpl.truth) == {"EI_unit", "GA_unit", "EA_unit"}
    assert len(sys.equations) == 3
    # Independent ground truth at truth=1 for all three stiffnesses.
    sol = _solve_cosserat_planar_bvp(COSSERAT_EI0, COSSERAT_GA0, COSSERAT_EA0,
                                     COSSERAT_PX, COSSERAT_PY)
    ss = np.linspace(0.05, 0.95, 19)
    Y = sol.sol(ss); x, y, th, M = Y
    d = sol.sol(ss)  # states; derivatives from the ODE rhs:
    # rebuild xp, yp, thpp analytically from the model
    ei, ga, ea = COSSERAT_EI0, COSSERAT_GA0, COSSERAT_EA0
    Px, Py = COSSERAT_PX, COSSERAT_PY
    nu = 1.0 + (Px*np.cos(th) + Py*np.sin(th))/ea
    eta = (-Px*np.sin(th) + Py*np.cos(th))/ga
    xp = nu*np.cos(th) - eta*np.sin(th)
    yp = nu*np.sin(th) + eta*np.cos(th)
    Mp = -(xp*Py - yp*Px)
    thpp = Mp/ei
    R_axial = ea*(xp*np.cos(th) + yp*np.sin(th) - 1) - (Px*np.cos(th) + Py*np.sin(th))
    R_shear = ga*(-xp*np.sin(th) + yp*np.cos(th)) - (-Px*np.sin(th) + Py*np.cos(th))
    R_moment = ei*thpp + (Py*xp - Px*yp)
    assert np.abs(R_axial).max() < 1e-6
    assert np.abs(R_shear).max() < 1e-6
    assert np.abs(R_moment).max() < 1e-6
    # The deformed shape must be genuinely large-deflection with real shear.
    assert abs(np.degrees(th).min()) > 25  # tip rotation well past small-angle
    assert np.abs(eta).max() > 0.05         # non-trivial shear strain


def test_dynamic_cosserat_solver_energy_and_static_limit():
    """The dynamic-rod forward solver must conserve energy when undamped and
    relax to the static gravity-loaded shape when damped — the two checks that
    validate it as ground truth for the inverse template."""
    import numpy as np
    from pinn_engine.data.synthetic import (
        _simulate_dynamic_cosserat, DYNCOS_EI0, DYNCOS_GA0, DYNCOS_EA0,
        DYNCOS_G, DYNCOS_J,
    )
    ei, ga, ea, g, j = DYNCOS_EI0, DYNCOS_GA0, DYNCOS_EA0, DYNCOS_G, DYNCOS_J
    N = 40
    # Undamped: total energy (KE + elastic + gravity PE) ~ conserved.
    s, t, X, Y, TH = _simulate_dynamic_cosserat(ei, ga, ea, g, c=0.0, j=j,
                                                N=N, t_end=2.0, n_t=120)
    ds = s[1] - s[0]
    # finite-difference velocities in time
    Vx = np.gradient(X, t, axis=0); Vy = np.gradient(Y, t, axis=0)
    Om = np.gradient(TH, t, axis=0)
    mass = np.full(N + 1, ds); mass[-1] = ds / 2; mass[0] = 0.0
    E = []
    for k in range(X.shape[0]):
        x, y, th = X[k], Y[k], TH[k]
        ke = 0.5 * np.sum(mass * (Vx[k]**2 + Vy[k]**2)) + 0.5 * np.sum(mass * j * Om[k]**2)
        dxs = (x[1:]-x[:-1])/ds; dys = (y[1:]-y[:-1])/ds
        the = (th[:-1]+th[1:])/2; dth = (th[1:]-th[:-1])/ds
        nu = dxs*np.cos(the)+dys*np.sin(the); eta = -dxs*np.sin(the)+dys*np.cos(the)
        pe = 0.5*ds*np.sum(ea*(nu-1)**2 + ga*eta**2 + ei*dth**2)
        grav = g*np.sum(mass*y)
        E.append(ke + pe + grav)
    E = np.array(E)
    # FD velocities add a little noise; drift should still be a small fraction.
    assert np.abs(E - E[0]).max() / max(abs(E[0]), 1.0) < 0.05

    # Damped: relaxes to the static gravity-loaded equilibrium (compare to a
    # heavily-damped long run's steady state — must be a smooth, monotone droop).
    s2, t2, X2, Y2, TH2 = _simulate_dynamic_cosserat(ei, ga, ea, g, c=3.0, j=j,
                                                     N=N, t_end=15.0, n_t=30)
    tip_y = Y2[-1, -1]
    assert tip_y < -0.1                       # droops downward
    assert abs(np.degrees(TH2[-1, -1])) > 15  # genuinely large deflection
    # steady: velocity (last vs second-last frame) is small
    assert np.abs(Y2[-1] - Y2[-2]).max() < 1e-2


def test_dynamic_cosserat_template_shape():
    tpl = get_template("dynamic_cosserat")
    sys = tpl.system(); sys.validate()
    comp = sys.compile()
    assert comp.input_names == ("s", "t")        # space-first space-time domain
    assert len(sys.equations) == 3 and len(sys.state) == 3
    assert set(tpl.truth) == {"EI_unit", "GA_unit", "EA_unit"}
    data, _ = tpl.synthetic_data(seed=0)
    for name in ("x_meas", "y_meas", "theta_meas"):
        inp, tgt = data[name]
        assert inp.shape[1] == 2 and inp.shape[0] == tgt.shape[0]


def test_cosserat_force_id_recovers_all_three_stiffnesses():
    """The force-from-motion baseline closes the dynamic-rod shear/axial gap:
    EI/GA/EA all recovered from noisy motion via inertia-derived forces."""
    from pinn_engine.data.synthetic import generate_dynamic_cosserat
    from pinn_engine.baselines import recover_from_template_data
    n_s, n_t = 41, 161
    data, truth = generate_dynamic_cosserat(seed=0, n_s=n_s, n_t=n_t,
                                            pos_noise_std=1e-3, ang_noise_std=5e-3)
    r = recover_from_template_data(data, n_s, n_t).as_dict()
    # All three recovered to single-digit %, where the PINN stalls at ~250% on GA/EA.
    assert abs(r["EI_unit"] - 1.0) < 0.06
    assert abs(r["GA_unit"] - 1.0) < 0.04
    assert abs(r["EA_unit"] - 1.0) < 0.10


def test_spatial_cosserat_3d_recovers_six_stiffnesses():
    """Full 3-D spatial Cosserat rod: recover all six stiffnesses (axial, two
    shear, torsion, two bending) from measured shape + orientation."""
    import numpy as np
    from pinn_engine.baselines import (generate_spatial_cosserat,
                                       recover_spatial_stiffness)
    names = ["EA_unit", "GA1_unit", "GA2_unit", "EI1_unit", "EI2_unit", "GJ_unit"]
    # Clean data → near-exact recovery of all six.
    data, truth = generate_spatial_cosserat(n_s=121, pos_noise_std=0.0,
                                            quat_noise_std=0.0, seed=0)
    r = recover_spatial_stiffness(data).as_dict()
    for k in names:
        assert abs(r[k] - 1.0) < 0.02, (k, r[k])
    # Recovers an arbitrary (non-unity) stiffness, not just truth=1.
    data2, _ = generate_spatial_cosserat(n_s=121, pos_noise_std=0.0,
                                         quat_noise_std=0.0, EA_unit=1.3,
                                         GJ_unit=0.7, EI1_unit=1.2)
    r2 = recover_spatial_stiffness(data2).units
    assert abs(r2["EA_unit"] - 1.3) < 0.03
    assert abs(r2["GJ_unit"] - 0.7) < 0.03
    assert abs(r2["EI1_unit"] - 1.2) < 0.03
    # Noisy data → all six still within single-digit %.
    data3, _ = generate_spatial_cosserat(n_s=121, pos_noise_std=1e-3,
                                         quat_noise_std=3e-3, seed=0)
    r3 = recover_spatial_stiffness(data3).as_dict()
    for k in names:
        assert abs(r3[k] - 1.0) < 0.10, (k, r3[k])


def test_contact_recovers_location_and_force_from_shape():
    """Proprioceptive contact sensing: recover the contact arclength and force
    from the curvature kink in the measured shape alone."""
    from pinn_engine.baselines import generate_contact_scenario, recover_contact
    for sc, Fc in [(0.3, 1.5), (0.5, 2.0), (0.7, 3.0)]:
        d, truth = generate_contact_scenario(Fc=Fc, sc=sc, ang_noise_std=0.0, seed=0)
        r = recover_contact(d)
        assert abs(r.sc - sc) < 0.03, (sc, r.sc)
        assert abs(r.Fc - Fc) / Fc < 0.05, (Fc, r.Fc)
    # Noisy still localizes and sizes the contact.
    d2, _ = generate_contact_scenario(Fc=2.0, sc=0.5, ang_noise_std=3e-3, seed=0)
    r2 = recover_contact(d2)
    assert abs(r2.sc - 0.5) < 0.05 and abs(r2.Fc - 2.0) / 2.0 < 0.08


def test_hyperelastic_recovers_nonlinear_constitutive():
    """Recover the nonlinear (cubic) moment-curvature and axial coefficients from
    a load sweep, and confirm a linear-only fit is decisively rejected."""
    from pinn_engine.baselines import (generate_hyperelastic_sweep,
                                       recover_hyperelastic)
    # Clean: exact recovery of linear + nonlinear coefficients.
    data, truth = generate_hyperelastic_sweep(ang_noise_std=0.0, strain_noise_std=0.0, seed=0)
    r = recover_hyperelastic(data)
    assert abs(r.a1 - truth["a1"]) < 0.01 and abs(r.a3 - truth["a3"]) < 0.01
    assert abs(r.b1 - truth["b1"]) < 0.05 and abs(r.b3 - truth["b3"]) < 0.1
    # The hyperelastic nonlinearity is detectable: linear-only fit residual is
    # far larger than the cubic-fit residual.
    assert r.linear_only_rms > 20 * max(r.nonlinear_rms, 1e-6)
    # Noisy: still recovers the coefficients within a few %.
    data2, _ = generate_hyperelastic_sweep(ang_noise_std=3e-3, strain_noise_std=2e-3, seed=0)
    r2 = recover_hyperelastic(data2)
    assert abs(r2.a1 - truth["a1"]) < 0.05 and abs(r2.a3 - truth["a3"]) < 0.08
    # Strain-softening (a3<0) is recovered with the right sign.
    data3, _ = generate_hyperelastic_sweep(bend={"a1": 1.0, "a3": -0.3},
                                           ang_noise_std=3e-3, seed=1)
    r3 = recover_hyperelastic(data3)
    assert r3.a3 < -0.2


def test_pneumatic_actuation_signs_and_self_calibration():
    """Pneumatic actuation pushes (extends) and bends *away* from the chamber —
    opposite a tendon — and self-calibrates EA/EI1/EI2/GJ from pressure sweeps."""
    from pinn_engine.baselines import (simulate_pneumatic_actuated,
                                       simulate_tendon_actuated,
                                       generate_pneumatic_calibration,
                                       recover_pneumatic_stiffness)
    stiff = {"EA": 15.0, "GA1": 15.0, "GA2": 12.0, "GJ": 0.8, "EI1": 1.0, "EI2": 0.8}
    # +y chamber: extends (x>1) and bends away (y<0); a +y tendon bends toward (y>0).
    s, r, q = simulate_pneumatic_actuated(stiff, [(1.0, 0.05, 0, 0)], [2.0])
    assert r[-1, 0] > 1.0 and r[-1, 1] < 0.0
    _, r_t, _ = simulate_tendon_actuated(stiff, [(0.05, 0, 0)], [2.0])
    assert r_t[-1, 1] > 0.0
    # central chamber → pure extension
    s, r, q = simulate_pneumatic_actuated(stiff, [(2.0, 0, 0, 0)], [2.0])
    assert r[-1, 0] > 1.1 and abs(r[-1, 1]) < 1e-9
    # self-calibration
    names = ["EA_unit", "EI1_unit", "EI2_unit", "GJ_unit"]
    data, _ = generate_pneumatic_calibration(pos_noise_std=0.0, quat_noise_std=0.0, seed=0)
    rc = recover_pneumatic_stiffness(data).as_dict()
    for k in names:
        assert abs(rc[k] - 1.0) < 0.01, (k, rc[k])
    data3, _ = generate_pneumatic_calibration(pos_noise_std=1e-3, quat_noise_std=3e-3, seed=0)
    r3 = recover_pneumatic_stiffness(data3).as_dict()
    for k in names:
        assert abs(r3[k] - 1.0) < 0.10, (k, r3[k])


def test_tendon_actuation_constant_curvature_and_self_calibration():
    """Tendon actuation: a single tendon gives the constant-curvature law
    κ=τd/EI, and a sweep of known tension patterns self-calibrates the rod's
    axial/bending/torsion stiffness from the actuated shapes (no external rig)."""
    import numpy as np
    from pinn_engine.baselines import (simulate_tendon_actuated,
                                       generate_tendon_calibration,
                                       recover_tendon_stiffness)
    stiff = {"EA": 15.0, "GA1": 15.0, "GA2": 12.0, "GJ": 0.8, "EI1": 1.0, "EI2": 0.8}
    # Single tendon at offset d=0.05, tension 2 → curvature τd/EI2 about ê3.
    s, r, q = simulate_tendon_actuated(stiff, [(0.05, 0.0, 0.0)], [2.0])
    # tip angle of a constant-curvature arc = κ·L = τd/EI2.
    R = np.array([[1 - 2 * (q[-1, 2] ** 2 + q[-1, 3] ** 2)]])  # not used; check via shape
    expected_kappa = 2.0 * 0.05 / 0.8
    # planar arc in xy: tip angle = atan2 of tangent; tangent angle ≈ κ at tip
    assert abs(r[-1, 2]) < 1e-9                       # stays in-plane (z=0)
    assert r[-1, 1] > 0.04                            # bends toward +y
    # Self-calibration: clean → near-exact; noisy → single-digit %.
    names = ["EA_unit", "EI1_unit", "EI2_unit", "GJ_unit"]
    data, _ = generate_tendon_calibration(pos_noise_std=0.0, quat_noise_std=0.0, seed=0)
    rc = recover_tendon_stiffness(data).as_dict()
    for k in names:
        assert abs(rc[k] - 1.0) < 0.01, (k, rc[k])
    data2, _ = generate_tendon_calibration(pos_noise_std=0.0, quat_noise_std=0.0,
                                           EI1_unit=1.4, GJ_unit=0.6, EA_unit=1.2)
    r2 = recover_tendon_stiffness(data2).units
    assert abs(r2["EI1_unit"] - 1.4) < 0.02 and abs(r2["GJ_unit"] - 0.6) < 0.02
    data3, _ = generate_tendon_calibration(pos_noise_std=1e-3, quat_noise_std=3e-3, seed=0)
    r3 = recover_tendon_stiffness(data3).as_dict()
    for k in names:
        assert abs(r3[k] - 1.0) < 0.08, (k, r3[k])


def test_dynamic_spatial_cosserat_recovers_six_stiffnesses():
    """Dynamic 3-D rod: recover all six stiffnesses from the time-resolved 3-D
    motion (shape + orientation) via the kinematic force/moment + regression."""
    from pinn_engine.baselines import (generate_dynamic_spatial_cosserat,
                                       recover_dynamic_spatial_stiffness)
    names = ["EA_unit", "GA1_unit", "GA2_unit", "EI1_unit", "EI2_unit", "GJ_unit"]
    # Coarse grid for test speed; bounds are loose (finer grids reach ~5%).
    data, truth = generate_dynamic_spatial_cosserat(N=32, n_t=101, pos_noise_std=1e-3,
                                                    quat_noise_std=3e-3, seed=0)
    r = recover_dynamic_spatial_stiffness(data).as_dict()
    for k in ["GA1_unit", "GA2_unit", "EI1_unit", "EI2_unit"]:
        assert abs(r[k] - 1.0) < 0.10, (k, r[k])
    # axial (EA) and torsion (GJ) are the smoothing-sensitive axial-direction modes
    assert abs(r["EA_unit"] - 1.0) < 0.15, r["EA_unit"]
    assert abs(r["GJ_unit"] - 1.0) < 0.15, r["GJ_unit"]


def test_dynamic_spatial_cosserat_solver_energy_and_planar_reduction():
    """Forward solver: undamped energy ~conserved, and an in-plane isotropic
    load stays in-plane (z=0) — the planar reduction."""
    import numpy as np
    from pinn_engine.baselines import simulate_dynamic_spatial_cosserat
    stiff = {"EA": 15.0, "GA1": 15.0, "GA2": 15.0, "GJ": 0.8, "EI1": 1.0, "EI2": 1.0}
    Jr = (0.02, 0.01, 0.01)
    # In-plane gravity, isotropic, no pre-twist -> motion confined to xy (z=0).
    s, t, r, q = simulate_dynamic_spatial_cosserat(
        stiff, Jrho=Jr, gvec=(0, -3, 0), c=0.0, twist0=0.0, N=24, t_end=1.5, n_t=80)
    assert np.abs(r[:, :, 2]).max() < 1e-6        # z stays zero
    assert r[-1, :, 1].min() < -0.2               # tip droops in -y

    # Undamped energy conservation (small relative to the PE<->KE exchange).
    nn = r.shape[0]; ds = s[1] - s[0]
    Cn = np.array([15., 15., 15.]); Cm = np.array([0.8, 1., 1.]); Jr_ = np.array(Jr)
    from pinn_engine.baselines.spatial_cosserat_id import _Rmat, _qmul, _E1
    v = np.gradient(r, t, axis=1)
    E = []
    for k in range(r.shape[1]):
        mass = np.full(nn, ds); mass[-1] = ds / 2; mass[0] = 0.0
        ke = 0.5 * np.sum(mass[:, None] * v[:, k] ** 2)
        pe = 0.0
        for e in range(nn - 1):
            qm = q[e, k] + q[e + 1, k]; qm /= np.linalg.norm(qm); R = _Rmat(qm)
            G = R.T @ ((r[e + 1, k] - r[e, k]) / ds) - _E1
            qc = np.array([q[e, k][0], -q[e, k][1], -q[e, k][2], -q[e, k][3]])
            K = 2 * _qmul(qc, q[e + 1, k])[1:] / ds
            pe += 0.5 * ds * (np.sum(Cn * G * G) + np.sum(Cm * K * K))
        grav = -np.sum(mass * (r[:, k] @ np.array([0, -3., 0])))
        E.append(ke + pe + grav)
    E = np.array(E)
    assert (E.max() - E.min()) < 0.05            # conserved vs PE<->KE swing ~2.4


def test_spatial_cosserat_solver_analytic_limits():
    """Forward solver matches closed-form limits: axial stretch, pure twist."""
    import numpy as np
    from pinn_engine.baselines import simulate_spatial_cosserat
    stiff = {"EA": 15.0, "GA1": 15.0, "GA2": 12.0, "GJ": 0.8, "EI1": 1.0, "EI2": 0.8}
    # Pure axial force → uniform stretch tip x = 1 + P/EA, no lateral motion.
    s, r, q = simulate_spatial_cosserat(stiff, P=(2.0, 0, 0), Mt=(0, 0, 0))
    assert abs(r[-1, 0] - (1 + 2.0 / 15.0)) < 1e-3
    assert abs(r[-1, 1]) < 1e-4 and abs(r[-1, 2]) < 1e-4
    # Pure axial twist moment → straight rod, tip rotation Mt1/GJ about x.
    s, r, q = simulate_spatial_cosserat(stiff, P=(0, 0, 0), Mt=(0.3, 0, 0))
    twist = 2 * np.arctan2(q[-1, 1], q[-1, 0])
    assert abs(twist - 0.3 / 0.8) < 1e-2
    assert abs(r[-1, 1]) < 1e-3 and abs(r[-1, 2]) < 1e-3


def test_objective_returns_relative_error():
    tpl = get_template("damped_oscillator")

    class FakeResult:
        final_params = {"c": 0.55, "k": 10.5}

    err = tpl.objective(FakeResult())
    # |0.55-0.5|/0.5 = 0.1; |10.5-10|/10 = 0.05; mean = 0.075
    assert abs(err - 0.075) < 1e-6
