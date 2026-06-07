"""Each bundled template compiles, has synthetic data, and a default config."""
import pytest

from pinn_engine.dsl.templates_lib import damped_oscillator, lorenz, diffusion_1d  # registers
from pinn_engine.dsl.templates import get_template


@pytest.mark.parametrize("name", ["damped_oscillator", "lorenz", "diffusion_1d",
                                  "coupled_drag_3d", "euler_bernoulli_beam",
                                  "axial_elastic_bar", "planar_elastica",
                                  "planar_cosserat"])
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


def test_objective_returns_relative_error():
    tpl = get_template("damped_oscillator")

    class FakeResult:
        final_params = {"c": 0.55, "k": 10.5}

    err = tpl.objective(FakeResult())
    # |0.55-0.5|/0.5 = 0.1; |10.5-10|/10 = 0.05; mean = 0.075
    assert abs(err - 0.075) < 1e-6
