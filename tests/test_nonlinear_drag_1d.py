"""Smoke tests for the 1-DOF nonlinear-drag template."""
import warnings
warnings.filterwarnings("ignore")

import numpy as np

from pinn_engine.dsl.templates_lib import nonlinear_drag_1d  # registers
from pinn_engine.dsl.templates import get_template


def test_nonlinear_drag_1d_compiles():
    tpl = get_template("nonlinear_drag_1d")
    sys = tpl.system()
    sys.validate()
    comp = sys.compile()
    assert comp.state_names == ["u"]
    assert set(comp.unknown_names) == {"c_lin", "c_quad"}
    assert comp.unknown_bounds["c_lin"][1] == 0.0
    assert comp.unknown_bounds["c_lin"][0] < 0.0  # negative-drag convention


def test_nonlinear_drag_1d_synthetic_data():
    tpl = get_template("nonlinear_drag_1d")
    data, truth = tpl.synthetic_data(seed=42)
    assert truth == {"c_lin": -10.0, "c_quad": -30.0}
    t, u = data["u_meas"]
    assert t.shape == u.shape
    assert t[0] == 0.0
    # Body starts from rest and accelerates to steady-state ≈ 0.43.
    assert abs(u[0]) < 0.05
    assert 0.3 < u[-1] < 0.55


def test_nonlinear_drag_1d_init_not_truth():
    """Bounds intentionally offset so PINA's midpoint init isn't trivially at truth."""
    tpl = get_template("nonlinear_drag_1d")
    comp = tpl.system().compile()
    assert comp.unknown_inits["c_lin"] != -10.0
    assert comp.unknown_inits["c_quad"] != -30.0
