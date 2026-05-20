"""Smoke tests for the Fossen 1-DOF surge template."""
import warnings
warnings.filterwarnings("ignore")

import numpy as np

from pinn_engine.dsl.templates_lib import fossen_surge  # registers
from pinn_engine.dsl.templates import get_template


def test_fossen_surge_compiles():
    tpl = get_template("fossen_surge")
    sys = tpl.system()
    sys.validate()
    comp = sys.compile()
    assert comp.state_names == ["u"]
    assert set(comp.unknown_names) == {"X_u", "X_uu"}
    assert comp.unknown_bounds["X_u"][1] == 0.0
    assert comp.unknown_bounds["X_u"][0] < 0.0  # Fossen convention: negative


def test_fossen_surge_synthetic_data():
    tpl = get_template("fossen_surge")
    data, truth = tpl.synthetic_data(seed=42)
    assert truth == {"X_u": -10.0, "X_uu": -30.0}
    t, u = data["u_meas"]
    assert t.shape == u.shape
    assert t[0] == 0.0
    # Vehicle starts from rest and accelerates to steady-state ≈ 0.43 m/s.
    assert abs(u[0]) < 0.05
    assert 0.3 < u[-1] < 0.55


def test_fossen_surge_init_not_truth():
    """Bounds intentionally offset so PINA's midpoint init isn't trivially at truth."""
    tpl = get_template("fossen_surge")
    comp = tpl.system().compile()
    assert comp.unknown_inits["X_u"] != -10.0
    assert comp.unknown_inits["X_uu"] != -30.0
