"""Each bundled template compiles, has synthetic data, and a default config."""
import pytest

from pinn_engine.dsl.templates_lib import damped_oscillator, lorenz, diffusion_1d  # registers
from pinn_engine.dsl.templates import get_template


@pytest.mark.parametrize("name", ["damped_oscillator", "lorenz", "diffusion_1d"])
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


def test_objective_returns_relative_error():
    tpl = get_template("damped_oscillator")

    class FakeResult:
        final_params = {"c": 0.55, "k": 10.5}

    err = tpl.objective(FakeResult())
    # |0.55-0.5|/0.5 = 0.1; |10.5-10|/10 = 0.05; mean = 0.075
    assert abs(err - 0.075) < 1e-6
