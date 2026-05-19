"""Test the sympy → torch compilation of the DSL."""
import pytest
import torch
from pina import LabelTensor

from pinn_engine.dsl import Variable, Parameter, Unknown, Sensor, System, SystemValidationError


def test_minimal_system_validates_and_compiles():
    t = Variable("t")
    x = Variable("x", depends_on=t)
    m = Parameter("m", value=1.0)
    c = Unknown("c", bounds=(0.0, 5.0))
    k = Unknown("k", bounds=(0.0, 100.0))
    sys = System(
        state=[x],
        equations=[m * x.dd + c * x.d + k * x],
        sensors=[Sensor("x_meas", observes=x)],
    )
    sys.validate()
    comp = sys.compile()
    assert comp.state_names == ["x"]
    assert comp.input_name == "t"
    assert set(comp.unknown_names) == {"c", "k"}
    assert comp.unknown_bounds["c"] == (0.0, 5.0)
    assert comp.unknown_inits["c"] == 2.5
    assert len(comp.physics_residuals) == 1
    assert len(comp.equation_hash) == 64  # sha256 hex


def test_validation_rejects_undeclared_symbol():
    import sympy as sp
    t = Variable("t")
    x = Variable("x", depends_on=t)
    foo = sp.Symbol("foo")  # not declared via the DSL
    sys = System(
        state=[x],
        equations=[x.dd + foo * x],
        sensors=[Sensor("x_meas", observes=x)],
    )
    with pytest.raises(SystemValidationError, match="undeclared"):
        sys.validate()


def test_validation_rejects_sensor_on_undeclared_variable():
    t = Variable("t")
    x = Variable("x", depends_on=t)
    y = Variable("y", depends_on=t)
    sys = System(
        state=[x],
        equations=[x.dd + x],
        sensors=[Sensor("y_meas", observes=y)],  # y not in state
    )
    with pytest.raises(SystemValidationError, match="not in"):
        sys.validate()


def test_compiled_residual_evaluates_on_label_tensors():
    """Residual callable runs on LabelTensors and returns a tensor of the right shape."""
    t = Variable("t")
    x = Variable("x", depends_on=t)
    m = Parameter("m", value=1.0)
    c = Unknown("c", bounds=(0.0, 5.0))
    k = Unknown("k", bounds=(0.0, 100.0))
    sys = System(
        state=[x],
        equations=[m * x.dd + c * x.d + k * x],
        sensors=[Sensor("x_meas", observes=x)],
    )
    comp = sys.compile()

    # Build a toy network and run the residual once.
    net = torch.nn.Sequential(torch.nn.Linear(1, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    t_pts = torch.linspace(0, 1, 16).reshape(-1, 1).requires_grad_(True)
    input_lt = LabelTensor(t_pts, labels=["t"])
    y = net(input_lt)
    output_lt = LabelTensor(y, labels=["x"])
    params_ = {
        "c": torch.tensor([0.5], requires_grad=True),
        "k": torch.tensor([10.0], requires_grad=True),
    }
    r = comp.physics_residuals[0](input_lt, output_lt, params_=params_)
    assert r.shape == (16, 1)
    assert r.requires_grad
