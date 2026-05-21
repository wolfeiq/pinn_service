"""Regression tests for the multi-variable (PDE) DSL extension.

The build supports both ODE (single input variable) and PDE (multi-input)
problems. These tests cover the PDE path that the Cosserat rod template
exercises in production.
"""
import warnings
warnings.filterwarnings("ignore")

import pytest
import torch
from pina import LabelTensor

from pinn_engine.dsl import Variable, Parameter, Unknown, Sensor, System


def test_variable_accepts_tuple_depends_on():
    s = Variable("s")
    t = Variable("t")
    u = Variable("u", depends_on=(s, t))
    assert u.depends_on_all == (s, t)
    assert u.depends_on == s   # first dep, back-compat


def test_variable_diff_w_r_t_specific_var():
    s = Variable("s")
    t = Variable("t")
    u = Variable("u", depends_on=(s, t))
    du_dt = u.diff(t)
    du_ds_2 = u.diff(s, 2)
    assert du_dt.variable_count == ((t, 1),)
    assert du_ds_2.variable_count == ((s, 2),)


def test_variable_diff_rejects_undeclared_var():
    s = Variable("s")
    t = Variable("t")
    u = Variable("u", depends_on=(s,))
    with pytest.raises(AttributeError, match="doesn't depend on"):
        u.diff(t)


def test_pde_system_compiles_with_multi_input():
    s = Variable("s")
    t = Variable("t")
    u = Variable("u", depends_on=(s, t))
    rho = Parameter("rho", value=1000.0)
    E = Unknown("E", bounds=(1e5, 1e7))
    system = System(
        state=[u],
        equations=[rho * u.diff(t, 2) - E * u.diff(s, 2)],
        sensors=[Sensor("u_meas", observes=u)],
    )
    system.validate()
    comp = system.compile()
    assert comp.is_pde
    assert comp.input_names == ("s", "t")


def test_pde_residual_evaluates_partial_derivatives_correctly():
    """∂²u/∂t² and ∂²u/∂s² of a known closed-form solution should match."""
    s = Variable("s")
    t = Variable("t")
    u = Variable("u", depends_on=(s, t))
    rho = Parameter("rho", value=1.0)
    E = Unknown("E", bounds=(0.5, 2.0))
    system = System(
        state=[u],
        equations=[rho * u.diff(t, 2) - E * u.diff(s, 2)],
        sensors=[Sensor("u_meas", observes=u)],
    )
    comp = system.compile()

    # Net = identity-style: u(s, t) = sin(s) * cos(t)
    # ∂²u/∂t² = -sin(s) * cos(t),  ∂²u/∂s² = -sin(s) * cos(t)
    # With ρ=1, E=1: residual = -sin·cos − (-sin·cos) = 0
    class WaveSolution(torch.nn.Module):
        def forward(self, x):
            # x is (N, 2) with columns (s, t)
            if hasattr(x, "labels"):
                s_col = x.extract(["s"])
                t_col = x.extract(["t"])
            else:
                s_col = x[..., 0:1]
                t_col = x[..., 1:2]
            return torch.sin(s_col) * torch.cos(t_col)

    net = WaveSolution()
    inp = torch.tensor([[0.5, 0.3], [0.7, 0.2], [1.0, 0.4]],
                       dtype=torch.float32, requires_grad=True)
    input_lt = LabelTensor(inp, labels=["s", "t"])
    y = net(input_lt)
    output_lt = LabelTensor(y, labels=["u"])
    params_ = {"E": torch.tensor([1.0], requires_grad=True)}
    r = comp.physics_residuals[0](input_lt, output_lt, params_=params_)
    # Wave eq residual on this solution must be 0 (up to autograd numerical noise).
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-5)
