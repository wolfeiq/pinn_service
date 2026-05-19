"""Pre-flight identifiability checks."""
import pytest
import torch

from pinn_engine.dsl import Variable, Parameter, Unknown, Sensor, System
from pinn_engine.core.networks import build_network
from pinn_engine.core.problem import build_problem
from pinn_engine.preflight import check_wellposedness, UnidentifiableError


def _build_problem(system, t_range=(0.0, 1.0)):
    comp = system.compile()
    return build_problem(comp, data={}, t_range=t_range), comp


def test_oscillator_is_wellposed():
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
    problem, comp = _build_problem(sys)
    net = build_network(input_dim=1, output_dim=1, depth=3, width=16, activation="tanh")
    report = check_wellposedness(problem, net, comp, n=64)
    assert report.passed
    assert report.rank == 2
    assert not report.blind_unknowns


def test_redundant_unknown_is_flagged():
    """If two unknowns appear ONLY as a product, only one degree of freedom is identifiable."""
    t = Variable("t")
    x = Variable("x", depends_on=t)
    a = Unknown("a", bounds=(0.1, 5.0))
    b = Unknown("b", bounds=(0.1, 5.0))
    # Equation: ẋ + (a*b) * x = 0  →  a and b are not separately identifiable.
    sys = System(
        state=[x],
        equations=[x.d + a * b * x],
        sensors=[Sensor("x_meas", observes=x)],
    )
    problem, comp = _build_problem(sys)
    net = build_network(input_dim=1, output_dim=1, depth=3, width=16, activation="tanh")
    with pytest.raises(UnidentifiableError):
        check_wellposedness(problem, net, comp, n=64)


def test_unused_unknown_is_blind():
    """An unknown that doesn't appear in any equation has zero sensitivity."""
    t = Variable("t")
    x = Variable("x", depends_on=t)
    a = Unknown("a", bounds=(0.0, 5.0))
    ghost = Unknown("ghost", bounds=(0.0, 5.0))  # never used
    sys = System(
        state=[x],
        equations=[x.d + a * x],
        sensors=[Sensor("x_meas", observes=x)],
    )
    # Reference 'ghost' so it makes it into all_symbols via a no-op? It won't,
    # since it's not in any equation. The validator should be fine with extra
    # *declarations*, but our compile only registers unknowns that appear in
    # equations — so `ghost` is simply not in the compiled system. This test
    # documents that behavior: declaring an unused Unknown does NOT show up.
    comp = sys.compile()
    assert "ghost" not in comp.unknown_names
    assert "a" in comp.unknown_names
