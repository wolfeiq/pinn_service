"""PINA :class:`InverseProblem` adapter.

``build_problem(compiled, data, t_range)`` returns a freshly-built
PINA problem class instance (a subclass of ``TimeDependentProblem`` +
``InverseProblem``) wired up with the compiled physics residuals and the
sensor data.

This is the bridge between our DSL and PINA's runtime.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

from pina import Condition, LabelTensor
from pina.problem import TimeDependentProblem, InverseProblem
from pina.domain import CartesianDomain
from pina.equation import Equation

from pinn_engine.dsl.system import CompiledSystem


def _as_label_tensor(arr, labels: list[str]) -> LabelTensor:
    """Wrap a 1-D or 2-D array as a PINA LabelTensor with given column labels."""
    t = torch.as_tensor(arr, dtype=torch.float32)
    if t.ndim == 1:
        t = t.reshape(-1, 1)
    return LabelTensor(t, labels=list(labels))


def build_problem(
    compiled: CompiledSystem,
    data: Dict[str, Tuple[Any, Any]],
    t_range: Tuple[float, float],
) -> InverseProblem:
    """Construct a PINA problem instance from a compiled DSL system + data.

    Parameters:
        compiled: the result of ``System.compile()``.
        data: mapping ``sensor_name -> (input_array, observation_array)``.
        t_range: ``(t_min, t_max)`` for the temporal collocation domain.

    Returns:
        an instantiated PINA problem.
    """
    state_names = list(compiled.state_names)
    input_name = compiled.input_name

    temporal_domain = CartesianDomain({input_name: [float(t_range[0]), float(t_range[1])]})
    unknown_parameter_domain = CartesianDomain(
        {
            name: [float(lo), float(hi)]
            for name, (lo, hi) in compiled.unknown_bounds.items()
        }
    )

    conditions: Dict[str, Condition] = {}

    # One physics condition per residual.
    for i, residual_fn in enumerate(compiled.physics_residuals):
        conditions[f"physics_{i}"] = Condition(
            domain=temporal_domain,
            equation=Equation(residual_fn),
        )

    # One data condition per sensor that has data.
    for sens in compiled.sensors:
        if sens.name not in data:
            continue
        t_arr, obs_arr = data[sens.name]
        input_lt = _as_label_tensor(t_arr, labels=[input_name])
        # The observed quantity gets the state-variable's name as label so PINA
        # can match it against the network's named output.
        observed_label = sens.observes.name if hasattr(sens.observes, "name") else sens.name
        target_lt = _as_label_tensor(obs_arr, labels=[observed_label])
        conditions[f"data_{sens.name}"] = Condition(input=input_lt, target=target_lt)

    attrs = {
        "output_variables": state_names,
        "temporal_domain": temporal_domain,
        "unknown_parameter_domain": unknown_parameter_domain,
        "conditions": conditions,
    }
    cls = type("CompiledInverseProblem", (TimeDependentProblem, InverseProblem), attrs)
    problem = cls()

    # PINA initializes unknown_parameters with U(0, range_hi) + range_lo, which
    # gives the wrong distribution unless lo == 0. Override to the requested init.
    for name, init in compiled.unknown_inits.items():
        problem.unknown_parameters[name] = torch.nn.Parameter(
            torch.tensor([float(init)], requires_grad=True)
        )

    return problem
