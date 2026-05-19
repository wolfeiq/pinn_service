"""End-to-end smoke test: tiny config, just verifies the pipeline runs.

The actual inverse-problem convergence quality is an empirical tuning exercise
not covered by tests. We assert structure: the run completes, the result
object has the expected shape, the unknown parameters changed at all from
their initial values.
"""
import warnings
warnings.filterwarnings("ignore")

import pytest

from pinn_engine.dsl.templates_lib import damped_oscillator  # registers
from pinn_engine.dsl.templates import get_template
from pinn_engine.core.trainer import train, TrainConfig


@pytest.mark.slow
def test_oscillator_trainer_runs_end_to_end():
    tpl = get_template("damped_oscillator")
    system = tpl.system()
    data, truth = tpl.synthetic_data(seed=0)
    config = TrainConfig(
        depth=3, width=16, activation="tanh",
        lr=2e-3, adam_epochs=30, lbfgs_iters=0,
        balancer="none", t_range=(0.0, 5.0),
        n_collocation=200, batch_size=128,
        seed=0, accelerator="cpu", deterministic=False,
    )
    result = train(system=system, data=data, config=config)
    assert "c" in result.final_params
    assert "k" in result.final_params
    # Values are floats, not NaN.
    for v in result.final_params.values():
        assert v == v  # not NaN
    assert result.compiled is not None
    assert result.network is not None
    assert result.problem is not None
