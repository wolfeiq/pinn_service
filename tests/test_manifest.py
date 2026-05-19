"""Manifest write/read round-trip."""
import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path
import pytest

from pinn_engine.dsl.templates_lib import damped_oscillator  # registers
from pinn_engine.dsl.templates import get_template
from pinn_engine.core.trainer import train, TrainConfig
from pinn_engine.repro import write_manifest, read_manifest


@pytest.mark.slow
def test_manifest_roundtrip(tmp_path):
    tpl = get_template("damped_oscillator")
    system = tpl.system()
    data, _ = tpl.synthetic_data(seed=0)
    config = TrainConfig(
        depth=3, width=8, activation="tanh",
        lr=2e-3, adam_epochs=5, lbfgs_iters=0,
        balancer="none", t_range=(0.0, 1.0),
        n_collocation=64, batch_size=64,
        seed=0, accelerator="cpu",
    )
    result = train(system=system, data=data, config=config)
    path = write_manifest(template="damped_oscillator", result=result, data=data, out_dir=tmp_path)
    assert path.exists()
    m = read_manifest(path)
    assert m.template == "damped_oscillator"
    assert m.seed == 0
    assert "c" in m.final_params and "k" in m.final_params
    # JSON-loadable
    parsed = json.loads(path.read_text())
    assert parsed["template"] == "damped_oscillator"
