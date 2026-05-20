"""Smoke tests for ONNX / TorchScript export."""
import warnings
warnings.filterwarnings("ignore")
import logging
for n in ("pytorch_lightning", "lightning.pytorch", "pina"):
    logging.getLogger(n).setLevel(logging.ERROR)

import json
import pytest

from pinn_engine.dsl.templates_lib import damped_oscillator  # registers
from pinn_engine.dsl.templates import get_template
from pinn_engine.core.trainer import train


@pytest.fixture(scope="module")
def tiny_result():
    """Train a 10-epoch toy run; return the TrainResult."""
    tpl = get_template("damped_oscillator")
    data, _ = tpl.synthetic_data(seed=0)
    cfg = tpl.default_config().model_copy(update={
        "seed": 0, "accelerator": "cpu", "adam_epochs": 10, "lbfgs_iters": 0,
    })
    return train(system=tpl.system(), data=data, config=cfg)


@pytest.mark.slow
def test_to_onnx_roundtrip(tmp_path, tiny_result):
    from pinn_engine.export import to_onnx
    out = to_onnx(tiny_result, tmp_path / "model.onnx")
    assert out.exists()
    sidecar = out.with_suffix(".json")
    assert sidecar.exists()
    meta = json.loads(sidecar.read_text())
    assert meta["format"] == "onnx"
    assert "c" in meta["discovered_parameters"]
    assert "k" in meta["discovered_parameters"]


@pytest.mark.slow
def test_to_torchscript_roundtrip(tmp_path, tiny_result):
    from pinn_engine.export import to_torchscript
    out = to_torchscript(tiny_result, tmp_path / "model.pt")
    assert out.exists()
    sidecar = out.with_suffix(".json")
    assert sidecar.exists()
    meta = json.loads(sidecar.read_text())
    assert meta["format"] == "torchscript"


@pytest.mark.slow
def test_export_no_network_raises(tmp_path, tiny_result):
    """If TrainResult has no network, export should refuse to write a file."""
    from pinn_engine.export import to_onnx
    tiny_result.network = None
    with pytest.raises(ValueError, match="no network"):
        to_onnx(tiny_result, tmp_path / "model.onnx")
