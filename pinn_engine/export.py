"""Model export: ONNX and TorchScript.

After training, the inverse PINN is two things:
  1. A neural network that maps ``t`` (and possibly spatial coords) to a
     state vector ``y(t)``.
  2. A set of discovered unknown parameters (``c``, ``k``, etc.).

For edge deployment the network is the heavy artifact — ONNX or
TorchScript both express it portably. The discovered parameters are
small JSON / floats; we attach them alongside the model file so a
downstream consumer doesn't have to re-load the manifest.

Two formats:
* **ONNX** — broad runtime support (ONNX Runtime, TensorRT, browser via
  onnxruntime-web, mobile via NNAPI). The right answer for production
  edge deployment.
* **TorchScript** — narrower scope (any environment that links libtorch)
  but lossless for arbitrary PyTorch graphs. Good for C++ pipelines.

Each export is verified by running the original and exported models on
the same input and asserting agreement within float tolerance.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch


class ExportVerificationError(RuntimeError):
    """Raised when the exported model's predictions differ from the original."""


def _input_spec(result) -> Tuple[Tuple[int, ...], list[str]]:
    """Get (example_input_shape, input_variable_names) from a TrainResult."""
    compiled = result.compiled
    if compiled is None:
        raise ValueError("TrainResult has no compiled system; can't infer input shape")
    return (1, 1), [compiled.input_name]  # Phase-1+2: time-only ODE input


def to_onnx(
    result,
    path: str | Path,
    *,
    opset: int = 17,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> Path:
    """Export the trained network to ONNX. Verifies round-trip.

    Parameters:
        result: a :class:`TrainResult` from :func:`pinn_engine.core.trainer.train`.
        path: output ``.onnx`` file path.
        opset: ONNX opset version. 17 covers everything PINN networks need.
        rtol, atol: tolerance for the round-trip check (vs. in-process model).

    Returns:
        The path written. A sibling ``<path>.json`` is also written with
        the discovered parameters and metadata.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    network = result.network
    if network is None:
        raise ValueError("TrainResult has no network; nothing to export")

    network = network.cpu().eval()
    input_shape, input_names = _input_spec(result)
    example = torch.zeros(*input_shape, dtype=torch.float32)

    torch.onnx.export(
        network,
        example,
        path.as_posix(),
        input_names=input_names,
        output_names=result.compiled.state_names if result.compiled else ["y"],
        opset_version=opset,
        dynamic_axes={input_names[0]: {0: "batch"}},
    )

    # Verify round-trip
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(
            "onnxruntime is required for round-trip verification. "
            "pip install onnxruntime"
        ) from e

    sess = ort.InferenceSession(path.as_posix(), providers=["CPUExecutionProvider"])
    test = torch.randn(8, input_shape[1], dtype=torch.float32)
    with torch.no_grad():
        torch_out = network(test).detach().cpu().numpy()
    onnx_out = sess.run(None, {input_names[0]: test.numpy()})[0]

    if not np.allclose(torch_out, onnx_out, rtol=rtol, atol=atol):
        max_abs = float(np.abs(torch_out - onnx_out).max())
        raise ExportVerificationError(
            f"ONNX round-trip differs from torch by max-abs {max_abs:.2e}"
        )

    # Sidecar JSON with the discovered parameters + metadata.
    sidecar = path.with_suffix(".json")
    payload = {
        "format": "onnx",
        "opset": opset,
        "input_variables": input_names,
        "output_variables": result.compiled.state_names if result.compiled else [],
        "discovered_parameters": dict(result.final_params),
        "run_id": result.run_id,
        "torch_version": torch.__version__,
    }
    sidecar.write_text(json.dumps(payload, indent=2))
    return path


def to_torchscript(
    result,
    path: str | Path,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> Path:
    """Export the trained network to TorchScript via :func:`torch.jit.trace`.

    Verifies round-trip against the in-process model. Sidecar JSON written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    network = result.network
    if network is None:
        raise ValueError("TrainResult has no network; nothing to export")

    network = network.cpu().eval()
    input_shape, input_names = _input_spec(result)
    example = torch.zeros(*input_shape, dtype=torch.float32)

    scripted = torch.jit.trace(network, example)
    scripted.save(path.as_posix())

    # Verify round-trip.
    loaded = torch.jit.load(path.as_posix())
    test = torch.randn(8, input_shape[1], dtype=torch.float32)
    with torch.no_grad():
        torch_out = network(test).detach().cpu().numpy()
        scripted_out = loaded(test).detach().cpu().numpy()

    if not np.allclose(torch_out, scripted_out, rtol=rtol, atol=atol):
        max_abs = float(np.abs(torch_out - scripted_out).max())
        raise ExportVerificationError(
            f"TorchScript round-trip differs from torch by max-abs {max_abs:.2e}"
        )

    sidecar = path.with_suffix(".json")
    payload = {
        "format": "torchscript",
        "input_variables": input_names,
        "output_variables": result.compiled.state_names if result.compiled else [],
        "discovered_parameters": dict(result.final_params),
        "run_id": result.run_id,
        "torch_version": torch.__version__,
    }
    sidecar.write_text(json.dumps(payload, indent=2))
    return path


__all__ = ["to_onnx", "to_torchscript", "ExportVerificationError"]
