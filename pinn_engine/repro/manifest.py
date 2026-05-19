"""Reproducibility manifests: every run writes one, every run can be verified.

A manifest captures *just enough* to re-run and assert the same discovered
parameters: git SHA, seed, equation/config/data hashes, library versions,
discovered parameters with uncertainty, and a pointer to callback outputs.
"""
from __future__ import annotations

import datetime
import json
import os
import platform
import subprocess
import sys
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl

from pinn_engine.repro.hashing import hash_config, hash_data


MANIFEST_DIR = Path(__file__).resolve().parents[2] / "manifests"


@dataclass
class Manifest:
    run_id: str
    timestamp: str
    git_sha: str
    git_dirty: bool
    template: str
    template_hash: str
    data_hash: str
    config_hash: str
    seed: int
    torch_version: str
    python_version: str
    pina_version: str
    lightning_version: str
    automl_study: Optional[str] = None
    trial_number: Optional[int] = None
    final_params: Dict[str, Dict[str, float]] = field(default_factory=dict)
    final_loss: float = float("nan")
    convergence_epoch: int = -1
    callback_outputs_path: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def _git_info() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=str(MANIFEST_DIR.parent),
        ).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, cwd=str(MANIFEST_DIR.parent),
        ).strip())
        return sha, dirty
    except Exception:
        return "no-git", False


def _pkg_version(name: str) -> str:
    try:
        import pkg_resources
        return pkg_resources.get_distribution(name).version
    except Exception:
        return "unknown"


def write_manifest(
    template: str,
    result,
    data: Dict[str, Any],
    automl_study: Optional[str] = None,
    trial_number: Optional[int] = None,
    out_dir: Optional[Path] = None,
) -> Path:
    """Build a :class:`Manifest` from a :class:`TrainResult` and write it to disk.

    Returns the path to the JSON file.
    """
    out_dir = Path(out_dir) if out_dir is not None else MANIFEST_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    sha, dirty = _git_info()

    # Discovered parameters: include tail-mean/std from ParamConfidence if available
    final_params: Dict[str, Dict[str, float]] = {}
    pc = result.callback_outputs.get("param_confidence") if hasattr(result, "callback_outputs") else None
    pc_summary = (pc or {}).get("summary") if isinstance(pc, dict) else None
    for name, val in result.final_params.items():
        if pc_summary and name in pc_summary:
            final_params[name] = {
                "final": float(val),
                "mean": float(pc_summary[name]["tail_mean"]),
                "std": float(pc_summary[name]["tail_std"]),
            }
        else:
            final_params[name] = {"final": float(val), "mean": float(val), "std": 0.0}

    m = Manifest(
        run_id=result.run_id,
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        git_sha=sha,
        git_dirty=dirty,
        template=template,
        template_hash=result.compiled.equation_hash if result.compiled else "unknown",
        data_hash=hash_data(data),
        config_hash=hash_config(result.config) if result.config else "unknown",
        seed=result.config.seed if result.config else -1,
        torch_version=_pkg_version("torch"),
        python_version=platform.python_version(),
        pina_version=_pkg_version("pina-mathlab"),
        lightning_version=_pkg_version("pytorch-lightning"),
        automl_study=automl_study,
        trial_number=trial_number,
        final_params=final_params,
        final_loss=float(result.final_loss) if result.final_loss == result.final_loss else float("nan"),
    )

    # Save callback raw arrays alongside the manifest as an npz.
    cb_path = out_dir / f"{m.run_id}_callbacks.npz"
    try:
        np.savez(cb_path, **{k: np.asarray(v, dtype=object) for k, v in result.callback_outputs.items()})
        m.callback_outputs_path = str(cb_path.name)
    except Exception:
        m.callback_outputs_path = None

    path = out_dir / f"{m.run_id}.json"
    with open(path, "w") as f:
        json.dump(asdict(m), f, indent=2, default=str)
    return path


def read_manifest(path: str) -> Manifest:
    with open(path) as f:
        d = json.load(f)
    return Manifest(**d)


class ManifestWriterCallback(pl.Callback):
    """Optional callback that writes the manifest on ``on_train_end``.

    The CLI calls :func:`write_manifest` directly with the TrainResult, but
    when running outside the CLI you can attach this callback for the same
    effect.
    """

    name = "manifest_writer"

    def __init__(self, template: str = "ad_hoc"):
        super().__init__()
        self.template = template
        self.output: Dict[str, Any] = {}

    def on_train_end(self, trainer, pl_module):
        # The trainer entry point owns the manifest in this engine; this
        # callback is a convenience for users who skip the entry point.
        self.output["note"] = "Use write_manifest(result, data) for full manifest."
