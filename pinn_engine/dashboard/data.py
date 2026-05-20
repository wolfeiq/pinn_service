"""Manifest + callback NPZ + Optuna DB loaders for the dashboard.

Pure data-layer code — no Streamlit. Lets us unit-test these helpers
without spinning up a server.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def manifests_dir() -> Path:
    """Default manifests directory: ``<repo>/manifests``."""
    return Path(__file__).resolve().parents[2] / "manifests"


def list_manifests(directory: Optional[Path] = None) -> List[Path]:
    """Return all run manifests (``*.json``), most recent first."""
    directory = directory or manifests_dir()
    files = [p for p in directory.glob("*.json") if not p.name.startswith("optuna_")]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def list_optuna_dbs(directory: Optional[Path] = None) -> List[Path]:
    """Return all Optuna study databases (``optuna_*.db``)."""
    directory = directory or manifests_dir()
    return sorted(directory.glob("optuna_*.db"), key=lambda p: p.stat().st_mtime, reverse=True)


@dataclass
class RunData:
    """All available data for a single training run."""

    path: Path
    manifest: Dict[str, Any]
    callback_outputs: Dict[str, Any]   # per-callback dict; may be empty

    @property
    def run_id(self) -> str:
        return self.manifest.get("run_id", self.path.stem)

    @property
    def template(self) -> str:
        return self.manifest.get("template", "?")

    @property
    def final_params(self) -> Dict[str, Dict[str, float]]:
        return self.manifest.get("final_params", {})


def load_run(manifest_path: Path) -> RunData:
    """Load a manifest + its sidecar callback NPZ (if present)."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())

    cb_outputs: Dict[str, Any] = {}
    cb_rel = manifest.get("callback_outputs_path")
    if cb_rel:
        cb_path = manifest_path.parent / cb_rel
        if cb_path.exists():
            try:
                arr = np.load(cb_path, allow_pickle=True)
                for key in arr.files:
                    item = arr[key]
                    # The arrays were saved as dtype=object; unwrap the 0-d
                    # object array to get the underlying dict/list.
                    try:
                        cb_outputs[key] = item.item() if item.shape == () else item.tolist()
                    except Exception:
                        cb_outputs[key] = None
            except Exception:
                cb_outputs = {}

    return RunData(path=manifest_path, manifest=manifest, callback_outputs=cb_outputs)


def load_optuna_study(db_path: Path):
    """Load an Optuna study from its SQLite DB file. Returns the Study object."""
    import optuna
    name = db_path.stem.replace("optuna_", "")
    return optuna.load_study(
        study_name=name,
        storage=f"sqlite:///{db_path.absolute()}",
    )
