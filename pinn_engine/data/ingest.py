"""Sensor data ingestion: CSV / npz / HDF5 → ``dict[str, (t, obs)]``.

The exact format is intentionally permissive — most users land here from a CSV
they exported from their robot's logger. We give them three entry points:

* :func:`load_csv(path, t_col, sensor_cols)` — common case.
* :func:`load_npz(path)` — ``.npz`` with keys matching sensor names.
* :func:`load_hdf5(path)` — HDF5 with ``t`` and per-sensor datasets.

:func:`validate_against_system` ensures the loaded data has the sensors the
system expects.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pinn_engine.dsl.system import System


def load_csv(
    path: str,
    t_col: str = "t",
    sensor_cols: Optional[List[str]] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load sensor data from a CSV. Time column is shared across all sensors.

    If ``sensor_cols`` is None, every non-``t_col`` column is treated as a sensor
    (and its name becomes the sensor name).
    """
    import pandas as pd

    df = pd.read_csv(path)
    if t_col not in df.columns:
        raise KeyError(f"Time column {t_col!r} not in CSV; have {list(df.columns)}")
    t = df[t_col].to_numpy(dtype=np.float32)
    cols = sensor_cols if sensor_cols is not None else [c for c in df.columns if c != t_col]
    return {col: (t, df[col].to_numpy(dtype=np.float32)) for col in cols}


def load_npz(path: str, t_key: str = "t") -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load sensor data from a ``.npz``. ``t_key`` is the shared time array."""
    arr = np.load(path)
    if t_key not in arr.files:
        raise KeyError(f"Time array {t_key!r} not in npz; have {arr.files}")
    t = arr[t_key].astype(np.float32)
    return {k: (t, arr[k].astype(np.float32)) for k in arr.files if k != t_key}


def load_hdf5(path: str, t_key: str = "t") -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load sensor data from an HDF5 file with ``t`` and per-sensor datasets."""
    import h5py

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    with h5py.File(path, "r") as f:
        if t_key not in f:
            raise KeyError(f"Time dataset {t_key!r} not in HDF5; have {list(f.keys())}")
        t = np.asarray(f[t_key], dtype=np.float32)
        for k in f.keys():
            if k == t_key:
                continue
            out[k] = (t, np.asarray(f[k], dtype=np.float32))
    return out


def load_data(path: str, **kwargs) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Dispatch by file extension."""
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return load_csv(path, **kwargs)
    if suffix == ".npz":
        return load_npz(path, **kwargs)
    if suffix in (".h5", ".hdf5"):
        return load_hdf5(path, **kwargs)
    raise ValueError(f"Unsupported data file extension {suffix!r}; use .csv, .npz, or .h5/.hdf5")


def validate_against_system(
    data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    system: System,
) -> None:
    """Check that every declared sensor has a corresponding key in ``data``."""
    missing = [s.name for s in system.sensors if s.name not in data]
    if missing:
        raise KeyError(
            f"Data is missing sensors declared by the System: {missing!r}. "
            f"Provided keys: {list(data.keys())}"
        )
    for name, (t, x) in data.items():
        if np.shape(t)[0] != np.shape(x)[0]:
            raise ValueError(
                f"Sensor {name!r}: time and observation arrays have different "
                f"lengths ({np.shape(t)[0]} vs {np.shape(x)[0]})"
            )
