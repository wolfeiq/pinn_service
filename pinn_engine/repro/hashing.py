"""Deterministic hashing for equations, data, and configs.

We want two runs with identical inputs to share manifest content. Hashes are
SHA-256 hex strings — short to compare, collision-resistant for our purposes.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

import numpy as np


def sha256_of(payload: bytes) -> str:
    """SHA-256 hex digest of bytes."""
    return hashlib.sha256(payload).hexdigest()


def hash_system(system) -> str:
    """Hash the *compiled* system's structure (not its torch closures)."""
    compiled = system.compile() if hasattr(system, "compile") else system
    return compiled.equation_hash


def hash_data(data: Dict[str, Any]) -> str:
    """Hash the canonical bytes of a data dict.

    We sort sensor names, then concatenate the byte view of each numpy array
    in (t, obs) order. The result is stable across runs with the same arrays.
    """
    h = hashlib.sha256()
    for name in sorted(data.keys()):
        t, obs = data[name]
        h.update(name.encode())
        h.update(np.ascontiguousarray(t).astype(np.float32).tobytes())
        h.update(np.ascontiguousarray(obs).astype(np.float32).tobytes())
    return h.hexdigest()


def hash_config(config) -> str:
    """Hash a pydantic ``TrainConfig`` (or any dataclass-like) deterministically."""
    if hasattr(config, "model_dump"):
        payload = config.model_dump()
    elif hasattr(config, "dict"):
        payload = config.dict()
    elif hasattr(config, "__dict__"):
        payload = dict(config.__dict__)
    else:
        payload = config

    def _clean(o):
        # Convert non-JSON-serializable bits to strings deterministically.
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in sorted(o.items())}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (int, float, str, bool, type(None))):
            return o
        return str(o)

    return hashlib.sha256(json.dumps(_clean(payload), sort_keys=True).encode()).hexdigest()
