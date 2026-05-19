"""Extra pruning callbacks for AutoML trials.

The Optuna integration ships :class:`PyTorchLightningPruningCallback` which
handles intermediate-value reporting + Hyperband / Median pruning. We add two
catch-all callbacks that abort a trial when something concretely bad happens:

* :class:`NanGuard` — NaN loss → trial is unrecoverable; raise to abort.
* :class:`ParamDivergenceGuard` — any unknown blows past 10× its bound; the
  optimization has lost the plot and continuing wastes compute.

These complement Hyperband's resource ladder — Hyperband only knows about
the monitored metric, so it can't see "the unknown drifted to 10^9 while loss
crept down" type failures.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import optuna
import pytorch_lightning as pl


class NanGuard(pl.Callback):
    name = "nan_guard"

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if loss is None:
            return
        try:
            val = float(loss)
        except Exception:
            return
        if math.isnan(val) or math.isinf(val):
            raise optuna.TrialPruned(
                f"NaN/Inf loss at epoch={trainer.current_epoch}, batch={batch_idx}"
            )


class ParamDivergenceGuard(pl.Callback):
    name = "param_divergence_guard"

    def __init__(self, bounds: Dict[str, Tuple[float, float]], factor: float = 10.0):
        super().__init__()
        self.bounds = bounds
        self.factor = float(factor)

    def on_train_epoch_end(self, trainer, pl_module):
        problem = getattr(pl_module, "problem", None)
        if problem is None or not hasattr(problem, "unknown_parameters"):
            return
        for name, p in problem.unknown_parameters.items():
            lo, hi = self.bounds.get(name, (-math.inf, math.inf))
            mid = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo)
            val = float(p.detach().cpu().reshape(-1)[0].item())
            if abs(val - mid) > self.factor * half:
                raise optuna.TrialPruned(
                    f"Param {name!r} diverged: value={val:.3g} outside "
                    f"{self.factor}× the declared bound {(lo, hi)!r}"
                )
