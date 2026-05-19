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


class TrainLossPruningCallback(pl.Callback):
    """Report ``train_loss_epoch`` to Optuna and prune on its judgment.

    Stand-in for ``PyTorchLightningPruningCallback`` which only fires on
    validation epochs — but PINN inverse problems typically have no
    val_dataloader, so the upstream callback silently never reports.

    This version reports the train loss every ``report_every`` epochs.
    """

    name = "train_loss_pruning"

    def __init__(self, trial, report_every: int = 50, monitor: str = "train_loss_epoch"):
        super().__init__()
        self.trial = trial
        self.report_every = int(report_every)
        self.monitor = monitor

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.report_every != 0:
            return
        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            # Try a couple of fallbacks before giving up silently.
            for fallback in ("train_loss", "loss", "loss_epoch"):
                if fallback in metrics:
                    val = float(metrics[fallback])
                    break
            else:
                return
        else:
            val = float(metrics[self.monitor])
        if math.isnan(val) or math.isinf(val):
            return  # NanGuard handles it
        self.trial.report(val, step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(
                f"Pruned at epoch {epoch} with {self.monitor}={val:.4g}"
            )
