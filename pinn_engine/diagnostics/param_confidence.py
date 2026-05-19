"""Running mean/std of each discovered unknown over training.

The simplest of the four callbacks — and arguably the most important.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import torch

from pinn_engine.diagnostics.callbacks import DiagnosticCallback


class ParamConfidence(DiagnosticCallback):
    """Logs each ``problem.unknown_parameters[name]`` every epoch."""

    name = "param_confidence"

    def __init__(self, every_n_epochs: int = 1):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self._history: Dict[str, List[float]] = defaultdict(list)
        self._epochs: List[int] = []

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch % self.every_n_epochs) != 0:
            return
        problem = getattr(pl_module, "problem", None)
        if problem is None or not hasattr(problem, "unknown_parameters"):
            return
        self._epochs.append(int(trainer.current_epoch))
        for name, p in problem.unknown_parameters.items():
            with torch.no_grad():
                val = float(p.detach().cpu().reshape(-1)[0].item())
            self._history[name].append(val)

        # Refresh ``self.output`` so an external reader can pick up the latest.
        self.output = {
            "epochs": list(self._epochs),
            "history": {k: list(v) for k, v in self._history.items()},
        }

    def on_train_end(self, trainer, pl_module):
        if not self._history:
            return
        summary = {}
        for name, vals in self._history.items():
            if len(vals) >= 10:
                tail = vals[-max(1, len(vals) // 10) :]
                t = torch.tensor(tail)
                summary[name] = {
                    "final": vals[-1],
                    "tail_mean": float(t.mean()),
                    "tail_std": float(t.std(unbiased=False) if t.numel() > 1 else 0.0),
                }
            else:
                summary[name] = {"final": vals[-1], "tail_mean": vals[-1], "tail_std": 0.0}
        self.output["summary"] = summary
