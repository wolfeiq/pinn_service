"""Per-epoch dump of inverse-problem unknowns to a JSON file.

PINA doesn't store ``problem.unknown_parameters`` in the solver's
``state_dict``, so a SIGKILL or OOM mid-training loses every recovered
parameter value — the only place they exist is the live ``problem``
object that dies with the process. The last engine run (Cosserat
inverse, May 24 2026) got OOM-killed at epoch 68/100 and we couldn't
recover the E_unit estimate from the checkpoints.

This callback writes ``{name: value}`` to a JSON file at the end of
every training epoch so the latest recovered values survive a crash.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl


class UnknownsDumper(pl.Callback):
    name = "unknowns_dumper"

    def __init__(self, output_path: str | Path):
        super().__init__()
        self.output_path = Path(output_path)
        self.history: list[dict] = []
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _dump(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> Optional[dict]:
        problem = getattr(pl_module, "problem", None)
        if problem is None or not hasattr(problem, "unknown_parameters"):
            return None
        values = {
            name: float(p.detach().cpu().item())
            for name, p in problem.unknown_parameters.items()
        }
        record = {"epoch": int(trainer.current_epoch), "values": values}
        self.history.append(record)
        payload = {"latest": record, "history": self.history}
        tmp = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.output_path)
        return record

    def on_train_epoch_end(self, trainer, pl_module):
        self._dump(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        self._dump(trainer, pl_module)
