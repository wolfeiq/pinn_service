"""Live training status — for the dashboard to poll while a run is in flight.

Writes ``manifests/live_<run_id>.json`` every ``write_every`` epochs with
the current epoch, train loss, and discovered parameter values. The
dashboard's Train view polls this file on a short interval and renders
a live loss curve / parameter trajectory.

The file is replaced atomically (write to tmp, ``os.replace``) so a
concurrent reader can't see a half-written JSON.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from pinn_engine.diagnostics.callbacks import DiagnosticCallback


class LiveStatusCallback(DiagnosticCallback):
    """Periodically dump ``{epoch, loss, params}`` to a JSON file."""

    name = "live_status"

    def __init__(
        self,
        run_id: str,
        out_dir: Optional[Path] = None,
        write_every: int = 10,
    ):
        super().__init__()
        self.run_id = run_id
        self.out_dir = Path(out_dir) if out_dir else (
            Path(__file__).resolve().parents[2] / "manifests"
        )
        self.write_every = int(write_every)
        self._history: list[Dict[str, Any]] = []

    def _write(self, payload: Dict[str, Any]) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"live_{self.run_id}.json"
        fd, tmp = tempfile.mkstemp(prefix=f"live_{self.run_id}_", dir=self.out_dir)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = int(trainer.current_epoch)
        if (epoch + 1) % self.write_every != 0 and epoch != 0:
            return
        loss = None
        for k in ("train_loss_epoch", "train_loss", "loss_epoch", "loss"):
            if k in trainer.callback_metrics:
                try:
                    loss = float(trainer.callback_metrics[k])
                    break
                except Exception:
                    pass
        problem = getattr(pl_module, "problem", None)
        params: Dict[str, float] = {}
        if problem is not None and hasattr(problem, "unknown_parameters"):
            for n, p in problem.unknown_parameters.items():
                try:
                    params[n] = float(p.detach().cpu().reshape(-1)[0].item())
                except Exception:
                    pass
        entry = {"epoch": epoch, "loss": loss, "params": params}
        self._history.append(entry)
        payload = {
            "run_id": self.run_id,
            "history": self._history,
            "latest": entry,
            "status": "running",
        }
        self._write(payload)
        self.output = payload

    def on_train_end(self, trainer, pl_module):
        # Mark complete so the dashboard knows it can stop polling.
        try:
            path = self.out_dir / f"live_{self.run_id}.json"
            if path.exists():
                payload = json.loads(path.read_text())
                payload["status"] = "complete"
                path.write_text(json.dumps(payload))
        except Exception:
            pass
