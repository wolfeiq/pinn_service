"""Sensor residuals: measured - predicted per sensor over training."""
from __future__ import annotations

from typing import List

import numpy as np
import torch
from pina import LabelTensor

from pinn_engine.diagnostics.callbacks import DiagnosticCallback


class SensorResiduals(DiagnosticCallback):
    """Stores per-sensor (measured - predicted) snapshots periodically."""

    name = "sensor_residuals"

    def __init__(self, every_n_epochs: int = 100):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self._snapshots: List[dict] = []

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch % self.every_n_epochs) != 0:
            return
        compiled = getattr(pl_module, "_compiled_system", None)
        data_dict = getattr(pl_module, "_engine_data", None)
        net = getattr(pl_module, "model", None) or getattr(pl_module, "_model", None)
        if compiled is None or data_dict is None or net is None:
            return
        snap = {"epoch": int(trainer.current_epoch), "per_sensor": {}}
        for sens in compiled.sensors:
            if sens.name not in data_dict:
                continue
            t_arr, obs_arr = data_dict[sens.name]
            device = next(net.parameters()).device
            t_np = np.asarray(t_arr)
            if t_np.ndim == 2 and t_np.shape[1] > 1:
                # PDE input (N, n_inputs) — pass through verbatim.
                t = torch.as_tensor(t_np, dtype=torch.float32, device=device)
            else:
                t = torch.as_tensor(t_np, dtype=torch.float32, device=device).reshape(-1, 1)
            with torch.no_grad():
                y = net(t)
                output_lt = LabelTensor(y, labels=list(compiled.state_names))
                pred = compiled.sensor_observation_fns[sens.name](output_lt).cpu().numpy().reshape(-1)
            obs = np.asarray(obs_arr).reshape(-1)
            snap["per_sensor"][sens.name] = {
                "t": t_arr.tolist() if hasattr(t_arr, "tolist") else list(t_arr),
                "residual": (obs - pred).tolist(),
            }
        self._snapshots.append(snap)
        self.output = {"snapshots": self._snapshots}
