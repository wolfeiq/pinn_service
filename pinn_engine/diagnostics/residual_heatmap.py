"""Where in the domain the physics is violated.

Every ``every_n_epochs`` we sample collocation points uniformly from the
problem's temporal domain, evaluate each compiled physics residual, and store
the L2 magnitude. Visualization is the dashboard's job — we just record.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
from pina import LabelTensor

from pinn_engine.diagnostics.callbacks import DiagnosticCallback


class ResidualHeatmap(DiagnosticCallback):
    """Per-collocation-point physics-residual magnitude over training."""

    name = "residual_heatmap"

    def __init__(self, every_n_epochs: int = 100, n_points: int = 256):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.n_points = n_points
        self._snapshots: List[dict] = []

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch % self.every_n_epochs) != 0:
            return
        problem = getattr(pl_module, "problem", None)
        net = getattr(pl_module, "model", None) or getattr(pl_module, "_model", None)
        if problem is None or net is None:
            return
        compiled = getattr(pl_module, "_compiled_system", None)
        if compiled is None:
            # Trainer didn't stash the compiled system on the solver; skip.
            return
        td = problem.temporal_domain
        var = td.variables[0]
        lo, hi = td.range_[var]
        device = next(net.parameters()).device
        t = torch.linspace(lo, hi, self.n_points, dtype=torch.float32, device=device).reshape(-1, 1)
        with torch.enable_grad():
            input_lt = LabelTensor(t.clone().requires_grad_(True), labels=[var])
            y = net(input_lt)
            output_lt = LabelTensor(y, labels=list(compiled.state_names))
            params_ = problem.unknown_parameters
            residuals_sq = []
            for r_fn in compiled.physics_residuals:
                r = r_fn(input_lt, output_lt, params_=params_)
                residuals_sq.append((r.detach().cpu().numpy() ** 2).reshape(-1))
        snap = {
            "epoch": int(trainer.current_epoch),
            "t": t.detach().cpu().numpy().reshape(-1).tolist(),
            "residual_l2": [np.sqrt(r).tolist() for r in residuals_sq],
        }
        self._snapshots.append(snap)
        self.output = {"snapshots": self._snapshots}
