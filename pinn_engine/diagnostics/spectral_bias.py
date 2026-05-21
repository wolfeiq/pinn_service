"""Track high-frequency content of the network output over training.

PINNs are biased toward smooth functions — high-frequency components emerge
later in training. Watching the FFT magnitude across epochs makes that bias
visible and helps diagnose underfitting on oscillatory targets.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch

from pinn_engine.diagnostics.callbacks import DiagnosticCallback


class SpectralBias(DiagnosticCallback):
    """FFT magnitude of the network output sampled on a fixed grid."""

    name = "spectral_bias"

    def __init__(self, every_n_epochs: int = 200, n_points: int = 512):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.n_points = n_points
        self._snapshots: List[dict] = []
        self._t = None

    def on_train_start(self, trainer, pl_module):
        problem = getattr(pl_module, "problem", None)
        if problem is None:
            return
        td = problem.temporal_domain
        var = td.variables[0]
        lo, hi = td.range_[var]
        # Build an input tensor matching the network's expected input shape.
        # For ODE problems that's (N, 1); for PDE problems we sweep along the
        # temporal axis and fix spatial coords at their midpoint (so the FFT
        # reflects time-domain content).
        compiled = getattr(pl_module, "_compiled_system", None)
        input_names = list(getattr(compiled, "input_names", ()) or [var])
        if len(input_names) == 1:
            self._t = torch.linspace(lo, hi, self.n_points, dtype=torch.float32).reshape(-1, 1)
        else:
            spatial = getattr(problem, "spatial_domain", None)
            t_col = torch.linspace(lo, hi, self.n_points, dtype=torch.float32)
            cols = []
            for n_in in input_names:
                if n_in == var:
                    cols.append(t_col)
                else:
                    s_lo, s_hi = spatial.range_[n_in]
                    cols.append(torch.full_like(t_col, 0.5 * (s_lo + s_hi)))
            self._t = torch.stack(cols, dim=1)

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch % self.every_n_epochs) != 0 or self._t is None:
            return
        net = getattr(pl_module, "model", None) or getattr(pl_module, "_model", None)
        if net is None:
            return
        device = next(net.parameters()).device
        t = self._t.to(device)
        with torch.no_grad():
            y = net(t).cpu().numpy()  # (N, output_dim)
        mags = []
        for col in range(y.shape[1]):
            spec = np.fft.rfft(y[:, col] - y[:, col].mean())
            mags.append(np.abs(spec).tolist())
        self._snapshots.append({"epoch": int(trainer.current_epoch), "mag": mags})
        self.output = {"snapshots": self._snapshots, "n_points": self.n_points}
