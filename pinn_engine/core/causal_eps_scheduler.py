"""Wang 2022 §3.2 ε-annealing for CausalPINN, with collapse-recovery.

The temporal causal weight is ω_i = exp(-ε · Σ_{k<i} L_r(t_k)). The tension:

  * Too-large ε → ω collapses to 0 → physics term is silently muted (the
    PINA default ε=100 fails this way on wave-equation-scale residuals,
    and so does even ε=1 when a random network produces residuals ~1e7).
  * Too-small ε → causal ordering vanishes; degenerates to vanilla PINN.

Wang's recipe assumes well-scaled residuals (O(1)) and anneals ε *up* as
the network converges. We add the symmetric move: if min(ω) collapses
below `collapse_floor`, *divide* ε by `multiplier` until causal weighting
is non-trivial again. This auto-discovers the right scale regardless of
residual magnitude at init.

Original reference: Wang, Sankaran, Perdikaris, *Respecting causality for
training physics-informed neural networks.* CMAME 421 (2024) 116813,
arXiv:2203.07404.
"""
from __future__ import annotations

import pytorch_lightning as pl


class CausalEpsAnnealer(pl.Callback):
    def __init__(
        self,
        eps_max: float = 100.0,
        eps_min: float = 1e-12,
        threshold: float = 1e-2,
        collapse_floor: float = 1e-3,
        multiplier: float = 10.0,
        min_epochs_between_bumps: int = 20,
    ):
        super().__init__()
        self.eps_max = float(eps_max)
        self.eps_min = float(eps_min)
        self.threshold = float(threshold)
        self.collapse_floor = float(collapse_floor)
        self.multiplier = float(multiplier)
        self.min_epochs_between_bumps = int(min_epochs_between_bumps)
        self._last_change_epoch = -10**9
        # (epoch, eps, max_bucket, min_weight, action)
        self.history: list[tuple[int, float, float, float, str]] = []

    def on_train_epoch_end(self, trainer, pl_module):
        time_loss = getattr(pl_module, "_last_causal_time_loss", None)
        weights = getattr(pl_module, "_last_causal_weights", None)
        if time_loss is None or weights is None:
            return
        max_bucket = float(time_loss.max())
        min_weight = float(weights.min())
        current_eps = float(getattr(pl_module, "_eps", 0.0))

        action = "hold"
        cooldown_ok = (
            trainer.current_epoch - self._last_change_epoch
            >= self.min_epochs_between_bumps
        )

        if min_weight < self.collapse_floor and current_eps > self.eps_min:
            # Collapse: residuals too large for current ε. Shrink immediately
            # (no cooldown — collapse means we get zero gradient).
            new_eps = max(current_eps / self.multiplier, self.eps_min)
            pl_module._eps = new_eps
            self._last_change_epoch = trainer.current_epoch
            action = f"shrink→{new_eps:.2e}"
        elif (
            cooldown_ok
            and max_bucket < self.threshold
            and current_eps < self.eps_max
        ):
            # Converged enough at this ε — anneal up per Wang 2022.
            new_eps = min(current_eps * self.multiplier, self.eps_max)
            pl_module._eps = new_eps
            self._last_change_epoch = trainer.current_epoch
            action = f"grow→{new_eps:.2e}"

        self.history.append((
            trainer.current_epoch, current_eps, max_bucket, min_weight, action
        ))
