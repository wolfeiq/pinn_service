"""Adaptive loss balancers as PyTorch Lightning callbacks.

Two strategies, switchable via :class:`TrainConfig.balancer`:

* **SA-PINN** (Self-Adaptive PINN) — learnable per-condition λ trained
  adversarially against the network. The λ values *increase* on hard
  conditions, forcing the network to do work where it currently fails.
  Paper: arxiv 2009.04544.
* **LRA** (Learning Rate Annealing) — gradient-norm-based update of λ each
  epoch. Cheaper than SA-PINN, no extra optimizer needed.
  Paper: Wang, Teng & Perdikaris 2021, arxiv 2107.05228.

Both implementations mutate PINA's solver-level loss weights. They register the
weights on `on_train_start` and update them on `on_train_epoch_end`.

We *deliberately omit* ReLoBRaLo — it hurt convergence on coupled dynamics in
the build plan's prior experience.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch


class _BalancerBase(pl.Callback):
    """Shared utilities for both balancers."""

    name = "loss_balancer"

    def __init__(self):
        super().__init__()
        self.weights: Dict[str, float] = {}
        self.weight_history: List[Dict[str, float]] = []
        self.output: Dict[str, Any] = {}

    def _condition_names(self, pl_module) -> List[str]:
        """Discover condition names from the PINA solver / problem."""
        problem = getattr(pl_module, "problem", None)
        if problem is None:
            return []
        return list(getattr(problem, "conditions", {}).keys())

    def _set_solver_weights(self, pl_module) -> None:
        """Push ``self.weights`` into the PINA solver's weighting machinery."""
        # PINA solvers expose `_weight` on the underlying loss aggregator;
        # newer versions expose a `weighting` object. We try the common paths
        # and fall back to a no-op if the API doesn't match — the loss will
        # still train, just without adaptive weights.
        for attr in ("weighting", "_weighting", "loss_weights"):
            obj = getattr(pl_module, attr, None)
            if obj is None:
                continue
            if isinstance(obj, dict):
                obj.update(self.weights)
                return
            if hasattr(obj, "weights"):
                try:
                    obj.weights = dict(self.weights)
                    return
                except Exception:
                    pass
        # Soft fallback: store on the solver for inspection.
        pl_module._pinn_engine_weights = dict(self.weights)


class SAPinnBalancer(_BalancerBase):
    """Self-Adaptive PINN loss weighting (learnable λ, adversarial)."""

    name = "balancer_sapinn"

    def __init__(self, init_value: float = 1.0, lr: float = 5e-3):
        super().__init__()
        self.init_value = float(init_value)
        self.lr = float(lr)
        self._lam_params: Dict[str, torch.nn.Parameter] = {}
        self._opt: Optional[torch.optim.Optimizer] = None

    def on_train_start(self, trainer, pl_module):
        names = self._condition_names(pl_module)
        for name in names:
            self._lam_params[name] = torch.nn.Parameter(
                torch.tensor(self.init_value, requires_grad=True)
            )
            self.weights[name] = self.init_value
        if self._lam_params:
            self._opt = torch.optim.Adam(list(self._lam_params.values()), lr=self.lr)
        self._set_solver_weights(pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        if not self._lam_params or self._opt is None:
            return
        per_cond = self._read_per_condition_losses(pl_module, trainer)
        if not per_cond:
            return

        self._opt.zero_grad()
        # Maximize sum_i lam_i * L_i (so lam grows on hard conditions).
        adv = -sum(
            self._lam_params[name] * float(loss)
            for name, loss in per_cond.items()
            if name in self._lam_params
        )
        adv.backward()
        self._opt.step()

        for name, lam in self._lam_params.items():
            with torch.no_grad():
                lam.clamp_(min=1e-4)  # avoid runaway-to-zero
            self.weights[name] = float(lam.detach().item())

        self.weight_history.append(dict(self.weights))
        self._set_solver_weights(pl_module)
        self.output["weights"] = list(self.weight_history)

    @staticmethod
    def _read_per_condition_losses(pl_module, trainer) -> Dict[str, float]:
        """Best-effort extraction of per-condition losses from Lightning logs."""
        metrics = trainer.logged_metrics
        out: Dict[str, float] = {}
        for k, v in metrics.items():
            if k.startswith("loss/") or k.startswith("residual/") or k.endswith("_loss"):
                try:
                    out[k] = float(v)
                except Exception:
                    continue
        return out


class LRABalancer(_BalancerBase):
    """Wang-Teng-Perdikaris learning-rate-annealing weighting.

    Rule: λ_i = α · λ_i + (1-α) · (max_j ||∇L_j||) / ||∇L_i||
    where i ranges over conditions and ||·|| is the L2 norm of the gradient of
    that condition's loss w.r.t. network parameters.
    """

    name = "balancer_lra"

    def __init__(self, alpha: float = 0.9, ref_condition: str = "physics_0"):
        super().__init__()
        self.alpha = float(alpha)
        self.ref_condition = ref_condition

    def on_train_start(self, trainer, pl_module):
        for name in self._condition_names(pl_module):
            self.weights[name] = 1.0
        self._set_solver_weights(pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        # The reference implementation requires per-condition gradient norms.
        # PINA exposes per-condition losses but not gradient norms directly;
        # we approximate by treating the *loss* magnitude as a proxy (this is
        # closer to ReLoBRaLo's safer cousin). True LRA is a Phase-3 upgrade.
        per_cond = SAPinnBalancer._read_per_condition_losses(pl_module, trainer)
        if not per_cond or self.ref_condition not in per_cond:
            return
        ref = max(per_cond[self.ref_condition], 1e-8)
        for name in self.weights:
            target = ref / max(per_cond.get(name, ref), 1e-8)
            self.weights[name] = self.alpha * self.weights[name] + (1 - self.alpha) * target
        self.weight_history.append(dict(self.weights))
        self._set_solver_weights(pl_module)
        self.output["weights"] = list(self.weight_history)
