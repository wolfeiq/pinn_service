"""Cosine annealing for the unknown-parameters optimizer group.

Adam's per-parameter normalization makes inverse parameters hard to move
when bounds are wide. ``param_lr_scale`` (in :class:`TrainConfig`) gives
unknowns their own optimizer group with a larger LR — fast traversal of
the bounds. But a constant high LR overshoots the true value and
oscillates around it.

This callback anneals the unknowns' LR with a cosine schedule across the
Adam phase::

    lr(epoch) = base * (min_scale + (1 - min_scale) * 0.5*(1 + cos(π·t)))
              where t = epoch / max_epochs

At ``epoch=0`` LR is ``base``; at ``epoch=max_epochs`` LR is ``base * min_scale``.

The network's own ``param_group`` is untouched — only the unknowns'
group gets re-scheduled. Identified by checking which group contains a
torch.nn.Parameter listed in ``problem.unknown_parameters.values()``.
"""
from __future__ import annotations

import math

import pytorch_lightning as pl


class UnknownsParamLRScheduler(pl.Callback):
    """Cosine-anneal the unknowns' optimizer-group LR.

    Args:
        max_epochs: the Adam-phase epoch count (cosine completes over this).
        min_scale: floor scale at the end of training (e.g. ``0.05`` →
            LR ends at 5 % of its starting value).
    """

    name = "unknowns_lr_scheduler"

    def __init__(self, max_epochs: int, min_scale: float = 0.05):
        super().__init__()
        self.max_epochs = max(int(max_epochs), 1)
        self.min_scale = float(min_scale)
        self._base_lr: float | None = None
        self._unknowns_group_idx: int | None = None

    def _locate_unknowns_group(self, pl_module) -> int | None:
        """Find the optimizer's param_group whose params include unknowns."""
        problem = getattr(pl_module, "problem", None)
        if problem is None or not hasattr(problem, "unknown_parameters"):
            return None
        unk_ids = {id(p) for p in problem.unknown_parameters.values()}
        optimizers = pl_module.trainer.optimizers
        if not optimizers:
            return None
        opt = optimizers[0]
        for i, group in enumerate(opt.param_groups):
            for p in group["params"]:
                if id(p) in unk_ids:
                    return i
        return None

    def on_train_start(self, trainer, pl_module):
        idx = self._locate_unknowns_group(pl_module)
        if idx is None:
            return
        self._unknowns_group_idx = idx
        self._base_lr = trainer.optimizers[0].param_groups[idx]["lr"]

    def on_train_epoch_start(self, trainer, pl_module):
        if self._unknowns_group_idx is None or self._base_lr is None:
            return
        progress = min(trainer.current_epoch / self.max_epochs, 1.0)
        # Cosine from 1.0 (at progress=0) to min_scale (at progress=1).
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        scale = self.min_scale + (1.0 - self.min_scale) * cos_factor
        trainer.optimizers[0].param_groups[self._unknowns_group_idx]["lr"] = (
            self._base_lr * scale
        )
