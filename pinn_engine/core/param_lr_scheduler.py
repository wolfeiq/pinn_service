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

    def __init__(
        self,
        max_epochs: int,
        min_scale: float = 0.05,
        trigger_below: float | None = None,
        trigger_param: str | None = None,
        taper_epochs: int | None = None,
    ):
        super().__init__()
        self.max_epochs = max(int(max_epochs), 1)
        self.min_scale = float(min_scale)
        # Two-phase mode: stay at full LR while the watched unknown is above
        # ``trigger_below``; once it crosses, cosine-taper to min_scale.
        # ``trigger_param`` selects which unknown to watch (by name);
        # default = first unknown found. ``taper_epochs`` sets how many epochs
        # the post-trigger cosine takes to reach min_scale (default = all
        # remaining epochs). A short taper brakes fast — necessary here because
        # the unknown descends ~0.2/epoch at trigger and a full-budget cosine
        # stays near full LR too long, overshooting truth before it bites.
        self.trigger_below = float(trigger_below) if trigger_below is not None else None
        self.trigger_param = trigger_param
        self.taper_epochs = int(taper_epochs) if taper_epochs is not None else None
        self._base_lr: float | None = None
        self._unknowns_group_idx: int | None = None
        self._trigger_epoch: int | None = None
        self._watched_param = None

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
        # PINA wraps the optimizer in a ConstantLR warmup (factor 1/3 for the
        # first 5 epochs). Reading param_groups[idx]["lr"] here captures the
        # warmup-discounted value (1/3 of intended), which we'd then pin —
        # silently running the unknowns 3x too slow. Prefer the LR scheduler's
        # stored base_lrs[idx] (the true target LR) when available.
        base = trainer.optimizers[0].param_groups[idx]["lr"]
        for sch_cfg in (getattr(trainer, "lr_scheduler_configs", None) or []):
            sch = getattr(sch_cfg, "scheduler", None)
            base_lrs = getattr(sch, "base_lrs", None)
            if base_lrs is not None and idx < len(base_lrs):
                base = base_lrs[idx]
                break
        self._base_lr = base
        # Bind the watched parameter for two-phase mode.
        if self.trigger_below is not None:
            problem = getattr(pl_module, "problem", None)
            unknowns = getattr(problem, "unknown_parameters", None) if problem else None
            if unknowns:
                if self.trigger_param is not None and self.trigger_param in unknowns:
                    self._watched_param = unknowns[self.trigger_param]
                else:
                    self._watched_param = next(iter(unknowns.values()))

    def on_train_epoch_start(self, trainer, pl_module):
        if self._unknowns_group_idx is None or self._base_lr is None:
            return
        ep = trainer.current_epoch
        # Two-phase: check trigger before computing cosine progress.
        if self.trigger_below is not None and self._trigger_epoch is None:
            if self._watched_param is not None:
                current = float(self._watched_param.detach().item())
                if current <= self.trigger_below:
                    self._trigger_epoch = ep
            if self._trigger_epoch is None:
                # Pre-trigger: stay SILENT — do not write the LR. PINA's
                # ConstantLR warmup (1/3 for 5 epochs, then full) governs this
                # phase. Writing base_lr here would (a) override the warmup,
                # unleashing full LR against a cold network → overshoot (run
                # #15), and (b) inflate the value the warmup milestone ×3s at
                # epoch 5 → a 1.5 spike. Letting the warmup run is what made
                # the only clean escape (run #10) descend gradually.
                return
        # Compute cosine progress. In two-phase mode, the cosine spans
        # [trigger_epoch, trigger_epoch + taper_epochs] (or to max_epochs if
        # taper_epochs is None) so the taper happens over the post-escape
        # window. A short span brakes fast.
        start = self._trigger_epoch if self._trigger_epoch is not None else 0
        if self._trigger_epoch is not None and self.taper_epochs is not None:
            span = max(self.taper_epochs, 1)
        else:
            span = max(self.max_epochs - start, 1)
        progress = min((ep - start) / span, 1.0)
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        scale = self.min_scale + (1.0 - self.min_scale) * cos_factor
        trainer.optimizers[0].param_groups[self._unknowns_group_idx]["lr"] = (
            self._base_lr * scale
        )
