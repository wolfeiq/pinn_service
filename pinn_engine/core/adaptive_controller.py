"""Adaptive LR controller for inverse-problem unknowns.

Replaces hand-tuned ``param_lr_scale`` + the two-phase trigger/taper with a
single runtime control law: keep each unknown's per-epoch *velocity* inside a
healthy band by adapting its optimizer-group learning rate.

The three failure modes characterised on the Cosserat / diffusion problems all
reduce to "velocity out of band":

* **Frozen** (LR too small — e.g. Cosserat at lr_scale=1): velocity ≈ 0 while
  the network is still learning → *ramp LR up*.
* **Basin trap** (Cosserat's spurious E=2 plateau): velocity ≈ 0, but ramping
  LR escapes the (shallow) basin to a *better* solution — so the loss drops.
  At a *true* optimum, ramping the LR cannot reduce the loss further. That
  loss response — not velocity alone — is how the controller distinguishes
  "trapped" (keep ramping) from "converged" (latch and stop ramping). Velocity
  alone is insufficient: an unknown sitting exactly at truth also has ≈0
  velocity, and a velocity-only rule would wrongly keep ramping it off.
* **Overshoot / oscillation** (diffusion's early bounce, or full LR vs a cold
  network): velocity too high or sign-flipping → *brake LR*.

Velocity is measured relative to the unknown's bound width so the same band
works across differently-scaled unknowns. The controller is silent during
PINA's ConstantLR warmup (it is load-bearing — lets the network fit a coarse
solution before the unknown moves fast) and takes over afterwards.
"""
from __future__ import annotations

import math
from typing import Optional

import pytorch_lightning as pl


class AdaptiveUnknownsController(pl.Callback):
    """Velocity-band LR controller for the unknowns' optimizer group.

    Args:
        warmup_epochs: stay silent for this many epochs (let PINA's ConstantLR
            warmup run); start controlling afterwards.
        v_lo, v_hi: target band for per-epoch velocity, expressed as a fraction
            of each unknown's bound width. Below ``v_lo`` → too slow (ramp up);
            above ``v_hi`` → too fast (brake).
        lr_up, lr_down: multiplicative LR adjustments per epoch.
        min_mult, max_mult: clamp on the LR multiplier (relative to the base LR
            captured at warmup end).
        patience: while frozen, ramp the LR (probe) for this many epochs; if the
            loss has not improved over that probe, declare converged. A trap
            escapes within the probe (loss drops, resetting the counter); a true
            optimum does not.
        loss_eps: relative loss improvement (vs the best seen) that counts as
            "still productive" and resets the probe counter.
        converged_mult: LR multiplier to hold after convergence (small, so the
            unknown rests near the optimum instead of being ramped off it).
        stop_on_converge: if True, set ``trainer.should_stop`` once converged.
    """

    name = "adaptive_unknowns_controller"

    def __init__(
        self,
        warmup_epochs: int = 5,
        v_lo: float = 3e-3,
        v_hi: float = 4e-2,
        lr_up: float = 2.0,
        lr_down: float = 0.5,
        min_mult: float = 0.05,
        max_mult: float = 200.0,
        patience: int = 4,
        loss_eps: float = 1e-3,
        converged_mult: float = 0.2,
        stop_on_converge: bool = False,
    ):
        super().__init__()
        self.warmup_epochs = int(warmup_epochs)
        self.v_lo = float(v_lo)
        self.v_hi = float(v_hi)
        self.lr_up = float(lr_up)
        self.lr_down = float(lr_down)
        self.min_mult = float(min_mult)
        self.max_mult = float(max_mult)
        self.patience = int(patience)
        self.loss_eps = float(loss_eps)
        self.converged_mult = float(converged_mult)
        self.stop_on_converge = bool(stop_on_converge)
        self._best_loss: float = float("inf")

        self._group_idx: Optional[int] = None
        self._base_lr: Optional[float] = None
        self._params: dict = {}          # name -> torch.nn.Parameter
        self._ranges: dict = {}          # name -> bound width
        self._prev: dict = {}            # name -> previous value
        self._prev_v: dict = {}          # name -> previous signed velocity
        self._mult: float = 1.0
        self._stall: int = 0
        self._prev_loss: Optional[float] = None
        self.converged: bool = False
        self.history: list = []          # per-epoch telemetry for diagnostics

    # ------------------------------------------------------------------ setup
    def _locate_unknowns_group(self, pl_module) -> Optional[int]:
        problem = getattr(pl_module, "problem", None)
        if problem is None or not hasattr(problem, "unknown_parameters"):
            return None
        unk_ids = {id(p) for p in problem.unknown_parameters.values()}
        opts = pl_module.trainer.optimizers
        if not opts:
            return None
        for i, group in enumerate(opts[0].param_groups):
            if any(id(p) in unk_ids for p in group["params"]):
                return i
        return None

    def on_train_start(self, trainer, pl_module):
        idx = self._locate_unknowns_group(pl_module)
        if idx is None:
            return
        self._group_idx = idx
        # True target LR (not the ConstantLR-warmup-discounted value).
        base = trainer.optimizers[0].param_groups[idx]["lr"]
        for sch_cfg in (getattr(trainer, "lr_scheduler_configs", None) or []):
            sch = getattr(sch_cfg, "scheduler", None)
            base_lrs = getattr(sch, "base_lrs", None)
            if base_lrs is not None and idx < len(base_lrs):
                base = base_lrs[idx]
                break
        self._base_lr = base
        problem = pl_module.problem
        self._params = dict(problem.unknown_parameters)
        bounds = {}
        comp = getattr(pl_module, "_compiled_system", None)
        if comp is not None:
            bounds = dict(getattr(comp, "unknown_bounds", {}) or {})
        for name, p in self._params.items():
            lo, hi = bounds.get(name, (0.0, 1.0))
            self._ranges[name] = max(abs(hi - lo), 1e-9)
            self._prev[name] = float(p.detach().item())
            self._prev_v[name] = 0.0

    # ------------------------------------------------------------------ control
    def on_train_epoch_start(self, trainer, pl_module):
        if self._group_idx is None or self._base_lr is None:
            return
        ep = trainer.current_epoch
        if ep < self.warmup_epochs or self.converged:
            return

        # Aggregate velocity signal across all unknowns (one shared LR group).
        rel_vs = []
        osc = False
        for name, p in self._params.items():
            cur = float(p.detach().item())
            v = cur - self._prev[name]
            rel_vs.append(abs(v) / self._ranges[name])
            pv = self._prev_v[name]
            # Oscillation: velocity reversed sign with non-trivial magnitude.
            if pv != 0.0 and (v * pv) < 0 and abs(v) / self._ranges[name] > self.v_lo:
                osc = True
            self._prev[name] = cur
            self._prev_v[name] = v
        max_rel_v = max(rel_vs) if rel_vs else 0.0

        # Loss feedback: did the LR ramp recently buy us anything?
        loss = self._read_loss(trainer)
        improving = loss is not None and loss < self._best_loss * (1.0 - self.loss_eps)
        # Overshoot signal: a probe that pushed the loss *up* means the LR is
        # too high right now — brake hard and immediately, before it diverges.
        worse = (
            loss is not None and self._prev_loss is not None
            and loss > self._prev_loss * (1.0 + self.loss_eps)
        )
        if loss is not None:
            self._best_loss = min(self._best_loss, loss)
            self._prev_loss = loss

        # Update the multiplier.
        if osc or max_rel_v > self.v_hi or worse:
            self._mult *= self.lr_down            # brake (too fast / oscillating / diverging)
            self._stall = 0
        elif max_rel_v < self.v_lo:
            # Frozen / plateaued.
            if improving:
                # The current LR IS working — the loss is still dropping (the
                # network is fitting the field, the unknown will follow).
                # HOLD; do not ramp, or we destabilise a converging solution.
                self._stall = 0
            else:
                # Stagnant AND frozen → probe by ramping LR to test whether a
                # better (lower-loss) solution is reachable, i.e. a shallow
                # basin. If ramping reduces the loss, `improving` flips true
                # next epoch and we stop ramping. If it never does, the probe
                # counter runs out → genuine optimum → converged.
                self._mult *= self.lr_up
                self._stall += 1
        else:
            self._stall = 0                       # healthy band, hold
        self._mult = min(max(self._mult, self.min_mult), self.max_mult)

        # Converged: ramping the LR no longer reduces the loss → genuine
        # optimum, not a shallow basin. Rest the LR low so we don't drift off it.
        if self._stall >= self.patience and not self.converged:
            self.converged = True
            self._mult = self.converged_mult
            if self.stop_on_converge:
                trainer.should_stop = True

        lr = self._base_lr * self._mult
        trainer.optimizers[0].param_groups[self._group_idx]["lr"] = lr

        self.history.append({
            "epoch": ep, "max_rel_v": max_rel_v, "osc": osc, "loss": loss,
            "improving": improving, "worse": worse, "mult": self._mult, "lr": lr,
            "stall": self._stall, "converged": self.converged,
        })

    @staticmethod
    def _read_loss(trainer) -> Optional[float]:
        metrics = getattr(trainer, "callback_metrics", None) or {}
        for key in ("train_loss", "train_loss_epoch", "mean_loss", "loss"):
            if key in metrics:
                try:
                    return float(metrics[key])
                except Exception:
                    pass
        # Fall back to any key containing "loss".
        for k, v in metrics.items():
            if "loss" in str(k).lower():
                try:
                    return float(v)
                except Exception:
                    pass
        return None
