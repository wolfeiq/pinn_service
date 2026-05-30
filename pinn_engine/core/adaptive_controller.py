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
        lr_down: float = 0.6,
        probe_boost: float = 2.0,
        probe_window: int = 3,
        stall_patience: int = 2,
        escape_eps: float = 2e-2,
        worse_threshold: float = 0.5,
        data_worse_threshold: float = 0.1,
        min_mult: float = 0.02,
        max_mult: float = 50.0,
        converged_mult: float = 0.2,
        stop_on_converge: bool = False,
    ):
        super().__init__()
        self.warmup_epochs = int(warmup_epochs)
        self.v_lo = float(v_lo)               # below → stalled (maybe probe)
        self.v_hi = float(v_hi)               # above → too fast (brake)
        self.lr_down = float(lr_down)         # brake factor (recoverable)
        self.probe_boost = float(probe_boost) # LR ×this during a probe (bounded)
        self.probe_window = int(probe_window) # epochs to hold a probe
        self.stall_patience = int(stall_patience)  # stalled epochs before probing
        self.escape_eps = float(escape_eps)   # loss-drop over a probe that counts as "productive"
        # A jump this large counts as real divergence (brake hard); set high so
        # noisy/causal-loss jitter doesn't spuriously brake a healthy descent.
        self.worse_threshold = float(worse_threshold)
        # *Data*-loss rise that counts as the unknown drifting past truth. Data
        # loss is what pins the unknown to observations; total loss can fall
        # while data rises (physics polishing masks the drift) — so monitor
        # data loss separately and tighter.
        self.data_worse_threshold = float(data_worse_threshold)
        self.min_mult = float(min_mult)
        self.max_mult = float(max_mult)
        self.converged_mult = float(converged_mult)
        self.stop_on_converge = bool(stop_on_converge)

        self._group_idx: Optional[int] = None
        self._base_lr: Optional[float] = None
        self._params: dict = {}          # name -> torch.nn.Parameter
        self._ranges: dict = {}          # name -> bound width
        self._prev: dict = {}            # name -> previous value
        self._prev_v: dict = {}          # name -> previous signed velocity
        self._base_mult: float = 1.0     # committed LR multiplier
        self._stall: int = 0
        self._state: str = "descend"     # descend | probe | converged
        self._probe_t: int = 0
        self._probe_loss0: Optional[float] = None
        self._probe_val0: dict = {}      # unknown values at probe entry
        self._prev_loss: Optional[float] = None
        self._prev_data_loss: Optional[float] = None
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

        loss = self._read_loss(trainer)
        data_loss = self._read_data_loss(trainer)
        # Real divergence (not noise): brake hard regardless of state.
        diverging = (
            loss is not None and self._prev_loss is not None
            and loss > self._prev_loss * (1.0 + self.worse_threshold)
        )
        # Data-loss rising = the unknown drifted past where the observations
        # want it. The total-loss divergence brake misses this when physics
        # loss is still falling fast enough to mask the data-fit drift —
        # which is how run #3 overshot truth into the lower bound.
        data_worse = (
            data_loss is not None and self._prev_data_loss is not None
            and data_loss > self._prev_data_loss * (1.0 + self.data_worse_threshold)
        )

        # ---- state machine -------------------------------------------------
        # eff_mult is the multiplier actually applied this epoch; base_mult is
        # the committed level we return to between probes.
        if self._state == "descend":
            if osc or max_rel_v > self.v_hi or diverging or data_worse:
                # Overshoot — brake the committed level (recoverable via probes).
                self._base_mult *= self.lr_down
                self._stall = 0
            elif max_rel_v < self.v_lo:
                # Stalled. After a little patience, probe: is more LR productive?
                self._stall += 1
                if self._stall >= self.stall_patience:
                    self._state = "probe"
                    self._probe_t = 0
                    self._probe_loss0 = loss if loss is not None else self._prev_loss
                    self._probe_val0 = {n: float(p.detach().item())
                                        for n, p in self._params.items()}
            else:
                self._stall = 0                    # healthy band — hold
            eff_mult = self._base_mult

        elif self._state == "probe":
            self._probe_t += 1
            eff_mult = self._base_mult * self.probe_boost
            # Abort early if the boost is clearly diverging or pushing the
            # unknown away from the data fit.
            if diverging or data_worse:
                self._base_mult *= self.lr_down
                self._state = "descend"
                self._stall = 0
            elif self._probe_t >= self.probe_window:
                dropped = (
                    loss is not None and self._probe_loss0 is not None
                    and loss < self._probe_loss0 * (1.0 - self.escape_eps)
                )
                # Did the *unknown* move under the boosted LR? A loss drop with a
                # motionless unknown is just the network polishing the field —
                # not an escape. At a true optimum the gradient ~0 so the unknown
                # won't move however high the LR; in a basin/creep it will.
                moved = max(
                    abs(float(p.detach().item()) - self._probe_val0.get(n, 0.0))
                    / self._ranges[n]
                    for n, p in self._params.items()
                ) > self.v_lo
                if dropped and moved:
                    # Higher LR is productively moving the unknown to a better
                    # solution → commit it and keep descending.
                    self._base_mult *= self.probe_boost
                    self._state = "descend"
                    self._stall = 0
                else:
                    # More LR can't reduce the loss by moving the unknown →
                    # genuine optimum.
                    self._state = "converged"
                    self.converged = True
                    self._base_mult = self.converged_mult
                    if self.stop_on_converge:
                        trainer.should_stop = True
        else:  # converged
            eff_mult = self.converged_mult

        self._base_mult = min(max(self._base_mult, self.min_mult), self.max_mult)
        eff_mult = min(max(eff_mult, self.min_mult), self.max_mult)
        if loss is not None:
            self._prev_loss = loss
        if data_loss is not None:
            self._prev_data_loss = data_loss

        lr = self._base_lr * eff_mult
        trainer.optimizers[0].param_groups[self._group_idx]["lr"] = lr

        self.history.append({
            "epoch": ep, "max_rel_v": max_rel_v, "osc": osc, "loss": loss,
            "data_loss": data_loss, "state": self._state,
            "diverging": diverging, "data_worse": data_worse,
            "base_mult": self._base_mult, "eff_mult": eff_mult, "lr": lr,
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

    @staticmethod
    def _read_data_loss(trainer) -> Optional[float]:
        """Sum of per-condition data losses (keys like ``data_<sensor>_loss_epoch``).
        PINA logs each data condition separately; we aggregate. Returns None if
        no data-condition keys are present."""
        metrics = getattr(trainer, "callback_metrics", None) or {}
        total, found = 0.0, False
        for k, v in metrics.items():
            ks = str(k).lower()
            if ks.startswith("data_") and "loss" in ks and ks.endswith("_epoch"):
                try:
                    total += float(v)
                    found = True
                except Exception:
                    pass
        return total if found else None
