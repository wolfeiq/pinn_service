"""Dynamic loss weighting schemes, implemented as PINA ``WeightingInterface``s.

PINA's solver calls ``weighting.aggregate(losses)`` once per training step,
where ``losses`` is a dict of per-condition tensor losses with the autograd
graph attached. The aggregator calls ``weights_update(losses)`` to refresh
the per-condition λ values, then returns ``Σ λ_i · L_i`` for backward.

Two schemes here:

* **SA-PINN** — learnable λ trained adversarially against the network. λ
  increases on hard conditions, forcing the network to do work where it
  currently fails. Reference: McClenny & Braga-Neto 2020, arXiv:2009.04544.
* **LRA** (learning-rate annealing) — per-condition λ updated from the
  ratio of gradient norms each epoch. The reference condition's gradient
  norm sets the "scale"; other λ values are EMA-blended toward
  ``max ||∇L_j|| / ||∇L_i||``. Reference: Wang, Teng & Perdikaris 2021,
  arXiv:2107.05228.

ReLoBRaLo (the build plan's third option) is deliberately omitted — it hurt
convergence on coupled dynamics in prior experiments.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
from pina.loss import WeightingInterface


class SAPinnWeighting(WeightingInterface):
    """Self-adaptive PINN: each condition gets a learnable scalar λ.

    The λ values are stored as :class:`torch.nn.Parameter` and updated by a
    dedicated Adam optimizer maximising ``Σ λ_i · L_i.detach()`` (clamped
    positive). The result is a free-running adversarial loop that gives more
    weight to conditions where the network currently fails.

    Returns scalar weights to the aggregator — the λ updates don't propagate
    into the network's main backward (only the resulting weights multiply
    the loss terms).
    """

    def __init__(
        self,
        lam_init: float = 1.0,
        lr: float = 5e-3,
        clamp_min: float = 1e-4,
        update_every_n_epochs: int = 1,
    ):
        super().__init__(update_every_n_epochs=update_every_n_epochs, aggregator="sum")
        self.lam_init = float(lam_init)
        self.lr = float(lr)
        self.clamp_min = float(clamp_min)
        self._lams: Dict[str, torch.nn.Parameter] = {}
        self._opt: Optional[torch.optim.Optimizer] = None
        # History of weight dicts for callback inspection.
        self.history: list[Dict[str, float]] = []

    def weights_update(self, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        # Lazy init: create one nn.Parameter per condition the first time we
        # see it. (PINA reveals condition names only at first aggregate call.)
        for name in losses.keys():
            if name not in self._lams:
                self._lams[name] = torch.nn.Parameter(
                    torch.tensor(self.lam_init, requires_grad=True)
                )
        if self._opt is None and self._lams:
            self._opt = torch.optim.Adam(list(self._lams.values()), lr=self.lr)

        # Adversarial update: gradient of ``-Σ λ_i · L_i.detach()`` w.r.t. λ
        # is ``-L_i.detach()``; Adam descends, so λ moves up where loss is
        # large. Detach the losses so this step doesn't pollute the network
        # backward graph.
        if self._opt is not None:
            self._opt.zero_grad()
            adv = -sum(
                self._lams[name] * float(loss.detach())
                for name, loss in losses.items()
            )
            adv.backward()
            self._opt.step()
            with torch.no_grad():
                for lam in self._lams.values():
                    lam.clamp_(min=self.clamp_min)

        weights = {name: float(self._lams[name].detach().item()) for name in losses}
        self.history.append(dict(weights))
        return weights


class LRAWeighting(WeightingInterface):
    """Learning-rate annealing weighting (Wang, Teng & Perdikaris 2021).

    For each condition ``i`` compute ``||∇L_i||₂`` w.r.t. the network's
    parameters. Then::

        λ_i ← α · λ_i + (1 − α) · (max_j ||∇L_j||) / ||∇L_i||

    Conditions whose gradients are small relative to the loudest get scaled
    up; conditions that dominate get scaled down. Exponential moving average
    smooths the trajectory.

    Cheap caveat: this requires one extra ``torch.autograd.grad`` call per
    condition per training step. For PINN inverse problems with 2-4
    conditions it's a small overhead.
    """

    def __init__(
        self,
        alpha: float = 0.9,
        lam_init: float = 1.0,
        update_every_n_epochs: int = 1,
    ):
        super().__init__(update_every_n_epochs=update_every_n_epochs, aggregator="sum")
        self.alpha = float(alpha)
        self.lam_init = float(lam_init)
        self._lams: Dict[str, float] = {}
        self.history: list[Dict[str, float]] = []

    def weights_update(self, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        # Solver-aware: we need the network's parameters for the gradients.
        model = self.solver.model

        # Compute ||∇L_i|| per condition. retain_graph=True so the network's
        # subsequent backward of the aggregated loss still has the graph.
        params = [p for p in model.parameters() if p.requires_grad]
        grad_norms: Dict[str, float] = {}
        for name, loss in losses.items():
            if not loss.requires_grad:
                grad_norms[name] = 1.0
                continue
            try:
                grads = torch.autograd.grad(
                    loss,
                    params,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )
            except RuntimeError:
                grad_norms[name] = 1.0
                continue
            sq = 0.0
            for g in grads:
                if g is not None:
                    sq += float(g.detach().pow(2).sum())
            grad_norms[name] = sq ** 0.5

        max_norm = max(grad_norms.values()) if grad_norms else 1.0
        new_weights: Dict[str, float] = {}
        for name in losses:
            current = self._lams.get(name, self.lam_init)
            target = max_norm / max(grad_norms[name], 1e-8)
            blended = self.alpha * current + (1.0 - self.alpha) * target
            self._lams[name] = blended
            new_weights[name] = blended

        self.history.append(dict(new_weights))
        return new_weights


def build_weighting(
    name: str,
    cond_weights: Dict[str, float] | None = None,
) -> WeightingInterface | None:
    """Factory: 'none' | 'sapinn' | 'lra' | 'scalar' → weighting instance.

    ``'scalar'`` returns a :class:`pina.loss.ScalarWeighting` initialized with
    ``cond_weights`` (the static per-condition dict). The dynamic balancers
    ignore ``cond_weights`` — they manage their own λ values.
    """
    from pina.loss import ScalarWeighting

    if name == "none":
        return None
    if name == "scalar":
        if cond_weights is None:
            raise ValueError("scalar weighting requires cond_weights")
        return ScalarWeighting(cond_weights)
    if name == "sapinn":
        return SAPinnWeighting()
    if name == "lra":
        return LRAWeighting()
    raise ValueError(f"Unknown weighting {name!r}")
