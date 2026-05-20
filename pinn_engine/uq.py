"""Ensemble uncertainty quantification.

Run the same training problem N times with different seeds. The
distribution of discovered parameters across the N runs is a Monte
Carlo estimate of the inverse-problem's uncertainty: a wide spread on
parameter ``c`` says "the data and physics together don't tightly
constrain ``c``", a tight spread says they do.

This is the cheapest serious UQ available — Bayesian PINNs and HMC
posteriors are theoretically richer but typically 10-100× more
expensive and shaped largely by their priors anyway. For a robotics
engineer asking "how confident am I in this discovered friction
coefficient?", an ensemble mean ± std is the right level of detail.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from pinn_engine.core.trainer import TrainConfig, TrainResult, train
from pinn_engine.dsl.system import System


@dataclass
class EnsembleResult:
    """The output of an ensemble training run.

    Attributes:
        members: the individual :class:`TrainResult`\\ s, one per seed.
        seeds: the seed list (in order).
        parameter_estimates: ``{name: {mean, std, values}}`` summary across
            the ensemble for every discovered parameter.
        n_models: number of ensemble members.
    """

    members: List[TrainResult]
    seeds: List[int]
    parameter_estimates: Dict[str, Dict[str, Any]]
    n_models: int

    def summary_table(self) -> str:
        """Format the ensemble summary as a readable text table."""
        lines = [f"Ensemble of {self.n_models} models (seeds {self.seeds})"]
        lines.append(f"{'Parameter':>15}  {'Mean':>12}  {'Std':>12}  {'CV':>8}")
        lines.append("-" * 52)
        for name, stats in self.parameter_estimates.items():
            mean = stats["mean"]
            std = stats["std"]
            cv = std / abs(mean) if abs(mean) > 1e-9 else float("inf")
            lines.append(f"{name:>15}  {mean:>12.4g}  {std:>12.4g}  {cv:>7.2%}")
        return "\n".join(lines)


def train_ensemble(
    system: System,
    data: Dict[str, Any],
    config: TrainConfig,
    n_models: int = 5,
    seeds: Optional[Sequence[int]] = None,
    callbacks: Optional[list] = None,
) -> EnsembleResult:
    """Train ``n_models`` PINNs at distinct seeds, aggregate the unknowns.

    Parameters:
        system, data, config: the inverse problem (same as :func:`train`).
        n_models: number of ensemble members (default 5).
        seeds: explicit seed list. If None, uses ``range(n_models)``.
        callbacks: optional list of Lightning callbacks, passed through.

    Returns:
        :class:`EnsembleResult` with per-parameter mean / std / CV.
    """
    seeds = list(seeds) if seeds is not None else list(range(n_models))
    if len(seeds) != n_models:
        n_models = len(seeds)

    members: List[TrainResult] = []
    for seed in seeds:
        cfg = config.model_copy(update={"seed": seed})
        result = train(system=system, data=data, config=cfg, callbacks=callbacks)
        members.append(result)

    # Aggregate per-parameter stats.
    if not members:
        raise RuntimeError("No ensemble members were trained")
    param_names = list(members[0].final_params.keys())
    estimates: Dict[str, Dict[str, Any]] = {}
    for name in param_names:
        vals = np.array([m.final_params[name] for m in members], dtype=np.float64)
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        estimates[name] = {
            "mean": float(vals.mean()),
            "std": std,
            "values": vals.tolist(),
        }
    return EnsembleResult(
        members=members,
        seeds=seeds,
        parameter_estimates=estimates,
        n_models=n_models,
    )


__all__ = ["train_ensemble", "EnsembleResult"]
