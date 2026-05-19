"""Optuna study orchestration.

``run_search(template_name, ...)`` builds the study, defines the objective,
runs it, and returns the :class:`optuna.study.Study` for inspection.

The objective:
1. Samples a :class:`TrainConfig` from the template's ``automl_space``.
2. Generates synthetic data (using the template's defaults).
3. Runs the trainer with a :class:`PyTorchLightningPruningCallback` plus our
   :class:`NanGuard` and :class:`ParamDivergenceGuard`.
4. Returns ``template.objective(result)`` — typically mean abs-rel-error
   against the known truth.

Manifests are written for every completed trial under ``manifests/``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import optuna
from optuna.samplers import TPESampler

# The integration package re-exports the PL pruning callback.
try:
    from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback
except ImportError:  # older optuna versions
    from optuna.integration import PyTorchLightningPruningCallback  # type: ignore

from pinn_engine.automl.pruning import NanGuard, ParamDivergenceGuard
from pinn_engine.diagnostics import default_bundle
from pinn_engine.dsl.templates import get_template
from pinn_engine.repro.manifest import write_manifest, MANIFEST_DIR


def _objective(trial, template_name: str, data, monitor: str = "train_loss"):
    template = get_template(template_name)
    config = template.automl_space(trial)
    system = template.system()
    bounds = template.unknown_bounds

    callbacks = default_bundle() + [
        NanGuard(),
        ParamDivergenceGuard(bounds=bounds),
        PyTorchLightningPruningCallback(trial, monitor=monitor),
    ]

    # Local import to avoid circular dependencies on tooling-only paths.
    from pinn_engine.core.trainer import train

    result = train(system=system, data=data, config=config, callbacks=callbacks)
    write_manifest(
        template=template_name,
        result=result,
        data=data,
        automl_study=trial.study.study_name,
        trial_number=trial.number,
    )
    return template.objective(result)


def run_search(
    template_name: str,
    n_trials: int = 20,
    study_name: Optional[str] = None,
    storage_dir: Optional[Path] = None,
    data=None,
    seed: int = 42,
    monitor: str = "train_loss",
) -> optuna.Study:
    """Run an Optuna study over the template's AutoML space."""
    template = get_template(template_name)
    if data is None:
        data, _truth = template.synthetic_data(seed=seed)

    storage_dir = Path(storage_dir) if storage_dir is not None else MANIFEST_DIR
    storage_dir.mkdir(parents=True, exist_ok=True)
    study_name = study_name or f"{template_name}_search"
    storage = f"sqlite:///{storage_dir / f'optuna_{study_name}.db'}"

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=TPESampler(seed=seed),
        pruner=optuna.pruners.HyperbandPruner(min_resource=200, max_resource=4000),
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(
        lambda t: _objective(t, template_name, data, monitor=monitor),
        n_trials=n_trials,
        catch=(RuntimeError,),  # don't let one bad trial kill the whole search
    )
    return study
