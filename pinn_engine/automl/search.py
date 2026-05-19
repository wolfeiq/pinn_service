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

from pinn_engine.automl.pruning import (
    NanGuard, ParamDivergenceGuard, TrainLossPruningCallback,
)
from pinn_engine.diagnostics import default_bundle
from pinn_engine.dsl.templates import get_template
from pinn_engine.repro.manifest import write_manifest, MANIFEST_DIR


# Fixed seeds the multi-seed objective averages over. Three is the literature
# norm (Auto-PINN, AutoPINN papers both average over 3-5 seeds). Pinned values
# rather than sampled so config "X scores 0.05%" is reproducible across runs.
MULTI_SEED_SEEDS = (42, 137, 2718)


def _objective(trial, template_name: str, data, monitor: str = "train_loss_epoch"):
    """Multi-seed AutoML objective: each trial averages over MULTI_SEED_SEEDS.

    Why: optimizing over ``seed`` lets Optuna pick lucky-seed artifacts. The
    "best" config in a single-seed search may not generalize to other seeds.
    Averaging over 3 fixed seeds makes the objective a robust property of the
    architecture+lr+weighting combo, not of the specific RNG draw.

    Pruning still works: we report the running average to Optuna after each
    seed (step=0, 1, 2), so a trial with a hopeless first seed gets pruned
    before wasting compute on the other two.
    """
    template = get_template(template_name)
    config = template.automl_space(trial)
    system = template.system()
    bounds = template.unknown_bounds

    from pinn_engine.core.trainer import train

    objectives = []
    for step, seed in enumerate(MULTI_SEED_SEEDS):
        config_seeded = config.model_copy(update={"seed": seed})

        callbacks = default_bundle() + [
            NanGuard(),
            ParamDivergenceGuard(bounds=bounds),
        ]

        result = train(system=system, data=data, config=config_seeded, callbacks=callbacks)
        obj_value = template.objective(result)
        objectives.append(obj_value)

        write_manifest(
            template=template_name,
            result=result,
            data=data,
            automl_study=trial.study.study_name,
            trial_number=trial.number,
        )

        # Report the running average; Hyperband prunes if this trial's
        # population position is bad enough at the current "rung" (seed step).
        running_avg = sum(objectives) / len(objectives)
        trial.report(running_avg, step=step)
        if trial.should_prune():
            raise optuna.TrialPruned(
                f"Pruned after {step + 1}/{len(MULTI_SEED_SEEDS)} seeds "
                f"(running avg = {running_avg:.4g})"
            )

    return sum(objectives) / len(objectives)


def run_search(
    template_name: str,
    n_trials: int = 20,
    study_name: Optional[str] = None,
    storage_dir: Optional[Path] = None,
    data=None,
    seed: int = 42,
    monitor: str = "train_loss_epoch",
) -> optuna.Study:
    """Run an Optuna study over the template's AutoML space."""
    template = get_template(template_name)
    if data is None:
        data, _truth = template.synthetic_data(seed=seed)

    storage_dir = Path(storage_dir) if storage_dir is not None else MANIFEST_DIR
    storage_dir.mkdir(parents=True, exist_ok=True)
    study_name = study_name or f"{template_name}_search"
    storage = f"sqlite:///{storage_dir / f'optuna_{study_name}.db'}"

    # Pruner operates on the seed step (0, 1, 2) inside _objective rather than
    # on training epochs. MedianPruner is the simpler choice here: prune a
    # trial after the first seed if its running average is worse than the
    # median trial at the same step. The earliest we can prune is after seed
    # 1, so n_startup_trials must cover the initial population.
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,  # let the first 5 trials run all seeds
            n_warmup_steps=0,
            interval_steps=1,
        ),
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(
        lambda t: _objective(t, template_name, data, monitor=monitor),
        n_trials=n_trials,
        catch=(RuntimeError,),  # don't let one bad trial kill the whole search
    )
    return study
