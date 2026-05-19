"""AutoML search-space resolution from a template.

Each template provides its own ``automl_space(trial)`` returning a
:class:`TrainConfig`. This module is a thin layer that resolves a
``template_name`` to that callable.
"""
from __future__ import annotations

from pinn_engine.dsl.templates import get_template


def trial_to_config(template_name: str, trial):
    """Sample a :class:`TrainConfig` from a Optuna ``trial`` for ``template_name``."""
    cls = get_template(template_name)
    return cls.automl_space(trial)
