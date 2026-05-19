"""Template registry — maps names like ``"damped_oscillator"`` to template classes.

Each template provides:
* a :class:`~pinn_engine.dsl.System`,
* a default AutoML search space (sampled from an Optuna ``trial``),
* a default :class:`~pinn_engine.core.trainer.TrainConfig`,
* an objective the AutoML run minimizes.

Importing :mod:`pinn_engine.dsl.templates_lib` registers all bundled templates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Type, Any

from pinn_engine.dsl.system import System


@dataclass
class TemplateMeta:
    name: str
    cls: Type[Any]


registry: Dict[str, TemplateMeta] = {}


def register_template(name: str):
    """Decorator: register a template under ``name``."""

    def deco(cls):
        if name in registry:
            raise ValueError(f"Template {name!r} already registered.")
        registry[name] = TemplateMeta(name=name, cls=cls)
        cls._registry_name = name
        return cls

    return deco


def get_template(name: str):
    """Resolve a template class. Imports the bundled lib on first call."""
    if not registry:
        # Trigger registration of bundled templates.
        import pinn_engine.dsl.templates_lib  # noqa: F401
    if name not in registry:
        raise KeyError(
            f"Template {name!r} not found. Known: {sorted(registry.keys())}"
        )
    return registry[name].cls
