"""1D diffusion inverse problem: discover diffusivity ``D`` from ``u(t, x)``.

PDE: ``∂u/∂t = D · ∂²u/∂x²``.

**Phase-1 status:** the time-only Problem adapter in ``core/problem.py`` does
not yet handle space+time problems — that's a Phase-3 extension. This template
is registered so the registry is complete and the synthetic data generator is
exercised by tests, but ``system()`` / ``default_config()`` will raise until
the spatial adapter lands.
"""
from __future__ import annotations

from pinn_engine.dsl.symbols import Variable, Unknown, Sensor
from pinn_engine.dsl.system import System
from pinn_engine.dsl.templates import register_template
from pinn_engine.core.trainer import TrainConfig


@register_template("diffusion_1d")
class Diffusion1D:
    """1D diffusion. PDE template; spatial adapter is Phase 3."""

    truth = {"D": 0.1}
    unknown_bounds = {"D": (1e-3, 1.0)}

    @staticmethod
    def system() -> System:
        raise NotImplementedError(
            "diffusion_1d requires the space+time problem adapter (Phase 3). "
            "Synthetic data is available via Diffusion1D.synthetic_data()."
        )

    @staticmethod
    def default_config() -> TrainConfig:
        raise NotImplementedError(
            "diffusion_1d requires the space+time problem adapter (Phase 3)."
        )

    @staticmethod
    def synthetic_data(seed: int = 0):
        from pinn_engine.data.synthetic import generate_diffusion_1d
        return generate_diffusion_1d(seed=seed)

    @staticmethod
    def automl_space(trial):
        raise NotImplementedError("diffusion_1d AutoML pending Phase-3 adapter.")

    @staticmethod
    def objective(result) -> float:
        return abs(result.final_params["D"] - Diffusion1D.truth["D"]) / 0.1
