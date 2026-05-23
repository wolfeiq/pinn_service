"""Auto-generate AutoML search spaces from a compiled :class:`System`.

The build plan used to require every template to hand-write an
``automl_space(trial)`` method. For genuinely new problems that's
friction: the user writes the equations in the DSL but still has to
hand-craft a search space.

This module produces a search space *automatically* from the structure
of the compiled system — number of unknowns, bound magnitudes, whether
it's a PDE, how many state variables. The heuristics are conservative
but cover the common case: drop a System in, get a sensible AutoML run
back.

A user can still override by writing a custom ``automl_space``; this is
the safety-net default.
"""
from __future__ import annotations

from typing import Callable, Any

from pinn_engine.dsl.system import CompiledSystem
from pinn_engine.core.trainer import TrainConfig


def _scale_from_bounds(bounds: tuple[float, float]) -> float:
    """Order-of-magnitude scale of a bound interval (for setting lam_data range)."""
    lo, hi = bounds
    width = abs(hi - lo)
    mid = max(abs(0.5 * (lo + hi)), 1e-6)
    return float(width / mid)   # relative width


def auto_search_space(compiled: CompiledSystem) -> Callable[[Any], TrainConfig]:
    """Return an ``automl_space(trial)`` callable tuned to the compiled system.

    Heuristics applied:

    * Depth: 3-6 for ODEs, 4-8 for PDEs (PDEs need more representational
      capacity; PINN literature consistently uses deeper nets there).
    * Width: ``[32, 64, 128]`` for ODEs, ``[64, 128, 256]`` for PDEs.
    * Activation: always the full set {tanh, sintanh, swish}. ``sin``
      added if any unknown's bound spans more than 3 orders of magnitude
      (suggests the signal may have high-frequency content).
    * ``lam_data_init``: search range broadens when bounds are loose
      relative to scale (per :func:`_scale_from_bounds`). Tight bounds
      get [10, 100]; loose get [10, 10000].
    * Balancer: always the full set {none, lra, sapinn}.
    * Epochs: 800 ODE, 2000 PDE — PDEs need more time.
    * Collocation count: scales with ``n_state × log(n_unknowns)``.
    """
    is_pde = compiled.is_pde
    n_unknowns = len(compiled.unknown_names)
    n_state = len(compiled.state_names)

    # Detect whether any unknown spans multiple decades — suggests a
    # broad scale that may need sin() activation or wider lam_data.
    max_rel_width = max(
        (_scale_from_bounds(b) for b in compiled.unknown_bounds.values()),
        default=1.0,
    )
    high_dynamic_range = max_rel_width > 5.0

    activations = ["tanh", "sintanh", "swish"]
    if high_dynamic_range:
        activations = activations + ["sin"]

    if is_pde:
        depth_lo, depth_hi = 4, 8
        widths = [64, 128, 256]
        adam_epochs = 2000
        coll = max(2000, 1000 * n_state)
        batch = 1024
    else:
        depth_lo, depth_hi = 3, 6
        widths = [32, 64, 128]
        adam_epochs = 800
        coll = max(1000, 500 * n_state)
        batch = 512

    if high_dynamic_range:
        lam_lo, lam_hi = 10.0, 10000.0
    else:
        lam_lo, lam_hi = 10.0, 1000.0

    def _space(trial) -> TrainConfig:
        # Fourier features only when the dynamic range suggests high-frequency
        # content. Otherwise pure MLP — the canonical PINN setup.
        if high_dynamic_range or is_pde:
            ff = trial.suggest_categorical("fourier_features", [0, 16, 32, 64])
            ff_sigma = trial.suggest_float("fourier_sigma", 0.5, 5.0, log=True) if ff > 0 else 1.0
        else:
            ff, ff_sigma = 0, 1.0

        # PDE inverse parameters typically need a separate (larger) LR to
        # traverse their bounds in a reasonable epoch budget — Adam's per-
        # parameter normalisation otherwise leaves the unknowns crawling.
        # ODEs usually don't need this; keep default 1.0 unless we know
        # better. Range stays log-scale and well-clear of overshoot bounds
        # we discovered empirically on Cosserat (scale=100+ overshoots).
        if is_pde:
            param_lr_scale = trial.suggest_float("param_lr_scale", 1.0, 50.0, log=True)
        else:
            param_lr_scale = 1.0

        return TrainConfig(
            depth=trial.suggest_int("depth", depth_lo, depth_hi),
            width=trial.suggest_categorical("width", widths),
            activation=trial.suggest_categorical("activation", activations),
            lr=trial.suggest_float("lr", 5e-4, 5e-3, log=True),
            lam_data_init=trial.suggest_float("lam_data_init", lam_lo, lam_hi, log=True),
            lam_physics_init=1.0,
            balancer=trial.suggest_categorical("balancer", ["none", "lra", "sapinn"]),
            adam_epochs=adam_epochs,
            lbfgs_iters=0,
            n_collocation=coll,
            batch_size=batch,
            fourier_features=ff,
            fourier_sigma=ff_sigma,
            param_lr_scale=param_lr_scale,
        )

    _space.__name__ = "auto_search_space"
    _space.__doc__ = (
        f"Auto-generated search space: "
        f"{'PDE' if is_pde else 'ODE'}, "
        f"{n_unknowns} unknowns, "
        f"{n_state} state vars, "
        f"high_dynamic_range={high_dynamic_range}"
    )
    return _space


__all__ = ["auto_search_space"]
