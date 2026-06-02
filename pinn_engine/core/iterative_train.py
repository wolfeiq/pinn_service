"""Iterative bound-tightening refinement loop.

Inverse-PINN convergence is bottlenecked by the *width* of the unknown's
search range — PINA initializes at the bound midpoint, so wide bounds put
the initial guess far from truth and the optimizer wastes epochs traversing
to the right region. ``iterative_train`` wraps :func:`train` to address this:

1. Run with the user's initial bounds → result ``θ₁``.
2. Tighten bounds around ``θ₁`` (shrink by ``tighten_factor``) and set the
   init to ``θ₁``.
3. Re-train → result ``θ₂``, hopefully tighter.
4. Repeat ``n_iters`` times.

Each iteration starts where the last finished, with a narrower search range,
so subsequent iterations converge faster and more precisely. Useful for
partial-identifiability problems (coupled-drag-style) and for sharpening Cosserat
beyond the controller's cap-limited equilibrium.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pinn_engine.core.trainer import TrainConfig, train, TrainResult


@dataclass
class IterativeResult:
    iterations: List[TrainResult]
    bounds_history: List[Dict[str, tuple]]
    final_params: Dict[str, float]


def iterative_train(
    system: Any,
    data: Dict[str, Any],
    base_config: TrainConfig,
    n_iters: int = 3,
    tighten_factor: float = 0.4,
    min_range: float = 1e-4,
    callbacks_factory=None,
) -> IterativeResult:
    """Run ``n_iters`` rounds of train(), tightening bounds around each
    result.

    Args:
        system, data: as in :func:`train`.
        base_config: starting :class:`TrainConfig`. The first iteration uses
            it as-is; subsequent iterations override
            ``unknown_bounds_override`` and ``unknown_inits_override``.
        n_iters: how many iterations to run (>=1).
        tighten_factor: each iteration shrinks the bound width by this
            factor, centered on the previous result. e.g. ``0.4`` keeps 40%
            of the previous range. Clamped to the original outer bounds so
            we never expand past where the user said is valid.
        min_range: don't shrink below this absolute width per dimension
            (prevents the search from collapsing to a single point).
        callbacks_factory: optional ``() -> list`` returning a fresh list of
            callbacks per iteration (so e.g. an AdaptiveUnknownsController
            instance isn't reused across runs).

    Returns:
        :class:`IterativeResult` with per-iter ``TrainResult``s and the
        bounds-history.
    """
    # Discover the original bounds so we can clip to them each round.
    compiled = system.compile()
    outer_bounds: Dict[str, tuple] = dict(compiled.unknown_bounds)

    cfg = deepcopy(base_config)
    bounds = dict(outer_bounds)
    inits: Dict[str, float] = {}

    results: List[TrainResult] = []
    bounds_history: List[Dict[str, tuple]] = []

    for it in range(int(n_iters)):
        # First iteration: leave config alone (uses compiled defaults).
        # Later iterations: pass narrowed bounds + previous result as init.
        if it == 0:
            cfg.unknown_bounds_override = None
            cfg.unknown_inits_override = None
        else:
            cfg.unknown_bounds_override = dict(bounds)
            cfg.unknown_inits_override = dict(inits)

        callbacks = callbacks_factory() if callbacks_factory else None
        bounds_history.append(dict(bounds))
        result = train(system, data, cfg, callbacks=callbacks)
        results.append(result)

        # Compute next iteration's bounds: shrink around the result, clip to
        # the outer bounds the user originally allowed.
        new_bounds: Dict[str, tuple] = {}
        new_inits: Dict[str, float] = {}
        for name, (lo_out, hi_out) in outer_bounds.items():
            theta = float(result.final_params.get(name, 0.5 * (lo_out + hi_out)))
            cur_lo, cur_hi = bounds.get(name, (lo_out, hi_out))
            cur_range = cur_hi - cur_lo
            new_range = max(cur_range * tighten_factor, min_range)
            new_lo = max(lo_out, theta - 0.5 * new_range)
            new_hi = min(hi_out, theta + 0.5 * new_range)
            # If clipping squashed one side, expand the other to keep new_range.
            if new_hi - new_lo < new_range - 1e-9:
                if new_lo == lo_out:
                    new_hi = min(hi_out, lo_out + new_range)
                elif new_hi == hi_out:
                    new_lo = max(lo_out, hi_out - new_range)
            new_bounds[name] = (new_lo, new_hi)
            new_inits[name] = theta
        bounds = new_bounds
        inits = new_inits

    return IterativeResult(
        iterations=results,
        bounds_history=bounds_history,
        final_params=results[-1].final_params,
    )
