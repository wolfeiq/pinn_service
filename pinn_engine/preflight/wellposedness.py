"""Well-posedness pre-flight: identifiability check before any training.

Given a freshly-initialized network and an :class:`InverseProblem` with
unknown parameters ``θ ∈ R^p``, we sample collocation points and compute the
**residual sensitivity matrix**::

    S[i, k] = ∂r_i / ∂θ_k    shape (N · n_eqs, p)

Why the residual (and not the sensor predictions): the network output is
*independent* of θ — only the physics residual ties θ to the predictions.
Identifiability of θ is therefore captured by the rank of S.

If ``rank(S) < p`` or ``cond(S)`` is huge, the unknowns are not jointly
recoverable from these equations and we raise :class:`UnidentifiableError`
*before* burning training cycles.

Computation: clean centered finite differences. We use a single small ε
because the residual is *linear* in θ for templates that satisfy the DSL
grammar (which is the common case for inverse PINNs); the finite-difference
error is then zero up to floating point. For non-linear-in-θ residuals (rare
in inverse problems by design) the same procedure is a perfectly serviceable
sensitivity estimate.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from pina import LabelTensor

from pinn_engine.dsl.system import CompiledSystem


class UnidentifiableError(RuntimeError):
    """Raised when the declared unknowns can't be recovered from the equations."""


class BoundsTooWideWarning(UserWarning):
    """Loose bounds give a bad initial guess (PINA inits at the midpoint).

    Not an error — the run will proceed. But convergence is likely slower than
    it could be. Tightening bounds to a 2-3× range around your best guess
    typically lands the discovered value in the first ~hundred epochs.
    """


@dataclass
class WellposednessReport:
    rank: int
    n_unknowns: int
    condition_number: float
    column_norms: Dict[str, float] = field(default_factory=dict)
    blind_unknowns: List[str] = field(default_factory=list)
    passed: bool = True

    def __str__(self) -> str:
        lines = [
            "Well-posedness pre-flight:",
            f"  rank(S) = {self.rank} / {self.n_unknowns}",
            f"  cond(S) = {self.condition_number:.2e}",
            f"  passed  = {self.passed}",
        ]
        if self.blind_unknowns:
            lines.append(
                "  Unknowns with ~zero sensitivity: " + ", ".join(self.blind_unknowns)
            )
        return "\n".join(lines)


def _sample_collocation(problem, n: int, compiled=None) -> torch.Tensor:
    """Sample ``n`` collocation points uniformly from the problem's input domain.

    For ODE problems (single time variable) this is a linspace along ``t``.
    For PDE problems (e.g. ``(s, t)``) this is a uniform grid over the joint
    spatial+temporal domain, flattened to ``(N, n_inputs)``.

    If ``compiled`` is provided the column order matches
    ``compiled.input_names``; otherwise we use the problem's introspection.
    """
    td = problem.temporal_domain
    t_var = td.variables[0]
    t_lo, t_hi = td.range_[t_var]
    spatial = getattr(problem, "spatial_domain", None)
    if spatial is None:
        return torch.linspace(t_lo, t_hi, n, dtype=torch.float32).reshape(-1, 1)

    # PDE case: tensor of size (N, n_inputs) with columns in compiled.input_names order.
    if compiled is not None:
        input_order = list(compiled.input_names)
    else:
        input_order = list(spatial.variables) + [t_var]

    n_side = max(2, int(round(n ** 0.5)))
    cols = []
    for name in input_order:
        if name == t_var:
            lo, hi = t_lo, t_hi
        else:
            lo, hi = spatial.range_[name]
        cols.append(torch.linspace(lo, hi, n_side, dtype=torch.float32))
    grid = torch.stack(torch.meshgrid(*cols, indexing="ij"), dim=-1).reshape(-1, len(cols))
    return grid


def _evaluate_residuals(
    network: torch.nn.Module,
    compiled: CompiledSystem,
    t_pts: torch.Tensor,
    theta: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Run all physics residuals once and return a flat (n_pts*n_eqs,) tensor.

    The network must be called on the *LabelTensor* (not the raw tensor) so
    that the autograd graph is rooted at ``input_lt`` and per-column
    derivatives composed during residual evaluation are valid.
    """
    t_grad = t_pts.clone().detach().requires_grad_(True)
    input_lt = LabelTensor(t_grad, labels=list(compiled.input_names) or [compiled.input_name])
    y = network(input_lt)
    output_lt = LabelTensor(y, labels=list(compiled.state_names))
    residual_chunks = []
    for r_fn in compiled.physics_residuals:
        r = r_fn(input_lt, output_lt, params_=theta)
        residual_chunks.append(r.reshape(-1))
    return torch.cat(residual_chunks)


def check_wellposedness(
    problem,
    network: torch.nn.Module,
    compiled: CompiledSystem,
    n: int = 128,
    eps: float = 1e-3,
    cond_threshold: float = 1e8,
    rel_blind_threshold: float = 1e-8,
) -> WellposednessReport:
    """Run the check. Raises :class:`UnidentifiableError` on failure.

    Returns a :class:`WellposednessReport` on success.
    """
    unknown_names = list(compiled.unknown_names)
    if not unknown_names:
        # No unknowns means nothing to identify — trivially well-posed.
        return WellposednessReport(rank=0, n_unknowns=0, condition_number=1.0)

    # Bounds-too-wide soft check. Loose bounds = bad init (PINA picks
    # midpoint) = slow / no convergence. Heuristic: fire when the absolute
    # width is large (>20 units) OR the midpoint is large (>50 in magnitude).
    # These thresholds catch the (0, 100)+ class of bounds that empirically
    # stall the damped oscillator; they're intentionally lenient to avoid
    # false positives on legitimately large-magnitude problems.
    for name in unknown_names:
        lo, hi = compiled.unknown_bounds[name]
        width = hi - lo
        mid = 0.5 * (lo + hi)
        if width > 20.0 or abs(mid) > 50.0:
            warnings.warn(
                f"Unknown {name!r} bounds {(lo, hi)!r} are wide (width "
                f"{width:.1f}, midpoint {mid:.3g}); PINA initializes at the "
                f"midpoint, which may be far from truth and slow convergence. "
                f"Consider tighter bounds (~2-3× your best guess) for faster "
                f"discovery.",
                BoundsTooWideWarning,
                stacklevel=2,
            )

    device = torch.device("cpu")  # one-shot — keep it simple
    net = network.to(device)
    was_training = net.training
    net.eval()

    t_pts = _sample_collocation(problem, n, compiled=compiled).to(device)

    # Baseline residual at θ₀.
    theta0 = {
        name: problem.unknown_parameters[name].detach().clone().to(device)
        for name in unknown_names
    }
    base = _evaluate_residuals(net, compiled, t_pts, theta0).detach().cpu().numpy()

    # Centered differences: S[:, k] = (r(θ + ε·e_k) - r(θ - ε·e_k)) / (2 ε)
    cols = []
    for k, name in enumerate(unknown_names):
        scale = max(1.0, abs(float(theta0[name].reshape(-1)[0].item())))
        dtheta = eps * scale

        theta_p = {
            n_: (t.clone() + (dtheta if n_ == name else 0.0)) for n_, t in theta0.items()
        }
        theta_m = {
            n_: (t.clone() - (dtheta if n_ == name else 0.0)) for n_, t in theta0.items()
        }
        r_p = _evaluate_residuals(net, compiled, t_pts, theta_p).detach().cpu().numpy()
        r_m = _evaluate_residuals(net, compiled, t_pts, theta_m).detach().cpu().numpy()
        cols.append((r_p - r_m) / (2.0 * dtheta))
    S = np.stack(cols, axis=1)  # shape (n_outputs, n_unknowns)

    if was_training:
        net.train()

    rank = int(np.linalg.matrix_rank(S, tol=1e-6))
    try:
        sv = np.linalg.svd(S, compute_uv=False)
        cond = float(sv[0] / max(sv[-1], 1e-30))
    except Exception:
        cond = float("inf")

    norms = {name: float(np.linalg.norm(S[:, k])) for k, name in enumerate(unknown_names)}
    max_norm = max(norms.values()) if norms else 1.0
    blind = [
        n_ for n_, v in norms.items()
        if v < max(rel_blind_threshold * max_norm, 1e-12)
    ]

    passed = rank >= len(unknown_names) and cond < cond_threshold
    report = WellposednessReport(
        rank=rank,
        n_unknowns=len(unknown_names),
        condition_number=cond,
        column_norms=norms,
        blind_unknowns=blind,
        passed=passed,
    )

    if not passed:
        msg = ["Inverse problem is not well-posed:"]
        msg.append(f"  rank(S) = {rank} but {len(unknown_names)} unknowns declared")
        msg.append(f"  cond(S) = {cond:.2e}  (threshold {cond_threshold:.0e})")
        if blind:
            msg.append(f"  Unknowns with ~zero residual sensitivity: {blind!r}")
        msg.append("Fix: add a sensor/equation that depends on the missing unknown(s), "
                   "or remove unknown(s) that don't appear in any equation.")
        raise UnidentifiableError("\n".join(msg))

    return report
