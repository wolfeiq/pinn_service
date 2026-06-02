"""Cramér-Rao Lower Bound preflight diagnostic.

Given a forward simulator ``y = h(θ) + ε`` with Gaussian sensor noise, the
*best possible* uncertainty on the recovered θ — achieved by any unbiased
estimator, including a perfectly-trained PINN — is bounded by the Cramér-Rao
inequality:

    Cov(θ_hat) ≥ F⁻¹,
    F        =  Sᵀ · diag(1/σ²) · S       (Fisher information)
    S[k, i]  =  ∂y_k / ∂θ_i                (sensitivity matrix)

This is a *preflight* check — it doesn't need training, just N+1 forward
simulations (truth + one perturbation per unknown). It catches the most
common failure mode in inverse-problem setups: **the data physically cannot
identify the parameters**. If CRLB SE on θ_i is, say, 30% of |θ_i|, then no
PINN, EKF, or hand-tuned recipe will do better — the data is the limit.

Examples (verified empirically — see ``compute_template_crlb_report``):
  * Damped oscillator: tiny CRLB SE → empirical convergence matches.
  * 1-DOF nonlinear-drag: moderate CRLB SE on c_quad → empirically partial-id.
  * 3-DOF coupled-drag: c_y has large CRLB SE → empirically c_y is the stuck unknown.

The estimator uses central finite differences for sensitivity; for noisy or
nonlinear forwards, ``perturb_rel`` may need tuning, but 1e-3 is robust.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np


@dataclass
class CRLBResult:
    """Per-unknown lower bounds on standard error and 95% CI half-width.

    Attributes:
        se          : {name: lower-bound standard error}
        se_relative : {name: SE / |truth|}  (interpretable as "best achievable
                      relative error" assuming an unbiased estimator)
        ci95_half   : {name: 1.96 * SE}     (best-achievable 95% CI half-width)
        fisher_info : the Fisher information matrix in unknown order
        covariance  : the inverse Fisher, i.e. CRLB covariance
        unknown_order : the ordering of unknowns in the matrices
    """
    se: Dict[str, float]
    se_relative: Dict[str, float]
    ci95_half: Dict[str, float]
    fisher_info: np.ndarray
    covariance: np.ndarray
    unknown_order: List[str]

    def summary_table(self) -> str:
        lines = [f"{'Unknown':>10}  {'truth':>10}  {'CRLB SE':>10}  {'SE/|truth|':>10}  {'95% CI half':>12}"]
        lines.append("-" * 60)
        for name in self.unknown_order:
            se = self.se[name]
            rel = self.se_relative[name]
            ci = self.ci95_half[name]
            lines.append(f"{name:>10}  {'':>10}  {se:>10.4g}  {rel:>10.2%}  {ci:>12.4g}")
        return "\n".join(lines)


def compute_crlb(
    forward_fn: Callable[[Dict[str, float], int], Dict[str, Tuple[np.ndarray, np.ndarray]]],
    truth: Dict[str, float],
    noise_stds: Dict[str, float],
    perturb_rel: float = 1e-3,
    perturb_abs_floor: float = 1e-6,
    seed: int = 0,
) -> CRLBResult:
    """Compute the Cramér-Rao lower bound on the unknowns of an inverse problem.

    Args:
        forward_fn: callable ``(truth_dict, seed) -> data_dict``, where
            ``data_dict[sensor] = (input_array, observation_array)`` (matches
            the engine's synthetic-data generator contract).
        truth: ``{unknown_name: true_value}``.
        noise_stds: ``{sensor_name: sigma}`` — the Gaussian noise level on each
            sensor (used to weight the Fisher information).
        perturb_rel: per-unknown relative perturbation for the finite-difference
            sensitivity. Default 1e-3 (0.1%).
        perturb_abs_floor: minimum absolute perturbation (used when |truth| < 1e-6
            so we don't divide by zero).
        seed: forward-simulator seed (held fixed across baseline + perturbations
            so the only thing varying is the unknown's value).

    Returns:
        :class:`CRLBResult` with per-unknown SE bounds.
    """
    names = list(truth.keys())

    # 1) Baseline simulation at truth — use seed=0 for noise but we'll subtract
    #    the *clean* signal by also running with seed=None… actually for CRLB we
    #    want the *signal* sensitivity, not noise. Easiest: use the same seed
    #    for both baseline and perturbed, so noise cancels in the finite
    #    difference (y_pert - y_base) / δ.
    baseline = forward_fn(dict(truth), seed)

    # 2) Per-unknown central-difference sensitivity.
    sens_per_unknown: Dict[str, Dict[str, np.ndarray]] = {}
    for name in names:
        val = float(truth[name])
        delta = perturb_rel * abs(val) if abs(val) > perturb_abs_floor else perturb_abs_floor
        truth_plus = dict(truth); truth_plus[name] = val + delta
        truth_minus = dict(truth); truth_minus[name] = val - delta
        data_plus = forward_fn(truth_plus, seed)
        data_minus = forward_fn(truth_minus, seed)
        sens: Dict[str, np.ndarray] = {}
        for sensor_name in baseline:
            _, y_plus = data_plus[sensor_name]
            _, y_minus = data_minus[sensor_name]
            sens[sensor_name] = (np.asarray(y_plus, dtype=np.float64)
                                 - np.asarray(y_minus, dtype=np.float64)) / (2 * delta)
        sens_per_unknown[name] = sens

    # 3) Assemble the stacked sensitivity matrix S [n_obs_total × n_unknowns],
    #    weight rows by 1/σ_k for each sensor block.
    columns: List[np.ndarray] = []
    for name in names:
        per_sensor = sens_per_unknown[name]
        weighted_rows: List[np.ndarray] = []
        for sensor_name, sens_vec in per_sensor.items():
            sigma = float(noise_stds.get(sensor_name, 1.0))
            weighted_rows.append(np.asarray(sens_vec, dtype=np.float64).reshape(-1) / sigma)
        columns.append(np.concatenate(weighted_rows))
    S_weighted = np.stack(columns, axis=1)   # (n_obs, n_unknowns)

    # 4) Fisher information & covariance. Use pseudoinverse for robustness when
    #    F is ill-conditioned (= partial-identifiability, exactly what we want
    #    to surface — the diagonal of pinv blows up, which is the signal).
    F = S_weighted.T @ S_weighted
    try:
        cov = np.linalg.pinv(F)
    except np.linalg.LinAlgError:
        cov = np.full_like(F, np.inf)
    diag = np.diag(cov).copy()
    diag[diag < 0] = np.nan   # numerical noise on a nearly-singular F
    se_vec = np.sqrt(diag)

    se = {name: float(se_vec[i]) for i, name in enumerate(names)}
    se_rel = {
        name: float(se_vec[i]) / max(abs(truth[name]), 1e-9)
        for i, name in enumerate(names)
    }
    ci = {name: 1.96 * v for name, v in se.items()}
    return CRLBResult(
        se=se,
        se_relative=se_rel,
        ci95_half=ci,
        fisher_info=F,
        covariance=cov,
        unknown_order=names,
    )


# -------- convenience: compute CRLB for a registered template -----------------


def compute_template_crlb(template_name: str, perturb_rel: float = 1e-3) -> CRLBResult:
    """Compute CRLB for one of the bundled inverse templates, using the template's
    own ``synthetic_data`` generator as the forward model. The sensor noise
    levels are pulled from the system's ``sensors[*].noise_std`` declarations.
    """
    from pinn_engine.dsl.templates import get_template
    tpl = get_template(template_name)

    # Discover the noise levels declared on the sensors.
    system = tpl.system()
    noise_stds: Dict[str, float] = {}
    for sens in system.sensors:
        # Pseudo-sensors (BC/IC) have noise_std == 0 — substitute a tiny floor
        # so we don't divide by zero in the Fisher weighting (these conditions
        # carry near-infinite weight, which is physically correct).
        sigma = float(getattr(sens, "noise_std", 0.0))
        noise_stds[sens.name] = max(sigma, 1e-6)

    # Discover truth from the template.
    truth = dict(getattr(tpl, "truth", {}))
    if not truth:
        raise ValueError(f"template {template_name!r} has no 'truth' attribute")

    # Wrap synthetic_data so it accepts a perturbed truth dict. Most generators
    # take their unknowns as kwargs (e.g. generate_nonlinear_drag_1d(c_lin=..., ...)),
    # so we forward via kwargs. The template's synthetic_data(seed) doesn't
    # accept truth overrides — go through pinn_engine.data.synthetic directly.
    from pinn_engine.data import synthetic as syn_mod
    # Map template name -> generator function name in synthetic.py.
    gen_map = {
        "damped_oscillator": "generate_damped_oscillator",
        "lorenz":            "generate_lorenz",
        "pendulum":          "generate_pendulum",
        "nonlinear_drag_1d": "generate_nonlinear_drag_1d",
        "coupled_drag_3d":   "generate_coupled_drag_3d",
        "diffusion_1d":      "generate_diffusion_1d",
        "cosserat_rod":      "generate_cosserat_rod",
    }
    gen_name = gen_map.get(template_name)
    if gen_name is None or not hasattr(syn_mod, gen_name):
        raise ValueError(f"no generator mapping for template {template_name!r}")
    gen = getattr(syn_mod, gen_name)

    # Cosserat truth is named 'E_unit' (dimensionless multiplier) but the generator
    # takes the dimensional E. Translate.
    truth_to_gen_kwargs = {
        "cosserat_rod": lambda t: {"E": float(t.get("E_unit", 1.0)) * 1.0e6},
    }
    translator = truth_to_gen_kwargs.get(template_name, lambda t: dict(t))

    def forward_fn(perturbed_truth, seed):
        kwargs = translator(perturbed_truth)
        result = gen(seed=seed, **kwargs)
        # Generators return either (data, truth) or just data — normalize.
        if isinstance(result, tuple):
            data = result[0]
        else:
            data = result
        return data

    return compute_crlb(forward_fn, truth, noise_stds, perturb_rel=perturb_rel)
