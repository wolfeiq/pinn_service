"""The training entry point.

``train(system, data, config, callbacks)`` is the one function the rest of the
engine (templates, examples, AutoML, CLI) calls. Internally it:

1. Compiles the system.
2. Runs the well-posedness pre-flight.
3. Builds a PINA :class:`InverseProblem`.
4. Builds the MLP via the network factory.
5. Runs Adam epochs with the optional loss balancer attached.
6. Runs an L-BFGS finetune with the balancer frozen.
7. Collects results into a :class:`TrainResult`.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from pydantic import BaseModel, Field

from pina import Trainer as PinaTrainer
from pina.solver import PINN as PinaPINN
from pina.solver import CausalPINN as PinaCausalPINN
from pina.optim import TorchOptimizer
from pina.loss import ScalarWeighting

from pinn_engine.core.weightings import SAPinnWeighting, LRAWeighting


class CausalLabeledDataPINN(PinaCausalPINN):
    """CausalPINN variant of LabeledDataPINN.

    Combines two engine-level fixes:
      * Column extraction for multi-output data conditions (same as
        :class:`LabeledDataPINN.loss_data`).
      * ``param_lr_scale`` for unknown parameters (same override as
        :class:`LabeledDataPINN.configure_optimizers`).

    Plus PINA's CausalPINN time-causal residual weighting (Wang 2022,
    arXiv:2203.07404) which empirically gives 10-100× iteration
    reduction on wave-equation / chaotic-system inverse problems.
    """

    _engine_param_lr_scale: float = 1.0

    def loss_data(self, input, target):
        out = self.forward(input)
        if hasattr(target, "labels") and hasattr(out, "labels"):
            cols = [c for c in target.labels if c in out.labels]
            if cols and len(cols) != len(out.labels):
                out = out.extract(cols)
        return self._loss_fn(out, target)

    def loss_phys(self, samples, equation):
        # Mirrors PINA's CausalPINN.loss_phys but stashes per-bucket
        # residuals and causal weights so the eps-annealer callback and
        # diagnostics can detect the failure mode where ω_i collapses
        # to ~0 and physics is silently muted.
        chunks, labels = self._split_tensor_into_chunks(samples)
        time_loss = []
        for chunk in chunks:
            chunk.labels = labels
            residual = self.compute_residual(samples=chunk, equation=equation)
            loss_val = self._loss_fn(
                torch.zeros_like(residual, requires_grad=True), residual
            )
            time_loss.append(loss_val)
        time_loss = torch.stack(time_loss)
        with torch.no_grad():
            weights = self._compute_weights(time_loss)
            self._last_causal_time_loss = time_loss.detach()
            self._last_causal_weights = weights.detach()
            max_bucket = float(time_loss.max())
            min_weight = float(weights.min())
            active = int((weights > 1e-3).sum())
            self.log("causal/max_bucket_loss", max_bucket, on_step=False, on_epoch=True)
            self.log("causal/min_weight", min_weight, on_step=False, on_epoch=True)
            self.log("causal/active_buckets", float(active), on_step=False, on_epoch=True)
            self.log("causal/eps", float(self._eps), on_step=False, on_epoch=True)
        return (weights * time_loss).mean()

    def configure_optimizers(self):
        from pina.problem import InverseProblem

        self.optimizer.hook(self.model.parameters())
        scale = float(getattr(self, "_engine_param_lr_scale", 1.0))
        if isinstance(self.problem, InverseProblem) and scale != 1.0:
            try:
                base_lr = self.optimizer.instance.param_groups[0]["lr"]
            except Exception:
                base_lr = 1e-3
            self.optimizer.instance.add_param_group({
                "params": [self._params[v] for v in self.problem.unknown_variables],
                "lr": base_lr * scale,
            })
        elif isinstance(self.problem, InverseProblem):
            self.optimizer.instance.add_param_group({
                "params": [self._params[v] for v in self.problem.unknown_variables]
            })
        self.scheduler.hook(self.optimizer)
        return ([self.optimizer.instance], [self.scheduler.instance])


class LabeledDataPINN(PinaPINN):
    """PINN solver that respects LabelTensor labels in data conditions.

    PINA's stock ``loss_data`` MSE's the full network output against the
    target. For multi-output systems (e.g. Lorenz: 3 states) with per-state
    sensors (each target is a single column), this broadcasts wrong:
    ``(N, 3)`` vs ``(N, 1)``. Here we extract only the columns of the network
    output whose labels appear in the target's labels before computing MSE.

    Also exposes ``param_lr_scale`` so unknown parameters can be optimised
    at a different learning rate than network weights.
    """

    # Multiplier on the base LR applied only to the unknown_parameters group.
    # The trainer sets this via attribute assignment after solver construction
    # since PINA's __init__ doesn't pass through extra kwargs.
    _engine_param_lr_scale: float = 1.0

    def loss_data(self, input, target):
        out = self.forward(input)
        if hasattr(target, "labels") and hasattr(out, "labels"):
            cols = [c for c in target.labels if c in out.labels]
            if cols and len(cols) != len(out.labels):
                out = out.extract(cols)
        return self._loss_fn(out, target)

    def configure_optimizers(self):
        """Override PINA's default to apply ``_engine_param_lr_scale``
        on the unknown_parameters' optimizer group."""
        from pina.problem import InverseProblem

        self.optimizer.hook(self.model.parameters())
        scale = float(getattr(self, "_engine_param_lr_scale", 1.0))
        if isinstance(self.problem, InverseProblem) and scale != 1.0:
            # Read the network's lr (the one PINA just hooked).
            try:
                base_lr = self.optimizer.instance.param_groups[0]["lr"]
            except Exception:
                base_lr = 1e-3
            scaled_lr = base_lr * scale
            self.optimizer.instance.add_param_group({
                "params": [self._params[v] for v in self.problem.unknown_variables],
                "lr": scaled_lr,
            })
        elif isinstance(self.problem, InverseProblem):
            # No scaling — fall back to PINA's default behaviour.
            self.optimizer.instance.add_param_group({
                "params": [self._params[v] for v in self.problem.unknown_variables]
            })
        self.scheduler.hook(self.optimizer)
        return ([self.optimizer.instance], [self.scheduler.instance])


class LBFGSInversePINN(LabeledDataPINN):
    """L-BFGS-compatible PINN solver for inverse problems.

    PINA's stock ``configure_optimizers`` attaches the network parameters
    to the optimizer and *then* adds ``unknown_parameters`` as a second
    ``param_group`` via ``add_param_group``. That's fine for Adam, but
    ``torch.optim.LBFGS`` asserts ``len(param_groups) == 1`` and crashes.

    This subclass merges network params + unknown_parameters into a single
    flat iterable BEFORE the optimizer is hooked, so L-BFGS sees one group
    and is happy. Inherits ``loss_data`` from :class:`LabeledDataPINN` so
    multi-output Lorenz-style problems still work.
    """

    def configure_optimizers(self):
        from pina.problem import InverseProblem

        all_params = list(self.model.parameters())
        if isinstance(self.problem, InverseProblem) and self._params is not None:
            for var in self.problem.unknown_variables:
                all_params.append(self._params[var])

        self.optimizer.hook(all_params)
        self.scheduler.hook(self.optimizer)
        return ([self.optimizer.instance], [self.scheduler.instance])

from pinn_engine.dsl.system import System, CompiledSystem
from pinn_engine.core.networks import build_network
from pinn_engine.core.problem import build_problem
from pinn_engine.preflight import check_wellposedness


class TrainConfig(BaseModel):
    """All hyperparameters for a single training run."""

    # Architecture
    depth: int = Field(default=4, ge=2, le=12)
    width: int = Field(default=64, ge=8, le=512)
    activation: str = Field(default="tanh")
    layer_norm: bool = True
    # Optional Fourier feature input encoding (0 = off; n > 0 adds 2n cols).
    fourier_features: int = Field(default=0, ge=0, le=256)
    fourier_sigma: float = Field(default=1.0, gt=0)

    # Optimization
    lr: float = Field(default=1e-3, gt=0)
    # Separate LR multiplier for unknown parameters. Network weights see
    # `lr`; unknowns see `lr * param_lr_scale`. Larger values let inverse
    # parameters traverse their search ranges faster — critical for PDE
    # inverse problems where Adam's per-parameter normalization otherwise
    # makes unknowns crawl. Default 1.0 = legacy single-LR behaviour.
    param_lr_scale: float = Field(default=1.0, gt=0)
    # Cosine-anneal the unknowns' LR from full scale → ``param_lr_min_scale ×
    # scale`` over the Adam phase. Prevents overshoot — start aggressive so the
    # unknown moves into the right basin, settle gently. 1.0 = no anneal.
    param_lr_min_scale: float = Field(default=1.0, gt=0, le=1.0)
    # Two-phase LR: hold full base LR until the watched unknown crosses
    # ``param_lr_trigger_below``, then start cosine taper from that epoch
    # over the remaining epoch budget. Useful when an unknown must escape
    # a spurious basin before braking — full LR keeps the gradient signal
    # alive through the basin, cosine kicks in only after escape. None =
    # standard single-phase cosine.
    param_lr_trigger_below: Optional[float] = None
    # Which unknown to watch for the trigger. None = first unknown.
    param_lr_trigger_param: Optional[str] = None
    # Solver: "pinn" (vanilla) or "causal" (time-causal residual weighting).
    # CausalPINN is the standard fix for wave/chaotic-PDE inverse problems
    # (Wang 2022, arXiv:2203.07404).
    solver_type: str = Field(default="pinn", pattern=r"^(pinn|causal)$")
    # CausalPINN ε. PINA's default is 100, which collapses ω_i = exp(-ε·Σ L_r)
    # to ~0 on epoch 1 for any non-trivial residual, silently muting physics.
    # Wang 2022 §3.2 recommends starting small (≈1) and annealing up.
    causal_eps: float = Field(default=1.0, gt=0)
    # If true, attach CausalEpsAnnealer: start at causal_eps, ×10 each time
    # max-bucket residual drops below causal_eps_threshold, cap at causal_eps_max.
    causal_eps_anneal: bool = False
    causal_eps_max: float = Field(default=100.0, gt=0)
    causal_eps_threshold: float = Field(default=1e-2, gt=0)
    adam_epochs: int = Field(default=2000, ge=0)
    lbfgs_iters: int = Field(default=0, ge=0)

    # Loss balancing
    balancer: str = Field(default="none", pattern=r"^(none|sapinn|lra)$")
    lam_data_init: float = 1.0
    lam_physics_init: float = 1.0

    # Domain
    t_range: Tuple[float, float] = (0.0, 1.0)
    spatial_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    n_collocation: int = 1000
    batch_size: int = 256

    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    accelerator: str = "auto"
    devices: int = 1

    # Output
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    log_every_n_steps: int = 50

    # Toggles
    skip_preflight: bool = False

    model_config = {"protected_namespaces": ()}  # allow "model_*" field names if added


@dataclass
class TrainResult:
    """What the trainer hands back."""

    run_id: str
    final_params: Dict[str, float]
    final_param_history: Dict[str, List[float]] = field(default_factory=dict)
    final_loss: float = float("nan")
    history: List[Dict[str, float]] = field(default_factory=list)
    callback_outputs: Dict[str, Any] = field(default_factory=dict)
    problem: Optional[Any] = None  # the PINA Problem instance
    network: Optional[Any] = None  # the trained MLP
    compiled: Optional[CompiledSystem] = None
    config: Optional[TrainConfig] = None
    weighting: Optional[Any] = None  # the PINA WeightingInterface — has .history for dynamics

    @property
    def objective_value(self) -> float:
        return self.final_loss


def _build_weighting(name: str, cond_weights: Dict[str, float]):
    """Pick the weighting strategy for the PINA solver.

    * ``"none"`` → static :class:`ScalarWeighting` using ``cond_weights``.
    * ``"sapinn"`` → dynamic SA-PINN with learnable λ per condition. Initialised
      from ``cond_weights`` so the user's static prior is the starting point.
    * ``"lra"`` → Wang-Teng-Perdikaris LRA via gradient-norm ratio.
    """
    if name == "none":
        return ScalarWeighting(cond_weights)
    if name == "sapinn":
        # Initialise λ from the static cond_weights (mean of values) so SA-PINN
        # starts from a known-good operating point rather than λ=1 everywhere.
        return SAPinnWeighting(lam_init=max(1.0, max(cond_weights.values()) / 10.0))
    if name == "lra":
        return LRAWeighting(alpha=0.9, lam_init=1.0)
    raise ValueError(f"Unknown balancer {name!r}")


def train(
    system: System,
    data: Dict[str, Tuple[Any, Any]],
    config: Optional[TrainConfig] = None,
    callbacks: Optional[List[pl.Callback]] = None,
) -> TrainResult:
    """Run a single training trial. See module docstring for the flow."""
    config = config or TrainConfig()
    callbacks = list(callbacks or [])

    pl.seed_everything(config.seed, workers=True)

    compiled = system.compile()
    problem = build_problem(
        compiled, data,
        t_range=config.t_range,
        spatial_ranges=config.spatial_ranges,
    )
    problem.discretise_domain(n=config.n_collocation, mode="random", domains="all")

    network = build_network(
        input_dim=len(compiled.input_names),  # 1 for ODE, 2+ for PDE
        output_dim=len(compiled.state_names),
        depth=config.depth,
        width=config.width,
        activation=config.activation,
        layer_norm=config.layer_norm,
        fourier_features=config.fourier_features,
        fourier_sigma=config.fourier_sigma,
    )

    if not config.skip_preflight:
        check_wellposedness(problem, network, compiled)

    # Build per-condition initial weights from TrainConfig (data conditions
    # get ``lam_data_init``, physics conditions get ``lam_physics_init``).
    # These become the static weights for balancer="none", and the starting
    # point for sapinn / lra dynamic balancers.
    cond_weights = {}
    for cname in problem.conditions.keys():
        if cname.startswith("data_"):
            cond_weights[cname] = float(config.lam_data_init)
        elif cname.startswith("physics_"):
            cond_weights[cname] = float(config.lam_physics_init)
        else:
            cond_weights[cname] = 1.0

    weighting = _build_weighting(config.balancer, cond_weights)

    is_causal = config.solver_type == "causal"
    SolverCls = CausalLabeledDataPINN if is_causal else LabeledDataPINN

    # Do we drive the unknowns' LR ourselves (cosine and/or two-phase trigger)?
    needs_scheduler = (
        (config.param_lr_min_scale < 1.0 or config.param_lr_trigger_below is not None)
        and config.param_lr_scale > 1.0
    )

    solver_kwargs = dict(
        problem=problem,
        model=network,
        optimizer=TorchOptimizer(torch.optim.Adam, lr=config.lr),
        weighting=weighting,
    )
    if needs_scheduler:
        # PINA defaults to ConstantLR(factor=1/3, total_iters=5) — a warmup
        # that (a) runs the LR at 1/3 for 5 epochs and (b) at its milestone
        # multiplies the *current* group LR by 3. When UnknownsParamLRScheduler
        # also writes the group LR, the two fight: the milestone turns our
        # 0.5 into 1.5 for one epoch. Pass a true-constant scheduler (factor 1,
        # no milestone) so our callback is the sole controller of the LR.
        from pina.optim import TorchScheduler
        solver_kwargs["scheduler"] = TorchScheduler(
            torch.optim.lr_scheduler.ConstantLR, factor=1.0, total_iters=1
        )
    if is_causal:
        # Without this, PINA defaults to eps=100 → ω_i collapse → physics ignored.
        eps_init = config.causal_eps if not config.causal_eps_anneal else config.causal_eps
        solver_kwargs["eps"] = eps_init
    solver = SolverCls(**solver_kwargs)
    # Make context available to our diagnostic callbacks.
    solver._compiled_system = compiled
    solver._engine_data = data
    # Stash the weighting on the solver so diagnostic callbacks can inspect
    # its history without re-grepping logs.
    solver._engine_weighting = weighting
    # Apply the separate-LR-for-unknowns config option.
    solver._engine_param_lr_scale = float(config.param_lr_scale)

    # Attach the unknowns' LR scheduler (cosine and/or two-phase trigger).
    # ``needs_scheduler`` was computed above so we could also swap PINA's
    # default warmup scheduler for a true constant before solver construction.
    if needs_scheduler:
        from pinn_engine.core.param_lr_scheduler import UnknownsParamLRScheduler
        callbacks.append(UnknownsParamLRScheduler(
            max_epochs=config.adam_epochs,
            min_scale=config.param_lr_min_scale,
            trigger_below=config.param_lr_trigger_below,
            trigger_param=config.param_lr_trigger_param,
        ))

    # Wang 2022 §3.2 ε-annealing: bump ε once max-bucket residual is small.
    if is_causal and config.causal_eps_anneal:
        from pinn_engine.core.causal_eps_scheduler import CausalEpsAnnealer
        callbacks.append(CausalEpsAnnealer(
            eps_max=config.causal_eps_max,
            threshold=config.causal_eps_threshold,
        ))

    trainer_adam = PinaTrainer(
        solver=solver,
        max_epochs=config.adam_epochs,
        accelerator=config.accelerator,
        devices=config.devices,
        deterministic=config.deterministic,
        log_every_n_steps=config.log_every_n_steps,
        callbacks=callbacks,
        enable_progress_bar=True,
        batch_size=config.batch_size,
    )
    trainer_adam.train()

    # Optional L-BFGS finetune. Inverse problems route through
    # LBFGSInversePINN, which merges model params + unknown_parameters
    # into a single optimizer param_group (torch.optim.LBFGS asserts
    # ``len(param_groups) == 1``).
    has_unknowns = bool(getattr(problem, "unknown_parameters", {}))
    if config.lbfgs_iters > 0:
        # Freeze the weighting before L-BFGS. Quasi-Newton expects a static
        # loss surface; an adaptive balancer (SAPinnWeighting / LRAWeighting)
        # changing weights every step breaks L-BFGS's history approximation.
        if isinstance(weighting, (SAPinnWeighting, LRAWeighting)) and weighting.history:
            frozen_weights = dict(weighting.history[-1])
            lbfgs_weighting = ScalarWeighting(frozen_weights)
        else:
            lbfgs_weighting = weighting  # already static (ScalarWeighting)

        SolverCls = LBFGSInversePINN if has_unknowns else LabeledDataPINN
        lbfgs_solver = SolverCls(
            problem=problem,
            model=network,
            optimizer=TorchOptimizer(
                torch.optim.LBFGS,
                lr=0.5,
                max_iter=config.lbfgs_iters,
                history_size=50,
                line_search_fn="strong_wolfe",
            ),
            weighting=lbfgs_weighting,
        )
        lbfgs_solver._compiled_system = compiled
        lbfgs_solver._engine_data = data
        lbfgs_solver._engine_weighting = lbfgs_weighting
        trainer_lbfgs = PinaTrainer(
            solver=lbfgs_solver,
            max_epochs=1,
            accelerator=config.accelerator,
            devices=config.devices,
            deterministic=config.deterministic,
            log_every_n_steps=config.log_every_n_steps,
            callbacks=callbacks,
            enable_progress_bar=True,
            batch_size=config.batch_size,
        )
        trainer_lbfgs.train()

    final_params = {
        name: float(p.detach().cpu().item())
        for name, p in problem.unknown_parameters.items()
    }
    final_loss = float("nan")
    if trainer_adam.logged_metrics:
        for k in ("train_loss_epoch", "train_loss", "loss_epoch", "loss", "mean_loss"):
            if k in trainer_adam.logged_metrics:
                final_loss = float(trainer_adam.logged_metrics[k])
                break

    return TrainResult(
        run_id=config.run_id,
        final_params=final_params,
        final_loss=final_loss,
        callback_outputs={
            getattr(cb, "name", cb.__class__.__name__): getattr(cb, "output", None)
            for cb in callbacks
            if hasattr(cb, "output")
        },
        problem=problem,
        network=network,
        compiled=compiled,
        config=config,
        weighting=weighting,
    )
