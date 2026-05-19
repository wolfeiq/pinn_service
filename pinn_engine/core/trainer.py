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
from pina.optim import TorchOptimizer
from pina.loss import ScalarWeighting


class LabeledDataPINN(PinaPINN):
    """PINN solver that respects LabelTensor labels in data conditions.

    PINA's stock ``loss_data`` MSE's the full network output against the
    target. For multi-output systems (e.g. Lorenz: 3 states) with per-state
    sensors (each target is a single column), this broadcasts wrong:
    ``(N, 3)`` vs ``(N, 1)``. Here we extract only the columns of the network
    output whose labels appear in the target's labels before computing MSE.
    """

    def loss_data(self, input, target):
        out = self.forward(input)
        if hasattr(target, "labels") and hasattr(out, "labels"):
            cols = [c for c in target.labels if c in out.labels]
            if cols and len(cols) != len(out.labels):
                out = out.extract(cols)
        return self._loss_fn(out, target)

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

    # Optimization
    lr: float = Field(default=1e-3, gt=0)
    adam_epochs: int = Field(default=2000, ge=0)
    lbfgs_iters: int = Field(default=0, ge=0)

    # Loss balancing
    balancer: str = Field(default="none", pattern=r"^(none|sapinn|lra)$")
    lam_data_init: float = 1.0
    lam_physics_init: float = 1.0

    # Domain
    t_range: Tuple[float, float] = (0.0, 1.0)
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

    @property
    def objective_value(self) -> float:
        return self.final_loss


def _select_balancer(name: str) -> Optional[pl.Callback]:
    from pinn_engine.core.loss_balancer import SAPinnBalancer, LRABalancer

    if name == "none":
        return None
    if name == "sapinn":
        return SAPinnBalancer()
    if name == "lra":
        return LRABalancer()
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
    problem = build_problem(compiled, data, t_range=config.t_range)
    problem.discretise_domain(n=config.n_collocation, mode="random", domains="all")

    network = build_network(
        input_dim=1,  # Phase 1+2 = time-dependent ODE problems only
        output_dim=len(compiled.state_names),
        depth=config.depth,
        width=config.width,
        activation=config.activation,
        layer_norm=config.layer_norm,
    )

    if not config.skip_preflight:
        check_wellposedness(problem, network, compiled)

    bal_cb = _select_balancer(config.balancer)
    if bal_cb is not None:
        callbacks.append(bal_cb)

    # Build per-condition static weights from TrainConfig (data conditions
    # get ``lam_data_init``, physics conditions get ``lam_physics_init``).
    cond_weights = {}
    for cname in problem.conditions.keys():
        if cname.startswith("data_"):
            cond_weights[cname] = float(config.lam_data_init)
        elif cname.startswith("physics_"):
            cond_weights[cname] = float(config.lam_physics_init)
        else:
            cond_weights[cname] = 1.0
    weighting = ScalarWeighting(cond_weights)

    solver = LabeledDataPINN(
        problem=problem,
        model=network,
        optimizer=TorchOptimizer(torch.optim.Adam, lr=config.lr),
        weighting=weighting,
    )
    # Make context available to our diagnostic callbacks.
    solver._compiled_system = compiled
    solver._engine_data = data

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

    # Optional L-BFGS finetune. Skipped when the problem has inverse
    # unknowns: PINA's solver attaches `unknown_parameters` as a second
    # `param_group`, but torch's L-BFGS asserts `len(param_groups) == 1`.
    # Until that's resolved upstream (or we ship a custom solver that
    # merges param groups for L-BFGS), Adam is the only supported phase
    # for inverse problems.
    has_unknowns = bool(getattr(problem, "unknown_parameters", {}))
    if config.lbfgs_iters > 0 and has_unknowns:
        import warnings as _w
        _w.warn(
            "L-BFGS finetune skipped: incompatible with PINA's InverseProblem "
            "(adds unknown_parameters as a second param_group, which torch.LBFGS "
            "does not support). Use Adam-only for inverse problems.",
            RuntimeWarning,
        )
    if config.lbfgs_iters > 0 and not has_unknowns:
        lbfgs_solver = PinaPINN(
            problem=problem,
            model=network,
            optimizer=TorchOptimizer(
                torch.optim.LBFGS,
                lr=0.5,
                max_iter=config.lbfgs_iters,
                history_size=50,
                line_search_fn="strong_wolfe",
            ),
        )
        trainer_lbfgs = PinaTrainer(
            solver=lbfgs_solver,
            max_epochs=1,
            accelerator=config.accelerator,
            devices=config.devices,
            deterministic=config.deterministic,
            log_every_n_steps=config.log_every_n_steps,
            callbacks=[cb for cb in callbacks if cb is not bal_cb],
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
    )
