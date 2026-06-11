"""Native PINN solver + collocation problem + training loop (PINA-free).

A compact, self-contained replacement for the PINA solver path. Handles ODE and
PDE inverse problems, multi-output systems, per-condition loss weighting, and a
separate learning rate for the unknown parameters. The physics residual and its
autograd derivatives come from the engine's own DSL compiler; this module just
samples collocation points, evaluates data + physics losses, and steps Adam.

This is the core Adam path. The advanced LR drivers (warmup, two-phase, adaptive
controller), CausalPINN, and RAR are layered separately as the migration
proceeds; the engine's existing PINA path remains the default until parity is
verified across every template.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from pinn_engine.core.native.labeltensor import LabelTensor


@dataclass
class PhysicsCondition:
    residual: Callable                 # residual(input_lt, output_lt, params) -> (N,1)
    input_names: Tuple[str, ...]       # column order of the collocation input
    ranges: Dict[str, Tuple[float, float]]


@dataclass
class DataCondition:
    input: LabelTensor                 # (N, n_inputs)
    target: torch.Tensor               # (N, n_obs)
    observed: List[str]                # state columns this sensor measures


@dataclass
class NativeProblem:
    """Container mirroring the attributes callbacks/diagnostics read."""
    state_names: List[str]
    input_names: Tuple[str, ...]
    physics: List[PhysicsCondition]
    data: Dict[str, DataCondition]
    unknown_parameters: Dict[str, torch.nn.Parameter]
    unknown_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    @property
    def unknown_variables(self) -> List[str]:
        return list(self.unknown_parameters.keys())


def _sample(cond: PhysicsCondition, n: int) -> LabelTensor:
    cols = []
    for name in cond.input_names:
        lo, hi = cond.ranges[name]
        cols.append(torch.rand(n, 1) * (hi - lo) + lo)
    x = torch.cat(cols, dim=1)
    x.requires_grad_(True)
    return LabelTensor(x, labels=list(cond.input_names))


@dataclass
class NativeTrainResult:
    final_params: Dict[str, float]
    final_loss: float
    history: List[Dict[str, float]] = field(default_factory=list)


def train_native(
    problem: NativeProblem,
    network: torch.nn.Module,
    *,
    lr: float = 1e-3,
    param_lr_scale: float = 1.0,
    adam_epochs: int = 5000,
    n_collocation: int = 1000,
    lam_data: float = 100.0,
    lam_physics: float = 1.0,
    warmup_epochs: int = 5,
    warmup_factor: float = 1.0 / 3.0,
    seed: int = 42,
    report_every: int = 0,
    report_cb: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> NativeTrainResult:
    """Adam training of a native inverse PINN. Returns the recovered params."""
    torch.manual_seed(seed)
    state_names = problem.state_names
    unknowns = problem.unknown_parameters
    net_params = list(network.parameters())
    unk_params = list(unknowns.values())

    # two param groups: network at `lr`, unknowns at `lr * param_lr_scale`.
    groups = [{"params": net_params, "lr": lr}]
    if unk_params:
        groups.append({"params": unk_params, "lr": lr * param_lr_scale})
    opt = torch.optim.Adam(groups)

    def forward(input_lt: LabelTensor) -> LabelTensor:
        return LabelTensor(network(input_lt), labels=list(state_names))

    history: List[Dict[str, float]] = []
    last = float("nan")
    for ep in range(adam_epochs):
        # load-bearing warmup: hold the unknowns' LR at warmup_factor for the
        # first few epochs so the network fits a coarse solution before the
        # unknowns move fast (mirrors PINA's ConstantLR warmup).
        if unk_params:
            opt.param_groups[1]["lr"] = (lr * param_lr_scale *
                                         (warmup_factor if ep < warmup_epochs else 1.0))
        opt.zero_grad()
        params = {k: v for k, v in unknowns.items()}

        phys_loss = torch.zeros(())
        for pc in problem.physics:
            ci = _sample(pc, n_collocation)
            out = forward(ci)
            r = pc.residual(ci, out, params)
            phys_loss = phys_loss + (r ** 2).mean()

        data_loss = torch.zeros(())
        for dc in problem.data.values():
            out = forward(dc.input)
            pred = out.extract(dc.observed)
            data_loss = data_loss + ((pred - dc.target) ** 2).mean()

        loss = lam_data * data_loss + lam_physics * phys_loss
        loss.backward()
        opt.step()
        # project unknowns back into their bounds (PINA clamps to the
        # unknown_parameter_domain; without this an unidentified-early unknown
        # can leave its feasible range and never return).
        with torch.no_grad():
            for name, p in unknowns.items():
                lo, hi = problem.unknown_bounds.get(name, (-float("inf"), float("inf")))
                p.clamp_(lo, hi)
        last = float(loss.detach())
        rec = {"epoch": ep, "loss": last,
               "data_loss": float(data_loss.detach()),
               "physics_loss": float(phys_loss.detach()),
               **{k: float(v.detach()) for k, v in unknowns.items()}}
        history.append(rec)
        if report_cb and report_every and (ep % report_every == 0 or ep == adam_epochs - 1):
            report_cb(ep, rec)

    return NativeTrainResult(
        final_params={k: float(v.detach()) for k, v in unknowns.items()},
        final_loss=last, history=history,
    )


def build_native_problem(compiled, data, t_range, spatial_ranges=None,
                         bounds_override=None, inits_override=None) -> NativeProblem:
    """Build a NativeProblem from a compiled DSL system + sensor data — the
    native analogue of ``core.problem.build_problem``."""
    input_name = compiled.input_name
    input_names = tuple(compiled.input_names)
    is_pde = compiled.is_pde
    ranges = {input_name: (float(t_range[0]), float(t_range[1]))}
    if is_pde:
        if not spatial_ranges:
            raise ValueError("PDE problem requires spatial_ranges")
        for n in input_names:
            if n != input_name:
                ranges[n] = (float(spatial_ranges[n][0]), float(spatial_ranges[n][1]))

    physics = [PhysicsCondition(residual=rf, input_names=input_names, ranges=ranges)
               for rf in compiled.physics_residuals]

    data_conds: Dict[str, DataCondition] = {}
    for sens in compiled.sensors:
        if sens.name not in data:
            continue
        inp, obs = data[sens.name]
        inp = np.asarray(inp)
        if inp.ndim == 1:
            inp = inp.reshape(-1, 1)
        cols = list(input_names) if is_pde else [input_name]
        input_lt = LabelTensor(inp, labels=cols)
        observed = sens.observes.name if hasattr(sens.observes, "name") else sens.name
        target = torch.as_tensor(np.asarray(obs).reshape(-1, 1), dtype=torch.float32)
        data_conds[sens.name] = DataCondition(input=input_lt, target=target,
                                              observed=[observed])

    eff_bounds = dict(compiled.unknown_bounds or {})
    if bounds_override:
        eff_bounds.update({k: (float(v[0]), float(v[1])) for k, v in bounds_override.items()
                           if k in eff_bounds})
    eff_inits = dict(compiled.unknown_inits or {})
    if inits_override:
        eff_inits.update({k: float(v) for k, v in inits_override.items() if k in eff_inits})
    unknowns = {name: torch.nn.Parameter(torch.tensor([float(eff_inits[name])]))
                for name in eff_bounds}

    return NativeProblem(state_names=list(compiled.state_names), input_names=input_names,
                         physics=physics, data=data_conds, unknown_parameters=unknowns,
                         unknown_bounds=eff_bounds)
