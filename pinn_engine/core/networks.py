"""MLP factory: depth × width × activation. Compatible with PINA's solver.

PINA's ``PINN`` accepts any ``torch.nn.Module`` whose forward takes a
:class:`pina.LabelTensor` and returns a :class:`pina.LabelTensor`. We use
PINA's ``FeedForward`` style: an MLP that takes a plain tensor in/out and
relies on PINA's machinery to wrap labels.

We build a thin custom MLP rather than using PINA's built-in network so the
``LayerNorm`` between hidden layers (which helps inverse-problem convergence)
and the activation factory are under our control.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from pinn_engine.core.activations import build_activation


class MLP(nn.Module):
    """A simple feed-forward network with optional LayerNorm between hidden layers."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        depth: int,
        width: int,
        activation: str = "tanh",
        layer_norm: bool = True,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError(f"depth must be >= 2, got {depth}")
        layers: list[nn.Module] = []
        prev = input_dim
        for i in range(depth - 1):
            layers.append(nn.Linear(prev, width))
            if layer_norm:
                layers.append(nn.LayerNorm(width))
            layers.append(build_activation(activation))
            prev = width
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        # Record meta for repro / manifest.
        self.depth = depth
        self.width = width
        self.activation_name = activation
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_network(
    input_dim: int,
    output_dim: int,
    depth: int = 4,
    width: int = 64,
    activation: str = "tanh",
    layer_norm: bool = True,
) -> MLP:
    """Factory wrapper used by the trainer and the AutoML objective."""
    return MLP(
        input_dim=input_dim,
        output_dim=output_dim,
        depth=depth,
        width=width,
        activation=activation,
        layer_norm=layer_norm,
    )
