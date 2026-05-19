"""Activation factory.

We add ``SinTanh`` (used to good effect in PINN time-series work) and ``Swish``
on top of torch's standard set. Looked up by short name.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SinTanh(nn.Module):
    """``sin(x) * tanh(x)`` — smooth, oscillatory, infinitely differentiable.

    Useful when the network output has both periodic and bounded character.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x) * torch.tanh(x)


class Swish(nn.Module):
    """``x * sigmoid(x)`` (a.k.a. SiLU). Smooth and self-gating."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class Sin(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


_ACTIVATIONS = {
    "tanh": nn.Tanh,
    "sin": Sin,
    "sintanh": SinTanh,
    "swish": Swish,
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}


def build_activation(name: str) -> nn.Module:
    """Return a fresh activation module by name."""
    try:
        return _ACTIVATIONS[name]()
    except KeyError as e:
        raise ValueError(
            f"Unknown activation {name!r}. Available: {sorted(_ACTIVATIONS)}"
        ) from e
