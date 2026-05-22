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


class FourierFeatureEmbedding(nn.Module):
    """Random Fourier-feature input encoding.

    Maps ``x ∈ R^d → [x, sin(B·x), cos(B·x)] ∈ R^{d + 2·m}`` where
    ``B ∈ R^{m×d}`` is sampled once from ``N(0, σ²)`` and held fixed.

    Helps PINNs fit high-frequency targets — the canonical "PINN can't
    learn fast oscillations" failure mode. Tancik et al. 2020
    ("Fourier Features Let Networks Learn High Frequency Functions in
    Low Dimensional Domains", arXiv:2006.10739) demonstrated this works
    well for implicit neural representations; the PINN literature
    (Wang-Wang-Perdikaris 2021, arXiv:2012.10047) adopted it directly.
    """

    def __init__(self, input_dim: int, n_features: int = 32, sigma: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.n_features = n_features
        self.sigma = float(sigma)
        # Random projection matrix B, frozen (no requires_grad).
        B = torch.randn(n_features, input_dim) * sigma
        self.register_buffer("B", B)

    @property
    def output_dim(self) -> int:
        return self.input_dim + 2 * self.n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, input_dim). proj: (N, n_features).
        proj = x @ self.B.T
        # Concatenate raw + sin(B·x) + cos(B·x).
        out = torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)
        return out


class MLP(nn.Module):
    """A simple feed-forward network with optional LayerNorm + Fourier features."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        depth: int,
        width: int,
        activation: str = "tanh",
        layer_norm: bool = True,
        fourier_features: int = 0,
        fourier_sigma: float = 1.0,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError(f"depth must be >= 2, got {depth}")

        # Optional Fourier-feature embedding before the first Linear.
        if fourier_features > 0:
            self.embedding: nn.Module = FourierFeatureEmbedding(
                input_dim=input_dim,
                n_features=fourier_features,
                sigma=fourier_sigma,
            )
            embed_out = input_dim + 2 * fourier_features
        else:
            self.embedding = nn.Identity()
            embed_out = input_dim

        layers: list[nn.Module] = []
        prev = embed_out
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
        self.fourier_features = fourier_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.embedding(x))


def build_network(
    input_dim: int,
    output_dim: int,
    depth: int = 4,
    width: int = 64,
    activation: str = "tanh",
    layer_norm: bool = True,
    fourier_features: int = 0,
    fourier_sigma: float = 1.0,
) -> MLP:
    """Factory wrapper used by the trainer and the AutoML objective."""
    return MLP(
        input_dim=input_dim,
        output_dim=output_dim,
        depth=depth,
        width=width,
        activation=activation,
        layer_norm=layer_norm,
        fourier_features=fourier_features,
        fourier_sigma=fourier_sigma,
    )
