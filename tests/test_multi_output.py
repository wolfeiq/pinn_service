"""Regression: data conditions on multi-state systems must respect output labels.

PINA's stock PINN.loss_data does ``MSE(forward(input), target)``. For a
3-state system (e.g. Lorenz) where each sensor observes a single state, the
target is shape ``(N, 1)`` but ``forward(input)`` is ``(N, 3)``. PyTorch
broadcasts the comparison silently and the resulting gradient is garbage.

This test ensures ``LabeledDataPINN.loss_data`` correctly extracts the
target's labeled columns from the network output before the MSE.
"""
import warnings
warnings.filterwarnings("ignore")

import torch
import pytest
from pina import LabelTensor

from pinn_engine.core.trainer import LabeledDataPINN


class _FakeProblem:
    """Minimal stub for what LabeledDataPINN needs from a problem."""
    conditions = {}
    unknown_parameters = {}


def _make_solver():
    """Build a LabeledDataPINN whose ``forward`` returns labeled (N, 3) outputs."""
    from pina.optim import TorchOptimizer
    from pinn_engine.core.networks import build_network

    net = build_network(input_dim=1, output_dim=3, depth=2, width=8, activation="tanh")
    # We instantiate the solver class without going through the full PINA init
    # path — we just want loss_data to be callable.
    class _Wrap(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            y = self.m(x)
            return LabelTensor(y, labels=["x", "y", "z"])
    return _Wrap(net)


def test_loss_data_extracts_matching_column():
    """target shape (N, 1) with label ['x'] must NOT broadcast against forward (N, 3)."""
    wrap = _make_solver()

    # Build a fresh LabeledDataPINN-shaped object by overriding forward and _loss_fn.
    class _Mini(LabeledDataPINN):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self._loss_fn = torch.nn.MSELoss()
            self._wrap = wrap
        def forward(self, x):
            return self._wrap(x)

    solver = _Mini()
    inp = LabelTensor(torch.randn(16, 1), labels=["t"])
    target = LabelTensor(torch.randn(16, 1), labels=["x"])

    # Should compute MSE on the 'x' column only, no broadcasting warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        loss = solver.loss_data(inp, target)

    assert loss.dim() == 0 or loss.numel() == 1, f"loss should be scalar, got {loss.shape}"


def test_loss_data_full_target_unchanged():
    """When target has all labels, behavior matches the parent class (no extract)."""
    wrap = _make_solver()

    class _Mini(LabeledDataPINN):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self._loss_fn = torch.nn.MSELoss()
            self._wrap = wrap
        def forward(self, x):
            return self._wrap(x)

    solver = _Mini()
    inp = LabelTensor(torch.randn(16, 1), labels=["t"])
    target = LabelTensor(torch.randn(16, 3), labels=["x", "y", "z"])

    loss = solver.loss_data(inp, target)
    assert loss.numel() == 1
