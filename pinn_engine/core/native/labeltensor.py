"""A minimal labelled tensor — the engine's drop-in for ``pina.LabelTensor``.

A ``LabelTensor`` is a ``torch.Tensor`` of shape ``(N, C)`` that carries a list
of column names (``labels``). The DSL residual compiler uses exactly two things:
``.labels`` (to find which column is which input/state variable) and
``.extract([name, ...])`` (to pull named columns, autograd-graph-preserving).
Everything else is plain torch.

We subclass ``torch.Tensor`` so a ``LabelTensor`` *is* a tensor (valid autograd
leaf, passes straight into an ``nn.Module``). Tensor ops generally return a bare
tensor or an unlabelled subclass instance — that's fine, because the engine only
ever reads ``.labels``/``.extract`` off tensors it constructed explicitly (the
network input, the network output, and sensor targets).
"""
from __future__ import annotations

from typing import List, Sequence

import torch


class LabelTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, labels: Sequence[str]):
        t = torch.as_tensor(data, dtype=torch.float32)
        if t.ndim == 1:
            t = t.reshape(-1, 1)
        obj = t.as_subclass(cls)
        obj._labels = list(labels)
        if obj.shape[-1] != len(obj._labels):
            raise ValueError(
                f"LabelTensor: {obj.shape[-1]} columns but {len(obj._labels)} labels "
                f"({list(labels)!r})")
        return obj

    # ``labels`` defaults to None on derived tensors so reads never crash.
    @property
    def labels(self) -> List[str]:
        return getattr(self, "_labels", None)

    @labels.setter
    def labels(self, value):
        self._labels = list(value)

    def extract(self, cols: Sequence[str]) -> "LabelTensor":
        """Return the named columns as a ``LabelTensor`` (preserves the graph)."""
        labels = self._labels
        idx = [labels.index(c) for c in cols]
        sub = torch.Tensor.__getitem__(self, (Ellipsis, idx))   # (N, len(cols)), grad-safe
        return LabelTensor(sub, labels=list(cols))

    # Keep torch's default behaviour for everything else; just don't try to
    # propagate labels through arbitrary ops (the engine re-labels explicitly).
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = super().__torch_function__(func, types, args, kwargs)
        # Derived tensors carry no labels (None) unless set explicitly.
        if isinstance(out, LabelTensor) and not hasattr(out, "_labels"):
            out._labels = None
        return out

    def __repr__(self):
        return f"LabelTensor(labels={getattr(self, '_labels', None)}, " \
               f"{torch.Tensor.__repr__(self)})"
