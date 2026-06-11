"""Native PINN core — the engine's own training stack, replacing PINA.

This subpackage provides the runtime pieces the engine used to borrow from PINA:
a labelled tensor, a collocation-domain sampler, a problem container, and a
PINN ``LightningModule``. The DSL→torch residual compiler, networks, adaptive
controller, CRLB, and RAR remain the engine's own and are unchanged.
"""
from pinn_engine.core.native.labeltensor import LabelTensor

__all__ = ["LabelTensor"]
