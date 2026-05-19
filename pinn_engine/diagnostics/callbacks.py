"""Base callback + ``default_bundle`` for diagnostics.

Every diagnostic is a :class:`pytorch_lightning.Callback`. Each one stores its
collected data on ``self.output`` so the trainer can copy it onto
``TrainResult.callback_outputs`` and the future Streamlit dashboard can
consume it without re-running anything.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytorch_lightning as pl


class DiagnosticCallback(pl.Callback):
    """Mixin: a Lightning callback that records data for downstream consumption.

    Subclasses set ``name`` (used as the key in ``TrainResult.callback_outputs``)
    and write to ``self.output`` (any json-serializable dict).
    """

    name: str = "diagnostic"

    def __init__(self):
        super().__init__()
        self.output: Dict[str, Any] = {}


def default_bundle() -> List[DiagnosticCallback]:
    """Return one fresh instance of each bundled diagnostic.

    Use::

        from pinn_engine.diagnostics import default_bundle
        result = train(system, data, config, callbacks=default_bundle())
        result.callback_outputs["param_confidence"]["history"]
    """
    from pinn_engine.diagnostics.residual_heatmap import ResidualHeatmap
    from pinn_engine.diagnostics.sensor_residuals import SensorResiduals
    from pinn_engine.diagnostics.param_confidence import ParamConfidence
    from pinn_engine.diagnostics.spectral_bias import SpectralBias

    return [
        ResidualHeatmap(),
        SensorResiduals(),
        ParamConfidence(),
        SpectralBias(),
    ]
