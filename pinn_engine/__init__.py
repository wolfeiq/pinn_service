"""pinn-engine: an open-source inverse PINN engine.

The product entry points live here so users can `from pinn_engine import ...`.
"""
from __future__ import annotations

__version__ = "0.1.0"

from pinn_engine.dsl.symbols import Variable, Parameter, Unknown, Sensor
from pinn_engine.dsl.system import System

__all__ = ["Variable", "Parameter", "Unknown", "Sensor", "System", "__version__"]
