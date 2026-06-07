"""Classical baselines for PINN vs. classical state-estimator comparison.

The build plan calls for benchmarking PINN inverse-parameter discovery
against EKF/ESKF — the dominant classical approach. This subpackage
provides those baselines so the comparison is on common ground (same
data, same problem statement, same metric).
"""
from pinn_engine.baselines.ekf import EKFInverseDampedOscillator, run_ekf_baseline
from pinn_engine.baselines.cosserat_force_id import (
    recover_stiffness_from_motion,
    recover_from_template_data,
    CosseratForceIDResult,
)

__all__ = [
    "EKFInverseDampedOscillator", "run_ekf_baseline",
    "recover_stiffness_from_motion", "recover_from_template_data",
    "CosseratForceIDResult",
]
