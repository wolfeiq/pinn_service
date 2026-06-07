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
from pinn_engine.baselines.spatial_cosserat_id import (
    simulate_spatial_cosserat,
    generate_spatial_cosserat,
    recover_spatial_stiffness,
    SpatialCosseratIDResult,
)
from pinn_engine.baselines.dynamic_spatial_cosserat_id import (
    simulate_dynamic_spatial_cosserat,
    generate_dynamic_spatial_cosserat,
    recover_dynamic_spatial_stiffness,
    DynamicSpatialIDResult,
)
from pinn_engine.baselines.tendon_actuated_id import (
    actuation_wrench,
    simulate_tendon_actuated,
    generate_tendon_calibration,
    recover_tendon_stiffness,
    TendonCalibrationResult,
)

__all__ = [
    "EKFInverseDampedOscillator", "run_ekf_baseline",
    "recover_stiffness_from_motion", "recover_from_template_data",
    "CosseratForceIDResult",
    "simulate_spatial_cosserat", "generate_spatial_cosserat",
    "recover_spatial_stiffness", "SpatialCosseratIDResult",
    "simulate_dynamic_spatial_cosserat", "generate_dynamic_spatial_cosserat",
    "recover_dynamic_spatial_stiffness", "DynamicSpatialIDResult",
    "actuation_wrench", "simulate_tendon_actuated",
    "generate_tendon_calibration", "recover_tendon_stiffness",
    "TendonCalibrationResult",
]
