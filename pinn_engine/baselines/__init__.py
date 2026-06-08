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
from pinn_engine.baselines.pneumatic_actuated_id import (
    pneumatic_wrench,
    simulate_pneumatic_actuated,
    generate_pneumatic_calibration,
    recover_pneumatic_stiffness,
    PneumaticCalibrationResult,
)
from pinn_engine.baselines.hyperelastic_rod_id import (
    simulate_hyperelastic_bending,
    generate_hyperelastic_sweep,
    recover_hyperelastic,
    HyperelasticIDResult,
)
from pinn_engine.baselines.contact_id import (
    simulate_contact,
    generate_contact_scenario,
    recover_contact,
    ContactIDResult,
)
from pinn_engine.baselines.contact_multi_id import (
    simulate_multi_contact,
    generate_multi_contact,
    recover_n_contacts,
    recover_contacts,
    MultiContactResult,
)
from pinn_engine.baselines.viscoelastic_rod_id import (
    creep_curvature,
    generate_creep_test,
    generate_dma_sweep,
    recover_creep,
    recover_dma,
    ViscoelasticIDResult,
)
from pinn_engine.baselines.viscohyperelastic_rod_id import (
    nonlinear_creep,
    generate_viscohyper_creep,
    recover_viscohyper,
    ViscoHyperIDResult,
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
    "pneumatic_wrench", "simulate_pneumatic_actuated",
    "generate_pneumatic_calibration", "recover_pneumatic_stiffness",
    "PneumaticCalibrationResult",
    "simulate_hyperelastic_bending", "generate_hyperelastic_sweep",
    "recover_hyperelastic", "HyperelasticIDResult",
    "simulate_contact", "generate_contact_scenario",
    "recover_contact", "ContactIDResult",
    "simulate_multi_contact", "generate_multi_contact",
    "recover_n_contacts", "recover_contacts", "MultiContactResult",
    "creep_curvature", "generate_creep_test", "generate_dma_sweep",
    "recover_creep", "recover_dma", "ViscoelasticIDResult",
    "nonlinear_creep", "generate_viscohyper_creep",
    "recover_viscohyper", "ViscoHyperIDResult",
]
