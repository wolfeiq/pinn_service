"""Tendon-actuated soft rod — self-calibration of stiffness from actuated shapes.

Cables routed at offsets from the centerline bend/twist the rod when tensioned
(the dominant continuum-manipulator drive). Because the actuation wrench is known
(commanded tensions x routing), commanding a sweep of tension patterns and
measuring the resulting shapes recovers the rod's axial/bending/torsion stiffness
with NO external test rig — the robot calibrates itself by moving.

  python3 scripts/exp_tendon_actuation.py
"""
from __future__ import annotations
import numpy as np

from pinn_engine.baselines import (simulate_tendon_actuated,
                                    generate_tendon_calibration,
                                    recover_tendon_stiffness)

NAMES = ["EA_unit", "EI1_unit", "EI2_unit", "GJ_unit"]
LAB = {n: n.split("_")[0] for n in NAMES}


def main():
    stiff = {"EA": 15.0, "GA1": 15.0, "GA2": 12.0, "GJ": 0.8, "EI1": 1.0, "EI2": 0.8}
    # constant-curvature sanity: single tendon, offset d, tension tau -> kappa=tau*d/EI
    d, tau = 0.05, 2.0
    s, r, q = simulate_tendon_actuated(stiff, [(d, 0, 0)], [tau])
    print(f"single tendon d={d}, tau={tau}: tip=({r[-1,0]:.3f},{r[-1,1]:.3f},{r[-1,2]:.3f})  "
          f"curvature tau*d/EI2 = {tau*d/stiff['EI2']:.3f}\n")

    print("self-calibration from actuated shapes (4 tendons, 16 tension patterns):")
    print(f"{'noise (pos,quat)':>18} | " + "  ".join(f"{LAB[n]:>6}" for n in NAMES))
    for pn, qn in [(0.0, 0.0), (1e-3, 3e-3), (2e-3, 6e-3)]:
        errs = {n: [] for n in NAMES}
        seeds = [0] if pn == 0 else range(5)
        for sd in seeds:
            data, _ = generate_tendon_calibration(pos_noise_std=pn, quat_noise_std=qn, seed=sd)
            r = recover_tendon_stiffness(data).as_dict()
            for n in NAMES:
                errs[n].append(100 * abs(r[n] - 1.0))
        tag = "clean" if pn == 0 else f"{pn:.0e},{qn:.0e}"
        cells = "  ".join(f"{np.mean(errs[n]):>5.1f}%" for n in NAMES)
        print(f"{tag:>18} | {cells}")
    print("\n(rel_err %; noisy rows averaged over 5 seeds)")
    print("shear GA is not tendon-excitable — use spatial_cosserat_id (external load).")


if __name__ == "__main__":
    main()
