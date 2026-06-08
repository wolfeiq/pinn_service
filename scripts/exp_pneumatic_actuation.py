"""Pneumatic soft rod — self-calibration of stiffness from pressurized shapes.

Pressurized chambers (PneuNets / fiber-reinforced) push the rod: a chamber at
offset d under pressure P extends the rod and bends it AWAY from the chamber
(opposite a tendon). The actuation wrench is known (P x chamber geometry), so a
sweep of pressure patterns + measured shapes recovers EA/EI1/EI2/GJ — the
pneumatic dual of tendon self-calibration.

  python3 scripts/exp_pneumatic_actuation.py
"""
from __future__ import annotations
import numpy as np
from pinn_engine.baselines import (generate_pneumatic_calibration,
                                    recover_pneumatic_stiffness)

NAMES = ["EA_unit", "EI1_unit", "EI2_unit", "GJ_unit"]
LAB = {n: n.split("_")[0] for n in NAMES}


def main():
    print("pneumatic self-calibration (4 chambers, 16 pressure patterns):")
    print(f"{'noise (pos,quat)':>18} | " + "  ".join(f"{LAB[n]:>6}" for n in NAMES))
    for pn, qn in [(0.0, 0.0), (1e-3, 3e-3), (2e-3, 6e-3)]:
        errs = {n: [] for n in NAMES}
        for sd in ([0] if pn == 0 else range(5)):
            data, _ = generate_pneumatic_calibration(pos_noise_std=pn, quat_noise_std=qn, seed=sd)
            r = recover_pneumatic_stiffness(data).as_dict()
            for n in NAMES:
                errs[n].append(100 * abs(r[n] - 1.0))
        tag = "clean" if pn == 0 else f"{pn:.0e},{qn:.0e}"
        print(f"{tag:>18} | " + "  ".join(f"{np.mean(errs[n]):>5.1f}%" for n in NAMES))
    print("\n(rel_err %; noisy rows averaged over 5 seeds)")
    print("pneumatic EXTENDS (axial strain > 0) where tendons compress — sign check.")


if __name__ == "__main__":
    main()
