"""Dynamic 3-D spatial Cosserat rod — recover all six stiffnesses from the
time-resolved 3-D motion (shape + orientation) of a soft rod.

The capstone of the rod suite: full geometrically-exact spatial rod *with
inertia*. A clamped, pre-twisted rod released under off-axis gravity swings,
bends in two planes, shears, and twists. In dynamics both the internal force and
moment are kinematic (from the measured accelerations and angular-momentum
rate), so the constitutive law is linear in the six stiffnesses and recovered by
least squares — the force/moment-from-motion identifier in full 3-D.

  python3 scripts/exp_dynamic_spatial_cosserat.py [N] [n_t]
"""
from __future__ import annotations
import sys
import numpy as np

from pinn_engine.baselines import (generate_dynamic_spatial_cosserat,
                                    recover_dynamic_spatial_stiffness)

NAMES = ["EA_unit", "GA1_unit", "GA2_unit", "EI1_unit", "EI2_unit", "GJ_unit"]
LAB = {n: n.split("_")[0] for n in NAMES}


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    n_t = int(sys.argv[2]) if len(sys.argv) > 2 else 161
    wt, ws = 21, 9
    print(f"dynamic 3-D Cosserat stiffness ID  (N={N}, n_t={n_t})")
    print("recover EA, GA1, GA2 (axial+shear) and GJ, EI1, EI2 (torsion+bending)\n")
    print(f"{'noise (pos,quat)':>18} | " + "  ".join(f"{LAB[n]:>6}" for n in NAMES))
    for pn, qn in [(0.0, 0.0), (1e-3, 3e-3)]:
        data, _ = generate_dynamic_spatial_cosserat(
            N=N, n_t=n_t, pos_noise_std=pn, quat_noise_std=qn, seed=0)
        r = recover_dynamic_spatial_stiffness(data, sg_window_t=wt, sg_window_s=ws).as_dict()
        tag = "clean" if pn == 0 else f"{pn:.0e},{qn:.0e}"
        cells = "  ".join(f"{100*abs(r[n]-1):>5.1f}%" for n in NAMES)
        print(f"{tag:>18} | {cells}")
    print("\n(rel_err %)  EA & GJ are the axial-direction modes (smoothing-sensitive)")


if __name__ == "__main__":
    main()
