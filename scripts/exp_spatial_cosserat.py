"""3-D spatial Cosserat rod — recover all six stiffnesses from measured shape +
orientation of a tip-loaded soft rod.

The full geometrically-exact spatial rod (bending in two planes, two shears,
extension, torsion). A tip-loaded cantilever is statically determinate, so the
internal force/moment follow from the measured shape + known tip wrench
(independent of the unknowns); the constitutive law is then linear in the six
stiffnesses and recovered by least squares — the 3-D generalisation of the
force-from-motion identifier.

  python3 scripts/exp_spatial_cosserat.py [n_s]
"""
from __future__ import annotations
import sys
import numpy as np

from pinn_engine.baselines import generate_spatial_cosserat, recover_spatial_stiffness

NAMES = ["EA_unit", "GA1_unit", "GA2_unit", "EI1_unit", "EI2_unit", "GJ_unit"]
LABEL = {"EA_unit": "EA", "GA1_unit": "GA1", "GA2_unit": "GA2",
         "EI1_unit": "EI1", "EI2_unit": "EI2", "GJ_unit": "GJ"}


def main():
    n_s = int(sys.argv[1]) if len(sys.argv) > 1 else 121
    print(f"3-D spatial Cosserat stiffness ID  (n_s={n_s})")
    print("recover EA, GA1, GA2 (axial+shear) and GJ, EI1, EI2 (torsion+bending)\n")
    print(f"{'noise (pos,quat)':>18} | " + "  ".join(f"{LABEL[n]:>6}" for n in NAMES))
    for pn, qn in [(0.0, 0.0), (1e-3, 3e-3), (2e-3, 6e-3)]:
        seeds = [0] if pn == 0 else range(5)
        errs = {n: [] for n in NAMES}
        for sd in seeds:
            data, _ = generate_spatial_cosserat(n_s=n_s, pos_noise_std=pn,
                                                quat_noise_std=qn, seed=sd)
            r = recover_spatial_stiffness(data).as_dict()
            for n in NAMES:
                errs[n].append(100 * abs(r[n] - 1.0))
        tag = "clean" if pn == 0 else f"{pn:.0e},{qn:.0e}"
        cells = "  ".join(f"{np.mean(errs[n]):>5.1f}%" for n in NAMES)
        print(f"{tag:>18} | {cells}")
    print("\n(rel_err %; noisy rows averaged over 5 seeds)")


if __name__ == "__main__":
    main()
