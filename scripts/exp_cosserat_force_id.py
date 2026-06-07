"""Force-from-motion identification for the dynamic Cosserat rod — the method
that closes the shear/axial gap the PINN training hits.

The internal force N(s,t) is kinematic (the integral of inertia from the free
tip), so it can be derived from the measured motion alone — independent of the
unknown stiffnesses. The constitutive law is then linear in EI/GA/EA and
recovered by least squares using only first spatial derivatives, sidestepping
the 2nd-derivative "explain-away" that stalls the collocation PINN on GA/EA.

  python3 scripts/exp_cosserat_force_id.py [n_s] [n_t]
"""
from __future__ import annotations
import sys
import numpy as np

from pinn_engine.data.synthetic import generate_dynamic_cosserat
from pinn_engine.baselines import recover_from_template_data

NAMES = ["EI_unit", "GA_unit", "EA_unit"]


def main():
    n_s = int(sys.argv[1]) if len(sys.argv) > 1 else 41
    n_t = int(sys.argv[2]) if len(sys.argv) > 2 else 161
    print(f"force-from-motion stiffness ID  (grid {n_s}x{n_t})\n")
    print(f"{'noise (pos,ang)':>20} | " + "  ".join(f"{n.split('_')[0]:>14}" for n in NAMES))
    for pn, an in [(0.0, 0.0), (1e-3, 5e-3), (2e-3, 1e-2)]:
        # average over a few seeds for the noisy rows
        seeds = [0] if pn == 0 else range(5)
        errs = {n: [] for n in NAMES}
        for sd in seeds:
            data, _ = generate_dynamic_cosserat(seed=sd, n_s=n_s, n_t=n_t,
                                                pos_noise_std=pn, ang_noise_std=an)
            r = recover_from_template_data(data, n_s, n_t).as_dict()
            for n in NAMES:
                errs[n].append(100 * abs(r[n] - 1.0))
        tag = "clean" if pn == 0 else f"{pn:.0e},{an:.0e}"
        cells = "  ".join(f"{np.mean(errs[n]):>12.2f}%" for n in NAMES)
        print(f"{tag:>20} | {cells}")
    print("\n(rel_err %; noisy rows averaged over 5 seeds)")
    print("cf. the PINN: EI recoverable but GA/EA stall at ~250% — see "
          "docs/dynamic_cosserat_experiments.md")


if __name__ == "__main__":
    main()
