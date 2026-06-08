"""Actuated-dynamics self-calibration for a soft rod.

When the actuation changes in time the rod has inertia and damping; its modal
curvature obeys  I_eff*kdd + c*kd + EI*k = M_act(t). Suddenly tensioning a tendon
and watching the rod ring and settle recovers stiffness EI, damping c, AND modal
inertia I_eff at once -- by regressing the (linear-in-parameters) equation of
motion against the measured curvature response.

  python3 scripts/exp_actuated_dynamics.py
"""
from __future__ import annotations
import numpy as np
from pinn_engine.baselines import generate_step_actuation, recover_actuated_dynamics


def main():
    print("dynamic self-calibration from a step-actuation ring-down\n")
    for truth in [{"EI": 1.0, "c": 0.05, "I_eff": 0.02},
                  {"EI": 2.0, "c": 0.12, "I_eff": 0.03}]:
        wn = (truth["EI"] / truth["I_eff"]) ** 0.5
        zeta = truth["c"] / (2 * (truth["EI"] * truth["I_eff"]) ** 0.5)
        print(f"  true: EI={truth['EI']} c={truth['c']} I_eff={truth['I_eff']}  "
              f"(omega_n={wn:.2f}, zeta={zeta:.3f})")
        print(f"{'noise':>8} | {'EI':>7} {'c':>8} {'I_eff':>8}")
        for ns in (0.0, 2e-3, 5e-3):
            d, _ = generate_step_actuation(params=truth, noise_std=ns, seed=0)
            r = recover_actuated_dynamics(d)
            tag = "clean" if ns == 0 else f"{ns:.0e}"
            print(f"{tag:>8} | {r.EI:>7.3f} {r.c:>8.4f} {r.I_eff:>8.4f}")
        print()
    print("EI (stiffness) is the most robust; c and I_eff come from kd, kdd and")
    print("need a bit cleaner data, but recover well from one ring-down.")


if __name__ == "__main__":
    main()
