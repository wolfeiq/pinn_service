"""Combined hyper- + visco-elastic identification.

Real silicone is BOTH nonlinear (hyperelastic) AND rate-dependent (viscoelastic).
A multi-level creep sweep separates the two: the instantaneous curvatures trace
the nonlinear elastic curve M_e(k)=a1*k+a3*k^3, the equilibrium curvatures give
the relaxation strength g_inf, and the creep rate gives tau. Neither a linear-
viscoelastic nor a nonlinear-elastic-only model fits both.

  python3 scripts/exp_viscohyperelastic.py
"""
from __future__ import annotations
import numpy as np
from pinn_engine.baselines import generate_viscohyper_creep, recover_viscohyper


def main():
    print("combined hyper+visco identification (multi-level creep sweep)\n")
    print(f"{'noise':>8} | {'a1':>7} {'a3':>7} {'g_inf':>7} {'tau':>7} | lin-fit resid")
    for truth in [{"a1": 1.0, "a3": 0.6, "g_inf": 0.5, "tau": 0.8},
                  {"a1": 0.8, "a3": 1.2, "g_inf": 0.35, "tau": 1.2}]:
        print(f"  true: a1={truth['a1']} a3={truth['a3']} g_inf={truth['g_inf']} tau={truth['tau']}")
        for ns in (0.0, 2e-3):
            d, _ = generate_viscohyper_creep(params=truth, noise_std=ns, seed=0)
            r = recover_viscohyper(d)
            tag = "clean" if ns == 0 else f"{ns:.0e}"
            print(f"{tag:>8} | {r.a1:>7.3f} {r.a3:>7.3f} {r.g_inf:>7.3f} {r.tau:>7.3f} | {r.linear_visco_rms:.3f}")
    print("\nlarge linear-fit residual => the instantaneous response is nonlinear")
    print("(hyperelastic); the creep over time => rate-dependent (viscoelastic).")


if __name__ == "__main__":
    main()
