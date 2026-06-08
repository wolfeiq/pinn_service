"""Hyperelastic soft-rod constitutive identification.

Real soft materials are hyperelastic — nonlinear, usually strain-stiffening. From
a load sweep (tip moments small->large) we recover the nonlinear moment-curvature
M(k)=a1*k + a3*k^3 (and axial N(e)=b1*e + b3*e^3), and show a linear-only fit is
decisively rejected -- detecting AND quantifying the hyperelasticity.

  python3 scripts/exp_hyperelastic.py
"""
from __future__ import annotations
import numpy as np
from pinn_engine.baselines import generate_hyperelastic_sweep, recover_hyperelastic


def main():
    print("hyperelastic constitutive identification (load sweep)\n")
    print(f"{'noise':>10} | {'a1 (EI)':>14} {'a3 (stiffen)':>14} {'b1 (EA)':>12} {'b3':>10} | lin/nl resid")
    for an, sn in [(0.0, 0.0), (3e-3, 2e-3), (6e-3, 4e-3)]:
        data, truth = generate_hyperelastic_sweep(ang_noise_std=an, strain_noise_std=sn, seed=0)
        r = recover_hyperelastic(data)
        tag = "clean" if an == 0 else f"{an:.0e}"
        print(f"{tag:>10} | {r.a1:>6.3f}/{truth['a1']:<6.2f} {r.a3:>6.3f}/{truth['a3']:<6.2f} "
              f"{r.b1:>5.2f}/{truth['b1']:<4.0f} {r.b3:>5.1f}/{truth['b3']:<4.0f} | "
              f"{r.linear_only_rms:.3f} / {r.nonlinear_rms:.4f}")
    print("\nlin resid >> nl resid  =>  the material is hyperelastic, not linear.")
    print("(a3>0 strain-stiffening; a3<0 strain-softening — both recovered.)")


if __name__ == "__main__":
    main()
