"""Viscoelastic soft-rod identification (rate-dependent material).

Real silicone/rubber creeps, relaxes, and dissipates energy. We model the rod's
bending as a Standard Linear Solid (Zener) and recover (E_inf, E1, tau) two
independent ways -- a creep test (constant actuation -> shape drifts over time)
and a DMA frequency sweep (oscillate -> phase lag -> storage/loss moduli) -- and
check they agree. The loss modulus peaks at omega = 1/tau (the VE fingerprint).

  python3 scripts/exp_viscoelastic.py
"""
from __future__ import annotations
import numpy as np
from pinn_engine.baselines import (generate_creep_test, recover_creep,
                                    generate_dma_sweep, recover_dma)


def main():
    truth = {"E_inf": 1.0, "E1": 1.5, "tau": 0.5}
    print(f"true material: E_inf={truth['E_inf']}, E1={truth['E1']}, tau={truth['tau']}  "
          f"(glassy E_g={truth['E_inf']+truth['E1']}, loss peak at omega=1/tau={1/truth['tau']:.1f})\n")
    print(f"{'method':>8} {'noise':>7} | {'E_inf':>7} {'E1':>7} {'tau':>7}")
    for ns in (0.0, 2e-3, 4e-3):
        dc, _ = generate_creep_test(params=truth, noise_std=ns, seed=0); rc = recover_creep(dc)
        dd, _ = generate_dma_sweep(params=truth, noise_std=ns, seed=0); rd = recover_dma(dd)
        tag = "clean" if ns == 0 else f"{ns:.0e}"
        print(f"{'creep':>8} {tag:>7} | {rc.E_inf:>7.3f} {rc.E1:>7.3f} {rc.tau:>7.3f}")
        print(f"{'DMA':>8} {tag:>7} | {rd.E_inf:>7.3f} {rd.E1:>7.3f} {rd.tau:>7.3f}")

    print("\nstorage E'(w) / loss E''(w) (clean DMA):")
    _, _ = None, None
    dd, _ = generate_dma_sweep(params=truth, noise_std=0.0, seed=0)
    rd = recover_dma(dd)
    print("   omega   E'(w)   E''(w)")
    for w, Ep, Epp in rd.storage_loss:
        mark = "  <- loss peak" if abs(w - 1 / truth["tau"]) < 1e-6 else ""
        print(f"   {w:>5.2f}  {Ep:>6.3f}  {Epp:>6.3f}{mark}")


if __name__ == "__main__":
    main()
