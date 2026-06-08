# Combined hyper- + visco-elastic identification

Real soft-robot materials are **both** nonlinear (hyperelastic) **and**
time-dependent (viscoelastic). This module identifies a rod with a *nonlinear
viscoelastic* bending law — the quasi-linear-viscoelastic (QLV / nonlinear
Standard Linear Solid) form standard for elastomers and soft tissue.

## Model

```
M(t) = g_∞·M_e(κ) + q,   q̇ = −q/τ + (1−g_∞)·dM_e/dt
M_e(κ) = a1·κ + a3·κ³                  (instantaneous nonlinear elastic)
```

`M_e` is the glassy nonlinear response; the reduced relaxation
`G(t)=g_∞+(1−g_∞)e^{−t/τ}` scales it from glassy (`G(0)=1`) to the equilibrium
fraction `g_∞`. Under a step moment `M_0` the curvature creeps from instantaneous
`κ_g` (`M_e(κ_g)=M_0`) to equilibrium `κ_∞` (`g_∞·M_e(κ_∞)=M_0`).

## Identification — a multi-level creep sweep separates the two physics

- instantaneous curvatures `κ_g(M_0)` trace the **nonlinear elastic** curve →
  recover `a1, a3` (`M_0 = a1κ_g + a3κ_g³`);
- equilibrium curvatures `κ_∞(M_0)` give the **relaxation strength**
  `g_∞ = M_0 / M_e(κ_∞)`;
- a 1-D fit of the creep ODE (with `a1,a3,g_∞` fixed) gives `τ`.

A *linear*-viscoelastic fit can't match the level-dependent instantaneous
response, and a *nonlinear-elastic-only* fit can't match the creep — only the
combined model fits both.

## Results (`scripts/exp_viscohyperelastic.py`)

| truth | noise | a1 | a3 | g_∞ | τ | linear-fit residual |
|---|---|---|---|---|---|---|
| 1.0/0.6/0.5/0.8 | clean | 1.000 | 0.600 | 0.500 | 0.797 | 0.143 |
| 1.0/0.6/0.5/0.8 | 2e-3 | 0.994 | 0.606 | 0.501 | 0.788 | 0.144 |
| 0.8/1.2/0.35/1.2 | 2e-3 | 0.792 | 1.211 | 0.363 | 1.097 | 0.204 |

All four parameters recovered (a1,a3 ~1%, g_∞ ~1%, τ ~1–8%). The large
linear-fit residual confirms the instantaneous response is genuinely nonlinear.

## Where this sits

This unifies the material story — a single model that is both hyperelastic and
viscoelastic, the realistic constitutive law for soft silicone. Composes the
hyperelastic (`a1,a3`) and viscoelastic (`g_∞,τ`) modules.

## Reproduce

```
python3 scripts/exp_viscohyperelastic.py
```

Module: `pinn_engine/baselines/viscohyperelastic_rod_id.py`.
