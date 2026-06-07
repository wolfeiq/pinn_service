# Dynamic planar Cosserat rod inverse: stiffness ID from motion

Recover bending / shear / axial stiffness (`EI_unit`, `GA_unit`, `EA_unit`,
truth = 1.0) of a soft rod from its **time-resolved motion** `x(s,t), y(s,t),
θ(s,t)`. This is the engine's most ambitious template: the dynamic (inertial)
geometrically-exact Simo-Reissner rod — a **multi-output, multi-unknown,
nonlinear PDE inverse over a 2-D space-time domain**, and the dynamics extension
of `planar_cosserat`.

## The model

A soft rod clamped at `s=0` is released from straight-horizontal under a
distributed gravity load; it swings down (~50° tip) and oscillates under light
viscous damping, settling toward the static droop. Dimensionless equations of
motion (time scale `T = L²√(ρA/EI_ref)`):

```
x_tt   = ∂Nx/∂s − c·x_t
y_tt   = ∂Ny/∂s − g − c·y_t
j·θ_tt = EI·θ_ss + (x_s·Ny − y_s·Nx)
Nx     = n1·cosθ − n2·sinθ,   n1 = EA·(ν−1)
Ny     = n1·sinθ + n2·cosθ,   n2 = GA·η
```

with stretch `ν = x_s cosθ + y_s sinθ` and shear `η = −x_s sinθ + y_s cosθ`.
The internal force components `Nx, Ny` are carried as **auxiliary network
outputs** so the momentum balance needs only a first spatial derivative of a
state (no hand-expanded force divergence). `g` (gravity), `c` (damping), and
`j = ρI/(ρA·L²)` (rotary inertia) are known. Five fields, five residuals.

**Why gravity, not a tip load.** A concentrated tip force on the near-massless
free-end node shock-excites fast axial/shear waves (max acceleration ~180,
axial strain spiking to ~0.9) — unrepresentable by a PINN. A distributed
gravity load starts the rod in smooth uniform free-fall (max acceleration ~8,
axial strain ≲0.16, shear ≲0.34): large-deflection but PINN-tractable.

## Forward solver (ground truth) — verified

Method of lines: staggered finite differences in `s` (strains/forces at element
midpoints, divergence at nodes) + adaptive RK45 in time
(`_simulate_dynamic_cosserat`). Three independent checks (in `test_templates.py`):

- **Energy conservation** (undamped): total energy drifts < 1e-6 relative.
- **Static limit** (damped): the steady state reproduces the static
  gravity-loaded rod shape to ~1e-5 — cross-validates against the static solver.
- **Residual transcription**: the template's five residuals vanish to ~0.3-1%
  of term scale on the solver fields (finite-difference truncation level; a sign
  error would read ~100%).

## Identifiability (CRLB preflight)

```
   Unknown   CRLB SE/|truth|
   EI_unit       0.02%
   GA_unit       0.05%
   EA_unit       0.02%
```

The space-time motion is enormously informative (1701 grid points × 3 fields) —
identifiability is a non-issue. The whole difficulty is **PINN training** over
the 2-D domain with five coupled fields.

## Results — EI recoverable; GA/EA training-limited (honest frontier result)

This is the hardest inverse in the engine, and the full 3-unknown problem is
**training-limited** — a real, documented frontier result (cf. the
training-limited `euler_bernoulli_beam` / `cosserat_rod` entries in ENGINE.md).
What we found, across many iterations:

| run | config | EI_unit | GA_unit | EA_unit |
|---|---|---|---|---|
| 3-field, free | scale=80, ep1000 | **0.93 (7%)** | 4.03 | 4.12 |
| 3-field, free | scale=80, ep2000 | 0.69 | 4.01 | 4.09 |
| 3-field, free + 41 sensors | scale=120, ep3000 | 0.24 | 3.54 | 3.60 |
| de-risk (GA,EA fixed=truth) | scale=30, ep1000 | 4.42 → (→1 slowly) | — | — |

- **EI (bending) is recoverable.** It picks up a strong, clean gradient from the
  angular-momentum residual and reaches ~7% within 1000 epochs.
- **GA, EA (shear, axial) are training-limited** — they freeze near ~3.5-4.0
  regardless of formulation, sensor density, LR, or `lam_physics`. Because they
  stay wrong, EI then *drifts off* truth (it compensates for the wrong
  stiffnesses through the coupled physics), which is why EI overshoots downward
  at longer budgets / higher LR.

### Diagnosis (the useful part)

1. **Auxiliary-force formulation kills the stiffness gradient.** The first
   design carried `Nx, Ny` as free network outputs with constitutive residuals
   `Nx = EA·(…)`. That residual is trivially satisfied by `Nx` *tracking* `EA`,
   so `EA`/`GA` get **zero gradient**. Switching to the **direct** formulation
   (expand `∂Nx/∂s` into 2nd derivatives of `x,y,θ`, so the stiffnesses appear in
   the momentum residuals tied to the data-anchored accelerations) is what made
   EI converge at all. Encoded in `build_system` via a programmatic chain-rule
   expansion (no hand algebra).
2. **GA/EA signal lives in under-resolved derivatives.** Shear/axial stiffness
   enters through `∂Nx/∂s`, i.e. the **2nd spatial derivatives of the
   near-rigid translation** `x, y` — small quantities the network can satisfy by
   adjusting its derivative field within the sensor-spacing/noise latitude
   (the dynamic analogue of the static "explain-away", but on derivatives).
   Denser spatial sampling (21→41) helped marginally; it did not close the gap.
   This is a *training/conditioning* gap, not an identifiability one — the CRLB
   floors are 0.02-0.05%.
3. **Load choice matters a lot.** A tip point load shock-excites fast
   axial/shear waves (accel ~180, axial strain ~0.9) and is hopeless for a PINN;
   distributed gravity (accel ~8) is what made the problem approachable.

### Open R&D (paths not yet tried to convergence)

- Curriculum on the unknowns (recover EI first, freeze, then GA/EA) — the
  de-risk run confirms EI→truth when GA/EA are known.
- Strain-derivative supervision or a weak-form / energy residual that exposes
  GA/EA without 2nd derivatives.
- Much longer budgets on GPU (these runs were CPU-bound at ~0.3-1.0 s/epoch).

**Bottom line (PINN):** bending stiffness is recoverable from motion in the
collocation PINN; full shear+axial recovery stalls there. The fix is below.

## Gap closed — force-from-motion identification

The shear/axial gap is **closed** by a direct, physics-informed estimator
(`pinn_engine/baselines/cosserat_force_id.py`,
`scripts/exp_cosserat_force_id.py`). Key insight: the internal force `N(s,t)`
is **kinematic** — linear-momentum balance integrated from the free tip gives

```
Nx(s,t) = −∫ₛᴸ (x_tt + c·x_t) ds'        Ny(s,t) = −∫ₛᴸ (y_tt + g + c·y_t) ds'
```

so `N` depends only on the measured accelerations and known `g, c` — **not on
the unknown stiffnesses**. With `N` derived from the motion, the constitutive
law is *linear* in the stiffnesses and recovered by least squares using only
**first** spatial derivatives (the strains) plus the angular balance for EI:

```
[Nx; Ny] = EA·(ν−1)·[cosθ; sinθ] + GA·η·[−sinθ; cosθ]
j·θ_tt − (x_s·Ny − y_s·Nx) = EI·θ_ss
```

This sidesteps the 2nd-derivative explain-away entirely (the divergence is
*integrated away* into `N`, which the data fixes). Derivatives are taken with
Savitzky-Golay local-polynomial fits (smooth + differentiate in one step) for
noise robustness. Results (grid 41×161, noisy rows averaged over 5 seeds):

| noise (pos, ang) | EI_unit | GA_unit | EA_unit |
|---|---|---|---|
| clean | 0.79% | 0.70% | 5.09% |
| 1e-3, 5e-3 | **2.08%** | **0.85%** | **5.55%** |
| 2e-3, 1e-2 | 6.05% | 1.24% | 6.85% |

All three stiffnesses recovered to single digits from noisy motion — where the
PINN stalls at ~250% on GA/EA. EA (axial, the smallest strain signal) is the
hardest and benefits most from dense time sampling (more accurate
accelerations). This is the recommended solver for the dynamic rod; the PINN
remains useful for smoothing the fields / recovering EI and as the
forward-consistency check.

**Methodological takeaway:** when a stiffness enters a PINN residual only
through a high-order derivative the network under-resolves, *integrate the
balance law to expose the parameter against a data-derived quantity* (here,
internal force from inertia) instead of fighting the derivative.

## Reproduce

```
python3 scripts/exp_dynamic_cosserat.py [epochs] [width] [ncol] [fourier]
```

Template: `pinn_engine/dsl/templates_lib/dynamic_cosserat.py`.
Data + solver: `generate_dynamic_cosserat` / `_simulate_dynamic_cosserat` in
`pinn_engine/data/synthetic.py`.
