# Full planar Cosserat rod inverse: bending + shear + axial stiffness

Recover **three** dimensionless stiffnesses — bending `EI_unit`, shear
`GA_unit`, axial `EA_unit` (truth = 1.0 each) — of a soft continuum rod from
its measured deformed shape `(x, y, θ)`. This is the geometrically-exact
Simo-Reissner planar rod **with shear and extension**, the step beyond
`planar_elastica` (inextensible, unshearable, bending only). It's also the
engine's **first multi-output, multi-unknown PDE inverse** — the
structural-mechanics analogue of `coupled_drag_3d`.

## The model

A slender rod is clamped at `s=0` and carries a constant tip load `(Px, Py)`
at the free end (no distributed load → the internal force is constant along the
rod and equal to the applied tip load). Writing the centerline by position
`(x, y)` and the cross-section by its orientation angle `θ` (which, with shear,
is *not* the centerline tangent), the Simo-Reissner balance laws collapse to
three residuals — each isolating one unknown:

```
axial    EA · (x'cosθ + y'sinθ − 1)  =  Px cosθ + Py sinθ      (constitutive)
shear    GA · (−x'sinθ + y'cosθ)     =  −Px sinθ + Py cosθ      (constitutive)
moment   EI · θ''  +  (Py x' − Px y') =  0                      (balance)
```

The first two are the constitutive laws read directly off the measured shape
(axial stretch `ν−1 = x'cosθ+y'sinθ−1`, shear strain `η = −x'sinθ+y'cosθ`);
the third is moment balance. Verified to vanish at machine precision (1e-15)
on an independent `solve_bvp` solution (see `test_templates.py`).

The unknowns are dimensionless multipliers on reference stiffness numbers
`EI0=1, GA0=15, EA0=15` (= stiffness·L²/EI_ref). The default tip load
`(Px, Py) = (2.5, −4)` drives the rod to a ~45° tip rotation with **~7-27%
shear** and **~17-31% axial** strain — a genuinely soft/thick rod where all
three deformation modes are active.

## Identifiability (CRLB preflight, 41 sensors)

```
   Unknown     CRLB SE  SE/|truth|
   EI_unit              0.29%
   GA_unit              0.53%
   EA_unit              0.11%
```

All three are well-identified — the soft-rod design ensures every stiffness
leaves a strong shape signature.

## Results

Sweep on CPU, seed 0 (`scripts/exp_planar_cosserat.py`):

| config | epochs | wall | EI_unit | GA_unit | EA_unit | mean rel_err |
|---|---|---|---|---|---|---|
| **fast** (fixed param_lr_scale=100) | 8000 | 628 s | 0.69% | 0.20% | 0.00% | **0.30%** |
| adaptive (controller) | 8000 | 469 s | 0.68% | 5.30% | 0.20% | 2.06% |
| adaptive (controller) | 12000 | 865 s | 17.18% | 12.53% | 188.86% | 72.86% ✗ |

CRLB floors: EI 0.29%, GA 0.53%, EA 0.11%.

## Findings

1. **The three unknowns converge at very different rates.** EI gets a strong,
   clean gradient from moment balance (R3) and converges first. GA and EA live
   in the *constitutive* residuals (R1, R2), which the network can partly
   satisfy by adjusting its own derivative field `x', y'` — so they only sharpen
   once the shape fit is accurate. Once the problem is well-conditioned (soft
   rod, large strains), a **fixed `param_lr_scale = 100` at 8000 epochs**
   converges all three cleanly to a mean **0.30%** — at the CRLB floor. This is
   the recommended setting.

2. **The adaptive controller is non-monotonic on this coupled 3-unknown
   problem.** Same controller config landed at mean 2.06% at 8000 epochs but
   **72.9%** at 12000 (fixed seed — pure epoch count): after EA had converged,
   the controller pushed it back out to 2.89. The per-unknown LR state machine,
   designed and tuned on single-unknown templates, does not yet handle three
   coupled unknowns whose residuals interact — it can sit on a good minimum or
   wander off it depending on the budget. Prefer the fixed scale here; treating
   the controller's coupled-multi-unknown stability is open R&D (cf. the
   CONVERGED-latch / drift-guard notes in ENGINE.md).

3. **The "explain-away" trap on a stiff unknown — and the fix.** An earlier
   design used a stiff axial response (`EA0=40`, axial strain only ~5%). EA got
   stuck near its midpoint init (rel_err **~380%**) in every run: the axial
   residual was small enough that the network neutralised it by nudging `x'`
   within the position-noise latitude, leaving EA unconstrained during training
   — even though CRLB said EA was identifiable. The cure was to **enlarge the
   strain signal**, not to push the optimiser harder: softening the rod
   (`EA0=15`, axial strain ~17-31%) and using denser/cleaner sensors dropped EA
   to **0.20%**. Lesson: when an inverse parameter stalls, check whether its
   *residual signal* is above the data-noise floor before blaming the LR.

4. **Levers that did NOT help.** Raising `lam_physics` to 10 (to amplify the
   unknown gradients) made things *worse* — it traded shape-fit accuracy for
   residual reduction, degrading the very `x', y'` the constitutive residuals
   depend on. The winning combination is a well-conditioned (soft-rod) problem +
   a fixed, generous `param_lr_scale` + enough epochs.

5. **Cheap collocation.** The solution is smooth, so `n_collocation = 512` is
   plenty — 2.7× faster per epoch than the 1500 default with no accuracy loss.

## Reproduce

```
python3 scripts/exp_planar_cosserat.py all        # full sweep
python3 scripts/exp_planar_cosserat.py fast       # recommended single config (0.30%)
```

Template: `pinn_engine/dsl/templates_lib/planar_cosserat.py`.
Data: `generate_planar_cosserat` in `pinn_engine/data/synthetic.py`.
