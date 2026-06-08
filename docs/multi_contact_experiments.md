# Multi-point contact estimation

Extends single-point proprioceptive sensing (`contact_id`) to **several
simultaneous contacts** — a soft finger wrapping an object, or a rod pressed
against multiple obstacles. Each point contact kinks the internal moment, so the
rod carries several curvature kinks; we localize and size all of them from the
measured shape.

## The exact variable

`m'(s) = −cosθ·n_y` and `∫cosθ ds = x(s)` (the horizontal coordinate), so the
bending moment `M(s) = EI·θ'(s)` is **exactly piecewise-linear in x** — slope
`−n_y` per segment, with a clean kink at each contact (no within-segment
curvature to confound the fit, unlike `κ(s)`). Fitting `M` vs `x` as a
continuous piecewise-linear function gives the contact positions (breakpoints)
and forces (`F_i` = jump in `n_y`).

`recover_n_contacts(data, n)` recovers a known number of contacts by coordinate
descent on the breakpoints; `recover_contacts(data)` adds a best-effort automatic
count (BIC over the contact number).

## Results (`scripts/exp_multi_contact.py`)

| true contacts | recovered (N known) |
|---|---|
| (0.30, 1.5), (0.65, 2.0) | (0.289, 1.47), (0.644, 2.06) |
| (0.25, 1.0), (0.50, 1.5), (0.75, 2.0) | (0.276, 1.05), (0.514, 1.41), (0.735, 1.64) |

With N known: contact locations to ~0.01–0.03 (of unit length) and forces to
~5–20% under shape noise. The automatic count is reliable for 1–2 contacts (it
may under-count denser scenes — a soft finger usually knows how many points it
grasps with).

## Where this sits

Completes the contact story: from single-point to multi-point whole-body tactile
sensing, the basis for soft grasping. Natural extension: distributed (continuous)
contact patches, where the shear varies continuously rather than in steps.

## Reproduce

```
python3 scripts/exp_multi_contact.py
```

Module: `pinn_engine/baselines/contact_multi_id.py`.
