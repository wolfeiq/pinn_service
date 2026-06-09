# Fisher-KPP reaction-diffusion inverse: recovering D and r

The Fisher-KPP equation is the archetypal **reaction-diffusion** PDE — and a
genuinely different domain from the engine's mechanics/transport templates:
population genetics (spread of an advantageous gene, Fisher 1937), invasive-
species ecology, combustion fronts, tumour growth.

## The problem

```
u_t = D·u_xx + r·u(1−u),   x ∈ [0,L],  u(0,t)=1,  u(L,t)=0
```

Linear diffusion `D·u_xx` plus a **logistic reaction** `r·u(1−u)` (the engine's
first quadratic-reaction term). It admits a travelling front advancing at the KPP
speed `~2√(rD)` with width `~√(D/r)`. Two unknowns: the front *speed* constrains
`rD`, the front *width* constrains `D/r`, so `D` (diffusion) and `r` (growth
rate) are **separately identifiable**. Inverse: recover both from a dense noisy
`u(x,t)` grid.

Ground truth from a stable method-of-lines solver (central differences + stiff
BDF); `u` stays bounded in `[0,1]` and the front propagates (verified in
`test_templates.py`).

## Identifiability (CRLB)

```
   Unknown   CRLB SE/|truth|
   D             0.31%
   r             0.15%
```

## Result — `r` recovered; `D` is the hard (sub-dominant) coefficient

This template exposes a real and instructive asymmetry between the two unknowns.

| unknown | PINN (12k ep) | direct FD regression (clean) | direct FD regression (noisy) |
|---|---|---|---|
| `r` (reaction) | **~1%** | **0.0%** | **~0.5%** |
| `D` (diffusion) | collapses to bound | 0.5013 (0.3%) | ~10–30% (biased low) |

**`r` is recovered robustly everywhere.** The logistic reaction `r·u(1−u)` is
*algebraic* in the measured field, so once the network fits `u`, `r` is pinned
directly — the PINN gives ~1%.

**`D` is training-limited in the PINN** — it sails through truth and collapses to
the lower bound. Diffusion is *sub-dominant* to the reaction here, so `D`
multiplies a small `u_xx` correction; the network can represent the front shape
itself and drive `D→0` (the "explain-away" failure seen on sub-dominant
high-derivative coefficients throughout this engine — cf. the rod GA/EA and the
dynamic-Cosserat work). Contrast `diffusion_1d`, where `D` is the *only* term
balancing `u_t` and recovers cleanly.

`D` **is identifiable** — a direct finite-difference least-squares on the clean
field recovers `D=0.5013, r=1.0000`. But it's the genuinely hard quantity: it
lives in a 2nd spatial derivative, so under measurement noise even the direct
regression is bias-variance-trapped (small smoothing window → noise blows up
`u_xx`; large window → smooths the front → both underestimate `D`).

**Takeaway:** in a multi-term reaction–diffusion inverse, the *reaction rate* is
easy and the *sub-dominant diffusion coefficient* is hard — for both PINN and
regression. A clean, honest result that reinforces the engine's recurring lesson
about sub-dominant high-derivative coefficients.

## Reproduce

```
python3 - <<'PY'
from pinn_engine.dsl.templates import get_template
from pinn_engine.core.trainer import train
tpl = get_template("fisher_kpp"); cfg = tpl.default_config(); cfg.skip_preflight = True
data, _ = tpl.synthetic_data(seed=0)
print(train(system=tpl.system(), data=data, config=cfg).final_params)
PY
```

Template: `pinn_engine/dsl/templates_lib/fisher_kpp.py`.
Data/solver: `generate_fisher_kpp` in `pinn_engine/data/synthetic.py`.
