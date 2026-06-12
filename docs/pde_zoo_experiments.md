# Beyond rods: a zoo of PDE inverse problems

A batch of templates added to show `pinn-engine` is a *general* inverse-PDE
solver across very different equation classes and domains — transport, dispersive
nonlinear waves, and quantitative finance — each recovering its parameters from a
dense noisy field.

| template | PDE | class | unknowns | result |
|---|---|---|---|---|
| `advection_diffusion_1d` | `u_t + v·u_x = D·u_xx` | linear transport | v, D | **v 0.1% / D 0.9%** |
| `kdv_1d` | `u_t + 6·u·u_x + δ·u_xxx = 0` | dispersive, 3rd-order, nonlinear | δ | **training-limited** (identifiable; PINN explain-away) |
| `black_scholes` | `V_t + (r−σ²/2)V_x + (σ²/2)V_xx − rV = 0` | finance (non-physics) | σ (implied vol) | **σ ~3%** |

(plus the earlier `burgers_1d` and `fisher_kpp` — nonlinear advection–diffusion
and reaction–diffusion.)

## Advection–diffusion — two separable unknowns, both recovered

`u_t + v·u_x = D·u_xx`. A Gaussian pulse **advects** at velocity `v` and
**broadens** with diffusivity `D`. Ground truth is the exact advected heat
kernel. The two effects are cleanly separable from the data (mean motion ↔ `v`,
broadening ↔ `D`), and unlike Fisher-KPP's sub-dominant diffusion, here `D` is
directly constrained by the visible spreading — **both** recover to <1% by epoch
1000 (v 0.10%, D 0.9% at the CRLB floor 0.08%).

## KdV — the first 3rd-order, dispersive template

`u_t + 6·u·u_x + δ·u_xxx = 0`, the birthplace of the **soliton**. This is the
engine's first **third-order** (`u_xxx`) residual — it exercises the autograd
path one order deeper than the wave/diffusion templates. Ground truth is the
exact single-soliton `u = 2δk²·sech²(k(x − 4δk²t − a))` (no solver error). Because
dispersion is a *leading-order* term in a soliton (the `δ·u_xxx` term balances the
nonlinear steepening), `δ` is cleanly identifiable.

**Honest result — δ is identifiable but PINN-training-limited.** The PINN drives
`δ` straight through truth to its lower bound (0.5 → 0.10 → bound): the network
represents the soliton's shape itself and explains the dispersion away. `δ` *is*
identifiable — a direct finite-difference regression on the clean soliton
recovers **δ = 0.504** (truth 0.5; CRLB floor 0.01%). This is the **third**
instance of the engine's recurring failure mode (after the rod sub-dominant
stiffnesses and Fisher-KPP's `D`), and the sharpest: **`δ` multiplies the
*highest-order* derivative (`u_xxx`), where spectral bias gives the network the
most freedom — so the explain-away is worse than for a 2nd-order coefficient.**
A clean rule of thumb emerges: *in a multi-term PDE inverse, the coefficient of
the highest-order derivative is the hardest for a collocation PINN, and severity
grows with derivative order.* (Compare `advection_diffusion_1d` above, where the
2nd-order `D` recovers cleanly — because the pulse's broadening *directly and
visibly* constrains it, rather than being a sub-dominant correction.)

## Black–Scholes — a non-physics inverse PDE

`V_t + (r−σ²/2)V_x + (σ²/2)V_xx − rV = 0` in log-price `x = ln S`. Same engine,
same DSL, same CRLB preflight — an entirely different field. Ground truth is the
exact European-call formula; the inverse recovers the **implied volatility `σ`**
(what options traders back out of market prices) from the price surface
`V(x,t)`. `σ` enters as the variance `σ²` in both the drift and diffusion terms.
Recovered to ~3% (CRLB floor 0.04%) — a clean demonstration that the engine is
not physics-specific.

## Reproduce

```python
from pinn_engine.dsl.templates import get_template
from pinn_engine.core.trainer import train
for name in ("advection_diffusion_1d", "kdv_1d", "black_scholes"):
    tpl = get_template(name); cfg = tpl.default_config(); cfg.skip_preflight = True
    data, _ = tpl.synthetic_data(seed=0)
    print(name, train(system=tpl.system(), data=data, config=cfg).final_params)
```

Templates in `pinn_engine/dsl/templates_lib/`; generators in
`pinn_engine/data/synthetic.py`.
