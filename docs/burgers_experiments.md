# Burgers' equation inverse: recovering viscosity ν

The 1-D viscous Burgers equation is *the* canonical nonlinear PDE and the classic
PINN benchmark (Raissi et al. 2019). It was **deferred** earlier in this repo
(the changelog: "FD forward solvers blow up at shock formation… future session
needed"). This template cracks it.

## The problem

```
u_t + u·u_x = ν·u_xx,   x ∈ [-1,1],  u(x,0) = −sin(πx),  u(±1,t) = 0
```

The advective term `u·u_x` steepens the profile into a sharp internal layer at
`x=0`; viscosity `ν·u_xx` smooths it. The balance sets the layer width
(`~ν`). Inverse problem: recover `ν` from a dense noisy `u(x,t)` grid.

## Cracking the forward solver

The naive non-conservative finite difference of `u·u_x` blows up at the steep
layer. The fix (`generate_burgers_1d`):

- **conservative flux form** `u_t = −½(u²)_x + ν·u_xx` (central differences),
- a **stiff implicit integrator** (BDF) — the diffusion term is stiff on a fine
  grid,
- a fine spatial grid (256 points) subsampled to the measurement grid.

Verified finite and bounded through the steepening at `ν = 0.05`, `0.1/π`, *and*
the hard benchmark `ν = 0.01/π` (max gradient grows 7 → 12 → 113 as ν drops; the
solution never blows up). See `test_templates.py`.

## Identifiability (CRLB)

```
   Unknown   CRLB SE/|truth|
   nu             0.13%
```

Dense `u(x,t)` data pins `ν` tightly.

## Result

| | ν | rel_err |
|---|---|---|
| truth | 0.05 | — |
| **recovered (PINN, 6000 ep)** | **0.04977** | **0.46%** |

The PINN reaches ~0.5% by epoch 1000 and settles near the CRLB floor. The
nonlinear self-advection `u·u_x` (the engine's first such residual term) poses no
problem for the *inverse* — the dense data anchors the field and the physics
residual pins `ν`. Fourier features (64) give the network the spectral capacity
to represent the steep internal layer.

## Reproduce

```
python3 - <<'PY'
from pinn_engine.dsl.templates import get_template
from pinn_engine.core.trainer import train
tpl = get_template("burgers_1d"); cfg = tpl.default_config(); cfg.skip_preflight = True
data, truth = tpl.synthetic_data(seed=0)
print(train(system=tpl.system(), data=data, config=cfg).final_params)
PY
```

Template: `pinn_engine/dsl/templates_lib/burgers_1d.py`.
Data/solver: `generate_burgers_1d` in `pinn_engine/data/synthetic.py`.
