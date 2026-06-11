# pinn-engine

**An inverse Physics-Informed Neural Network engine for parameter discovery.**

You write the governing equations you know (ODEs/PDEs) and mark the parameters
you *don't*. You hand it noisy sensor data. It returns the physical parameters
that make your model consistent with the measurements — and, uniquely, tells you
**up front whether the data can identify those parameters at all**.

> 🏛️ **Official page & white paper: [the13olympian.com](https://the13olympian.com)**
> · Full engineering reference: [`docs/ENGINE.md`](docs/ENGINE.md)

---

## What it's built for

Calibrating physical models from real measurements — wherever you have a
governing equation and want the parameters inside it:

- **Soft robotics** — continuum/Cosserat-rod stiffness, actuation, material, and
  contact identification.
- **Structural & material characterization** — beams, bars, elastic/hyperelastic/
  viscoelastic constitutive laws.
- **Dynamical systems** — damping, drag, reaction/growth rates, chaotic-system
  parameters.
- **Scientific inverse problems** generally — diffusion, advection, reaction.

## Why it's different

Most PINN tooling targets *forward* simulation and leaves you to hand-tune. This
engine is purpose-built for **inverse problems** and ships the things that
actually make them work in practice:

| | |
|---|---|
| **Symbolic equation DSL** | Declare physics once (`Variable`/`Parameter`/`Unknown`/`Sensor`); it compiles to torch residuals + autograd automatically. |
| **CRLB identifiability preflight** | Computes the *data-theoretic* best-case error per parameter **before** training — separating "your sensors can't do better" from "your training can." Few PINN frameworks have this. |
| **Adaptive unknowns-LR controller** | A runtime state machine that auto-tunes the unknown parameters' learning rate — removes the hand-tuning that eats weeks. |
| **AutoML + reproducibility** | Optuna/Hyperband architecture search, per-run manifests (git SHA, seeds, hashes, final params), ensemble UQ, ONNX/TorchScript export. |

## Highlights / what's been validated

Verified across **14 inverse templates** spanning very different physics:

- **Dynamics & chaos** — damped oscillator, pendulum, nonlinear/coupled drag,
  Lorenz system.
- **Nonlinear PDEs** — diffusion, **Burgers'** (advection–diffusion, recovered
  ν to 0.46%), **Fisher–KPP** (reaction–diffusion).
- **Structural mechanics** — elastic bar, Euler–Bernoulli beam, large-deflection
  elastica.
- **A full soft-robotics Cosserat-rod stack** — planar & 3-D, static & dynamic
  stiffness ID (bending/shear/axial/torsion); tendon & pneumatic actuation with
  **self-calibration**; hyperelastic, viscoelastic, and combined material ID; and
  **proprioceptive contact sensing** (where and how hard a soft finger is
  touching, from shape alone). See [`docs/ENGINE.md`](docs/ENGINE.md) and the
  per-topic notes in [`docs/`](docs/).

---

## Quick start

```bash
pip install -e .

# AutoML search on an inverse problem, with a live leaderboard:
pinn-engine search damped_oscillator --n-trials 20 --study demo
optuna-dashboard sqlite:///manifests/optuna_demo.db
```

Define and solve an inverse problem in a few lines:

```python
from pinn_engine.dsl import Variable, Parameter, Unknown, Sensor, System
from pinn_engine.core.trainer import train, TrainConfig
from pinn_engine.core.adaptive_controller import AdaptiveUnknownsController

t = Variable("t")
x = Variable("x", depends_on=t)
m = Parameter("m", value=1.0)
c = Unknown("c", bounds=(0.0, 5.0))      # ← discover these
k = Unknown("k", bounds=(0.0, 100.0))

system = System(
    state=[x],
    equations=[m * x.dd + c * x.d + k * x],          # m·ẍ + c·ẋ + k·x = 0
    sensors=[Sensor("x_meas", observes=x, noise_std=0.01)],
)

result = train(system, data={"x_meas": (t_arr, x_noisy)},
               config=TrainConfig(depth=4, width=64, adaptive_unknowns_lr=True),
               callbacks=[AdaptiveUnknownsController()])

print(result.final_params)   # {"c": 0.503, "k": 10.04}
```

More: `python examples/01_damped_oscillator.py`, `02_lorenz_inverse.py`,
`03_automl_search.py`.

## Know what's achievable *before* you train

```python
from pinn_engine.diagnostics.crlb import compute_template_crlb
print(compute_template_crlb("diffusion_1d").summary_table())
```

The CRLB preflight reports the Cramér–Rao lower bound on each unknown's standard
error. If your sensors physically can't identify a parameter better than X%, no
PINN, EKF, or hand-tuning will — and the diagnostic tells you that in seconds,
separating **data-limited** problems (raise the sensor quality) from
**training-limited** ones (push the optimizer).

> **Tip — bounds are a prior, not bookkeeping.** Unknowns initialize at the
> *midpoint* of their bounds, so `bounds=(0, 100)` for a truth of `10` starts you
> at `50`. Declare bounds at ~2–3× your best guess; the preflight warns when they
> are too wide. No good guess? `iterative_train()` shrinks bounds around each
> result and re-trains (took pendulum 19% → 0.43% in one extra pass).

---

## Bundled templates

| template | problem | unknowns |
|---|---|---|
| `damped_oscillator` | `m·ẍ + c·ẋ + k·x = 0` | c, k |
| `pendulum` | nonlinear pendulum with friction | c |
| `lorenz` | Lorenz chaotic system | σ, ρ, β |
| `nonlinear_drag_1d` | `m·u̇ = τ + c_lin·u + c_quad·u²` | c_lin, c_quad |
| `coupled_drag_3d` | planar 3-DOF rigid body + Coriolis | c_x, c_y, c_n |
| `diffusion_1d` | `u_t = D·u_xx` | D |
| `burgers_1d` | `u_t + u·u_x = ν·u_xx` (advection–diffusion) | ν |
| `fisher_kpp` | `u_t = D·u_xx + r·u(1−u)` (reaction–diffusion) | D, r |
| `axial_elastic_bar` | `EA·u'' + p₀ = 0` (static elasticity) | EA |
| `euler_bernoulli_beam` | `EI·w'''' = q₀` (linear beam) | EI |
| `cosserat_rod` | `ρ·u_tt = E·u_ss` (wave) | E |
| `planar_elastica` | `EI·θ'' = −P₀·cos θ` (large-deflection rod) | EI |
| `planar_cosserat` | full planar Cosserat rod (shear + extension) | EI, GA, EA |
| `dynamic_cosserat` | dynamic Cosserat rod (inertia + time) | EI, GA, EA |

The soft-robotics identification stack (3-D rods, actuation, material, contact)
lives in [`pinn_engine/baselines/`](pinn_engine/baselines/) with topic notes in
[`docs/`](docs/).

## Engine features

- **DSL** → torch residual compilation with autograd.
- **Adaptive unknowns-LR controller** (DESCEND/PROBE/CONVERGED state machine).
- **CRLB preflight** (forward-simulation Fisher-information bound).
- **L2 / Tikhonov prior** for partial identifiability; **iterative bound-tightening**.
- **Loss balancers** (static, SA-PINN, Wang–Teng–Perdikaris LRA).
- **CausalPINN** + ε-annealer (Wang 2022) for wave-equation inverse.
- **RAR** residual-adaptive collocation refinement.
- **AutoML** (Optuna + Hyperband), **ensemble UQ**, reproducibility manifests.
- **Export** (ONNX + TorchScript), **Streamlit dashboard**, and a CLI:
  `pinn-engine train | search | verify | dashboard`.

## Tech stack

Python · **PyTorch** + **PyTorch Lightning** · **PINA** (PINN runtime) ·
**Optuna** (AutoML) · **SymPy** (equation DSL) · **SciPy/NumPy** (ground-truth
solvers & diagnostics).

## Docs & references

- [the13olympian.com](https://the13olympian.com) — official page & white paper.
- [`docs/ENGINE.md`](docs/ENGINE.md) — full architecture, math, every knob, the
  validation table, and the complete R&D changelog.
- [`docs/`](docs/) — per-topic experiment notes.
- Design reference: DeepXDE (`docs/deepxde_patterns.md`); built on PINA.

## License

Apache-2.0.
