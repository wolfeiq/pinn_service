# pinn-engine

An open-source **inverse Physics-Informed Neural Network engine**: you write the
equations you know plus the parameters you want to discover, hand it noisy
sensor data, and it gives you back the physical parameters that make your
sensors consistent with your physics — with diagnostics, a Cramér-Rao preflight
that tells you the data-theoretic best you can hope for, and a reproducibility
manifest for every run.

For the full engineering reference (architecture, math, every knob, every
template, the changelog of every R&D iteration), see [`docs/ENGINE.md`](docs/ENGINE.md).

## Why this exists

PINNs work in research papers and break in practice. Loss balancing eats weeks,
architectures need bespoke tuning, well-posedness failures waste days of
training. `pinn-engine` is the infrastructure that makes PINNs usable for
inverse problems: a symbolic equation DSL, an auto-adaptive LR controller, an
L2 prior for partial identifiability, an iterative bound-tightening refinement
loop, a Cramér-Rao preflight diagnostic, AutoML, drop-in diagnostic callbacks,
and reproducibility-first run manifests.

Built on **PINA** + **PyTorch Lightning** + **Optuna**.

## Killer demo (one command)

```bash
pip install -e .
pinn-engine search damped_oscillator --n-trials 20 --study demo
optuna-dashboard sqlite:///manifests/optuna_demo.db
```

Searches MLP architectures + loss balancer + learning rate against the
inverse damped-oscillator problem, prunes bad trials with Hyperband, writes
a reproducibility manifest per trial, serves a real-time leaderboard.

## Examples

```bash
python examples/01_damped_oscillator.py    # discover c, k from noisy x(t)
python examples/02_lorenz_inverse.py       # discover σ, ρ, β from a chaotic trajectory
python examples/03_automl_search.py        # programmatic AutoML
```

## Concepts in 60 seconds

```python
from pinn_engine.dsl import Variable, Parameter, Unknown, Sensor, System

t = Variable("t")
x = Variable("x", depends_on=t)
m = Parameter("m", value=1.0)
c = Unknown("c", bounds=(0.0, 5.0))
k = Unknown("k", bounds=(0.0, 100.0))

system = System(
    state=[x],
    equations=[m * x.dd + c * x.d + k * x],
    sensors=[Sensor("x_meas", observes=x, noise_std=0.01)],
)

from pinn_engine.core.trainer import train, TrainConfig
from pinn_engine.core.adaptive_controller import AdaptiveUnknownsController

result = train(system, data={"x_meas": (t_arr, x_noisy)},
               config=TrainConfig(depth=4, width=64, activation="sintanh",
                                  adaptive_unknowns_lr=True),
               callbacks=[AdaptiveUnknownsController()])

print(result.final_params)   # {"c": 0.503, "k": 10.04}
```

## CRLB preflight: know what's achievable BEFORE training

```python
from pinn_engine.diagnostics.crlb import compute_template_crlb

r = compute_template_crlb("diffusion_1d")
print(r.summary_table())
# Reports the data-theoretic lower bound on each unknown's SE.
# If your sensors physically can't identify the parameter better than X%,
# no PINN, EKF, or hand-tuning will. CRLB tells you that up-front.
```

The CRLB diagnostic separates *data-limited* problems (already at the
theoretical floor — no improvement possible without better sensors) from
*training-limited* problems (large gap — room to push). See `docs/ENGINE.md`
for the validation table across all bundled templates.

## Why your bounds matter (read this first)

`Unknown("k", bounds=(0.0, 100.0))` is **not just bookkeeping** — it's an init
prior. PINA initializes every unknown at the **midpoint** of its declared
bounds. If the truth is `k=10` and you give it `bounds=(0, 100)`, the
optimizer starts at `k=50` — 5× the truth — and may never recover within a
normal training budget.

**Rule of thumb:** declare bounds at roughly **2-3× your best guess**. The
preflight will warn (`BoundsTooWideWarning`) if a bound's width exceeds 4×
the midpoint. Tighter bounds land the right answer in the first few hundred
epochs; loose bounds drift to a bound or stall mid-search. If you don't have
a good guess, wrap with `pinn_engine.core.iterative_train.iterative_train()`
— a meta-loop that shrinks bounds around each result and re-trains, validated
to take pendulum from 19% to 0.43% (45× tighter) in one extra iteration.

## Bundled templates

Twelve inverse-problem templates (see `docs/ENGINE.md` for full math + the best
results achieved through the engine):

| template | problem | unknowns |
|---|---|---|
| `damped_oscillator` | `m·ẍ + c·ẋ + k·x = 0` | c, k |
| `lorenz` | Lorenz chaotic system | σ, ρ, β |
| `pendulum` | nonlinear pendulum with friction | c |
| `nonlinear_drag_1d` | `m·u̇ = τ + c_lin·u + c_quad·u²` | c_lin, c_quad (partial-id) |
| `coupled_drag_3d` | planar 3-DOF rigid body with Coriolis coupling | c_x, c_y, c_n |
| `diffusion_1d` | `u_t = D·u_xx` | D |
| `cosserat_rod` | `ρ·u_tt = E·u_ss` (wave equation) | E_unit |
| `axial_elastic_bar` | `EA·u'' + p₀ = 0` (static elasticity) | EA_unit |
| `euler_bernoulli_beam` | `EI·w'''' = q₀` (linear beam bending) | EI_unit |
| `planar_elastica` | `EI·θ'' = −P₀·cos θ` (large-deflection soft rod) | EI_unit |
| `planar_cosserat` | full planar Cosserat rod (shear + extension) | EI_unit, GA_unit, EA_unit |
| `dynamic_cosserat` | dynamic Cosserat rod (inertia + time, space-time PDE) | EI_unit, GA_unit, EA_unit |

## Engine features

- **DSL**: declare physics symbolically (`Variable`, `Parameter`, `Unknown`,
  `Sensor`), compile to torch callables automatically.
- **Adaptive controller** (`AdaptiveUnknownsController`): runtime LR
  state machine (DESCEND/PROBE/CONVERGED) that auto-tunes the unknowns'
  LR — no hand-picked `param_lr_scale`.
- **L2 prior** (`TrainConfig.unknown_l2_prior` + `unknown_l2_anchor`):
  Tikhonov regularization for partial-identifiability problems.
- **Iterative refinement** (`iterative_train`): meta-loop that shrinks
  bounds around each result for precision sharpening.
- **CRLB preflight**: forward-simulation Fisher-information bound on each
  unknown's SE before any training.
- **CausalPINN** + ε-annealer (Wang 2022) for wave-equation inverse.
- **AutoML** with Hyperband pruning (Optuna).
- **Loss balancers**: static, SA-PINN, Wang–Teng–Perdikaris LRA.
- **Diagnostic callbacks**: per-epoch unknown JSON dump (survives OOM/SIGKILL),
  sensor residuals, parameter confidence trajectory, spectral bias.
- **Ensemble UQ**: train N seeds, get distribution over unknowns.
- **Reproducibility manifests**: git SHA, seeds, hashes, final params per run.
- **Export**: ONNX + TorchScript with round-trip verification.
- **Streamlit dashboard** over manifests + Optuna studies.
- **CLI**: `pinn-engine train | search | verify | dashboard`.

## Design references

- DeepXDE — read as design reference (`docs/deepxde_patterns.md`), not a
  runtime dependency.
- PINA — the runtime PINN library this engine extends.
- For the full math + R&D history, read [`docs/ENGINE.md`](docs/ENGINE.md).

## License

Apache-2.0.
