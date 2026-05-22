# pinn-engine

An open-source **inverse Physics-Informed Neural Network engine**: you write the equations you know plus the parameters you want to discover, hand it noisy sensor data, and it gives you back the physical parameters that make your sensors consistent with your physics — with uncertainty bounds, convergence diagnostics, and a reproducibility manifest for every run.

## Why this exists

PINNs work in research papers and break in practice. Loss balancing eats weeks, architectures need bespoke tuning, well-posedness failures waste days of training. `pinn-engine` is the boring-but-essential infrastructure that makes PINNs usable: a symbolic equation DSL, automated architecture search, a pre-flight identifiability check, drop-in diagnostic callbacks, and reproducibility-first run manifests.

Built on **PINA** + **PyTorch Lightning** so you inherit a mature solver/trainer foundation, with **Optuna** for AutoML and a small set of opinionated additions that turn "a PINN script" into "a tool a robotics engineer can actually deploy."

## Killer demo (one command)

```bash
pip install -e .
pinn-engine search damped_oscillator --n-trials 20 --study demo
optuna-dashboard sqlite:///manifests/optuna_demo.db
```

This searches MLP architectures + loss-balancer + learning rate against the inverse damped-oscillator problem, prunes bad trials with Hyperband, writes a reproducibility manifest per trial, and serves a real-time leaderboard.

## Three examples

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

from pinn_engine.core.trainer import train
from pinn_engine.diagnostics import default_bundle
from pinn_engine.core.trainer import TrainConfig

result = train(system, data={"x_meas": (t_arr, x_noisy)},
               config=TrainConfig(depth=4, width=64, activation="sintanh"),
               callbacks=default_bundle())

print(result.final_params)   # {"c": 0.503, "k": 10.04}
```

## Why your bounds matter (read this first)

`Unknown("k", bounds=(0.0, 100.0))` is **not just bookkeeping** — it's an init prior. PINA initializes every unknown to the **midpoint** of its declared bounds. If the truth is `k=10` and you give it `bounds=(0, 100)`, the optimizer starts at `k=50` — 5× the truth — and may never recover within a normal training budget.

**Rule of thumb:** declare bounds at roughly **2-3× your best guess**. The pre-flight will warn you (`BoundsTooWideWarning`) if a bound's relative width exceeds 4×. Tighter bounds typically land the right answer in the first few hundred epochs; loose bounds drift to a bound or stall mid-search.

This is the single biggest knob in practice. Our `damped_oscillator` template uses `c∈(0,1.5), k∈(0,20)` for a truth of `(0.5, 10)` — discovered to 0.1% error in 800 epochs on CPU.

## Maturity status

| Template | Status | Mean rel-err | Per-run wall-clock |
|---|---|---:|---:|
| `damped_oscillator` | **production** | 0.10 % | ~60 s CPU |
| `lorenz` | **production** | <0.02 % | ~13 min CPU |
| `pendulum` | production | n/a (untuned) | ~60 s CPU |
| `fossen_surge` | production | 9.8 % (partial identifiability — physics limit, not engine) | ~30 s CPU |
| `cosserat_rod` (PDE) | **beta** | converges to <10 % at 10k+ epochs (~30 min CPU); see `docs/literature_comparison.md` | ~30 min CPU |

PDE templates use the same engine, DSL, AutoML, dashboard — they just need
~10× more training budget than ODE templates to reach published-paper
accuracy. This is the standard story for PINN-on-PDE inverse problems
(see DD-PINN, Wang 2022 causal PINN — both run at 10k-100k epochs).

## What's in v0.1 (Phase 1 + 2)

- Sympy-based equation DSL with `Variable`, `Parameter`, `Unknown`, `Sensor`, `System`
- Three equation templates: damped oscillator, Lorenz, 1D diffusion
- PINA + PyTorch Lightning trainer with Adam → L-BFGS handoff
- SA-PINN and Wang-Teng-Perdikaris LRA loss balancers (switchable)
- Well-posedness pre-flight (analytical sensitivity matrix via `torch.func.jacrev`)
- Four-callback diagnostic bundle: residual heatmap, sensor residuals, parameter confidence, spectral bias
- Optuna AutoML with `PyTorchLightningPruningCallback` + Hyperband
- Reproducibility manifests (git SHA, seeds, hashes, final params + UQ)
- Typer CLI: `train | search | inspect | verify`

## What's deferred (later phases)

- Robotics templates (Fossen 6-DOF, Cosserat rod, rigid-body + contact)
- ROS 2 bag ingestion, ONNX / TorchScript export
- Streamlit dashboard
- Poseidon PFM initialization
- CasADi / Drake / acados MPC export

## Design references

- DeepXDE — read as design reference (see `docs/deepxde_patterns.md`), not a runtime dependency.
- PINA — runtime PINN library.
- The build plan: `/Users/mary/Downloads/claude_code_build_plan.md`.
- The product vision: `/Users/mary/Downloads/product_summary_pinn.md`.

## License

Apache-2.0.
