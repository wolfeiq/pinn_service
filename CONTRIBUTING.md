# Contributing to `pinn-engine`

Thanks for considering a contribution. The engine has three main contribution
paths — new **equation templates**, **bug reports**, and **feature proposals**
— each with a different bar and process. This file walks through all three,
plus the dev setup, conventions, and where to read the deep docs.

For the full engineering reference (architecture, math, every knob), see
[`docs/ENGINE.md`](docs/ENGINE.md).

---

## Dev setup

```bash
git clone https://github.com/wolfeiq/pinn_service
cd pinn_service
python3 -m venv .venv && source .venv/bin/activate
pip install -e .[dev]    # editable install + test deps
pytest tests/ -q          # 32 tests should pass in ~25 s
```

The engine targets Python 3.9+; tests run on CPU and don't need a GPU.

If you have a GPU, the engine auto-detects via Lightning's
`accelerator="auto"` — set `cfg.accelerator="cpu"` to force CPU for
deterministic comparisons.

---

## Contributing a new equation template

This is the highest-value contribution type — each template proves out a new
problem class. The engine bundles 7 templates today (3 ODEs, 1 partial-id
ODE, 1 coupled 3-DOF ODE, 2 PDEs); good additions would include nonlinear
PDEs, 2D/3D spatial PDEs, coupled multi-physics, or canonical inverse
problems from a domain we don't yet cover.

A template is **6 things in 4 files**. Here's the checklist (working example
in `pinn_engine/dsl/templates_lib/nonlinear_drag_1d.py`):

### 1. Synthetic data generator

`pinn_engine/data/synthetic.py` — add a function `generate_<your_name>` that
takes the unknowns as keyword arguments (with truth defaults) and returns
`(data_dict, truth_dict)`:

```python
def generate_my_problem(
    alpha: float = 1.0,
    beta: float = 2.0,
    n_samples: int = 1000,
    noise_std: float = 0.01,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    # forward-simulate with scipy.integrate.solve_ivp or a closed form
    # ...
    return (
        {"sensor_1": (t_array, y_noisy_array)},   # one entry per sensor
        {"alpha": alpha, "beta": beta},            # truth dict
    )
```

The **unknowns-as-kwargs** contract is what makes the CRLB preflight
(`pinn_engine.diagnostics.crlb`) work — it perturbs each unknown by ±δ via
those kwargs to compute the Fisher information sensitivity matrix. Don't
hard-code the truth values inside the function.

Also add an entry to `pinn_engine/data/__init__.py`'s import list and
`__all__`.

### 2. Template class

Create `pinn_engine/dsl/templates_lib/<your_name>.py`:

```python
from pinn_engine.dsl.symbols import Variable, Parameter, Unknown, Sensor
from pinn_engine.dsl.system import System
from pinn_engine.dsl.templates import register_template
from pinn_engine.core.trainer import TrainConfig


def build_system() -> System:
    t = Variable("t")
    x = Variable("x", depends_on=t)
    alpha = Unknown("alpha", bounds=(0.0, 2.0))
    beta  = Unknown("beta",  bounds=(0.0, 4.0))
    return System(
        state=[x],
        equations=[x.dd + alpha * x.d + beta * x],   # your physics
        sensors=[Sensor("sensor_1", observes=x, noise_std=0.01)],
    )


@register_template("my_problem")
class MyProblem:
    truth = {"alpha": 1.0, "beta": 2.0}
    unknown_bounds = {"alpha": (0.0, 2.0), "beta": (0.0, 4.0)}

    @staticmethod
    def system() -> System:
        return build_system()

    @staticmethod
    def default_config() -> TrainConfig:
        # Pick sane defaults for *this* problem. Match the activation,
        # depth, and lr_scale to the gradient regime of your physics.
        # ODEs typically: depth=4-5, width=64, lr=1e-3, adam_epochs=1000-2000.
        # PDEs typically: depth=5-6, width=64, fourier_features=32-64,
        #                  adam_epochs=5000-10000.
        return TrainConfig(
            depth=5, width=64, activation="tanh",
            lr=1e-3, adam_epochs=2000, lbfgs_iters=0,
            t_range=(0.0, 10.0),
            n_collocation=2000, batch_size=512,
            lam_data_init=10.0, lam_physics_init=1.0,
        )

    @staticmethod
    def synthetic_data(seed: int = 0):
        from pinn_engine.data.synthetic import generate_my_problem
        return generate_my_problem(seed=seed)

    @staticmethod
    def objective(result) -> float:
        # Mean relative error across unknowns — used by AutoML.
        errs = [abs(result.final_params[k] - v) / max(abs(v), 1e-6)
                for k, v in MyProblem.truth.items()]
        return float(sum(errs) / len(errs))

    @staticmethod
    def automl_space(trial):
        # Optional: defines the Optuna search space for `pinn-engine search`.
        return TrainConfig(
            depth=trial.suggest_int("depth", 3, 7),
            width=trial.suggest_categorical("width", [32, 64, 128]),
            activation=trial.suggest_categorical("activation", ["tanh", "sintanh", "swish"]),
            lr=trial.suggest_float("lr", 5e-4, 5e-3, log=True),
            lam_data_init=trial.suggest_float("lam_data_init", 10.0, 1000.0, log=True),
            lam_physics_init=1.0,
            balancer=trial.suggest_categorical("balancer", ["none", "lra", "sapinn"]),
            adam_epochs=2000, lbfgs_iters=0,
            t_range=(0.0, 10.0),
            n_collocation=2000, batch_size=512,
        )
```

### 3. Register it

Add to `pinn_engine/dsl/templates_lib/__init__.py`:

```python
from pinn_engine.dsl.templates_lib import (
    damped_oscillator,
    lorenz,
    # ...
    my_problem,   # ← add yours
)
```

### 4. Tests

Add an entry to `tests/test_templates.py`'s parametrization:

```python
@pytest.mark.parametrize("name", ["damped_oscillator", "lorenz", "diffusion_1d",
                                  "coupled_drag_3d", "my_problem"])
def test_template_system_and_data(name):
    ...
```

Optionally add a dedicated `tests/test_my_problem.py` with template-specific
sanity checks (steady-state values, conservation laws, etc.).

### 5. CRLB sanity check

```python
from pinn_engine.diagnostics.crlb import compute_template_crlb
print(compute_template_crlb("my_problem").summary_table())
```

If the CRLB SE on an unknown is huge (>50% relative), your sensor setup
doesn't physically identify that unknown — either add sensors, tighten the
noise model, or scope your truth/bounds differently. Don't ship a template
where the data can't possibly recover the unknowns.

### 6. (Optional) Brief experiment doc

If the template introduces a new failure mode worth documenting (a basin, a
partial-id case, a numerical pitfall), add a short
`docs/<your_name>_experiments.md` modeled on
`docs/cosserat_causal_experiments.md` or `docs/diffusion_1d_experiments.md`.

### What I'll ask in review

- Is the physics correctly stated and matches a citable source?
- Does the CRLB report look reasonable (no infinite SEs)?
- Does the default config converge on this problem in a normal budget?
- Are the bounds tight enough that PINA's midpoint init isn't 5× the truth?
- Does `pytest tests/` still pass?

---

## Reporting bugs

Please open a [GitHub issue](https://github.com/wolfeiq/pinn_service/issues)
with:

1. **A minimal reproducer** — the smallest `system + data + config` that
   exhibits the bug. Inline the synthetic data generator call if possible.
2. **Expected vs actual** — what `result.final_params` you got vs what you
   thought you'd get, plus the relative error.
3. **Environment** — Python version, OS, GPU (if any), output of
   `pip show pina pytorch-lightning torch optuna | grep Version`.
4. **Run artifacts** — `logs/<run_id>_live.json` and (if reachable)
   `logs/<run_id>_summary.json`. These survive OOM kills and let me see
   the per-epoch convergence trajectory.

**Before filing, check if it's an engine limit, not a bug.** Run the CRLB
preflight first:

```python
from pinn_engine.diagnostics.crlb import compute_template_crlb
print(compute_template_crlb("your_template").summary_table())
```

If your empirical rel_err matches CRLB SE within ~2-3×, the data physically
can't do better — that's not a bug, that's identifiability. The
[`docs/ENGINE.md`](docs/ENGINE.md) "Known limitations" section catalogs
several of these.

### Bugs I particularly care about

- **PINA-side metric logging at scale** — there's a known issue where
  per-condition `data_<sensor>_loss` keys are populated in short runs but
  `None` in long Cosserat runs (callback-order or multi-batch interaction
  not fully traced). If you can reproduce or fix this, please open a PR.
- **Adaptive controller drift-guard tuning** — the `convergence_window=20` /
  `drift_floor=5e-3` defaults are conservative and miss slow drifts (~0.001%
  per epoch on `coupled_drag_3d c_y`). Empirical tuning suggestions welcome.
- **The bounds-clamp timing window** — `_clamp_inverse_problem_params` fires
  after `optimizer.step` but before the next loss eval, so an unknown can
  transiently leave its declared bounds for one epoch. Hard-projection
  callback would fix it.

---

## Proposing features

Before writing code:

1. **Open a [GitHub issue](https://github.com/wolfeiq/pinn_service/issues)
   describing the feature** — what problem it solves, how it composes with
   existing features (adaptive controller, L2 prior, iterative refinement,
   CausalPINN), and what its scope boundary is.
2. **Skim [`docs/ENGINE.md`](docs/ENGINE.md)** to make sure the feature
   isn't already covered or directly contradicted by an existing one.
3. **For algorithmic R&D** (new loss balancer, sampling scheme,
   regularizer, etc.) — cite the paper / derivation, and describe what
   *signal* would tell us the feature is working (e.g. "should close the
   CRLB gap by ≥2× on a training-limited template").
4. **For new engine knobs** — argue why it can't be a callback. Callbacks
   compose; knobs proliferate.

Tagged-as-`good first issue` features I'd happily accept PRs for:

- **L-BFGS post-Adam refinement** for non-inverse problems (the
  `LBFGSInversePINN` solver merges param groups, but the routing is
  ad-hoc; cleaner integration with the controller would let well-posed
  templates do a final L-BFGS sharpening pass automatically).
- **Hard bounds enforcement** via a callback that clamps each
  `problem.unknown_parameters[name]` to its declared range on
  `on_train_batch_end`. Currently PINA only clamps before each
  `loss_data` eval (see "Known limitations" in `docs/ENGINE.md`).
- **Convergence quality metric** in `TrainResult` — `result.crlb_ratio`
  = empirical rel_err / CRLB SE per unknown. Makes "you have N× headroom"
  surfaced automatically.
- **Multi-start initialization** — wrap `train()` to run from N random
  initial points within bounds, pick the lowest-loss one. Compositional
  with everything else.

Features I'm cooler on (out of scope or higher complexity than they look):

- Bayesian / variational posteriors — the existing `train_ensemble`
  (Monte-Carlo seed ensemble) is the engine's preferred UQ. Bayesian
  PINNs are 10-100× more expensive and shaped largely by their priors.
- DSL extensions to non-sympy backends — the sympy-based DSL is the
  contract. New solvers / autodiff engines belong in PINA, not here.

---

## Code style + conventions

- **Format**: PEP 8, 4-space indent, max line length 100. We don't enforce
  a formatter, but `ruff format` produces output we'd accept.
- **Type hints**: required on public functions, optional on internal helpers.
- **Docstrings**: required on public functions and classes; first line is a
  single sentence imperative summary; longer body for non-obvious *why*.
- **Comments**: explain the *why*, not the *what*. Anywhere a config knob,
  threshold, or magic number lives, briefly explain how to choose it.
- **No silent behavior changes**: if you change a default, document it in
  the PR description and update `docs/ENGINE.md` if the change affects
  measured results.

### Commit messages

The git history is detailed by design — every commit message explains the
*why* of the change in enough depth that a future contributor can understand
the motivation without reading the diff. See the existing log for the tone.

Co-author tags are encouraged when AI assistance materially shaped the work
(per Anthropic's recommended attribution).

### Tests

- All new templates must be added to `tests/test_templates.py`'s
  parametrization (smoke test: compiles, has data, has default config).
- New engine features should add at least one test that exercises the
  feature end-to-end (`tests/test_adaptive_controller.py` is a good
  reference; marked `@pytest.mark.slow` if it takes more than ~60 s).
- Bug fixes should land with a regression test that fails on the old code
  and passes on the new.

Run the suite with `pytest tests/ -q`.

---

## PR process

1. Branch from `main`. Naming convention: `feature/<short-name>` or
   `fix/<short-name>`.
2. Make focused commits — one logical change per commit. Easier to review
   and easier to revert if needed.
3. Open a draft PR early if you want feedback on the approach before
   investing too much.
4. Link the issue it addresses (`Closes #N`).
5. CI / `pytest tests/` must pass before merge.

I'll review within a few days. For non-trivial PRs expect at least one
round of revision — usually about scope, naming, or making sure the
feature composes with the existing toolkit cleanly (controller + L2 prior +
iterative refinement should all still work after your change).

---

## Code of conduct

Be excellent. Specifically:

- Assume good intent. Most "this is broken" reports are someone using the
  engine differently than I expected.
- Disagreement is fine, dismissiveness is not.
- Stay on the technical question. If a discussion isn't moving the engine
  forward, I'll close the thread and ask to take it offline.

Harassment, personal attacks, or sustained derailing get warned once, then
banned.

---

## Questions

If you're not sure whether something is a bug, a feature request, or just
something you don't understand yet — open an issue with the `question`
label. No question is too basic.

For everything else: see [`docs/ENGINE.md`](docs/ENGINE.md).
