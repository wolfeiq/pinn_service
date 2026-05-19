# DeepXDE inverse-problem patterns — what to port, what to leave

Design-decision record. We do **not** depend on DeepXDE at runtime; PINA is the runtime. This document captures the design choices behind DeepXDE's mature inverse-problem implementation so we can port the *patterns* into our PINA-based engine.

References used:
- DeepXDE `examples/pinn_inverse/Lorenz_inverse.py` — canonical inverse setup
- DeepXDE `deepxde/callbacks.py::VariableValue` — discovered-parameter logging
- DeepXDE FAQ (https://deepxde.readthedocs.io/en/latest/user/faq.html) — inverse-problem pitfalls and "I failed to train" entries
- PINA `pina.problem.InverseProblem`, `pina.solver.PINN` source (this repo's runtime dependency)

---

## 1. How DeepXDE exposes unknown inverse parameters

DeepXDE uses an explicit **`external_trainable_variables`** list. Each unknown is wrapped:

```python
C1 = dde.Variable(1.0)            # an unknown, initialized to 1.0
C2 = dde.Variable(1.0)
C3 = dde.Variable(1.0)
external_trainable_variables = [C1, C2, C3]
```

These variables are then:
1. Closed over inside the user-supplied PDE residual function (so the equation depends on them).
2. Passed to `model.compile(..., external_trainable_variables=...)` so they're attached to the optimizer alongside the network weights.
3. Passed to `dde.callbacks.VariableValue(...)` so their discovered values are logged every `period` epochs.

The clarity here is the design lesson: **the unknown is explicit, not hidden inside a network's output**, and it has a separate code path through the optimizer + logging.

`VariableValue` is also worth porting in spirit: a single, drop-in callback that periodically logs each unknown to a file with a configurable precision. It hooks `on_train_begin / on_epoch_end / on_train_end` and handles each backend (TF1, TF2, PyTorch, JAX) uniformly.

## 2. PINA's analog and the gaps

PINA's `InverseProblem` exposes unknowns via:

```python
class InverseProblem(AbstractProblem):
    @abstractmethod
    def unknown_parameter_domain(self): ...        # CartesianDomain over unknowns

    # auto-initialized in __init__:
    self.unknown_parameters = {                     # dict[str, nn.Parameter]
        var: nn.Parameter(uniform-sample-from-domain) for var in unknown_variables
    }
```

PINA's `Equation` callable receives them via the `params_` argument:

```python
def residual(input_, output_, params_=None):
    c, k = params_['c'], params_['k']
    ...
```

**Comparison and gaps:**

| Concern | DeepXDE | PINA | Action for `pinn-engine` |
|---|---|---|---|
| Unknown declaration | `dde.Variable(initial_value)` | `unknown_parameter_domain` (a `CartesianDomain`) | Wrap PINA's mechanism behind our DSL's `Unknown(name, bounds=...)`. The DSL emits the domain. |
| Bounds | not enforced (any float) | range is initial uniform draw; no constraint during training | Our DSL's `bounds` becomes both the initialization range **and** an optional projection in the `ParamDivergenceGuard` callback. |
| Initial value | user-specified scalar | random uniform from domain | Add a `Unknown(..., init="midpoint" \| "random" \| float)` option in our DSL. |
| Parameter logging | `VariableValue` callback (built in) | not built in | Our `ParamConfidence` callback is the analog (richer: running mean/std per epoch). |
| Optimizer attachment | explicit list to `model.compile` | automatic — PINA's solver picks up everything in `problem.unknown_parameters` | Inherit PINA's behavior; nothing to do. |

Net: PINA covers the mechanics, but DeepXDE's *interface* is cleaner. The DSL we build smooths over PINA's `unknown_parameter_domain` boilerplate.

## 3. Loss-weighting choices in the canonical inverse examples

DeepXDE's `Lorenz_inverse.py` and `Navier_Stokes_inverse.py` are instructive on weighting:

- **No fancy adaptive weighting in the examples.** They use uniform weights and rely on the user reading the loss components from the screen + tuning. This is exactly the "months tuning λ" complaint the product summary calls out.
- **Both data and physics losses are on residual scale ~1.0** by design (e.g., normalize state variables to roughly unit magnitude before training). The implicit lesson: if you have to think about λ, you should first think about scaling.
- **Adam first, L-BFGS second** is the default sequence. L-BFGS converges sharply but only from a good neighborhood, and it's brittle when the loss surface still has scale mismatches.

What we port:
- The **Adam → L-BFGS** sequence is the trainer's default.
- The **scale-first, weight-second** philosophy: each template should non-dimensionalize residuals where possible. Loss balancers (SA-PINN, LRA) are second-line defense, not first-line.
- Weighting choices live in the *template*, not the user's code. A `damped_oscillator` template sets sensible defaults for its physics; users only override if they know what they're doing.

What we improve:
- **SA-PINN** (learnable λ, adversarial against the loss): drop-in for templates where scale-first isn't enough.
- **Wang-Teng-Perdikaris LRA**: gradient-norm-based λ updates each epoch. Same role, different mechanism.
- **ReLoBRaLo deliberately omitted** — the build plan's prior experience: it hurts on coupled dynamics.

## 4. Inverse-problem pitfalls (FAQ-derived) and our defenses

The DeepXDE FAQ has a recurring set of pitfalls. Mapping them to our engine's defenses:

| Pitfall | DeepXDE FAQ entry | Our defense |
|---|---|---|
| Unbalanced losses (data and physics terms differ by orders of magnitude) | "I failed to train the network … unbalanced losses" | Templates default to scale-normalized residuals. SA-PINN / LRA as opt-in second line. |
| Loss gets stuck / divergent training | "large training loss" | Optuna `HyperbandPruner` kills the bad trial; `NanGuard` callback aborts immediately on NaN. |
| Inverse parameter doesn't converge to truth | "Solve inverse problems with unknown parameters" thread (~30 linked issues) | **Well-posedness pre-flight (§1.5).** If the sensitivity matrix is rank-deficient or ill-conditioned, the user gets a clear error *before* burning training cycles. This is the single highest-leverage defense. |
| Parameter drifts wildly (beyond physical sense) | scattered across "training failures" thread | `ParamDivergenceGuard` callback: any unknown exceeds 10× its declared bound → trial pruned. |
| Architecture wrong for problem | "How to choose the network size/depth?" | AutoML search space (per template) covers depth ∈ [3,8], width ∈ {32,64,128,256}, activation ∈ {tanh, sin, sintanh, swish}. We don't make the user guess. |
| Reproducibility — "I got different results yesterday" | (implicit across many issues) | Reproducibility manifest (git SHA + seed + data hash + config hash); `pinn-engine verify <manifest>` is the regression test. |

The well-posedness pre-flight is the most important port-from-spirit decision: DeepXDE doesn't have one, but its FAQ is full of users who would have been saved hours by it. PINA also doesn't have one. This is one of our four named differentiators, and it's defensible on first principles.

---

## What we do **not** port

- DeepXDE's TF1/TF2/PaddlePaddle/JAX backend abstraction — we're PyTorch-only via PINA + Lightning.
- DeepXDE's `Model.compile` / `Model.train` API — Lightning's `Trainer.fit()` is the right abstraction for us.
- DeepXDE's geometry library — PINA's `CartesianDomain` and friends cover Phase 1+2 needs (ODE time domains, 1D spatial domains).
- The `Variable` wrapper itself — superseded by our `Unknown` symbol with bounds and identifiability metadata.

---

## Open questions revisited during implementation

- **Does PINA's `params_` argument pass through to all relevant callback hooks?** Confirmed yes via `Equation.residual(input_, output_, params_=...)`. Our DSL's compiled residual functions thread `params_` correctly.
- **Does PINA support per-condition weights at the solver level?** Yes — via the `weighting` argument to `PINN(...)`. Our loss balancers plug in there or via Lightning callbacks that mutate condition weights.
- **What about `SelfAdaptivePINN` in PINA's solver list?** PINA already ships a self-adaptive-PINN solver class. For Phase 1 we use the basic `PINN` solver + our own SA-PINN callback so the weighting mechanism is uniform across templates; we may swap to `SelfAdaptivePINN` later if it benchmarks better.
