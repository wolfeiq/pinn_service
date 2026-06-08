# `pinn-engine` — engineering reference

Single-file reference for the engine: changelog, architecture, capabilities,
training pipeline, and all the math. Authoritative as of 2026-06-02 (CRLB
diagnostic + drift-guard era); see the git log for newer changes.

---

## Table of contents

1. [What it is](#what-it-is)
2. [Changelog](#changelog)
3. [Architecture](#architecture)
4. [Capabilities](#capabilities)
5. [The training pipeline](#the-training-pipeline)
6. [The mathematics](#the-mathematics)
7. [Templates inventory](#templates-inventory)
8. [Public API surface](#public-api-surface)
9. [Known limitations + open issues](#known-limitations--open-issues)

---

## What it is

An **inverse Physics-Informed Neural Network engine**: you write the equations
you know plus the parameters you want to discover, hand it noisy sensor data,
and it gives you back the physical parameters that make your sensors consistent
with your physics — with diagnostics and a reproducibility manifest per run.

Built on **PINA** (PINN library) + **PyTorch Lightning** (training loop) +
**Optuna** (AutoML). The novel pieces in this repo are:

- A **symbolic equation DSL** so PDEs/ODEs are declared once and lowered to
  torch callables, sensors, and the PINA problem object automatically.
- A **runtime LR controller** (`AdaptiveUnknownsController`) that auto-tunes
  the unknowns' LR via velocity-band + loss-probe state machine, replacing
  hand-tuned `param_lr_scale` and the two-phase trigger/taper.
- A **Tikhonov L2 prior** on unknowns for partial-identifiability.
- An **iterative bound-tightening** meta-loop (`iterative_train`) for
  precision sharpening of well-posed problems.
- **CausalPINN** (Wang 2022) with an ε-annealer for wave-equation inverse.
- A **reproducibility manifest** + diagnostics (UQ ensemble, spectral bias,
  parameter confidence, sensor residuals) per run.

Bundled with 9 reference templates (3 ODE-only, 1 partial-id ODE, 1 coupled
3-DOF ODE, 2 dynamic PDEs, 2 static structural PDEs — see
[Templates inventory](#templates-inventory)).

---

## Changelog

Reverse chronological. Commit SHAs in parens. Major moments **bold**.

### Multi-point contact sensing (Jun 08, 2026)

- **`contact_multi_id` — several simultaneous contacts** (soft finger wrapping an
  object). Key trick: `M=EI·θ'` is *exactly piecewise-linear in x* (since
  `∫cosθ ds = x`), so a continuous PWL fit of `M` vs `x` reads off contact
  positions (breakpoints) and forces (slope jumps). `recover_n_contacts` (N
  known): locations ~0.01-0.03, forces ~5-20% noisy, for 1-3 contacts;
  `recover_contacts` adds a best-effort BIC count (reliable for 1-2). Extends the
  single-point sensor to whole-body multi-contact. See
  `docs/multi_contact_experiments.md`; demo `scripts/exp_multi_contact.py`.

### Combined hyper- + visco-elastic (Jun 08, 2026)

- **`viscohyperelastic_rod_id` — nonlinear *and* rate-dependent material** (QLV /
  nonlinear SLS): `M = g_∞·M_e(κ) + q`, `q̇=−q/τ+(1−g_∞)Ṁ_e`, `M_e=a1κ+a3κ³`. A
  multi-level creep sweep separates the physics — instantaneous curvatures
  `κ_g(M_0)` give the nonlinear elastic `a1,a3`; equilibrium `κ_∞(M_0)` give the
  relaxation strength `g_∞`; a 1-D ODE fit gives `τ`. All four to ~1% (clean);
  the large linear-fit residual confirms genuine nonlinearity. Unifies the
  material story (the realistic silicone constitutive law is both). See
  `docs/viscohyperelastic_experiments.md`; demo `scripts/exp_viscohyperelastic.py`.

### Viscoelastic material identification (Jun 08, 2026)

- **`viscoelastic_rod_id` — rate-dependent material.** Real soft materials creep,
  relax, and dissipate (hysteresis). Models the rod bending as a Standard Linear
  Solid (Zener): `M = E_∞·κ + q`, `q̇ = −q/τ + E_1·κ̇`. Recovers `(E_∞, E_1, τ)`
  **two independent ways** — a creep test (constant actuation → curvature drifts
  `M_0/E_g → M_0/E_∞` with retardation `τ·E_g/E_∞`) and a DMA frequency sweep
  (oscillate → phase lag → storage/loss moduli, loss peak at `ωτ=1`). Clean:
  exact; noisy: <1%; the two methods agree to <1% (consistency check). Completes
  the material story (linear → hyperelastic → viscoelastic). See
  `docs/viscoelastic_experiments.md`; demo `scripts/exp_viscoelastic.py`.

### Proprioceptive contact sensing (Jun 08, 2026)

- **`contact_id` — environment interaction.** A point contact makes the internal
  shear jump → the moment `m(s)=EI·κ(s)` has a slope kink at the contact. From
  the measured shape alone, a changepoint fit of κ(s) recovers **where** the rod
  touches (`s_c`) and **how hard** (`F_c`) — whole-body tactile sensing, no force
  sensor. Geometrically-exact: `κ'` jumps by `−cosθ(s_c)·F_c/EI`. Recovers `s_c`
  to ~1% of length and `F_c` to ~3% across locations/forces, robust to noise. See
  `docs/contact_experiments.md`; demo `scripts/exp_contact.py`. With this the
  soft-rod stack spans drive (tendon+pneumatic), nonlinear material, and contact.

### Hyperelastic material identification (Jun 08, 2026)

- **`hyperelastic_rod_id` — nonlinear constitutive identification.** Real soft
  materials are hyperelastic (nonlinear, strain-stiffening); a linear rod is only
  the small-strain limit. Models the symmetric leading nonlinearity
  `M(κ)=a1κ+a3κ³`, `N(ε)=b1ε+b3ε³` and recovers all four coefficients from a load
  sweep (tip moment → constant statically-determinate curvature). Clean: exact;
  noisy: ~1%. The **linear-only fit residual is ~100× the cubic residual** — the
  hyperelasticity is decisively detected and quantified, not assumed. Recovers
  strain-softening (`a3<0`) too. Closes the material-nonlinearity gap. See
  `docs/hyperelastic_experiments.md`; demo `scripts/exp_hyperelastic.py`.

### Pneumatic actuation (Jun 08, 2026)

- **`pneumatic_actuated_id` — the second soft-robot drive.** Pressurized chambers
  (PneuNets / fiber-reinforced): a chamber of area `A` at offset `(dy,dz)` under
  pressure `P` pushes with force `P·A` → `n_act=(+ΣPA,0,0)` (extension),
  `m_act=(ΣPA·h, ΣPA·dz, −ΣPA·dy)`. Opposite sign to a tendon — extends and bends
  *away* from the chamber (verified). Self-calibration from pressure sweeps:
  clean exact; noisy (pos 1e-3, quat 3e-3): **EA 0.2% / EI1 3.6% / EI2 2.6% /
  GJ 3.3%**. Both major drives (pull/push) now covered. See
  `docs/pneumatic_actuation_experiments.md`; demo `scripts/exp_pneumatic_actuation.py`.

### Tendon actuation — passive rod → soft robot (Jun 07, 2026)

- **`tendon_actuated_id` — first actuation capability.** Cables routed at
  cross-section offsets `(dy, dz)` (+ helix lever `h`) apply, under tension `τ`,
  a constant material wrench `n_act=(−Στ,0,0)`, `m_act=(Στh, −Στdz, Στdy)` →
  constant-strain (PCC) shape. Single tendon → `κ = τd/EI` (verified exactly).
- **Stiffness self-calibration.** The actuation wrench is *known* (commanded
  tensions × routing), so `wrench = C·strain` is linear in the stiffnesses:
  sweeping tension patterns and measuring the actuated shapes recovers
  `EA, EI1, EI2, GJ` with **no external test rig** — the robot calibrates itself
  by moving. Clean: exact; noisy (pos 1e-3, quat 3e-3): **EA 0.3% / EI1 2.3% /
  EI2 1.0% / GJ 3.2%**. Shear (GA) isn't tendon-excitable (use
  `spatial_cosserat_id`). Same recipe as the rest of the suite — known/derived
  wrench → linear regression for stiffness. See
  `docs/tendon_actuation_experiments.md`; demo `scripts/exp_tendon_actuation.py`.
  Open next for soft-robotics: pneumatic actuation, hyperelastic/viscoelastic
  materials, contact.

### Dynamic 3-D Cosserat rod — the rod suite completed (Jun 07, 2026)

- **`dynamic_spatial_cosserat_id` — the full dynamic 3-D rod** (`r(s,t)∈ℝ³`,
  `R(s,t)∈SO(3)` with inertia + 3-D rotational dynamics incl. the gyroscopic
  term), unifying the time-domain and spatial rods. Method-of-lines forward
  solver verified: undamped energy conserved; **reproduces the planar dynamic
  solver to ~1e-4**; in-plane isotropic load stays planar; off-axis + pre-twist
  gives genuine out-of-plane motion + dynamic torsion.
- **Recovers all six stiffnesses from 3-D motion** — in dynamics both the
  internal force *and* moment are kinematic (`n_sp` from accelerations, `m_sp`
  from the angular-momentum rate `dH/dt`), so the constitutive law is a linear
  regression. A **pre-twist** conditions torsion (GJ). Noisy motion (pos 1e-3,
  quat 3e-3, N=50): **EA 4.3% / GA1 2.3% / GA2 1.0% / EI1 1.3% / EI2 2.0% /
  GJ 3.3%**. See `docs/dynamic_spatial_cosserat_experiments.md`; demo
  `scripts/exp_dynamic_spatial_cosserat.py`.
- **The rod suite is now complete across geometry × time** (planar/3-D ×
  static/dynamic). The unifying recipe throughout: expose each stiffness against
  a statically-determinate or inertia-derived internal force/moment, turning the
  inverse into a well-conditioned linear regression rather than fighting an
  under-resolved high-order PINN residual. Open for full soft-robotics:
  actuation (tendon/pneumatic), hyperelastic/viscoelastic materials, contact.

### 3-D spatial Cosserat rod — full continuum-manipulator model (Jun 07, 2026)

- **`spatial_cosserat_id` — the full geometrically-exact 3-D rod** (Simo-Reissner),
  the model real continuum manipulators need: bending in two planes, two shears,
  extension, and **torsion** — six strains, six stiffnesses
  (EA, GA1, GA2, EI1, EI2, GJ). Centerline `r(s)∈ℝ³` + orientation `R(s)∈SO(3)`
  (quaternion). Forward BVP solver (`simulate_spatial_cosserat`) verified against
  closed-form limits: pure axial → stretch `1+P/EA`; pure twist → tip rotation
  `Mt₁/GJ`; transverse → planar elastica; quaternion norm holds to 1.0.
- **Recovers all six stiffnesses** (`recover_spatial_stiffness`) from measured
  shape + orientation: a tip-loaded cantilever is statically determinate, so the
  internal force/moment follow from the measured shape + known tip wrench
  (independent of the unknowns), making the constitutive law a linear regression.
  Clean: exact (0.0%); standard noise (pos 1e-3, quat 3e-3): **EA 0.2% / GA1 0.7%
  / GA2 0.5% / EI1 3.0% / EI2 0.5% / GJ 1.4%**. Recovers arbitrary non-unit
  stiffness too (e.g. EA=1.3, GJ=0.7 → exact). This is the 3-D generalisation of
  the force-from-motion identifier — the recurring lesson: expose a stiffness
  against a data-derived force/moment rather than an under-resolved PINN residual.
  See `docs/spatial_cosserat_experiments.md`; demo `scripts/exp_spatial_cosserat.py`.
  (Static 3-D; actuation / hyperelastic-viscoelastic materials / dynamic-3-D
  remain open for full soft-robotics coverage.)

### Dynamic Cosserat rod — first space-time multi-field inverse (Jun 07, 2026)

- **`dynamic_cosserat` template added (12th template)** — the time-domain
  (inertial) geometrically-exact Simo-Reissner rod, recovering EI/GA/EA from the
  **time-resolved motion** `x(s,t), y(s,t), θ(s,t)` of a soft rod swinging under
  gravity. The engine's first **nonlinear PDE inverse over a 2-D space-time
  domain**. Ships with a verified method-of-lines forward solver (undamped energy
  conserved to ~1e-7; damped steady state reproduces the static rod to ~1e-5).
- **EI (bending) recoverable (~7%); GA/EA (shear/axial) training-limited.** The
  full 3-unknown inverse is an honest frontier result — CRLB floors are tiny
  (EI 0.02% / GA 0.05% / EA 0.02%), so the gap is *training*, not
  identifiability. Two findings worth keeping:
  - **Auxiliary-force formulation gives the stiffnesses zero gradient.** Carrying
    `Nx, Ny` as free outputs makes the constitutive residual `Nx = EA·(…)`
    trivially satisfiable by `Nx` tracking `EA`. The fix — expand `∂Nx/∂s`
    directly into 2nd derivatives of `x,y,θ` (programmatic chain rule) so the
    stiffnesses sit in the data-anchored momentum residuals — is what makes EI
    converge.
  - **GA/EA signal lives in under-resolved translational 2nd derivatives**, so
    the net explains the residual away within the sensor/noise latitude (dynamic
    analogue of the static explain-away). Load choice is critical: a tip point
    load shock-excites fast waves (accel ~180); distributed gravity (accel ~8) is
    what makes it tractable. See `docs/dynamic_cosserat_experiments.md`; driver
    `scripts/exp_dynamic_cosserat.py`.
- (**gap closed**) **`cosserat_force_id` baseline** — the dynamic shear/axial
  gap is closed by a direct physics-informed estimator: the internal force
  `N(s,t)` is *kinematic* (the tip-integral of inertia), so it's derivable from
  the measured motion alone; the constitutive law is then linear in EI/GA/EA and
  recovered by least squares with only first spatial derivatives — **EI 2% / GA
  0.9% / EA 5.5%** on noisy motion (vs the PINN's ~250% on GA/EA). The lesson:
  when a parameter only enters via a high-order derivative the net
  under-resolves, integrate the balance law to expose it against a data-derived
  quantity rather than fighting the derivative.
  (`pinn_engine/baselines/cosserat_force_id.py`, `scripts/exp_cosserat_force_id.py`.)

### Full planar Cosserat rod — first multi-output multi-unknown PDE (Jun 07, 2026)

- **`planar_cosserat` template added (11th template)** — the geometrically-exact
  Simo-Reissner planar rod **with shear and extension**, recovering **three**
  stiffnesses at once: bending `EI_unit`, shear `GA_unit`, axial `EA_unit`
  (truth 1.0 each). The engine's **first multi-output (x, y, θ) multi-unknown
  *PDE* inverse** — the structural-mechanics analogue of `coupled_drag_3d`. For
  a tip-loaded cantilever (constant internal force) the balance laws collapse to
  three residuals that each isolate one unknown: axial + shear constitutive laws
  (read off the measured shape) and moment balance. Residuals verified to vanish
  at 1e-15 on an independent `solve_bvp` solution.
- **Solved to the CRLB floor: mean 0.30% rel_err** (EI 0.69% / GA 0.20% /
  EA 0.00%) with a fixed `param_lr_scale=100`, 8000 epochs, ncol=512, ~10 min
  CPU. CRLB floors EI 0.29% / GA 0.53% / EA 0.11%.
- **"Explain-away" finding.** An initial stiff-axial design (`EA0=40`, ~5% axial
  strain) left EA stuck at ~380% in every run — the network neutralised the tiny
  axial residual by nudging `x'` inside the position-noise band, so EA was never
  constrained during training *despite being CRLB-identifiable*. Fix was to
  enlarge the **signal**, not push the optimiser: soften the rod (`EA0=15`, axial
  strain ~17-31%) + denser/cleaner sensors → EA 0.20%. Lesson: when an unknown
  stalls, check its residual signal vs the data-noise floor before blaming LR.
- **Adaptive controller is non-monotonic on coupled multi-unknown problems.**
  Same controller config gave mean 2.06% at 8000 ep but **72.9%** at 12000 ep
  (fixed seed) — after EA converged the controller wandered it back out. The LR
  state machine, tuned on single-unknown templates, doesn't yet handle three
  interacting unknowns; fixed `param_lr_scale` is stable and preferred here.
  Open R&D. Raising `lam_physics` also hurt (trades shape-fit for residual).
  See `docs/cosserat_planar_experiments.md`; driver
  `scripts/exp_planar_cosserat.py`.

### Soft-robotics / large-deflection rod (Jun 07, 2026)

- **`planar_elastica` template added (10th template)** — the
  geometrically-exact, large-deflection soft-robot continuum rod the
  `cosserat_rod` docstring promised ("include curvature strains for the
  full soft-robotics formulation"). Static planar elastica cantilever in
  tangent-angle form: `EI·θ''(s) = −P₀·cos(θ)`, `θ(0)=0`, `θ'(L)=0`. First
  template with a **trig nonlinearity in a *spatial* (BVP) residual** — the
  `cos(θ)` is the geometric nonlinearity, distinct from the linear
  `euler_bernoulli_beam`. At load parameter `α = P₀L²/EI = 2.5` the tip
  rotates 51° and droops 0.56·L — squarely large-deflection (linear beam
  theory errs by tens of percent). Ground truth via `scipy.solve_bvp`;
  sensors measure `θ(s̃)` directly (the fiber-Bragg / IMU-array model).
- **Converges near the CRLB floor out of the box.** CRLB SE 0.46% (31 angle
  sensors, 1e-2 rad noise); engine hits **0.565%** with RAR / 0.648%
  baseline / 0.866% adaptive, all ~70 s on CPU. Plain tanh MLP,
  `param_lr_scale=20`, **no Fourier features / no two-phase LR / no causal
  weighting** — unlike the `cosserat_rod` wave equation, the static elastica
  is convex in `EI_unit` given the measured shape (`EI_unit = −α·cosθ/θ''`
  is a direct read-off), so there's no spurious basin. RAR composes with the
  nonlinear residual and gives the same kind of modest edge it gives on
  diffusion. More Adam budget (8000 ep) and L-BFGS both leave the converged
  minimum unchanged — the data-fit, not the optimizer budget, sets the floor.
- **Adaptive-controller trap reproduced + explained.** Enabling the
  controller while resetting `param_lr_scale` to 1.0 caps the unknown's LR at
  4e-3 (`max_mult=4`), too weak to leave the midpoint init → stalls at 4.75
  (**375% err**). Keeping the template's `param_lr_scale=20` as the
  controller base fixes it (0.866%). Concrete evidence for the standing "start
  the controller from the template's own `param_lr_scale`" guidance. See
  `docs/soft_robotics_elastica_experiments.md`; driver
  `scripts/exp_planar_elastica.py`.

### RAR + L-BFGS R&D (Jun 04, 2026)

- (**c5c5ecb** 2026-06-04) **RAR (residual-based adaptive collocation
  refinement)** added (Wu et al. 2022, CMAME 403, 115671). Callback
  draws a large uniform candidate pool from the physics domain every
  N epochs, evaluates `|residual|` via the solver's `compute_residual`,
  keeps the top-K candidates, mixes with a fraction of the existing
  points, and rewrites the dataset in-place via PinaTensorDataset's
  `update_data`. New `rar_*` TrainConfig knobs; all default off.
  **Empirical win on diffusion_1d (smooth-solution problem!): 0.64%
  → 0.14% rel_err (4.5×), only +2% wallclock overhead** at 2000-epoch
  budget. RAR composes with the controller, CausalPINN, L2 prior, and
  L-BFGS finetune.
- (**ed369cd** 2026-06-04) Controller `min_epochs_before_converged`
  knob added (default off). Sweep showed forcing the controller to
  stay in DESCEND/PROBE longer *hurts* on well-conditioned templates
  — coupled_drag_3d c_x drifts 1.8% → 11.2% as min_epochs goes
  0 → 2000. The earlier "premature convergence at ep14" was actually
  the controller correctly stepping aside; its continued intervention
  is what degrades the result. Knob kept available, default disabled.
- L-BFGS post-Adam **NULL on c_y plateau** (coupled_drag_3d): c_y stays
  at 21.86% → 21.82% with LBFGS-50. c_x tightens (0.40% → 0.11%) but
  the c_y plateau is structural — partial identifiability *in practice*
  at this data/network setup, not an optimizer issue. The L2 prior on
  c_y remains the right tool for this; second-order optimization can't
  rescue an information-limited unknown.
- L-BFGS post-Adam **NULL on euler_bernoulli_beam**: at the template's
  default 1500-epoch Adam budget, EI_unit is at 3.5 (vs truth 1.0,
  250% rel_err). LBFGS-50 leaves EI_unit at 3.5 and actually *raises*
  train loss (1.9 → 3336, line-search divergence). The fix for beam is
  *more Adam epochs* (4th-order PDE needs a longer Adam phase), not a
  second-order chaser. L-BFGS only refines an already-converged Adam
  minimum; it can't compensate for under-budgeted first-order training.

### Construction-templates era (Jun 02, 2026)

- (**257060a** 2026-06-02) **Two new construction-engineering templates**
  added — `euler_bernoulli_beam` (first 4th-order spatial PDE in the
  engine; rel_err currently training-limited) and `axial_elastic_bar`
  (2nd-order static elasticity; converges to 0.26% rel_err in 24 s, near
  the CRLB floor 0.03%). Both non-dimensionalised so loss landscape is
  well-conditioned.
- Attempted `burgers_1d` (nonlinear PDE) but deferred — Cole-Hopf at the
  canonical ν = 0.01/π has float64 precision issues, and finite-difference
  forward solvers blow up at shock formation. Future session needed.

### CRLB-driven R&D (Jun 02, 2026)

- (**e6a65d0** 2026-06-02) Controller drift-guard: before latching CONVERGED
  on a failed probe, check if any unknown has accumulated `drift_floor`
  drift over the last `convergence_window` epochs. If so, return to
  DESCEND. Scaffolded; thresholds need empirical tuning.
- (**fbe4e23** 2026-06-02) **CRLB preflight diagnostic** added
  (`pinn_engine.diagnostics.crlb.compute_template_crlb`). Reveals 25× CRLB
  headroom on diffusion, 36× on 3-DOF coupled-drag c_y, 70× on Cosserat — the
  engine's empirical results are far from data-theoretic limits on
  several templates.
- (**b568eca** 2026-06-02) `docs/ENGINE.md` (this file) added — single-file
  engineering reference.

### Adaptive control + regularization era (May 28 – Jun 02, 2026)

- (**f8a26be** 2026-06-02) L2 prior bug fix: partial anchor now only
  regularizes listed unknowns; previously auto-filled midpoints could
  silently pull other unknowns wrong.
- (**ca06504** 2026-06-01) **`coupled_drag_3d` template added** (7th template):
  coupled three-axis planar drag inverse, first multi-unknown coupled ODE.
- (**4eefc14** 2026-06-01) `iterative_train` meta-loop: shrinks bounds
  around each result and re-trains. Pendulum 19% → 0.43% in two iterations.
- (**0da13a7** 2026-06-01) **L2 (Tikhonov) prior on unknowns** added. the coupled-drag
  c_lin 13% → 0.78% with truth anchor.
- (**2e2b695** 2026-05-31) Adaptive controller validated across all 6
  templates; "use template's own `param_lr_scale`" guidance written.
- (**82999e2** 2026-05-31) **Cosserat basin auto-tune cracked**: adaptive
  controller lands E_unit at 0.92 (rel_err 8%), cap-mediated.
- (`7119ad7` 2026-05-31) Controller fix: data-loss read uses both
  `callback_metrics` + `logged_metrics`; `max_mult` capped at 4 to prevent
  exponential commit runaway; `escape_eps` raised 2% → 10%.
- (`314fef0` 2026-05-30) Data-loss brake added (catches monotonic overshoot
  the velocity-based brakes miss).
- (**d53877f** 2026-05-30) **Controller redesigned** as DESCEND/PROBE/CONVERGED
  state machine with bounded reversible probes.
- (`08c6886` 2026-05-30) Loss-worse brake threshold raised to 50% so noisy
  causal loss doesn't spuriously brake during healthy descent.
- (`63487ff` / `962cfdf` / **c5eacbf** 2026-05-29) `AdaptiveUnknownsController`
  added: runtime LR controller, validated on diffusion (1.6%) and ODEs (<0.5%).
- (**9ff93d8** 2026-05-29) **`diffusion_1d` template implemented** (was stubbed);
  converges to 1.6% with plain config — recipe generalizes to parabolic PDEs.

### Cosserat / two-phase recipe era (May 24 – May 28, 2026)

- (**9a826ff** 2026-05-28) **Cosserat solved (4.5% rel_err)** via two-phase
  LR recipe (run #16). Warmup + silent-pre-trigger + trigger E<2.0 + 5-epoch
  cosine brake.
- (`5aea820` 2026-05-28) Keep PINA's warmup; silent pre-trigger; add
  `taper_epochs` knob.
- (**66fdd2a** 2026-05-28) **LR-capture bug fixed**: scheduler was reading
  the warmup-discounted LR (×⅓) as base, secretly training the unknown 3×
  too slow.
- (`6122b7f` 2026-05-27) Cosserat causal: causal_eps=100 collapse bug fixed,
  `UnknownsDumper`, experiments doc.
- (`41c32ea` / `04343f7` 2026-05-26) PDE convergence work: cosine LR anneal,
  separate `param_lr_scale` for unknowns.
- (`542c8d6` / `e254db3` / `632be29` / `0ae9ee4` / `8ed36be` 2026-05-25-26)
  Cosserat hardening: non-dimensionalise, BC/IC as pseudo-sensors, Fourier
  features mandatory, "any problem" push.

### Phase 3 — PDE support (May 17 – May 23, 2026)

- (**0bdc980** 2026-05-23) **Phase 3+: PDE support** (space + time) and
  Cosserat rod template.
- (`e5f2e54` 2026-05-22) Diagnostic callbacks: PDE-aware input construction.
- (`321c318` / `0f80c63` 2026-05-21) **Phase 5: Streamlit dashboard**.
- (`f7eece8` / `b0d4e0d` / `c6c9271` / `36efa9f` 2026-05-20) 12-trial
  AutoML, EKF baseline, ensemble UQ + pendulum + ROS 2 bag ingestion, ONNX/
  TorchScript export.
- (`31a703e` 2026-05-18) Phase 3: 1-DOF nonlinear-drag surge inverse template.

### Phase 1+2 foundation (Apr 21 – May 17, 2026)

- (`78b7778` / `27e5cb7` / `7912e8a` / `be02a58` / `703422e` / `3db93ce` /
  `cea53cd` / `529cc67` 2026-05-12 – 17) MPS device sensitivity, L-BFGS
  inverse support, multi-seed AutoML, SA-PINN/LRA balancers wired,
  multi-seed AutoML, verify tolerance loosened, session report.
- (`b83b86f` / `84cd208` 2026-04-30 – 05-04) AutoML: custom pruning
  callback, monitor metric fix.
- (`cf89328` 2026-04-25) Multi-output data conditions fix for Lorenz.
- (**ff3a989** 2026-04-21) **Initial commit**: inverse PINN engine with
  AutoML (Phase 1+2).

---

## Architecture

```
pinn_engine/
├── dsl/                       Symbolic DSL — declare physics + sensors
│   ├── symbols.py             Variable / Parameter / Unknown / Sensor
│   ├── system.py              System builder + compile() → CompiledSystem
│   ├── compile.py             Lowers sympy → torch residual callables
│   ├── templates.py           Template registry
│   └── templates_lib/         The 7 bundled templates
│
├── core/                      Training engine
│   ├── trainer.py             TrainConfig + train(); subclassed PINA solvers
│   │                          (CausalLabeledDataPINN, LabeledDataPINN, LBFGSInversePINN)
│   ├── problem.py             build_problem() — lowers CompiledSystem +
│   │                          data → PINA InverseProblem; supports
│   │                          bounds_override / inits_override
│   ├── networks.py            MLP + Fourier-feature input encoding
│   ├── activations.py         Custom activations (sintanh)
│   ├── weightings.py          SAPinnWeighting, LRAWeighting balancers
│   ├── adaptive_controller.py AdaptiveUnknownsController state machine
│   ├── iterative_train.py     iterative_train() refinement meta-loop
│   ├── param_lr_scheduler.py  Manual two-phase LR scheduler (legacy/precision)
│   ├── causal_eps_scheduler.py  CausalPINN ε-annealer (Wang 2022)
│   └── unknowns_dumper.py     Per-epoch unknown-parameter JSON dump
│
├── data/
│   └── synthetic.py           Reference data generators per template
│
├── automl/                    Optuna search infra
│   ├── search.py              SearchStudy + objective wrapper
│   ├── pruning.py             TrainLossPruningCallback (Hyperband)
│   ├── space.py               Per-template search space construction
│   └── auto_space.py          "Any problem" auto-search-space heuristic
│
├── diagnostics/               Drop-in callbacks
│   ├── sensor_residuals.py
│   ├── spectral_bias.py
│   └── param_confidence.py
│
├── repro/                     Reproducibility manifests
│   ├── manifest.py
│   └── hashing.py
│
├── dashboard/                 Streamlit live dashboard (Phase 5)
│   ├── app.py
│   └── data.py
│
├── uq.py                      Deep-ensemble uncertainty
├── export.py                  ONNX + TorchScript export with round-trip verify
└── cli.py                     `pinn-engine train|search|verify|dashboard|...`
```

**Data flow (one training run):**

```
Template ──→ System ──compile()──→ CompiledSystem (residual fns, sensor obs fns,
                                                    unknown bounds + inits)
   │
   └─── synthetic_data() ─→ {sensor_name: (input_array, target_array)}
              │
              ▼
      build_problem(compiled, data, t_range, spatial_ranges,
                    bounds_override, inits_override)
              │
              ▼
      PINA InverseProblem instance (conditions: physics_0, data_<sensor>...)
              │
              ▼
      Solver (CausalLabeledDataPINN | LabeledDataPINN | LBFGSInversePINN)
              │
              ▼
      PINA Trainer + callbacks:
        - UnknownsDumper                  (per-epoch JSON)
        - AdaptiveUnknownsController      (LR control state machine)
        - CausalEpsAnnealer (if causal)   (Wang 2022 ε-anneal)
        - UnknownsParamLRScheduler (legacy two-phase)
              │
              ▼
      TrainResult { final_params, final_loss, problem, network, compiled, ... }
```

---

## Capabilities

What the engine handles **today** (2026-06-02):

**Problem classes:**

- **ODE inverse** with 1+ unknowns, optionally coupled (Lorenz 3 unknowns,
  3-DOF coupled-drag 3 coupled drag coefficients).
- **PDE inverse** with 1 spatial + 1 temporal variable; data conditions,
  physics conditions, IC and BC as pseudo-sensors.
- **Partial-identifiability problems** (data doesn't uniquely pin all
  unknowns) — addressed via the L2 prior with explicit anchor.

**Inverse-problem solving tools:**

- `AdaptiveUnknownsController` — auto-tune the unknowns' LR at runtime
  (no per-problem hand tuning of `param_lr_scale`).
- L2 prior (`unknown_l2_prior` + `unknown_l2_anchor`) for ill-posed cases.
- `iterative_train()` — meta-loop that shrinks bounds and re-trains for
  precision sharpening.
- Manual two-phase LR (`param_lr_trigger_below` + `param_lr_taper_epochs`)
  — precision tuning when you know the basin shape.
- CausalPINN solver (`solver_type="causal"`) + bi-directional ε-anneal
  for wave-equation / chaotic inverse problems.
- Loss balancers: `"none"`, `"sapinn"` (self-adaptive PINN), `"lra"`
  (learning-rate annealing).
- L-BFGS post-Adam refinement via `LBFGSInversePINN` (merges param groups
  so torch.LBFGS accepts the inverse-problem set-up).

**Architecture features:**

- MLP with Fourier feature input encoding (critical for PDE high-frequency
  content).
- Custom activations including `sintanh` (oscillation-friendly).
- LayerNorm (default on).

**AutoML:**

- Optuna study with Hyperband pruning (`TrainLossPruningCallback` — reports
  train-loss every 100 epochs since PINN inverse has no validation set).
- Per-template `automl_space()` constructors; `auto_space.py` heuristic for
  templates that don't have one yet.
- Multi-seed evaluation (variance + median).

**Diagnostics + reproducibility:**

- Per-epoch unknown-parameter JSON dump (survives OOM/SIGKILL).
- **CRLB preflight** (`pinn_engine.diagnostics.crlb.compute_template_crlb`):
  Cramér-Rao lower bound on each unknown's standard error, computed by
  forward simulation only (no training). Tells you the best possible
  uncertainty the data + sensor noise can support, *before* spending hours
  training. Separates data-limited problems (engine at CRLB floor → no
  improvement possible without better sensors/data) from training-limited
  problems (large gap → room to improve via more epochs / tuning /
  regularization).
- Sensor-residual diagnostic callback.
- Spectral-bias diagnostic (frequency-domain network output analysis).
- Parameter-confidence diagnostic (per-epoch unknown tracking).
- Deep-ensemble UQ (`pinn_engine.uq`).
- Reproducibility manifest per run (hashes everything: equation canonical
  form, data, config, network init).
- ONNX + TorchScript export with round-trip verification (tolerance 5%
  default — non-deterministic ops on CPU/MPS make 0.1% unrealistic).

**Tooling:**

- `pinn-engine train|search|verify|dashboard|...` CLI (Typer).
- Streamlit dashboard over manifests + Optuna studies (live training
  progress + equation editor).
- ROS 2 bag ingestion for real vehicle data.

**Outputs:**

- Per-unknown point estimate + final loss.
- Per-epoch unknown trajectory (JSON).
- Lightning CSV log per run.
- (Optional) ensemble UQ posterior over unknowns.
- (Optional) ONNX/TorchScript model export.

---

## The training pipeline

Step-by-step, what happens inside `train(system, data, config, callbacks=…)`:

1. **Seed everything** (`pl.seed_everything(config.seed)`).
2. **Compile the system** (`system.compile()`):
   - Validates: state has `depends_on`, sensors observe declared states,
     equations only reference declared symbols, bounds are valid.
   - Lowers each sympy `equation` to a torch callable that takes
     `(samples, network, unknown_dict, params)` and returns the residual.
   - Returns a `CompiledSystem` with `state_names`, `input_names`,
     `unknown_names`, `unknown_bounds`, `unknown_inits`, `residuals`,
     `obs_fns`, `eq_hash`.
3. **Build the PINA problem** (`build_problem()`):
   - Constructs the `CartesianDomain`s: temporal + (PDE only) spatial.
   - Builds `unknown_parameter_domain` from `compiled.unknown_bounds`
     (with optional override from `config.unknown_bounds_override`).
   - Creates one `DomainEquationCondition` per physics residual (named
     `physics_0`, `physics_1`, …) and one `TensorInputTensorTargetCondition`
     per data sensor (named `data_<sensor>`).
   - Instantiates a class derived from `(SpatialProblem)? + TimeDependentProblem +
     InverseProblem`.
   - Overrides the PINA-default unknown initialization (which is U(0, hi-lo)
     + lo — wrong distribution unless lo == 0) with `compiled.unknown_inits`
     (or `config.unknown_inits_override`).
4. **Discretise physics domain**: `problem.discretise_domain(n=n_collocation,
   mode="random", domains="all")` — random uniform samples.
5. **Build the network**: MLP of `(input_dim, ..., output_dim)` with the
   chosen activation, layer-norm, and Fourier-feature input encoding
   (`fourier_features` cosine/sine projections, `fourier_sigma` width).
6. **Pre-flight check** (`check_wellposedness(problem, network, compiled)`):
   - Warns on overly-wide bounds (more than 20× truth implies the midpoint
     init is hopelessly far).
   - Sanity-checks the residual evaluates to a finite tensor.
   - Sanity-checks sensors have data in the dict.
   - (Skippable via `config.skip_preflight`.)
7. **Build the weighting**: `ScalarWeighting`, `SAPinnWeighting`, or
   `LRAWeighting` with per-condition initial weights set from
   `lam_data_init` / `lam_physics_init`.
8. **Instantiate the solver**:
   - `CausalLabeledDataPINN` if `solver_type="causal"` (passes `eps` —
     defaults to `1.0`, not PINA's collapse-prone `100`).
   - `LabeledDataPINN` otherwise.
   - `LBFGSInversePINN` if a post-Adam L-BFGS phase is requested
     (`lbfgs_iters > 0`); this subclass merges network + unknown param
     groups so `torch.optim.LBFGS` (which asserts a single param group)
     accepts the inverse problem.
9. **Engine-level attributes** stashed on the solver (PINA's `__init__`
   doesn't pass through extras):
   - `_engine_param_lr_scale` (separate LR multiplier for unknowns).
   - `_engine_unknown_l2_prior` (Tikhonov weight).
   - `_engine_unknown_l2_anchors` (per-unknown anchors).
   - `_compiled_system`, `_engine_data`, `_engine_weighting` (for diagnostics).
10. **Wire callbacks**:
    - `UnknownsDumper` (always).
    - `UnknownsParamLRScheduler` if (`param_lr_min_scale<1.0` or
      `param_lr_trigger_below` set) and not `adaptive_unknowns_lr`.
    - `AdaptiveUnknownsController` if `adaptive_unknowns_lr=True`.
    - `CausalEpsAnnealer` if `solver_type="causal"` and
      `causal_eps_anneal=True`.
    - User-supplied callbacks last.
11. **PINA Trainer.fit(solver)** runs Adam for `adam_epochs` epochs;
    optionally L-BFGS for `lbfgs_iters` after.
12. **Read final unknowns** from `problem.unknown_parameters`.
13. **Compose `TrainResult`** with point estimates, final loss, compiled
    system, network, problem, weighting, config, and the run id.

---

## The mathematics

### Inverse problem formulation

Given a system of PDEs/ODEs in state `u(x, t)` with unknown parameters `θ`:

```
F[u, ∇u, ∇²u, …; θ] = 0       (governing physics)
B[u]|∂Ω           = 0          (boundary conditions)
I[u]|t=0          = u₀         (initial conditions)
y_k = h_k(u(x_k, t_k)) + ε_k   (sensor k, noise ε_k)
```

Find `θ` (and a network approximation `u_NN ≈ u`) that minimizes the
composite PINN loss:

```
ℒ(θ, NN) = λ_phys · 𝔼_(x,t)∈Ω [ ||F[u_NN; θ](x,t)||² ]              (physics)
         + Σ_k λ_data,k · 𝔼_data [ ||h_k(u_NN(x_k, t_k)) − y_k||² ]   (data)
         +       λ_L2  · Σ_θ (θ − θ_anchor)²                          (Tikhonov)
```

`u_NN` is a multi-layer perceptron with Fourier-feature input encoding;
`∇u_NN`, `∇²u_NN` come from `torch.autograd`.

### Loss decomposition (per PINA)

PINA represents each loss term as a `Condition`:

- `DomainEquationCondition` for physics (`physics_i`): the residual `F`
  is evaluated at `n_collocation` random points in the domain.
- `TensorInputTensorTargetCondition` for sensors (`data_<sensor>`):
  `||u_NN(x_data, t_data) − y_data||²` is computed exactly at the data
  points.

The weighting (`ScalarWeighting` / `SAPinnWeighting` / `LRAWeighting`)
aggregates condition losses to the scalar that backprops.

### Network: MLP + Fourier features

The network input `(x, t)` is first lifted by random Fourier features:

```
φ(x, t) = [sin(B · [x, t]), cos(B · [x, t])]
B ∈ ℝ^(F × d_input),  B_ij ~ 𝒩(0, σ²)
```

where `F = fourier_features` and `σ = fourier_sigma`. The lifted vector
`φ ∈ ℝ^(2F)` goes through `depth` linear layers of width `width` with a
chosen activation (`tanh`, `sintanh = (sin + tanh)/2`, `swish`, …) and
optional layer-norm.

**Why this matters for PDEs**: PINNs without Fourier features have severe
*spectral bias* (smooth network output → `u_ss ≈ 0` for the wave equation
→ ∂L/∂E ≈ 0 → unknown never updates). Mandatory for Cosserat; helpful
elsewhere.

### Optimizer + LR scheduling

- **Adam** for `adam_epochs` (default optimizer; per-parameter learning
  rates from gradient-magnitude moving averages).
- **PINA wraps the optimizer in `ConstantLR(factor=1/3, total_iters=5)`** —
  a 5-epoch warmup that runs the LR at ⅓ of base for epochs 0–4, then full
  base from epoch 5. **Load-bearing**: removing it (run #15) overshoots
  the wave-eq inverse to the lower bound in 3 epochs because the unknown
  moves too fast against a cold network.
- **Separate param-group LR for unknowns**: `lr_unknown = lr × param_lr_scale`.
  PINN inverse needs the unknowns to move faster than the network weights
  to escape Adam's per-parameter normalization throttling. The right
  `param_lr_scale` is problem-specific: Cosserat 500, the coupled-drag 1.0, default
  1.0. **Always use each template's own `default_config().param_lr_scale`**
  — forcing a universal value (e.g. 500 for the coupled-drag) blows up easy problems.
- **Optional cosine taper** (legacy two-phase precision option):
  ```
  scale(epoch) = min_scale + (1 − min_scale) · ½ · (1 + cos(π · progress))
  progress = (epoch − trigger_epoch) / taper_epochs   (or … / max_epochs if no trigger)
  ```
- **L-BFGS** optional refinement after Adam: second-order, line-search;
  uses the engine's `LBFGSInversePINN` (merges param groups).

### The adaptive controller — state machine + math

A runtime callback that adapts `lr_unknown = base_lr × eff_mult` based on
the observed *bounds-relative velocity* of each unknown and the loss
trajectory. Three states.

**Velocity signal**

```
v_i(t)        = θ_i(t) − θ_i(t−1)             (per-epoch velocity)
rel_v_i(t)    = |v_i(t)| / (b_hi_i − b_lo_i)  (relative to bound width)
max_rel_v(t)  = max_i rel_v_i(t)
osc(t)        = 1  iff  sign(v_i(t)) ≠ sign(v_i(t−1))  for some i with non-trivial step
```

**Loss signals**

```
loss(t)        = train_loss               (total)
data_loss(t)   = Σ_k data_<k>_loss        (per-condition data losses)
diverging(t)   = loss(t) > prev_loss · 1.5
data_worse(t)  = data_loss(t) > prev_data_loss · 1.1
improving(t)   = loss(t) < best_loss · 0.999
```

**State machine** (silent during `warmup_epochs` so PINA's warmup runs):

- **DESCEND** (default):
  - If `osc ∨ max_rel_v > v_hi ∨ diverging ∨ data_worse`:
    `base_mult ← base_mult · lr_down`   (brake; recoverable via probes)
  - Else if `max_rel_v < v_lo`:
    `stall ← stall + 1`. After `stall_patience` epochs → enter **PROBE**.
  - Else: hold.
  - Applied: `eff_mult = base_mult`.

- **PROBE**: temporarily boost `eff_mult = base_mult × probe_boost` for
  `probe_window` epochs. At window end, evaluate:
  - `dropped = loss < probe_loss₀ · (1 − escape_eps)`
  - `moved   = max_i |θ_i(t) − θ_i(probe_start)| / range_i > v_lo`
  - If `dropped ∧ moved` → commit: `base_mult ← base_mult · probe_boost`,
    back to DESCEND.
  - Else → **CONVERGED**: `base_mult ← converged_mult`, optionally stop.
  - If `diverging ∨ data_worse` mid-probe → abort: `base_mult ← base_mult ·
    lr_down`, back to DESCEND.

- **CONVERGED**: hold `eff_mult = converged_mult` (low LR, rest).

**Drift-guard against premature CONVERGED latching** (added 2026-06-02 after
the CRLB diagnostic revealed 3-DOF coupled-drag c_y has 36× headroom while the
controller was prematurely declaring convergence on it):

Before latching CONVERGED on a failed probe, check whether *any* unknown
has accumulated more than `drift_floor` of drift over the last
`convergence_window` epochs. If yes, go back to DESCEND — the slow
descent is real, the probe just doesn't add to it.

```
for name in unknowns:
    if |hist[name][-1] − hist[name][0]| / range[name] > drift_floor:
        → still drifting; revert to DESCEND, reset stall counter
    else:
        → not drifting; latch CONVERGED
```

Default `convergence_window = 20`, `drift_floor = 5e-3`. Per the
`docs/ENGINE.md` known-issues section, these defaults are conservative
and miss the ultra-slow drift seen on 3-DOF coupled-drag c_y (~0.001%/epoch);
tuning is open work.

**Cap**: `base_mult` clamped to `[min_mult, max_mult]` (defaults `0.02`,
`4.0`). The cap is what prevents the probe-commit loop from running away
exponentially on problems where the loss keeps dropping cosmetically (the
network polishing the field after the unknown has reached truth).

**The "moved AND dropped" gate is what distinguishes a *trap* from a *true
optimum*:** at a real optimum the unknown's gradient vanishes so it won't
move even under boosted LR (probe fails → CONVERGED); in a shallow basin
the gradient is nonzero so it moves (probe succeeds → commit and keep
going).

### L2 prior on unknowns (Tikhonov regularization)

Adds to the training loss:

```
ℒ_prior(θ) = λ · Σ_{i ∈ anchors} (θ_i − a_i)²
```

`λ = unknown_l2_prior` (default `0.0` = disabled). `a_i = unknown_l2_anchor[i]`
(only the unknowns explicitly named in the dict get a prior; if the dict
is `None`, the legacy "all unknowns anchored at bound midpoint" behavior
fires). The term is added inside our `training_step` override; logged as
`unknown_l2_prior`.

**Use it for partial-identifiability**: when the data doesn't uniquely
pin the unknowns, λ pulls the solution toward your prior belief `a`.
Validated:

- 1-DOF nonlinear-drag, λ=1, anchor=truth: c_lin 13% → **0.78%**.
- 3-DOF coupled-drag, λ=1, anchor=full truth: c_lin 0.00%, c_y 1.4%, c_n 0.01%.
- 3-DOF coupled-drag, λ=0.5, anchor={c_y: −30} only: c_y 22% → **1.37%**, c_lin and
  c_n unaffected by the prior (3.4% each from the data signal alone).

### Causal weighting (Wang 2022, arXiv:2203.07404)

For temporal problems, naive PINN minimizes residuals at all times jointly
— which lets the network learn an incoherent solution that satisfies
average physics but violates causality (later-time errors influence
earlier-time training). CausalPINN sorts collocation points by time into
buckets and weights each bucket by

```
ω_i = exp(−ε · Σ_{k ≤ i} L_phys(t_k))
loss_phys = Σ_i ω_i · L_phys(t_i)
```

so a bucket is only "active" (high ω) once earlier buckets are well-fit.
`ε` controls how strict the cascade is.

**The `eps=100` bug**: PINA's default `eps=100` collapses ω to ~0 by
epoch 1 for any non-trivial residual, silently muting the physics term.
The engine defaults `causal_eps=1.0` (or `1e-8` for the Cosserat wave
equation where initial residuals are O(1e7)) and uses a bi-directional
ε-annealer:

```
if max_bucket_loss < threshold:   ε ← min(ε · 10, ε_max)   # tighten
elif ω_min < ω_min_threshold:     ε ← ε / 10               # loosen
```

### Iterative bound-tightening

Wrapper: `iterative_train(system, data, base_config, n_iters, tighten_factor)`.
At each iteration:

```
θ_k        = train(system, data, cfg_k).final_params
range_k    = bounds_k.hi − bounds_k.lo
range_{k+1} = max(range_k · tighten_factor, min_range)
bounds_{k+1} = clip([θ_k − range_{k+1}/2,  θ_k + range_{k+1}/2],  outer_bounds)
init_{k+1}  = θ_k
```

Each round starts inside a narrower bracket re-centered on the previous
result, with that result as the initial value. Validated on pendulum:
iter 1 (`bounds=(0, 1)`) `c=0.357` (19.1%) → iter 2 (`bounds=(0.157, 0.557)`)
`c=0.301` (**0.43%**, 45× tighter) → iter 3 (`(0.221, 0.381)`) converged.

**Caveat**: does *not* help partial-identifiability. If the first run
deterministically lands at the wrong basin (e.g., the coupled-drag `c_lin=−8.4` vs
truth `−10`), tightening around `−8.4` just rediscovers the wrong basin
more reliably. For partial-id, use the L2 prior with an explicit anchor.

### Cramér-Rao lower bound (CRLB preflight)

For an inverse problem `y_k = h_k(u(x_k, t_k); θ) + ε_k` with
`ε_k ~ 𝒩(0, σ_k²)`, any unbiased estimator `θ̂` satisfies the Cramér-Rao
inequality:

```
Cov(θ̂)  ≥  F⁻¹
F        =  Sᵀ · diag(1/σ²) · S                 (Fisher information)
S[k, i]  =  ∂y_k / ∂θ_i                         (sensitivity matrix)
```

The diagonal of `F⁻¹` gives the best-achievable per-parameter variance;
`SE_i = √diag(F⁻¹)_i`. Importantly, **this bound applies to any estimator**
— PINN, EKF, Kalman smoother, hand-tuned recipe. If CRLB SE on `c_lin` is 5%,
no algorithm will do better with that data + noise.

**Implementation** (`pinn_engine/diagnostics/crlb.py`):

1. Forward-simulate at truth via the template's data generator → baseline
   observations `y_k(θ)`.
2. For each unknown `θ_i`, simulate at `θ ± δ_i` (central difference,
   `δ_i = perturb_rel · |θ_i|`). Hold the seed fixed so the noise term
   cancels in the finite difference.
3. Sensitivity per sensor: `S[k, i] = (y_k(θ + δ_i) − y_k(θ − δ_i)) / (2·δ_i)`.
4. Stack into a single `(n_obs, n_unknowns)` matrix with rows weighted by
   `1/σ_k` per sensor block.
5. `F = Sᵀ S`, then `Cov = pinv(F)` (pseudoinverse so ill-conditioned
   partial-id cases produce large-but-non-NaN SEs that surface the
   problem).
6. Return `SE_i`, `SE_i / |θ_i|`, and the `95% CI half-width = 1.96·SE_i`.

Cost: `2N+1` forward simulations for `N` unknowns. For ODEs that's seconds;
for the Cosserat wave-eq simulator (FD scheme) it's also under a second.

**Validation across all 7 templates** (2026-06-02) — empirical convergence
vs CRLB SE per unknown reveals where the engine is and isn't at the
data-theoretic limit:

| template | unknown | CRLB SE (rel) | best empirical | gap | reading |
|---|---|---|---|---|---|
| `damped_oscillator` | c, k | 0.15% / 0.02% | 0.11% / 0.01% | ~1× | at CRLB floor |
| `lorenz` | σ, ρ, β | <0.005% | <0.05% | ~1× | at CRLB floor |
| `pendulum` | c | 0.14% | 0.43% | 3× | small headroom |
| `diffusion_1d` | D | **0.063%** | 1.57% (50ep) / 0.24% (200ep) | 4-25× | **epoch-starved** |
| `nonlinear_drag_1d` | c_lin, c_quad | 5.5% / 4.6% | 13% / 15% | 2-3× | near floor (partial-id real) |
| `coupled_drag_3d` | c_lin, **c_y**, c_n | 0.19% / **0.62%** / 0.12% | 1.8% / 14% / 6.6% | up to **36×** | **c_y is *training*-limited, not partial-id** |
| `cosserat_rod` | E_unit | **0.11%** | 4.5% hand-tuned, 8% adaptive | **40-70×** | **huge training-limited headroom** |

The Cosserat finding is the most important — both the hand-tuned 4.5% and
the adaptive controller's 8% are far from the data-theoretic floor (0.11%).
PINN training is the bottleneck, not identifiability. That's a real future
R&D opportunity.

### Loss balancers

- **`"none"` (default)**: per-condition static weights from `lam_data_init`
  and `lam_physics_init`. Wrapped as PINA's `ScalarWeighting`.
- **`"sapinn"`** (Self-Adaptive PINN, McClenny–Braga 2020): learnable
  per-collocation-point weights `λ_i(t)` that the optimizer pushes UP
  where the residual is hard (adversarial balancing). Implemented as
  `SAPinnWeighting` plugging into PINA's `WeightingInterface`.
- **`"lra"`** (Learning Rate Annealing, Wang 2021): scales physics vs data
  losses by the ratio of their gradient norms, computed periodically.
  Implemented as `LRAWeighting`.

### Synthetic data generators

For each template, `data/synthetic.py` provides a forward simulator:

- **Damped oscillator**: `solve_ivp` on `m·ẍ + c·ẋ + k·x = 0`.
- **Lorenz**: `solve_ivp` on the 3-D chaotic system.
- **Pendulum**: `solve_ivp` on `I·θ̈ + c·θ̇ + mgL·sin(θ) = 0`.
- **1-DOF nonlinear-drag**: `solve_ivp` on `m·u̇ = τ_u + c_lin·u + c_quad·u²`.
- **3-DOF coupled-drag**: `solve_ivp` on the coupled (`u̇`, `v̇`, `ṙ`) system
  with Coriolis terms `m22·v·r`, `m11·u·r`, `(m22 − m11)·u·v`.
- **Cosserat rod (wave eq)**: explicit finite difference (central in space,
  leapfrog in time, CFL-bounded `dt`) on `ρ·u_tt = E·u_ss` with
  `u(0,t) = 0`, `u_s(L,t) = 0`, `u(s,0)` = Gaussian bump.
- **Diffusion 1-D**: closed-form Gaussian solution to `u_t = D·u_xx`
  spreading from a narrow IC.

All add Gaussian noise; all are reproducible from seed.

---

## Templates inventory

12 bundled inverse templates (`pinn_engine/dsl/templates_lib/`):

| name | physics | unknowns | best result via engine |
|---|---|---|---|
| `damped_oscillator` | `m·ẍ + c·ẋ + k·x = 0` | c, k | **c 0.11%, k 0.01%** (adaptive) |
| `lorenz` | Lorenz σ, ρ, β | σ, ρ, β | **all <0.05%** (adaptive) |
| `pendulum` | `I·θ̈ + c·θ̇ + mgL·sin(θ) = 0` | c | **0.43%** (adaptive + iterative) |
| `nonlinear_drag_1d` | `m·u̇ + drag` | c_lin, c_quad | 13% adaptive; **0.78%** + L2 prior (truth anchor) |
| `coupled_drag_3d` | planar three-axis planar + Coriolis | c_lin, c_y, c_n | adaptive 1.8/22/6.6%; **0/1.4/0%** + L2 prior (full truth) |
| `diffusion_1d` | `u_t = D·u_xx` | D | **1.6%** at 50 ep / **0.24%** at 200 ep / **0.10%** at 200 ep + L2 prior anchor=truth (= CRLB floor 0.063%) |
| `cosserat_rod` | `ρ·u_tt = E·u_ss` (wave) | E_unit | **4.5%** (hand-tuned two-phase, run #16); 8% (adaptive, cap-limited) |
| `axial_elastic_bar` | `EA·u'' + p₀ = 0` (static, clamped-free) | EA_unit | **0.26%** in 24 s on CPU (near CRLB 0.03%) |
| `euler_bernoulli_beam` | `EI·w'''' = q₀` (static, simply-supported) | EI_unit | training-limited (CRLB 0.10%; engine convergence is slow on the 4th-order autograd path) |
| `planar_elastica` | `EI·θ'' = −P₀·cos θ` (geometrically-exact large-deflection rod) | EI_unit | **0.565%** (RAR) / 0.648% (baseline) — near CRLB floor 0.46%, ~70 s on CPU |
| `planar_cosserat` | full Simo-Reissner rod (shear + extension), 3 residuals | EI_unit, GA_unit, EA_unit | **mean 0.30%** (EI 0.69% / GA 0.20% / EA 0.00%) at CRLB floor; fixed scale=100, ~10 min CPU |
| `dynamic_cosserat` | **dynamic** Simo-Reissner rod (inertia + time), 2-D space-time | EI_unit, GA_unit, EA_unit | PINN: EI ~7%, GA/EA stall; **force-from-motion solver: EI 2% / GA 0.9% / EA 5.5%** (noisy) — gap closed |

---

## Public API surface

### Core entry points

```python
from pinn_engine.dsl.templates import get_template
from pinn_engine.core.trainer import TrainConfig, train, TrainResult
from pinn_engine.core.iterative_train import iterative_train, IterativeResult
from pinn_engine.core.adaptive_controller import AdaptiveUnknownsController
from pinn_engine.core.unknowns_dumper import UnknownsDumper
from pinn_engine.diagnostics.crlb import compute_crlb, compute_template_crlb, CRLBResult
from pinn_engine.uq import train_ensemble, EnsembleResult   # Monte-Carlo seed-ensemble UQ
```

### Minimal usage

```python
tpl = get_template("diffusion_1d")
system = tpl.system()
data, truth = tpl.synthetic_data(seed=0)
cfg = tpl.default_config()
cfg.adaptive_unknowns_lr = True
result = train(system, data, cfg, callbacks=[AdaptiveUnknownsController()])
print(result.final_params)
```

### CRLB preflight (do this BEFORE training)

```python
from pinn_engine.diagnostics.crlb import compute_template_crlb
r = compute_template_crlb("diffusion_1d")
print(r.summary_table())
# Reports: per-unknown SE lower bound + relative + 95% CI half-width.
# If SE_relative on your unknown is, say, 30%, your sensors physically
# cannot identify it better than that — no PINN, EKF, or hand-tuning will.
```

### Iterative refinement

```python
res = iterative_train(system, data, cfg, n_iters=3, tighten_factor=0.4,
                      callbacks_factory=lambda: [AdaptiveUnknownsController()])
print(res.final_params)
```

### L2 prior (partial-id)

```python
cfg.unknown_l2_prior = 0.5
cfg.unknown_l2_anchor = {"c_y": -30.0}    # only c_y regularized
```

### Selected `TrainConfig` knobs

| field | default | what |
|---|---|---|
| `depth`, `width`, `activation` | 4, 64, `"tanh"` | MLP geometry |
| `fourier_features`, `fourier_sigma` | 0, 1.0 | Fourier-feature input encoding |
| `lr`, `param_lr_scale` | 1e-3, 1.0 | base LR + multiplier on the unknowns' param group |
| `adam_epochs`, `lbfgs_iters` | 2000, 0 | optimizer budgets |
| `balancer` | `"none"` | `"none"`, `"sapinn"`, `"lra"` |
| `lam_data_init`, `lam_physics_init` | 1.0, 1.0 | initial weights per condition class |
| `solver_type` | `"pinn"` | `"pinn"` or `"causal"` (Wang 2022) |
| `causal_eps`, `causal_eps_anneal` | 1.0, False | causal weighting + ε-annealer |
| `t_range`, `spatial_ranges`, `n_collocation` | (0,1), None, 1000 | physics domain + sampling |
| `param_lr_min_scale`, `param_lr_trigger_below`, `param_lr_taper_epochs` | 1.0, None, None | manual two-phase LR (precision option) |
| `adaptive_unknowns_lr` | False | runtime LR controller |
| `unknown_l2_prior`, `unknown_l2_anchor` | 0.0, None | Tikhonov |
| `unknown_bounds_override`, `unknown_inits_override` | None, None | iterative refinement hooks |
| `seed`, `deterministic` | 42, True | reproducibility |
| `accelerator`, `devices` | `"auto"`, 1 | Lightning device |
| `skip_preflight` | False | bypass identifiability check |

### CLI

```
pinn-engine train <template> [options]
pinn-engine search <template> --n-trials N --study NAME
pinn-engine verify <manifest>
pinn-engine dashboard
```

---

## Known limitations + open issues

1. **PINA data-condition logging at scale**: per-condition `data_<sensor>_loss`
   keys are present in short runs but were `None` in some long Cosserat runs;
   the data-loss brake therefore didn't fire in iter #5. Root cause not
   fully traced (probably callback-order / multi-batch CSV-schema interaction);
   the `max_mult=4` cap is the working safety until this is fixed.
2. **Cosserat is *training*-limited, not data-limited.** CRLB SE on E_unit is
   0.11%; both the adaptive controller (8%) and the hand-tuned two-phase
   recipe (4.5%) are 40-70× off. Closing this is real R&D — candidates:
   working data-loss brake, RAR adaptive collocation, much longer training
   budget at the template default (10000 ep vs the 50 we ran). See
   `docs/cosserat_causal_experiments.md` for the historical R&D arc.
3. **Bounds-clamp timing**: PINA clamps unknowns in `loss_data` (which runs
   *after* `optimizer.step`), so an unknown can transiently leave its
   bounds for one epoch before being clamped. Diffusion was observed at
   D=−0.002 once. Cosmetic; recovery is automatic.
4. **Wide-bound midpoint anchor**: the L2 prior with the default `None`
   anchor (= bound midpoints) can pull unknowns *away* from truth when
   the bounds are wide and asymmetric (1-DOF nonlinear-drag's midpoint is farther
   from truth than the baseline). Honest fix: pass an explicit anchor; the
   feature shines with a real prior.
5. **6-DOF rigid-body inverse not yet a template.** `coupled_drag_3d` is the
   largest coupled-multi-unknown ODE in the engine; full 6-DOF rigid body
   (with added-mass coupling, multi-rate sensors, body↔world-frame
   rotations) would be the natural next step.
6. **Experiment scripts have been epoch-starved.** Several R&D runs set
   `adam_epochs=50` for fast iteration; this is far below the template's
   intended production budget (e.g. `diffusion_1d` defaults to 10000,
   `cosserat_rod` to 10000). Diffusion at 200 ep is **7× tighter** than at
   50 ep (1.71% → 0.24%); 3-DOF coupled-drag c_y at 8000 ep is 1.6× tighter
   than at 2000 ep. The empirical baselines in this doc reflect a mix of
   debug and production budgets — check `cfg.adam_epochs` before drawing
   "the engine can't do better" conclusions.
7. **Adaptive controller's CONVERGED latch fires too eagerly on slow
   coupled descents.** Drift-guard scaffolded (commit `e6a65d0`) but its
   `convergence_window=20` / `drift_floor=5e-3` defaults are conservative
   — 3-DOF coupled-drag c_y's ~0.001%/epoch drift is below the floor. Tuning
   open.
