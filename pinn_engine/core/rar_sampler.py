"""Residual-based adaptive refinement (RAR) for collocation points.

Standard PINN training samples collocation points uniformly from the
physics domain and never moves them. When the solution has localized
features — shocks, boundary layers, sharp interfaces — the uniform
sample wastes capacity on smooth regions and under-samples the hard
ones. The residual is exactly the signal of where the network is
struggling, so resampling biased toward high-residual regions
concentrates compute where it matters.

This callback implements the RAR variant from Wu et al. 2022 ("A
comprehensive study of non-adaptive and residual-based adaptive
sampling for physics-informed neural networks", *Comput. Methods Appl.
Mech. Engrg.* 403, 115671): every ``refresh_every_epochs``, draw a
large candidate pool, evaluate the equation residual at each candidate,
keep the top-K by |residual|, mix with a fraction of the existing
points (avoid catastrophic forgetting of well-fit regions), and write
the new set back to the dataset.

This is purely a sampling change — the loss, network, and optimizer are
untouched. It composes with the adaptive controller, CausalPINN, the
L2 prior, and L-BFGS finetune.
"""
from __future__ import annotations

import torch
import lightning.pytorch as pl
from pina import LabelTensor


class RARSampler(pl.Callback):
    """Periodically replace physics collocation points with high-residual ones.

    Args:
        refresh_every_epochs:
            Resample cadence. Defaults to 200. Set to a divisor of total
            epochs so the last refresh happens before convergence.
        candidate_pool:
            Number of uniform candidates to draw each refresh. Larger
            pool → better top-K coverage, more wallclock per refresh.
            10× the collocation count is a good rule of thumb.
        keep_old_fraction:
            Fraction of existing collocation points to retain (random
            subset). Prevents pure-greedy collapse onto a single hot
            spot; the rest of the budget goes to top-residual candidates.
            0.5 is the Wu 2022 default.
        warmup_epochs:
            Skip resampling for the first ``warmup_epochs``. Resampling
            before the network has a meaningful residual signal just
            adds noise.
    """

    def __init__(
        self,
        refresh_every_epochs: int = 200,
        candidate_pool: int = 10_000,
        keep_old_fraction: float = 0.5,
        warmup_epochs: int = 100,
    ):
        super().__init__()
        if not 0.0 <= keep_old_fraction < 1.0:
            raise ValueError(f"keep_old_fraction must be in [0, 1); got {keep_old_fraction}")
        if refresh_every_epochs <= 0:
            raise ValueError(f"refresh_every_epochs must be > 0; got {refresh_every_epochs}")
        self.refresh_every = int(refresh_every_epochs)
        self.candidate_pool = int(candidate_pool)
        self.keep_old_fraction = float(keep_old_fraction)
        self.warmup_epochs = int(warmup_epochs)
        # Diagnostics: how many resamples happened and the median residual
        # magnitude at each (so users can see RAR actually targeting something).
        self.refresh_history: list[dict] = []

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1  # +1 so first refresh is at epoch == refresh_every
        if epoch <= self.warmup_epochs:
            return
        if epoch % self.refresh_every != 0:
            return

        problem = pl_module.problem
        ds = trainer.datamodule.train_dataset
        if ds is None or not hasattr(ds, "conditions_dict"):
            return

        # We need gradients through inputs for autograd-based residuals.
        # Don't touch param grads on the solver — switch the model to eval is
        # wrong (some activations differ), so keep train mode but disable
        # the optimizer step for this forward.
        pl_module.model.eval()
        try:
            for cond_name, cond_obj in problem.conditions.items():
                if not cond_name.startswith("physics_"):
                    continue
                self._refresh_one(trainer, pl_module, ds, cond_name, cond_obj, epoch)
        finally:
            pl_module.model.train()

    # ------------------------------------------------------------------
    def _refresh_one(self, trainer, pl_module, ds, cond_name, cond_obj, epoch):
        if cond_name not in ds.conditions_dict:
            return
        current_input = ds.conditions_dict[cond_name]["input"]
        n_current = int(current_input.shape[0])
        # Cap n_new by candidate_pool so we never short-fill (which would
        # cause a length drift in the dataset). If the pool is too small,
        # increase keep_old_fraction implicitly for this refresh.
        max_new = min(self.candidate_pool, n_current)
        n_new = min(n_current - int(self.keep_old_fraction * n_current), max_new)
        n_keep = n_current - n_new

        # 1. Draw a large uniform candidate pool from the physics domain.
        # PINA stores condition.domain as the *string key* into
        # problem.domains, not the domain object itself.
        domain_key = cond_obj.domain
        domain = pl_module.problem.domains[domain_key]
        # PINA CartesianDomain.sample returns a LabelTensor.
        candidates = domain.sample(self.candidate_pool, "random").sort_labels()

        # 2. Compute |residual| on the candidates. Match the model's device
        # (PINA Trainer moves the model to MPS/CUDA/CPU per trainer.accelerator;
        # CartesianDomain.sample returns CPU tensors).
        device = next(pl_module.model.parameters()).device
        candidates_input = candidates.to(device).clone()
        candidates_input.requires_grad_(True)

        # Use the solver's compute_residual so labels, parameter dict, and
        # any solver-specific output wrapping match the training path.
        equation = cond_obj.equation
        try:
            residual = pl_module.compute_residual(samples=candidates_input,
                                                  equation=equation)
        except Exception:
            # Fallback: call the equation directly with the raw model output.
            output = pl_module.forward(candidates_input)
            params = getattr(pl_module.problem, "unknown_parameters", None) or None
            try:
                residual = equation.residual(candidates_input, output, params)
            except TypeError:
                residual = equation.residual(candidates_input, output)

        # residual may be a LabelTensor or plain tensor; squeeze last dim if it's 1.
        r = residual.detach()
        if r.ndim == 2 and r.shape[1] == 1:
            r = r.squeeze(-1)
        elif r.ndim == 2 and r.shape[1] > 1:
            # multi-component physics in one Equation — take L2 norm.
            r = torch.linalg.vector_norm(r, dim=1)
        mag = r.abs()

        # 3. Pick top-K candidates by |residual|.
        if n_new >= mag.numel():
            top_idx = torch.arange(mag.numel())
        else:
            top_idx = torch.topk(mag, k=n_new, largest=True).indices
        new_pts = candidates_input.detach()[top_idx]
        # restore labels
        new_pts = LabelTensor(new_pts, labels=list(candidates.labels))

        # 4. Randomly retain n_keep of the current collocation set. Put
        # everything on the dataset's device (= current_input.device) for cat;
        # the trainer will move to model device at batch time.
        ds_device = current_input.device
        if n_keep > 0:
            keep_idx = torch.randperm(n_current)[:n_keep]
            kept = current_input[keep_idx]
            combined_raw = torch.cat([
                kept.as_subclass(torch.Tensor).to(ds_device),
                new_pts.as_subclass(torch.Tensor).to(ds_device),
            ], dim=0)
            combined = LabelTensor(combined_raw, labels=list(current_input.labels))
        else:
            combined = LabelTensor(
                new_pts.as_subclass(torch.Tensor).to(ds_device),
                labels=list(current_input.labels),
            )

        # 5. Write back into the dataset. Sanity: keep length stable so the
        # PinaSampler / Collator don't choke on a changed conditions_length.
        assert combined.shape[0] == n_current, (
            f"RAR length drift: was {n_current}, now {combined.shape[0]}"
        )
        ds.conditions_dict[cond_name]["input"] = combined

        # 6. Bookkeeping: log median residual at refresh time (= what we picked).
        med_resid_picked = float(mag[top_idx].median().detach().cpu().item())
        med_resid_pool = float(mag.median().detach().cpu().item())
        self.refresh_history.append({
            "epoch": int(epoch),
            "cond": cond_name,
            "n_replaced": int(n_new),
            "median_residual_picked": med_resid_picked,
            "median_residual_pool": med_resid_pool,
        })
