"""``pinn-engine`` Typer CLI.

Subcommands:
* ``train <template>`` — single training run, writes a manifest.
* ``search <template>`` — Optuna AutoML over the template's search space.
* ``inspect <manifest.json>`` — pretty-print a manifest.
* ``verify <manifest.json>`` — re-run with the same config + seed and assert the
  discovered parameters match.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from pinn_engine.dsl.templates import get_template

app = typer.Typer(add_completion=False, no_args_is_help=True, help="pinn-engine CLI")
console = Console()


@app.command()
def train(
    template: str = typer.Argument(..., help="Template name (e.g. damped_oscillator)"),
    seed: int = typer.Option(42, help="Random seed."),
    adam_epochs: Optional[int] = typer.Option(None, help="Override adam_epochs."),
    lbfgs_iters: Optional[int] = typer.Option(None, help="Override lbfgs_iters."),
    manifest: Optional[Path] = typer.Option(None, help="Write manifest to this file."),
    skip_preflight: bool = typer.Option(False, help="Skip the well-posedness check."),
    export_onnx: Optional[Path] = typer.Option(None, help="After training, export the network to this .onnx path."),
    export_torchscript: Optional[Path] = typer.Option(None, help="After training, export the network to this .pt path (TorchScript)."),
) -> None:
    """Run a single training trial for a template's synthetic data."""
    tpl = get_template(template)
    system = tpl.system()
    data, truth = tpl.synthetic_data(seed=seed)
    config = tpl.default_config().model_copy(update={"seed": seed})
    if adam_epochs is not None:
        config = config.model_copy(update={"adam_epochs": adam_epochs})
    if lbfgs_iters is not None:
        config = config.model_copy(update={"lbfgs_iters": lbfgs_iters})
    if skip_preflight:
        config = config.model_copy(update={"skip_preflight": True})

    from pinn_engine.core.trainer import train as _train
    from pinn_engine.diagnostics import default_bundle
    from pinn_engine.repro.manifest import write_manifest

    result = _train(system=system, data=data, config=config, callbacks=default_bundle())

    out = write_manifest(template=template, result=result, data=data)
    if manifest is not None:
        Path(manifest).write_bytes(Path(out).read_bytes())
        out = manifest

    _print_result(result, truth, out)

    if export_onnx is not None:
        from pinn_engine.export import to_onnx
        path = to_onnx(result, export_onnx)
        rprint(f"[bold]Exported ONNX:[/bold] {path} (sidecar: {path.with_suffix('.json')})")
    if export_torchscript is not None:
        from pinn_engine.export import to_torchscript
        path = to_torchscript(result, export_torchscript)
        rprint(f"[bold]Exported TorchScript:[/bold] {path} (sidecar: {path.with_suffix('.json')})")


@app.command()
def search(
    template: str = typer.Argument(..., help="Template name."),
    n_trials: int = typer.Option(20, help="Number of Optuna trials."),
    study: str = typer.Option("demo", help="Study name (becomes sqlite filename)."),
    seed: int = typer.Option(42, help="Sampler seed."),
) -> None:
    """Run AutoML search over a template's hyperparameter space."""
    from pinn_engine.automl.search import run_search

    rprint(f"[bold]Running AutoML search:[/bold] template={template} n_trials={n_trials}")
    s = run_search(template_name=template, n_trials=n_trials, study_name=study, seed=seed)
    rprint(f"\n[bold green]Best trial:[/bold green] #{s.best_trial.number}  value={s.best_value:.4g}")
    rprint(f"Best params: {s.best_trial.params}")
    rprint(
        f"\nOpen the dashboard:\n  "
        f"[cyan]optuna-dashboard sqlite:///manifests/optuna_{study}.db[/cyan]"
    )


@app.command()
def inspect(manifest: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    """Pretty-print a manifest."""
    data = json.loads(Path(manifest).read_text())
    t = Table(title=f"Manifest {data.get('run_id')}", show_header=True)
    t.add_column("Field")
    t.add_column("Value")
    for key in (
        "timestamp", "git_sha", "git_dirty", "template", "template_hash",
        "data_hash", "config_hash", "seed",
        "torch_version", "python_version", "pina_version", "lightning_version",
        "automl_study", "trial_number", "final_loss",
    ):
        v = data.get(key)
        if isinstance(v, str) and len(v) > 60:
            v = v[:8] + "…" + v[-8:]
        t.add_row(key, str(v))
    console.print(t)

    fp = data.get("final_params", {})
    if fp:
        t2 = Table(title="Discovered parameters", show_header=True)
        t2.add_column("Name")
        t2.add_column("Final")
        t2.add_column("Tail mean")
        t2.add_column("Tail std")
        for name, vals in fp.items():
            t2.add_row(name, f"{vals['final']:.4g}", f"{vals['mean']:.4g}", f"{vals['std']:.2g}")
        console.print(t2)


@app.command()
def verify(
    manifest: Path = typer.Argument(..., exists=True, readable=True),
    tol: float = typer.Option(5e-2, help="Per-parameter relative tolerance. Default 5%% — tighter is unrealistic with non-deterministic ops on MPS/CPU."),
    adam_epochs: Optional[int] = typer.Option(None, help="Override the template's default adam_epochs for the re-run."),
) -> None:
    """Re-run the trial described by ``manifest`` and assert the discovered params match."""
    data = json.loads(Path(manifest).read_text())
    template = data["template"]
    seed = data["seed"]
    truth_params = {k: v["final"] for k, v in data["final_params"].items()}

    rprint(f"[bold]Verifying:[/bold] template={template} seed={seed}")
    tpl = get_template(template)
    system = tpl.system()
    syn_data, _ = tpl.synthetic_data(seed=seed)
    update = {"seed": seed, "accelerator": "cpu", "deterministic": True}
    if adam_epochs is not None:
        update["adam_epochs"] = adam_epochs
    config = tpl.default_config().model_copy(update=update)

    from pinn_engine.core.trainer import train as _train

    result = _train(system=system, data=syn_data, config=config)

    ok = True
    for name, prev in truth_params.items():
        new = result.final_params.get(name, float("nan"))
        rel = abs(new - prev) / max(abs(prev), 1e-6)
        good = rel < tol
        ok &= good
        rprint(f"  {name}: prev={prev:.6g}  new={new:.6g}  rel_err={rel:.2e}  {'✓' if good else '✗'}")
    if ok:
        rprint("[bold green]VERIFIED[/bold green]: all parameters match within tolerance.")
    else:
        rprint(f"[bold red]MISMATCH[/bold red]: at least one parameter exceeded tol={tol:.0e}")
        raise typer.Exit(code=1)


def _print_result(result, truth, manifest_path):
    t = Table(title="Training Result", show_header=True)
    t.add_column("Parameter"); t.add_column("Discovered"); t.add_column("Truth"); t.add_column("Rel err")
    for name, val in result.final_params.items():
        true = truth.get(name)
        rel = abs(val - true) / max(abs(true), 1e-6) if true is not None else float("nan")
        t.add_row(name, f"{val:.5g}", f"{true:.5g}" if true is not None else "—", f"{rel:.2e}" if true is not None else "—")
    console.print(t)
    rprint(f"[dim]Manifest: {manifest_path}[/dim]")


if __name__ == "__main__":
    app()
