"""Streamlit entry point for the pinn-engine dashboard.

Run via the CLI::

    pinn-engine dashboard

Or directly::

    streamlit run pinn_engine/dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Make the package importable when launched via `streamlit run` from anywhere.
_pkg_root = Path(__file__).resolve().parents[2]
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from pinn_engine.dashboard.data import (  # noqa: E402
    list_manifests,
    list_optuna_dbs,
    load_run,
    load_optuna_study,
    manifests_dir,
    RunData,
)


st.set_page_config(
    page_title="pinn-engine",
    page_icon="🔬",
    layout="wide",
)


# ============================================================ sidebar


def _sidebar_pick_manifests() -> List[Path]:
    st.sidebar.markdown("### Manifests")
    st.sidebar.caption(f"`{manifests_dir()}`")
    available = list_manifests()
    if not available:
        st.sidebar.warning("No manifests found.")
        return []
    labels = [f"{p.stem[:8]}… ({p.stat().st_mtime:.0f})" for p in available]
    selected_labels = st.sidebar.multiselect(
        "Pick one or more runs",
        labels,
        default=labels[:1],
    )
    label_to_path = dict(zip(labels, available))
    return [label_to_path[s] for s in selected_labels]


def _sidebar_pick_view() -> str:
    return st.sidebar.radio(
        "View",
        ["Overview", "Parameters", "Convergence", "Compare", "AutoML", "Equations", "Train"],
        index=0,
    )


# ============================================================ views


def _view_overview(runs: List[RunData]) -> None:
    st.title("pinn-engine")
    st.markdown("Phase-5 dashboard over `manifests/`")
    if not runs:
        st.info("Pick a manifest from the sidebar.")
        return
    run = runs[0]
    m = run.manifest
    cols = st.columns(3)
    cols[0].metric("template", m.get("template", "?"))
    cols[1].metric("final loss", f"{m.get('final_loss', float('nan')):.4g}")
    cols[2].metric("seed", str(m.get("seed", "?")))

    st.markdown("#### Manifest")
    st.dataframe(
        pd.DataFrame(
            [
                {"field": k, "value": str(v)[:120]}
                for k, v in m.items()
                if k not in {"final_params"}
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )


def _view_parameters(runs: List[RunData]) -> None:
    st.title("Discovered parameters")
    if not runs:
        st.info("Pick a manifest.")
        return

    run = runs[0]
    fp = run.final_params
    if not fp:
        st.warning("No discovered parameters recorded.")
        return

    rows = [
        {"name": n, "final": v["final"], "mean": v["mean"], "std": v["std"]}
        for n, v in fp.items()
    ]
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)

    # Bar chart of final values with ±σ error bars
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["name"],
            y=df["final"],
            error_y={"type": "data", "array": df["std"]},
            name="discovered ±σ",
        )
    )
    fig.update_layout(
        title="Discovered values (±σ from ensemble or training tail)",
        yaxis_title="value",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def _view_convergence(runs: List[RunData]) -> None:
    st.title("Convergence diagnostics")
    if not runs:
        st.info("Pick a manifest.")
        return

    run = runs[0]
    cb = run.callback_outputs

    # Parameter trajectory (ParamConfidence)
    pc = cb.get("param_confidence")
    if isinstance(pc, dict) and pc.get("history"):
        st.subheader("Parameter trajectory")
        epochs = pc.get("epochs", [])
        fig = go.Figure()
        for name, vals in pc["history"].items():
            fig.add_trace(go.Scatter(x=epochs, y=vals, name=name, mode="lines"))
        fig.update_layout(xaxis_title="epoch", yaxis_title="parameter value", height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No `param_confidence` data in this run.")

    # Residual heatmap
    rh = cb.get("residual_heatmap")
    if isinstance(rh, dict) and rh.get("snapshots"):
        st.subheader("Physics residual over training")
        snaps = rh["snapshots"]
        t_grid = np.array(snaps[0]["t"])
        epochs = np.array([s["epoch"] for s in snaps])
        # Use the first equation's residual; multi-equation could tabify later.
        mat = np.array([s["residual_l2"][0] for s in snaps])
        fig = go.Figure(
            data=go.Heatmap(
                x=t_grid,
                y=epochs,
                z=mat,
                colorscale="Viridis",
                colorbar={"title": "|r|"},
            )
        )
        fig.update_layout(
            xaxis_title="collocation t",
            yaxis_title="epoch",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No `residual_heatmap` data in this run.")

    # Sensor residuals (latest snapshot)
    sr = cb.get("sensor_residuals")
    if isinstance(sr, dict) and sr.get("snapshots"):
        st.subheader("Sensor residuals (latest epoch)")
        latest = sr["snapshots"][-1]
        fig = go.Figure()
        for sensor_name, snap in latest.get("per_sensor", {}).items():
            fig.add_trace(
                go.Scatter(
                    x=snap["t"],
                    y=snap["residual"],
                    name=sensor_name,
                    mode="lines",
                )
            )
        fig.update_layout(
            xaxis_title="t",
            yaxis_title="measured − predicted",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No `sensor_residuals` data in this run.")

    # Spectral bias
    sb = cb.get("spectral_bias")
    if isinstance(sb, dict) and sb.get("snapshots"):
        st.subheader("Spectral bias")
        snaps = sb["snapshots"]
        epochs = np.array([s["epoch"] for s in snaps])
        # Use the first output column.
        mat = np.array([s["mag"][0] for s in snaps])
        freqs = np.arange(mat.shape[1])
        fig = go.Figure(
            data=go.Heatmap(
                x=freqs,
                y=epochs,
                z=np.log10(np.array(mat) + 1e-12),
                colorscale="Plasma",
                colorbar={"title": "log10 |F|"},
            )
        )
        fig.update_layout(
            xaxis_title="frequency bin",
            yaxis_title="epoch",
            title="FFT magnitude of network output (log scale)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No `spectral_bias` data in this run.")


def _view_compare(runs: List[RunData]) -> None:
    st.title("Compare runs")
    if len(runs) < 2:
        st.info("Pick 2+ manifests in the sidebar to compare.")
        return

    rows = []
    for r in runs:
        m = r.manifest
        row = {
            "run_id": r.run_id[:8],
            "template": r.template,
            "seed": m.get("seed", "?"),
            "final_loss": m.get("final_loss", float("nan")),
            "automl_study": m.get("automl_study"),
            "trial_number": m.get("trial_number"),
        }
        for name, vals in r.final_params.items():
            row[f"{name}_final"] = vals["final"]
            row[f"{name}_std"] = vals["std"]
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Grouped bar chart per parameter
    all_params = set()
    for r in runs:
        all_params.update(r.final_params.keys())
    for p in sorted(all_params):
        fig = go.Figure()
        for r in runs:
            v = r.final_params.get(p)
            if v is None:
                continue
            fig.add_trace(
                go.Bar(
                    x=[r.run_id[:8]],
                    y=[v["final"]],
                    error_y={"type": "data", "array": [v["std"]]},
                    name=r.run_id[:8],
                )
            )
        fig.update_layout(
            title=f"{p} across runs (±σ)",
            yaxis_title=p,
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)


def _view_automl() -> None:
    st.title("AutoML studies")
    dbs = list_optuna_dbs()
    if not dbs:
        st.info("No Optuna study databases found.")
        return
    labels = [f"{p.stem.replace('optuna_', '')}" for p in dbs]
    pick = st.selectbox("Pick a study", labels)
    db = dbs[labels.index(pick)]
    try:
        study = load_optuna_study(db)
    except Exception as e:
        st.error(f"Failed to load study: {e}")
        return

    trials = study.trials
    rows = []
    for t in trials:
        row = {
            "number": t.number,
            "state": t.state.name,
            "value": t.value,
        }
        row.update(t.params)
        rows.append(row)
    df = pd.DataFrame(rows)
    st.dataframe(df.sort_values("value", na_position="last"), hide_index=True, use_container_width=True)

    st.markdown(
        f"For the live leaderboard: "
        f"`optuna-dashboard sqlite:///{db.absolute()}`"
    )

    # Scatter plot of trial values
    fig = go.Figure()
    completed = [t for t in trials if t.state.name == "COMPLETE"]
    pruned = [t for t in trials if t.state.name == "PRUNED"]
    if completed:
        fig.add_trace(
            go.Scatter(
                x=[t.number for t in completed],
                y=[t.value for t in completed],
                mode="markers",
                name=f"completed ({len(completed)})",
                marker={"size": 10},
            )
        )
    if pruned:
        fig.add_trace(
            go.Scatter(
                x=[t.number for t in pruned],
                y=[t.value if t.value is not None else float("nan") for t in pruned],
                mode="markers",
                name=f"pruned ({len(pruned)})",
                marker={"size": 10, "symbol": "x"},
            )
        )
    fig.update_layout(
        xaxis_title="trial",
        yaxis_title="objective",
        title=f"Study `{pick}` — best value: {study.best_value:.4g}",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================ equations view


_DEFAULT_DSL = '''\
# Edit your inverse problem here. The engine looks for a name `system`.
from pinn_engine.dsl import Variable, Parameter, Unknown, Sensor, System

t = Variable("t")
x = Variable("x", depends_on=t)
m = Parameter("m", value=1.0)
c = Unknown("c", bounds=(0.0, 1.5))
k = Unknown("k", bounds=(0.0, 20.0))

system = System(
    state=[x],
    equations=[m * x.dd + c * x.d + k * x],
    sensors=[Sensor("x_meas", observes=x, noise_std=0.01)],
)
'''


def _view_equations() -> None:
    """Equation editor. Lets the user paste DSL code and compile it."""
    st.title("Equations")
    st.caption(
        "Declare an inverse problem in the engine's symbolic DSL. The text "
        "below is `exec`'d in a sandboxed namespace; the dashboard then "
        "calls `system.compile()` and reports the structure."
    )
    code = st.text_area(
        "DSL code",
        value=st.session_state.get("dsl_code", _DEFAULT_DSL),
        height=320,
        key="dsl_code",
    )
    if st.button("Compile & validate", type="primary"):
        # Sandboxed namespace: only the DSL primitives are available.
        from pinn_engine.dsl import Variable, Parameter, Unknown, Sensor, System
        from pinn_engine.dsl.system import SystemValidationError
        import sympy as _sp

        ns: dict = {
            "Variable": Variable,
            "Parameter": Parameter,
            "Unknown": Unknown,
            "Sensor": Sensor,
            "System": System,
            "sympy": _sp,
            "sp": _sp,
        }
        try:
            exec(code, ns)
        except Exception as e:
            st.error(f"`exec` failed: {type(e).__name__}: {e}")
            return
        system = ns.get("system")
        if system is None:
            st.error("No `system` variable was assigned in your code.")
            return
        try:
            comp = system.compile()
        except SystemValidationError as e:
            st.error(f"Validation failed: {e}")
            return
        except Exception as e:
            st.error(f"Compilation failed: {type(e).__name__}: {e}")
            return
        st.success("System compiled.")
        cols = st.columns(3)
        cols[0].metric("state variables", ", ".join(comp.state_names))
        cols[1].metric("unknowns", ", ".join(comp.unknown_names))
        cols[2].metric("physics residuals", len(comp.physics_residuals))
        st.json(
            {
                "state": comp.state_names,
                "input": comp.input_name,
                "unknowns": comp.unknown_names,
                "bounds": {k: list(v) for k, v in comp.unknown_bounds.items()},
                "inits": comp.unknown_inits,
                "sensors": [
                    {"name": s.name, "noise_std": s.noise_std}
                    for s in comp.sensors
                ],
                "equation_hash": comp.equation_hash[:16] + "…",
            }
        )
        # Stash for the Train view to pick up.
        st.session_state["compiled_system_code"] = code


# ============================================================ train view (live)


def _view_train() -> None:
    """Kick off a training run from the dashboard and stream live progress."""
    import json
    import subprocess

    st.title("Train")
    st.caption(
        "Launch a training run on a registered template. The CLI process is "
        "spawned in the background and writes live status every 10 epochs "
        "to `manifests/live_<run_id>.json`; the plots below auto-refresh."
    )

    from pinn_engine.dsl.templates import registry, get_template

    if not registry:
        import pinn_engine.dsl.templates_lib  # noqa: F401

    template_names = sorted(registry.keys())
    template = st.selectbox("Template", template_names, index=0)
    cols = st.columns(3)
    seed = cols[0].number_input("Seed", value=42, step=1, min_value=0, max_value=99_999)
    adam_epochs = cols[1].number_input("Adam epochs", value=800, step=100, min_value=50)
    lbfgs_iters = cols[2].number_input("L-BFGS iters", value=0, step=10, min_value=0)

    if st.button("Start training", type="primary"):
        cmd = [
            sys.executable, "-m", "pinn_engine.cli", "train", template,
            "--seed", str(seed),
            "--adam-epochs", str(adam_epochs),
            "--lbfgs-iters", str(lbfgs_iters),
        ]
        proc = subprocess.Popen(cmd, cwd=str(_pkg_root))
        st.session_state["train_pid"] = proc.pid
        st.session_state["train_started_at"] = pd.Timestamp.now()
        st.success(f"Started PID {proc.pid}. Polling for live status…")

    # Show live status of the most recent live_*.json (within last 5 minutes).
    live_files = sorted(
        manifests_dir().glob("live_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not live_files:
        st.info("No live training files yet — kick off a run above.")
        return

    pick = st.selectbox(
        "Live status file",
        [p.name for p in live_files[:10]],
        index=0,
    )
    live_path = manifests_dir() / pick
    try:
        payload = json.loads(live_path.read_text())
    except Exception as e:
        st.warning(f"Couldn't read {pick}: {e}")
        return

    status = payload.get("status", "?")
    latest = payload.get("latest", {})
    cols = st.columns(4)
    cols[0].metric("status", status)
    cols[1].metric("epoch", latest.get("epoch", "?"))
    cols[2].metric(
        "loss",
        f"{latest.get('loss'):.4g}" if latest.get("loss") is not None else "—",
    )
    cols[3].metric("history len", len(payload.get("history", [])))

    history = payload.get("history", [])
    if history:
        epochs = [h["epoch"] for h in history]
        losses = [h["loss"] for h in history]

        # Loss curve
        fig_loss = go.Figure(
            go.Scatter(x=epochs, y=losses, mode="lines+markers", name="loss")
        )
        fig_loss.update_layout(
            xaxis_title="epoch",
            yaxis_title="train loss",
            yaxis_type="log",
            height=300,
        )
        st.plotly_chart(fig_loss, use_container_width=True)

        # Parameter trajectory
        param_names = list(history[-1].get("params", {}).keys())
        if param_names:
            fig_p = go.Figure()
            for name in param_names:
                fig_p.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=[h["params"].get(name, float("nan")) for h in history],
                        mode="lines",
                        name=name,
                    )
                )
            fig_p.update_layout(
                xaxis_title="epoch",
                yaxis_title="parameter value",
                height=300,
            )
            st.plotly_chart(fig_p, use_container_width=True)

    # Auto-refresh every 2 s while running.
    if status == "running":
        import time
        time.sleep(2)
        st.rerun()


# ============================================================ main


def main() -> None:
    paths = _sidebar_pick_manifests()
    view = _sidebar_pick_view()
    runs = [load_run(p) for p in paths]

    if view == "Overview":
        _view_overview(runs)
    elif view == "Parameters":
        _view_parameters(runs)
    elif view == "Convergence":
        _view_convergence(runs)
    elif view == "Compare":
        _view_compare(runs)
    elif view == "AutoML":
        _view_automl()
    elif view == "Equations":
        _view_equations()
    elif view == "Train":
        _view_train()


if __name__ == "__main__":
    main()
