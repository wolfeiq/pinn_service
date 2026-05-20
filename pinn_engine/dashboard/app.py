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
        ["Overview", "Parameters", "Convergence", "Compare", "AutoML"],
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


if __name__ == "__main__":
    main()
