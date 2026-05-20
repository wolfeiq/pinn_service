"""Streamlit dashboard — Phase-5 visualization layer.

Reads run manifests + callback NPZ files + Optuna study DBs and presents
a multi-page UI. Launched via the CLI::

    pinn-engine dashboard

Internally this just runs ``streamlit run pinn_engine/dashboard/app.py``;
the CLI subcommand wraps the path-resolution boilerplate.
"""
