"""Programmatic AutoML search over the damped-oscillator template.

Equivalent to running::

    pinn-engine search damped_oscillator --n-trials 12

but called from Python so you can inspect the study object afterwards.
"""
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

from pinn_engine.automl import run_search


def main():
    study = run_search(
        template_name="damped_oscillator",
        n_trials=12,
        study_name="oscillator_example",
        seed=42,
    )

    print("\n=== Search complete ===")
    print(f"Best trial:    #{study.best_trial.number}")
    print(f"Best value:    {study.best_value:.4g}")
    print(f"Best params:   {study.best_trial.params}")
    print(f"\nPruned trials: {sum(1 for t in study.trials if t.state.name == 'PRUNED')} "
          f"/ {len(study.trials)}")
    print(f"\nLeaderboard: optuna-dashboard sqlite:///manifests/optuna_oscillator_example.db")


if __name__ == "__main__":
    main()
