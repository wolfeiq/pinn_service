"""End-to-end smoke test for the auto-adaptive LR controller.

Uses diffusion_1d on CPU because it's fast (<60s) and exercises the full
DESCEND/PROBE/CONVERGED state machine. Asserts the controller converges D
(truth 0.1) to within reasonable tolerance and writes telemetry. Marked slow
so it doesn't run by default with the rest of the suite.
"""
import warnings
warnings.filterwarnings("ignore")

import pytest

from pinn_engine.dsl.templates_lib import diffusion_1d  # registers
from pinn_engine.dsl.templates import get_template
from pinn_engine.core.trainer import train
from pinn_engine.core.adaptive_controller import AdaptiveUnknownsController


@pytest.mark.slow
def test_adaptive_controller_converges_diffusion_1d():
    tpl = get_template("diffusion_1d")
    system = tpl.system()
    data, truth = tpl.synthetic_data(seed=0)
    cfg = tpl.default_config()
    cfg.accelerator = "cpu"
    cfg.adam_epochs = 50
    cfg.lbfgs_iters = 0
    cfg.param_lr_scale = 500.0   # universal starting scale
    cfg.deterministic = False

    ctrl = AdaptiveUnknownsController()
    result = train(system, data, cfg, callbacks=[ctrl])

    # The controller writes per-epoch telemetry.
    assert ctrl.history, "controller history is empty"
    last = ctrl.history[-1]
    for key in ("base_mult", "eff_mult", "state", "loss"):
        assert key in last, f"history entry missing {key}"

    # Convergence: D (truth 0.1) within 10% rel-err.
    D = result.final_params["D"]
    rel_err = abs(D - truth["D"]) / abs(truth["D"])
    assert rel_err < 0.10, f"adaptive controller failed diffusion_1d: D={D}, rel_err={rel_err}"
