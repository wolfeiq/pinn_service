"""Importing this package registers all bundled templates."""
from pinn_engine.dsl.templates_lib import (
    damped_oscillator,
    lorenz,
    diffusion_1d,
    nonlinear_drag_1d,
    coupled_drag_3d,
    pendulum,
    cosserat_rod,
    euler_bernoulli_beam,
    axial_elastic_bar,
    planar_elastica,
    planar_cosserat,
    dynamic_cosserat,  # noqa: F401
)  # noqa: F401
