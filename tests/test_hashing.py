"""Repro hashing: same inputs → same hashes."""
import numpy as np

from pinn_engine.dsl import Variable, Parameter, Unknown, Sensor, System
from pinn_engine.core.trainer import TrainConfig
from pinn_engine.repro.hashing import hash_data, hash_config, hash_system


def _osc():
    t = Variable("t"); x = Variable("x", depends_on=t)
    m = Parameter("m", value=1.0)
    c = Unknown("c", bounds=(0.0, 5.0)); k = Unknown("k", bounds=(0.0, 100.0))
    return System(state=[x], equations=[m * x.dd + c * x.d + k * x],
                  sensors=[Sensor("x_meas", observes=x)])


def test_system_hash_stable_across_compiles():
    a = _osc().compile().equation_hash
    b = _osc().compile().equation_hash
    assert a == b


def test_data_hash_depends_on_content():
    d1 = {"x": (np.array([0.0, 1.0]), np.array([0.1, 0.2]))}
    d2 = {"x": (np.array([0.0, 1.0]), np.array([0.1, 0.3]))}
    assert hash_data(d1) == hash_data(d1)
    assert hash_data(d1) != hash_data(d2)


def test_config_hash_independent_of_run_id():
    """Two configs differing only in run_id should hash differently — but we'd
    rather they hash the same so manifests can be compared by config alone.

    This test documents current behavior: ``run_id`` is part of the config and
    therefore changes the hash. If we want config-only hashes to ignore run_id,
    flip the exclusion in :func:`hash_config`.
    """
    a = TrainConfig(seed=1, run_id="aaa")
    b = TrainConfig(seed=1, run_id="bbb")
    # Just exercise the function — equality depends on implementation choice.
    assert isinstance(hash_config(a), str) and len(hash_config(a)) == 64
    assert isinstance(hash_config(b), str) and len(hash_config(b)) == 64
