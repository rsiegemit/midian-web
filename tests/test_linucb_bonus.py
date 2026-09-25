"""linucb_honest bonus="own" (post-hoc fix): the default keeps the pre-registered params (so existing rows keep their
id), and the fixed bonus no longer collapses at scale on a calibrated bernoulli world."""
import os

import pytest

from midian.methods import load_method


def test_default_params_unchanged():
    assert load_method("linucb_honest")().params == {"alpha": 1.0}
    assert load_method("linucb_honest")(bonus="own").params == {"alpha": 1.0, "bonus": "own"}


@pytest.mark.skipif(not os.environ.get("RTE_DATA"), reason="needs RTE_DATA populations")
def test_own_bonus_beats_context_bonus_at_1e4():
    from midian.run import run_method
    from midian.world import World
    C = f"{os.environ['RTE_DATA']}/populations/specialist_n1000_K16_seed1/S.npy"
    w = World(10000, 16, "specialist", 0.0, seed=1, backend="bernoulli", backend_kwargs={"calibrate_from": C})
    st = w.tasks(1000)
    old = run_method(w, st, {"name": "linucb_honest", "params": {}}, 3)["success"]
    new = run_method(w, st, {"name": "linucb_honest", "params": {"bonus": "own"}}, 3)["success"]
    assert new > old + 0.03
