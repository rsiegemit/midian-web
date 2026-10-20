"""Protocol tests for midian.backends.replay against a tiny fake cell table (no
download). Mirrors the 6-member protocol checked implicitly by bernoulli.py's
usage in World. A second test, skipped unless the real npz exists, loads it
and reports the measured S summary at n=1000."""
from __future__ import annotations

import os

import numpy as np
import pytest

from midian.backends.replay import ReplayBackend
from midian.config import RTE_DATA
from midian.world import Task, World
from tests.conftest import fake_cells as make_fake_cells

REAL_CELLS_PATH = os.path.join(RTE_DATA, "data", "routerbench_cells.npz")


@pytest.fixture(scope="module")
def fake_cells(tmp_path_factory):
    return make_fake_cells(str(tmp_path_factory.mktemp("replay") / "fake_cells.npz"))


DISTS = ["specialist", "heavy_tail", "bimodal", "correlated", "iid_uniform"]


@pytest.mark.parametrize("dist", DISTS)
def test_protocol_shapes(fake_cells, dist):
    n, K = 50, 8
    rng = np.random.default_rng(123)
    b = ReplayBackend(n=n, K=K, dist=dist, seed=0, rng=rng, cells_path=fake_cells)

    assert b.n == n
    assert b.K == K
    assert len(b.families) == K
    assert all(isinstance(f, str) for f in b.families)

    S = b.true_skill()
    assert S.shape == (n, K)
    assert np.all(S >= 0.0) and np.all(S <= 1.0)

    D = b.declared("programmatic")
    assert D.shape == (n, K)
    assert np.all(D >= 0.0) and np.all(D <= 1.0)
    D2 = b.declared("self_described")
    np.testing.assert_array_equal(D, D2)          # no LLM here: documented as identical

    assert b.model_id.shape == (n,)
    assert b.model_id.dtype == np.int8
    assert b.mask.shape == (n, K)
    assert b.mask.dtype == bool


def test_execute_deterministic_and_valid(fake_cells):
    n, K = 30, 8
    rng = np.random.default_rng(1)
    b = ReplayBackend(n=n, K=K, dist="specialist", seed=0, rng=rng, cells_path=fake_cells)
    task = Task(id=0, family=3, instance=17)
    o1 = b.execute(5, task)
    o2 = b.execute(5, task)
    assert o1 == o2
    assert o1 in (0, 1)
    # instance wraps by n_prompts[category] -> same effective outcome as instance + n_prompts
    n_prompts = int(b._n_prompts[3])
    task_wrapped = Task(id=0, family=3, instance=17 + n_prompts)
    assert b.execute(5, task_wrapped) == o1


def test_execute_many_shape_dtype(fake_cells):
    n, K = 40, 8
    rng = np.random.default_rng(2)
    b = ReplayBackend(n=n, K=K, dist="iid_uniform", seed=0, rng=rng, cells_path=fake_cells)
    agents = np.array([0, 1, 2, 3])
    families = np.array([0, 1, 2, 3])
    out = b.execute_many(agents[:, None], families[:, None], np.arange(5)[None, :] + 1000 * agents[:, None])
    assert out.shape == (4, 5)
    assert out.dtype == np.int8
    assert np.all((out == 0) | (out == 1))


def test_masked_category_uses_weakest_model(fake_cells):
    """When mask[a,f] is True, execute must read the per-category weakest
    model's outcome, not the agent's own model's outcome."""
    n, K = 10, 8
    rng = np.random.default_rng(3)
    b = ReplayBackend(n=n, K=K, dist="iid_uniform", seed=0, rng=rng, cells_path=fake_cells)
    a, f = 0, 0
    b.mask[a, f] = True
    task = Task(id=0, family=f, instance=5)
    n_prompts = int(b._n_prompts[f])
    row = int(b._row_start[f]) + (5 % n_prompts)
    expected = int(b._outcomes[row, int(b._weakest_model[f])])
    assert b.execute(a, task) == expected


def test_world_end_to_end(fake_cells):
    world = World(n=200, K=8, dist="specialist", beta=0.1, seed=0,
                  backend="replay", backend_kwargs={"cells_path": fake_cells})
    assert world.n == 200
    assert world.K == 8
    tasks = world.tasks(50)
    assert len(tasks) == 50
    for t in tasks:
        o = world.execute(0, t)
        assert o in (0, 1)
    stats = world.stats()
    assert stats["n"] == 200


def test_k_larger_than_table_caps(fake_cells):
    rng = np.random.default_rng(0)
    b = ReplayBackend(n=10, K=1000, dist="iid_uniform", seed=0, rng=rng, cells_path=fake_cells)
    assert b.K == 8          # capped to the fake table's K_full


@pytest.mark.skipif(not os.path.exists(REAL_CELLS_PATH), reason="real routerbench_cells.npz not downloaded")
def test_real_cells_summary():
    rng = np.random.default_rng(0)
    b = ReplayBackend(n=1000, K=64, dist="specialist", seed=0, rng=rng, cells_path=REAL_CELLS_PATH)
    S = b.true_skill()
    print(f"\n[replay real cells] n=1000 K={b.K} S: mean={S.mean():.3f} "
          f"std={S.std():.3f} min={S.min():.3f} max={S.max():.3f}")
    assert S.shape == (1000, b.K)
    assert np.all(S >= 0.0) and np.all(S <= 1.0)
