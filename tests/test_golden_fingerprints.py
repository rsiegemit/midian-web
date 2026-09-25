"""G1 guard: enumeration of a few cheap grids matches tests/golden/grid_fingerprints.tsv.

The grids cover `methods: all` (pkgutil listing + extra), mirror_of chains, `blocks:` with churn, and routereval
backend_kwargs. None has an RTE_DATA-expanded backend_kwargs, so no data assets are needed; the full check over all
197 grids is `python scripts/checks/grid_fingerprint.py --compare tests/golden/grid_fingerprints.tsv`."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "checks"))
import grid_fingerprint as gf  # noqa: E402

GOLDEN = os.path.join(ROOT, "tests", "golden", "grid_fingerprints.tsv")
GOLDEN_D1 = os.path.join(ROOT, "tests", "golden", "grid_fingerprints_d1.tsv")   # D1 ids of the 11 + new grids
CHEAP = ["smoke", "bernoulli_cost_smoke", "churn_n1000", "churn_n1000_fw_dd", "va_b_routereval5k", "lie_max_n1000",
         "routereval5k_norep_cal"]
RTE_DATA_DEP = {"bernoulli_1e7_cal", "bernoulli_b_probe", "bernoulli_b_sweep", "bernoulli_scale", "bernoulli_scale_v5",
                "linucb_fix_bernoulli_1e7", "pool_fill_bernoulli_1e7", "rivals5_bernoulli_1e7_cal",
                "rivals_b_bernoulli_1e7", "scale_100k", "va_b_bernoulli_1e7"}


@pytest.fixture(scope="module")
def cfg():
    return gf.run.load_config()


@pytest.fixture(scope="module")
def golden():
    return {**gf.read_tsv(GOLDEN), **gf.read_tsv(GOLDEN_D1)}


@pytest.mark.parametrize("grid", CHEAP)
def test_fingerprint_matches_golden(cfg, golden, grid):
    got = gf.fingerprint(cfg, grid)
    assert {c: str(got[c]) for c in gf.COLUMNS} == golden[grid]


def test_grid_set_and_rte_data_dependence(cfg, golden):
    """Same grids as the golden files; exactly the 11 known grids expand $RTE_DATA (decision D1)."""
    names = gf.grid_names(cfg)
    assert sorted(names) == sorted(golden)
    dep = {g for g in names if any(gf.env_dependent(b) for b in gf.run.blocks(cfg, g))}
    assert dep == RTE_DATA_DEP == {g for g, r in golden.items() if r["rte_data_dep"] == "1"}
