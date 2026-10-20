"""Shared test setup: repo on sys.path, a private LLM memo, and the markers of pyproject.toml assigned in one place.

Default run (`pytest -q`) deselects `slow` and `fleet`; `-m slow` runs the kNN / MLP router contracts (~25 min),
`-m ""` runs everything. `data` and `fwenv` tests skip themselves when $RTE_DATA assets or framework venvs are absent.
"""
import os
import sys
import tempfile

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
# never load the production memo (GBs)
os.environ.setdefault("RTE_LLM_CACHE", tempfile.mkdtemp(prefix="rte_test_memo_"))

SLOW_PARAMS = {"knn_router", "mlp_router"}                       # MiniLM-embedding routers in test_each_method
FLEET = {"test_live_execution", "test_live_execute_many_shape"}   # need a served vLLM fleet
FWENV = {"test_framework_pick_is_its_own_choice", "test_framework_picks_the_agent_the_model_named",
         "test_picks_the_first_candidate"}                         # need a framework venv under $RTE_DATA/env
ECHO = {"fw_echo", "test_parallel_prefetch_gives_exactly_the_sequential_picks"}   # echo worker in the base env
DATA_MODULES = {"test_linucb_bonus.py", "test_routereval_family_order.py"}
DATA = {"test_default_bit_identical", "test_calibrated_reproduces_live_stats",
        "test_calibrated_on_every_backend_and_lying_unchanged", "test_replay_split_disjoint_rows",
        "test_no_repeat_llmrouterbench", "test_shuffle_permutes_pool_deterministically",
        "test_shuffle_makes_declared_argmax_ties_random", "test_trueskill_golden_after_probe_fix",
        "test_real_cells_summary"}


def fake_cells(path, K=8, M=4, prompts=40):
    """A tiny RouterBench-shaped cell table in the layout scripts/data/02_download_routerbench.py writes (no download).
    test_erratum30's GOLDEN hashes it: change nothing here."""
    rng = np.random.default_rng(0)
    bias = rng.uniform(0.2, 0.9, M)
    np.savez(
        path,
        model_names=np.array([f"m{m}" for m in range(M)]),
        category_names=np.array([f"c{k}" for k in range(K)]),
        offsets=np.arange(K + 1, dtype=np.int64) * prompts,
        n_prompts=np.full(K, prompts, np.int64),
        outcomes=(rng.random((K * prompts, M)) < bias).astype(np.int8),
    )
    return path


def pytest_collection_modifyitems(config, items):
    for it in items:
        fn, params = it.originalname, set(map(str, getattr(getattr(it, "callspec", None), "params", {}).values()))
        if it.path.name == "test_each_method.py" and params & SLOW_PARAMS:
            it.add_marker(pytest.mark.slow)
        if fn in FLEET:
            it.add_marker(pytest.mark.fleet)
        if fn in FWENV:
            it.add_marker(pytest.mark.fwenv)
        if fn in DATA or it.path.name in DATA_MODULES:
            it.add_marker(pytest.mark.data)
        if (it.path.name == "test_each_method.py" and "fw_echo" in params) or fn in ECHO:
            it.add_marker(pytest.mark.fwenv)
