"""Shared test setup: repo on sys.path, a private LLM memo, and the markers of pyproject.toml assigned in one place.

Default run (`pytest -q`) deselects `slow` and `fleet`; `-m slow` runs the kNN / MLP router contracts (~25 min),
`-m ""` runs everything. `data` and `fwenv` tests skip themselves when $RTE_DATA assets or framework venvs are absent.
"""
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def _echo_env_missing():
    from rte.methods.frameworks._bridge import venv_python
    try:
        venv_python("rte")
        return False
    except RuntimeError:
        return True


def pytest_collection_modifyitems(config, items):
    echo_skip = None
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
            if echo_skip is None:
                echo_skip = _echo_env_missing()
            if echo_skip:
                it.add_marker(pytest.mark.skip(reason="echo worker interpreter not found ($RTE_DATA/env/rte)"))
