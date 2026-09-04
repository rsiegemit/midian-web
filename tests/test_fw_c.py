"""Frameworks-C rivals (smolagents, CAMEL, MetaGPT, AgentScope) driven through their own venvs against
scripts/mock_openai_server.py: no GPU, no vLLM. A framework whose venv is missing is skipped, not failed."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

import numpy as np
import pytest

from rte.budget import Budget
from rte.methods import load_method
from rte.world import World

RTE_DATA = os.environ.get("RTE_DATA", "/scratch/rte")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_picks = lambda st: {k: st[k] for k in ("picks", "fallbacks", "bad_name")}
FRAMEWORKS = {"fw_smolagents": "fw_smolagents", "fw_camel_workforce": "fw_camel",
              "fw_metagpt": "fw_metagpt", "fw_agentscope": "fw_agentscope"}
N_TASKS = 20


def _build(method, **kw):
    """A built method on the standard bernoulli world, or a skip if it declines to exist."""
    try:
        M = load_method(method)(**kw)
    except NotImplementedError as e:
        pytest.skip(f"{method}: {e}")
    w = World(100, 16, "specialist", 0.25, seed=1)
    M.build(w.view(M.needs), Budget(3))
    return M, w


@pytest.fixture(scope="module")
def mock_url():
    port = next(p for p in range(8300, 8400) if socket.socket().connect_ex(("127.0.0.1", p)))
    proc = subprocess.Popen([sys.executable, os.path.join(REPO, "scripts", "mock_openai_server.py"), str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        if socket.socket().connect_ex(("127.0.0.1", port)) == 0:
            break
        time.sleep(0.1)
    else:
        proc.kill(); raise RuntimeError("mock server did not start")
    yield f"http://127.0.0.1:{port}/v1"
    proc.terminate()
    try: proc.wait(timeout=10)
    except subprocess.TimeoutExpired: proc.kill()


@pytest.mark.parametrize("method,env", sorted(FRAMEWORKS.items()))
def test_picks_the_first_candidate(method, env, mock_url):
    """The mock always names the FIRST agent in the request. If the recipe really intercepts the
    framework's own selection, every route must land on cand[0] -- not merely somewhere in the top-k."""
    if not os.path.exists(os.path.join(RTE_DATA, "env", env, "bin", "python")):
        pytest.skip(f"venv missing: $RTE_DATA/env/{env} (run scripts/fw_envs/{env[3:]}.sh)")
    M, w = _build(method, base_url=mock_url)
    try:
        for t in w.tasks(N_TASKS):
            assert M.fetch(t) == int(M.retrieve(t)[0]), f"{method} did not return the framework's own pick"
        assert _picks(M.stats) == {"picks": N_TASKS, "fallbacks": 0, "bad_name": 0}, M.stats
    finally:
        M.bridge.close()


@pytest.mark.parametrize("method", sorted(FRAMEWORKS))
@pytest.mark.parametrize("choice,counter", [("not_a_candidate", "bad_name"), (None, "fallbacks")])
def test_bad_choice_falls_back_to_declared_argmax(method, choice, counter):
    """A framework that answers with a non-candidate (or nothing) must not poison the route: fall back
    to declared argmax among the retrieved k and say so in stats. No venv needed -- the bridge is mocked."""
    M, w = _build(method, base_url="http://127.0.0.1:1/v1")
    M.bridge.select = lambda *a, **kw: {"choice": choice}
    for t in w.tasks(5):
        cand = M.retrieve(t)
        assert M.fetch(t) == int(cand[np.argmax(M.view.declared[cand, t.family])])
    assert _picks(M.stats) == {"picks": 0, "fallbacks": 5 if counter == "fallbacks" else 0,
                               "bad_name": 5 if counter == "bad_name" else 0}, M.stats


@pytest.mark.parametrize("method", ["fw_crewai", "fw_magentic_one", "fw_google_adk"])
def test_no_delegation_is_a_failure_with_strict_success_zero(method):
    """A worker that reports "FAILURE: ..." (the framework answered instead of delegating) still routes the
    declared-argmax fallback (lenient accounting) but scores 0 under `success_strict`; picks score their outcome."""
    M, w = _build(method, base_url="http://127.0.0.1:1/v1")
    M.bridge.select = lambda *a, **kw: {"choice": None, "error": None, "raw": "FAILURE: answered itself"}
    tasks = w.tasks(4)
    for t in tasks[:2]:
        a = M.fetch(t); M.observe(t, a, 1)
    M.bridge.select = lambda task, cands, *a, **kw: {"choice": cands[0]["name"], "error": None, "raw": "ok"}
    for t in tasks[2:]:
        a = M.fetch(t); M.observe(t, a, 1)
    assert M.stats["failures"] == 2 and M.stats["picks"] == 2 and M.stats["fallbacks"] == 0
    assert M.stats["success_strict"] == 0.5 and M.stats["fallback_rate"] == 0.5
