"""SPEC §6A rows 1-4 (LangGraph, CrewAI, AutoGen, Magentic-One) end to end against
`scripts/mock_openai_server.py`, so no GPU is needed.

The mock always answers with the FIRST agent named in the request, so a worker that really drives its
framework's selection primitive returns candidate 0 of the retrieved top-k on every call. Asserting that
(not merely "some candidate") is what proves the recipe intercepts the selection rather than something
downstream of it. The fallback test needs no venv.
"""
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
from rte.methods.frameworks._bridge import venv_python
from rte.world import World

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N, K, DIST, BETA, SEED, QUERIES = 100, 16, "specialist", 0.25, 1, 20
CASES = [("fw_langgraph", "fw_langgraph"), ("fw_crewai", "fw_crewai"),
         ("fw_autogen", "fw_autogen"), ("fw_magentic_one", "fw_autogen")]


@pytest.fixture(scope="module")
def mock_url():
    """A mock OpenAI-compatible server on a free port, torn down after the module."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen([sys.executable, os.path.join(REPO, "scripts", "mock_openai_server.py"), str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(200):
        try:
            with socket.create_connection(("127.0.0.1", port), 0.2):
                break
        except OSError:
            time.sleep(0.05)
    else:
        proc.kill()
        pytest.fail("mock server did not come up")
    yield f"http://127.0.0.1:{port}/v1"
    proc.kill()
    proc.wait(timeout=10)


def _built(method, base_url, n=N):
    world = World(n, K, DIST, BETA, seed=SEED)
    m = load_method(method)(base_url=base_url)
    assert m.needs == frozenset({"declared"})
    m.build(world.view(m.needs), Budget(3))
    return world, m


@pytest.mark.parametrize("method,env", CASES)
def test_framework_picks_the_first_candidate(method, env, mock_url):
    try:
        venv_python(env)
    except RuntimeError as e:
        pytest.skip(str(e))
    world, m = _built(method, mock_url)
    try:
        before = world.ledger.snapshot()
        for task in world.tasks(QUERIES):
            assert int(m.fetch(task)) == int(m.retrieve(task)[0]), f"{method}: not the first retrieved candidate"
        assert m.stats == {"picks": QUERIES, "fallbacks": 0, "bad_name": 0}, \
            f"{method}: {m.stats}, bridge={m.bridge.stats}"
        # SPEC §6A ledger formula: per fetch, k descriptions compared, one supervisor hop, k+2 messages.
        cost = world.ledger.diff(before)
        assert (cost["probes"], cost["reports"]) == (0, 0)
        assert cost["comparisons"] == QUERIES * m.k
        assert cost["hops"] == QUERIES
        assert cost["messages"] == QUERIES * (m.k + 2)
    finally:
        m.bridge.close()


@pytest.mark.parametrize("n", [100, 1000])
@pytest.mark.parametrize("method,env", CASES)
def test_unmapped_choice_falls_back_to_declared_argmax(method, env, n):
    """A framework naming something that is not a candidate must not route to it."""
    world, m = _built(method, "http://127.0.0.1:1/v1", n=n)
    m.bridge.select = lambda *a, **kw: {"choice": "agent_999999", "error": None, "raw": None}
    task = world.tasks(1)[0]
    cand = m.retrieve(task)
    assert m.fetch(task) == int(cand[np.argmax(m.view.declared[cand, task.family])])
    assert m.stats == {"picks": 0, "fallbacks": 0, "bad_name": 1}
