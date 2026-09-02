"""End-to-end checks for the framework rivals in SPEC §6A rows 1-4 (LangGraph, CrewAI, AutoGen,
Magentic-One), driven against `scripts/mock_openai_server.py` so no GPU is needed.

The mock always picks the first agent named in the prompt / the first candidate tool, so a worker that
really drives its framework's selection primitive returns a candidate name on every call: we assert
`picks == QUERIES` (no fallbacks, no unmapped names) and that every returned id is in the retrieved top-k.
Skipped when the framework venv is absent.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

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
    url = f"http://127.0.0.1:{port}/v1"
    for _ in range(200):
        try:
            with socket.create_connection(("127.0.0.1", port), 0.2):
                break
        except OSError:
            time.sleep(0.05)
    else:
        proc.kill()
        pytest.fail("mock server did not come up")
    yield url
    proc.kill()
    proc.wait(timeout=10)


@pytest.mark.parametrize("method,env", CASES)
def test_framework_picks_every_time(method, env, mock_url):
    try:
        venv_python(env)
    except RuntimeError as e:
        pytest.skip(str(e))
    world = World(N, K, DIST, BETA, seed=SEED)
    m = load_method(method)(base_url=mock_url)
    assert m.needs == frozenset({"declared"})
    m.build(world.view(m.needs), Budget(3))
    try:
        for task in world.tasks(QUERIES):
            a = m.fetch(task)
            assert int(a) in {int(c) for c in m.retrieve(task)}, f"{method}: {a} not in the retrieved top-k"
        assert m.stats == {"picks": QUERIES, "fallbacks": 0, "bad_name": 0}, \
            f"{method}: {m.stats}, bridge={m.bridge.stats}"
    finally:
        m.bridge.close()
