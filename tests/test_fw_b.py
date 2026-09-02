"""Framework rivals whose venvs `scripts/fw_envs/{maf,openai_agents,google_adk,llamaindex}.sh` build, driven
against `scripts/mock_openai_server.py` -- no GPU, no vLLM. The mock always routes to the FIRST agent named in
the request, so asserting the pick IS the first retrieved candidate proves we intercept the framework's real
selection rather than any default. The fallback test needs no venv: it drives `FrameworkMethod` directly."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request

import numpy as np
import pytest

from rte.budget import Budget
from rte.methods import load_method
from rte.world import World

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RTE_DATA = os.environ.get("RTE_DATA", "/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte")
N, K, DIST, BETA, SEED, Q = 100, 16, "specialist", 0.25, 1, 20
NAMES = ["fw_maf", "fw_openai_agents", "fw_google_adk", "fw_llamaindex"]
# HandoffBuilder keeps its targets in a set, so MAF's handoff tool order is not the candidate order and the
# mock's fixed policy cannot land on candidate 0; every other recipe presents the roster in candidate order.
CASES = [("fw_maf", {}, True), ("fw_maf", {"mode": "handoff"}, False), ("fw_openai_agents", {}, True),
         ("fw_google_adk", {}, True), ("fw_llamaindex", {}, True), ("fw_llamaindex", {"mode": "handoff"}, True)]


def _built(name):
    py = os.path.join(RTE_DATA, "env", load_method(name).env, "bin", "python")
    return py if os.path.exists(py) else pytest.skip(f"venv for {name} not built (scripts/fw_envs/*.sh)")


@pytest.fixture(scope="module")
def mock_url():
    port = next(p for p in range(8200, 8300) if socket.socket().connect_ex(("127.0.0.1", p)) != 0)
    proc = subprocess.Popen([sys.executable, os.path.join(REPO, "scripts", "mock_openai_server.py"), str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}/v1"
    for _ in range(100):
        try:
            urllib.request.urlopen(url + "/models", timeout=1).read()
            break
        except OSError:
            time.sleep(0.1)
    yield url
    proc.kill(); proc.wait()


@pytest.mark.parametrize("name,params,first", CASES,
                         ids=[f"{n}-{p.get('mode', 'default')}" for n, p, _ in CASES])
def test_framework_picks_the_agent_the_model_named(name, params, first, mock_url):
    _built(name)
    world = World(N, K, DIST, BETA, seed=SEED)
    method = load_method(name)(base_url=mock_url, **params)
    method.build(world.view(method.needs), Budget(3))
    try:
        for task in world.tasks(Q):
            topk = [int(a) for a in method.retrieve(task)]
            chosen = method.fetch(task)
            assert chosen == topk[0] if first else chosen in topk, f"{name} {params}: {chosen} not in {topk[:3]}"
        assert method.stats == {"picks": Q, "fallbacks": 0, "bad_name": 0}, f"bridge {method.bridge.stats}"
    finally:
        method.bridge.close()


@pytest.mark.parametrize("name", NAMES)
def test_unknown_choice_falls_back_to_declared_argmax(name):
    """A framework that answers with a name outside the top-k must not route there: fall back, count it."""
    world = World(N, K, DIST, BETA, seed=SEED)
    method = load_method(name)(base_url="http://127.0.0.1:1/v1")     # never contacted: the bridge is stubbed
    method.build(world.view(method.needs), Budget(3))
    method.bridge.select = lambda *a, **kw: {"choice": "agent_999999", "error": None, "raw": None}
    tasks = world.tasks(5)
    for task in tasks:
        topk = method.retrieve(task)
        assert method.fetch(task) == int(topk[np.argmax(method.view.declared[topk, task.family])])
    assert method.stats == {"picks": 0, "fallbacks": 0, "bad_name": len(tasks)}
    assert world.ledger.messages == N + len(tasks) * (method.k + 2)   # n at build, k + 2 per fetch
