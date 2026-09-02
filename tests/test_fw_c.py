"""End-to-end checks for the Frameworks-C rivals (smolagents, CAMEL, MetaGPT, AgentScope):
each worker is driven through its OWN venv against scripts/mock_openai_server.py, so no GPU and
no vLLM are needed. A framework whose venv has not been built is skipped, not failed."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

import pytest

from rte.budget import Budget
from rte.methods import load_method
from rte.world import World

RTE_DATA = os.environ.get("RTE_DATA", "/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMEWORKS = {"fw_smolagents": "fw_smolagents", "fw_camel_workforce": "fw_camel",
              "fw_metagpt": "fw_metagpt", "fw_agentscope": "fw_agentscope"}
N_TASKS = 20


def _free_port(lo=8300, hi=8400):
    for p in range(lo, hi):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)):
                return p
    raise RuntimeError("no free port in 8300-8399")


@pytest.fixture(scope="module")
def mock_url():
    port = _free_port()
    proc = subprocess.Popen([sys.executable, os.path.join(REPO, "scripts", "mock_openai_server.py"), str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                break
        time.sleep(0.1)
    else:
        proc.kill(); raise RuntimeError("mock server did not start")
    yield f"http://127.0.0.1:{port}/v1"
    proc.terminate()
    try: proc.wait(timeout=10)
    except subprocess.TimeoutExpired: proc.kill()


@pytest.mark.parametrize("method,env", sorted(FRAMEWORKS.items()))
def test_framework_picks_within_topk(method, env, mock_url):
    py = os.path.join(RTE_DATA, "env", env, "bin", "python")
    if not os.path.exists(py):
        pytest.skip(f"venv missing: {py} (run scripts/fw_envs/{env[3:]}.sh)")
    cls = load_method(method)
    try:
        M = cls(base_url=mock_url)
    except NotImplementedError as e:
        pytest.skip(f"{method}: {e}")
    w = World(100, 16, "specialist", 0.25, seed=1)
    M.build(w.view(M.needs), Budget(3))
    try:
        for t in w.tasks(N_TASKS):
            a = M.fetch(t)
            assert a in {int(x) for x in M.retrieve(t)}, f"{method} routed outside its own top-k"
        assert M.stats["picks"] == N_TASKS, f"{method} did not select every time: {M.stats}"
    finally:
        M.bridge.close()
