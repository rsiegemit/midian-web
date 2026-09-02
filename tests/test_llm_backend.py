"""CPU-only tests for the llm backend. Nothing here needs a GPU or a served endpoint;
the two tests that do are skipped unless $RTE_DATA/endpoints.json exists.

What is checked:
  * reasoning-gym instances are deterministic from (family, instance) and the verifier scores
    the gold answer 1 and junk 0 -- for EVERY family in the K=16 list;
  * profile drawing is seeded and matches the shape each skill distribution claims;
  * prompt construction honours the handicaps (exemplar withheld, tool removed, budget capped)
    and identical signatures produce byte-identical prompts (the memo's correctness condition);
  * answer extraction and both tools.
"""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from rte.backends.llm import (FAMILIES_16, LADDER, LLMBackend, build_prompt, default_families,
                              draw_profiles, extract_answer, family_exemplar, find_tool_call,
                              instance_entry, run_calculator, run_python, score, signature)

RTE_DATA = Path(os.environ.get("RTE_DATA", "/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte"))


def _served() -> list[str]:
    """Models the fleet is actually serving right now. An endpoints.json left behind by a finished
    job is an empty dict, not a missing file, so presence of the file proves nothing."""
    try:
        from rte import llm_client
        return sorted(llm_client.endpoints())
    except Exception:                                        # noqa: BLE001
        return []


needs_endpoints = pytest.mark.skipif(
    not _served(), reason="no vLLM endpoints served; run scripts/serve_smoke.sbatch first")


# --------------------------------------------------------------------------- reasoning gym
@pytest.mark.parametrize("family", FAMILIES_16)
def test_instance_is_deterministic_and_gold_scores_one(family):
    a = instance_entry(family, 4242)
    b = instance_entry(family, 4242)
    assert a["question"] == b["question"]
    assert a["answer"] == b["answer"]
    assert a["answer"] is not None, f"{family} has no gold answer"
    assert score(family, 4242, str(a["answer"])) == 1
    assert score(family, 4242, "zzz_not_an_answer") == 0


def test_different_instances_give_different_problems():
    qs = {instance_entry("basic_arithmetic", i)["question"] for i in range(20)}
    assert len(qs) > 10


def test_score_never_raises_on_malformed_answers():
    # prime_factorization's verifier does int() on the answer; a junk answer must score 0, not crash.
    for bad in ["", "   ", "<answer></answer>", "I don't know", "3 x 5 x ???"]:
        assert score("prime_factorization", 77, bad) == 0


def test_default_families_are_registered_and_unique():
    fams = default_families(16)
    assert len(fams) == 16 and len(set(fams)) == 16
    from reasoning_gym.factory import DATASETS
    assert all(f in DATASETS for f in fams)


# --------------------------------------------------------------------------- profiles
def test_specialist_has_exactly_three_unhandicapped_families():
    profs = draw_profiles(200, 16, "specialist", seed=1)
    assert len(profs) == 200
    assert all(len(p["specialty"]) == 3 for p in profs)
    assert all(p["model"] in LADDER for p in profs)
    assert all(p["tool"] in ("calculator", "python", "none") for p in profs)


def test_profile_draw_is_seeded_and_reproducible():
    a = draw_profiles(50, 16, "specialist", seed=7)
    b = draw_profiles(50, 16, "specialist", seed=7)
    c = draw_profiles(50, 16, "specialist", seed=8)
    assert a == b
    assert a != c


def test_heavy_tail_is_one_in_ten_big_unhandicapped_experts():
    profs = draw_profiles(2000, 16, "heavy_tail", seed=3)
    experts = [p for p in profs if len(p["specialty"]) == 16]
    assert 0.06 < len(experts) / len(profs) < 0.15
    assert all(p["model"] in ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct",
                              "google/gemma-2-9b-it") for p in experts)
    assert all(p["tool"] == "python" for p in experts)
    others = [p for p in profs if p not in experts]
    assert all(p["specialty"] == [] and p["tool"] == "none" for p in others)
    assert all(p["model"] in ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct")
               for p in others)


def test_bimodal_is_twenty_percent_seven_b_with_tools():
    profs = draw_profiles(2000, 16, "bimodal", seed=5)
    good = [p for p in profs if p["model"] == "Qwen/Qwen2.5-7B-Instruct"]
    assert 0.15 < len(good) / len(profs) < 0.26
    assert all(p["tool"] == "python" and len(p["specialty"]) == 16 for p in good)
    bad = [p for p in profs if p["model"] == "Qwen/Qwen2.5-0.5B-Instruct"]
    assert len(good) + len(bad) == len(profs)
    assert all(p["tool"] == "none" and p["specialty"] == [] for p in bad)


def test_correlated_specialty_is_group_level():
    for p in draw_profiles(100, 16, "correlated", seed=2):
        groups = {f % 4 for f in p["specialty"]}
        for g in groups:                                 # a good group is good in ALL its families
            assert all(f in p["specialty"] for f in range(16) if f % 4 == g)


def test_iid_uniform_specialty_is_unstructured():
    profs = draw_profiles(500, 16, "iid_uniform", seed=4)
    sizes = np.array([len(p["specialty"]) for p in profs])
    assert 6 < sizes.mean() < 10 and sizes.std() > 1.0


def test_unknown_dist_raises():
    with pytest.raises(ValueError):
        draw_profiles(10, 4, "nope", seed=0)


# --------------------------------------------------------------------------- prompts / handicaps
def test_handicap_withholds_exemplar_description_and_tool():
    p = {"id": 0, "model": LADDER[0], "specialty": [0], "tool": "python"}
    assert signature(p, 0, 512, 512) == (LADDER[0], False, "python", 512)
    assert signature(p, 1, 512, 512) == (LADDER[0], True, "none", 512)

    q_ex, _ = family_exemplar("gcd")
    ok = build_prompt("gcd", "Q?", handicapped=False, tool="python")[0]["content"]
    hand = build_prompt("gcd", "Q?", handicapped=True, tool="none")[0]["content"]
    assert "Worked example" in ok and q_ex[:40] in ok
    assert "Worked example" not in hand
    assert "gcd" in ok and "gcd" not in hand          # the family description goes too
    assert "Python sandbox" in ok and "Python sandbox" not in hand
    # both keep the answer-format instruction: the verifier is shared, so format compliance
    # must not be part of the handicap
    assert "<answer>" in ok and "<answer>" in hand


def test_token_budget_cap_is_opt_in():
    """The budget cap was measured to be non-monotone (it HELPED a 0.5B on syllogism), so it must
    be off unless a caller asks for it."""
    p = {"id": 0, "model": LADDER[0], "specialty": [0], "tool": "none"}
    b = LLMBackend(n=1, K=2, dist="specialist", seed=1, rng=np.random.default_rng(1),
                   families=["gcd", "basic_arithmetic"])
    assert b.handicap_max_tokens == b.max_tokens
    b2 = LLMBackend(n=1, K=2, dist="specialist", seed=1, rng=np.random.default_rng(1),
                    families=["gcd", "basic_arithmetic"], handicap_max_tokens=160)
    assert b2.handicap_max_tokens == 160
    assert signature(p, 1, 512, 160)[3] == 160


def test_calculator_tool_appears_only_for_calculator_agents():
    calc = build_prompt("gcd", "Q?", False, "calculator")[0]["content"]
    none = build_prompt("gcd", "Q?", False, "none")[0]["content"]
    assert "<calc>" in calc and "Python sandbox" not in calc
    assert "<calc>" not in none and "Python sandbox" not in none


def test_same_signature_gives_identical_prompt():
    """The memo is keyed on the signature, so equal signatures MUST give equal prompts."""
    p1 = {"id": 0, "model": LADDER[2], "specialty": [0, 1, 2], "tool": "calculator"}
    p2 = {"id": 9, "model": LADDER[2], "specialty": [2, 5, 7], "tool": "calculator"}
    assert signature(p1, 2, 512, 512) == signature(p2, 2, 512, 512)
    q = instance_entry("gcd", 11)["question"]
    assert build_prompt("gcd", q, False, "calculator") == build_prompt("gcd", q, False, "calculator")
    # and different handicap status must NOT collide
    assert signature(p1, 3, 512, 512) != signature(p1, 2, 512, 512)


def test_user_message_is_the_raw_question():
    q = instance_entry("chain_sum", 5)["question"]
    msgs = build_prompt("chain_sum", q, False, "none")
    assert msgs[1] == {"role": "user", "content": q}
    assert msgs[0]["role"] == "system"


# --------------------------------------------------------------------------- extraction
@pytest.mark.parametrize("text,want", [
    ("blah <answer>42</answer>", "42"),
    ("<answer>1</answer> then <answer>2</answer>", "2"),
    ("<ANSWER> 7 </ANSWER>", "7"),
    ("no tags here\nfinal line", "final line"),
    ("", ""),
    ("<answer>\n-3.5\n</answer>", "-3.5"),
])
def test_extract_answer(text, want):
    assert extract_answer(text) == want


def test_find_tool_call_skips_when_already_answered():
    assert find_tool_call("<calc>1+1</calc>", "calculator") == "1+1"
    assert find_tool_call("<calc>1+1</calc><answer>2</answer>", "calculator") is None
    assert find_tool_call("```python\nprint(1)\n```", "python").strip() == "print(1)"
    assert find_tool_call("<calc>1+1</calc>", "none") is None


# --------------------------------------------------------------------------- tools
def test_calculator_evaluates_and_refuses_non_arithmetic():
    assert run_calculator("2 + 3 * 4") == "14"
    assert run_calculator("(10 - 4) / 2") == "3.0"
    assert run_calculator("2^10") == "1024"
    assert run_calculator("__import__('os').system('echo hi')").startswith("ERROR")
    assert run_calculator("open('/etc/passwd')").startswith("ERROR")
    assert run_calculator("9 ** 9999").startswith("ERROR")


def test_python_tool_runs_and_contains_failures():
    assert run_python("print(6*7)") == "42"
    assert run_python("while True: pass").startswith("ERROR: timed out")
    assert "network disabled" in run_python(
        "import socket\ntry:\n socket.socket()\nexcept OSError as e:\n print(e)")


# --------------------------------------------------------------------------- backend wiring
def test_backend_exposes_the_protocol():
    b = LLMBackend(n=8, K=4, dist="specialist", seed=1, rng=np.random.default_rng(1))
    assert b.n == 8 and b.K == 4 and len(b.families) == 4
    for member in ("true_skill", "declared", "execute", "execute_many", "stats"):
        assert callable(getattr(b, member))
    assert b.stats()["llm_executions"] == 0


def test_backend_registers_itself_for_llm_supervisor():
    from rte.backends import llm as m
    b = LLMBackend(n=4, K=4, dist="specialist", seed=2, rng=np.random.default_rng(2))
    assert m.current_backend() is b


def test_declared_programmatic_is_noisy_S(tmp_path):
    b = LLMBackend(n=6, K=4, dist="specialist", seed=3, rng=np.random.default_rng(3),
                   population_dir=str(tmp_path))
    S = np.zeros((6, 4), dtype=np.float32)
    S[:, 0] = 0.8
    np.save(tmp_path / "S.npy", S)
    D = b.declared("programmatic")
    assert D.shape == (6, 4)
    assert np.all((D >= 0) & (D <= 1))
    assert abs(float(D[:, 0].mean()) - 0.8) < 0.1
    with pytest.raises(ValueError):
        b.declared("nonsense")


# --------------------------------------------------------------------------- live (skipped w/o fleet)
@pytest.fixture()
def live_ladder(monkeypatch):
    """Pin the model ladder to what the fleet actually serves, so these tests work against the
    one-model smoke fleet as well as the full seven-model fleet."""
    from rte.backends import llm as m
    served = _served()
    monkeypatch.setattr(m, "LADDER", served)
    monkeypatch.setattr(m, "SMALL", served)
    monkeypatch.setattr(m, "BIG", served)
    monkeypatch.setattr(m, "LARGE_MODELS", frozenset())
    return served


@needs_endpoints
def test_live_single_execution(live_ladder):
    from rte.world import Task
    b = LLMBackend(n=2, K=2, dist="specialist", seed=1, rng=np.random.default_rng(1),
                   families=["basic_arithmetic", "gcd"])
    out = b.execute(0, Task(0, 0, 12345))
    assert out in (0, 1)
    assert b.stats()["llm_executions"] == 1


@needs_endpoints
def test_live_execute_many_shape(live_ladder):
    b = LLMBackend(n=3, K=2, dist="specialist", seed=1, rng=np.random.default_rng(1),
                   families=["basic_arithmetic", "gcd"])
    out = b.execute_many(np.array([0, 1, 2]), np.array([0, 0, 0]), reps=2,
                         rng=np.random.default_rng(0))
    assert out.shape == (3, 2)
    assert set(np.unique(out)) <= {0, 1}


# --------------------------------------------------------------------------- mock fleet (no GPU)
class _MockHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible chat endpoint. Answers `<answer>42</answer>`, except that a
    calculator agent's FIRST turn gets a `<calc>` block back so the tool round is exercised."""
    calls = 0
    supervisor_reply = None

    def log_message(self, *a):
        pass

    def do_GET(self):
        self._send({"ok": True})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        msgs = req.get("messages", [])
        system = msgs[0]["content"] if msgs else ""
        already_tooled = any("Tool result:" in (m.get("content") or "") for m in msgs)
        if "routing supervisor" in system and type(self).supervisor_reply is not None:
            content = type(self).supervisor_reply
        elif "<calc>" in system and not already_tooled:
            content = "Let me compute. <calc>21*2</calc>"
        else:
            content = "<answer>42</answer>"
        type(self).calls += 1
        self._send({"id": "x", "object": "chat.completion", "model": req.get("model", "m"),
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": content}}]})

    def _send(self, body):
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture()
def mock_fleet(tmp_path, monkeypatch):
    from rte import llm_client
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _MockHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    ep = tmp_path / "endpoints.json"
    ep.write_text(json.dumps({m: url for m in LADDER}))
    monkeypatch.setattr(llm_client, "ENDPOINTS_PATH", ep)
    monkeypatch.setattr(llm_client, "ENDPOINT_DIR", tmp_path / "endpoints.d")
    monkeypatch.setattr(llm_client, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(llm_client, "_endpoints_cache", None)
    monkeypatch.setattr(llm_client, "_endpoints_mtime", None)
    monkeypatch.setattr(llm_client, "_dbs", {})
    monkeypatch.setattr(llm_client, "_db_locks", {})
    monkeypatch.setattr(llm_client, "_clients", {})
    llm_client.reset_stats()
    _MockHandler.calls = 0
    yield llm_client
    srv.shutdown()


def test_client_memoizes_across_calls(mock_fleet):
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    a = mock_fleet.complete(LADDER[0], msgs, max_tokens=32, cache_key="k1")
    b = mock_fleet.complete(LADDER[0], msgs, max_tokens=32, cache_key="k1")
    assert a == b == "<answer>42</answer>"
    assert mock_fleet.stats()["generations"] == 1, "second call should have been served from disk"
    assert mock_fleet.stats()["hits"] == 1


def test_complete_batch_deduplicates_repeated_keys(mock_fleet):
    """40 requests, 2 distinct keys -> 2 generations. This is the property the whole
    measurement budget rests on."""
    msgs = [[{"role": "system", "content": "s"}, {"role": "user", "content": f"u{i % 2}"}]
            for i in range(40)]
    keys = [f"key{i % 2}" for i in range(40)]
    out = mock_fleet.complete_batch(LADDER[0], msgs, keys, max_tokens=32)
    assert len(out) == 40 and all(o == "<answer>42</answer>" for o in out)
    assert mock_fleet.stats()["generations"] == 2, mock_fleet.stats()
    assert _MockHandler.calls == 2


def test_backend_end_to_end_against_mock(mock_fleet):
    from rte.world import Task
    b = LLMBackend(n=4, K=2, dist="specialist", seed=1, rng=np.random.default_rng(1),
                   families=["basic_arithmetic", "gcd"])
    o = b.execute(0, Task(0, 0, 999))
    assert o in (0, 1)
    out = b.execute_many(np.arange(4), np.zeros(4, int), reps=2, rng=np.random.default_rng(0))
    assert out.shape == (4, 2) and set(np.unique(out)) <= {0, 1}
    st = b.stats()
    assert st["llm_executions"] == 9 and st["llm_generations"] > 0


def test_tool_round_trip_against_mock(mock_fleet):
    """A calculator agent gets a <calc> block back, we evaluate it, and the follow-up turn
    supplies the final answer -- two generations for one execution."""
    from rte.world import Task
    b = LLMBackend(n=1, K=1, dist="specialist", seed=1, rng=np.random.default_rng(1),
                   families=["basic_arithmetic"])
    b.profiles[0] = {"id": 0, "model": LADDER[0], "specialty": [0], "tool": "calculator"}
    b.execute(0, Task(0, 0, 555))
    assert b.stats()["llm_tool_calls"] == 1
    assert _MockHandler.calls == 2


def test_no_endpoints_raises_a_clear_error(tmp_path, monkeypatch):
    from rte import llm_client
    monkeypatch.setattr(llm_client, "ENDPOINTS_PATH", tmp_path / "nope.json")
    monkeypatch.setattr(llm_client, "ENDPOINT_DIR", tmp_path / "nope.d")
    monkeypatch.setattr(llm_client, "_endpoints_cache", None)
    monkeypatch.setattr(llm_client, "_endpoints_mtime", None)
    assert not llm_client.have_endpoints()
    with pytest.raises(llm_client.NoEndpointsError, match="no vLLM endpoints"):
        llm_client.endpoints()


# --------------------------------------------------------------------------- llm_supervisor
def test_llm_supervisor_end_to_end(mock_fleet, tmp_path):
    """World -> llm backend -> descriptions -> TF-IDF shortlist -> supervisor pick, no GPU."""
    from rte.budget import Budget
    from rte.methods.llm_supervisor import LLMSupervisor
    from rte.world import World

    w = World(n=6, K=2, dist="specialist", beta=0.0, seed=1, backend="llm",
              backend_kwargs={"families": ["basic_arithmetic", "gcd"], "measure_probes": 2,
                              "measure_probes_large": 2, "population_dir": str(tmp_path / "pop"),
                              "concurrency": 4})
    w.backend.descriptions()
    view = w.view({"declared"})
    m = LLMSupervisor(top_k=4)
    m.build(view, Budget())

    before = view.ledger.snapshot()
    a = m.fetch(w.tasks(1)[0])
    d = view.ledger.diff(before)
    assert 0 <= a < w.n
    assert d["comparisons"] == 4 and d["hops"] == 1
    assert d["probes"] == 0 and d["reports"] == 0     # it is a declared-channel method


def test_llm_supervisor_picks_the_index_the_model_names(mock_fleet, tmp_path):
    from rte.budget import Budget
    from rte.methods.llm_supervisor import LLMSupervisor
    from rte.world import World

    _MockHandler.supervisor_reply = "2"
    try:
        w = World(n=6, K=2, dist="specialist", beta=0.0, seed=4, backend="llm",
                  backend_kwargs={"families": ["basic_arithmetic", "gcd"], "measure_probes": 1,
                                  "measure_probes_large": 1, "population_dir": str(tmp_path / "p2"),
                                  "concurrency": 4})
        w.backend.descriptions()
        view = w.view({"declared"})
        m = LLMSupervisor(top_k=5)
        m.build(view, Budget())
        task = w.tasks(1)[0]
        assert m.fetch(task) == int(m._retrieve(int(task.family))[2])
    finally:
        _MockHandler.supervisor_reply = None


def test_llm_supervisor_needs_descriptions(monkeypatch):
    from rte.backends import llm as m
    from rte.budget import Budget
    from rte.methods.llm_supervisor import LLMSupervisor
    monkeypatch.setattr(m, "_CURRENT", None)

    class FakeView:
        n, K, families = 5, 2, ["basic_arithmetic", "gcd"]
        declared = np.zeros((5, 2), dtype=np.float32)
    with pytest.raises(RuntimeError, match="per-agent descriptions"):
        LLMSupervisor().build(FakeView(), Budget())
