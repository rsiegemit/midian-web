"""CPU-only checks for the llm backend. No GPU; a local mock OpenAI server stands in for vLLM.
The two live tests are skipped unless a fleet is actually serving.

Contract checks (CONTRACT "Correctness checks"): the verifier scores gold 1.0 and junk 0.0 for
EVERY family adapter; (family, instance) regenerates identical question text; the memo returns the
identical answer and counts a hit; profile shape matches each skill distribution.
"""
from __future__ import annotations

import json
import pathlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from rte.backends import families, prompts, tools
from rte.backends.llm import LLMBackend, current_backend
from rte.backends.population import TOOL, bands, draw_profiles, ladder, signature

LADDER = bands(ladder())[0]


def _served() -> list[str]:
    """An endpoints.json left behind by a finished job is an empty dict, not a missing file, so
    the file existing proves nothing."""
    try:
        from rte import llm_client
        return sorted(llm_client.endpoints())
    except Exception:                                            # noqa: BLE001
        return []


needs_endpoints = pytest.mark.skipif(not _served(), reason="no vLLM endpoints served")


# --------------------------------------------------------------------------- families / verifier
@pytest.mark.parametrize("family", families.FAMILIES_16)
def test_verifier_scores_gold_one_and_junk_zero(family):
    e = families.entry(family, 4242)
    assert e["answer"] is not None, f"{family} has no gold answer"
    a = families.adapter(family)
    assert a.score(str(e["answer"]), e) == pytest.approx(1.0)
    assert a.score("zzz_not_an_answer", e) < 0.99
    assert families.correct(family, 4242, str(e["answer"])) == 1
    assert families.correct(family, 4242, "zzz_not_an_answer") == 0


@pytest.mark.parametrize("family", families.FAMILIES_16)
def test_instance_text_is_deterministic(family):
    assert families.question(family, 4242) == families.question(family, 4242)


def test_different_instances_give_different_problems():
    assert len({families.question("basic_arithmetic", i) for i in range(20)}) > 10


def test_score_never_raises_on_malformed_answers():
    # prime_factorization's verifier calls int() on the answer: junk is wrong, not a crash.
    for bad in ["", "   ", "I don't know", "3 x 5 x ???"]:
        assert families.correct("prime_factorization", 77, bad) == 0


def test_difficulty_params_are_applied():
    """Stock `basic_arithmetic` reaches 6 terms and 4 digits and scored 0.25 on the 7B; the cap
    in families.PARAMS is what puts a specialty family in the 0.70-0.95 band."""
    assert families.PARAMS["basic_arithmetic"] == {"max_terms": 3, "max_digits": 2}
    for i in range(30):
        q = families.question("basic_arithmetic", i)
        assert sum(c.isdigit() for c in q.replace(" ", "")) <= 8      # <=4 terms x 2 digits
    assert families.adapter("basic_arithmetic").params == (("max_digits", 2), ("max_terms", 3))
    assert families.adapter("gcd").params == ()


def test_dead_families_are_out_of_the_k16_list():
    for dead in ("leg_counting", "caesar_cipher", "base_conversion", "bitwise_arithmetic",
                 "spell_backward", "word_sorting"):
        assert dead not in families.FAMILIES_16 and dead in families.FAMILIES_64


def test_family_lists_are_unique_and_sized():
    assert len(families.names(16)) == 16
    assert len(families.FAMILIES_64) == len(set(families.FAMILIES_64)) == 64
    assert "propositional_logic" not in families.FAMILIES_64      # answer=None, unsolvable
    assert "graph_color" not in families.FAMILIES_64


# --------------------------------------------------------------------------- population
def test_specialist_has_exactly_three_unhandicapped_families():
    p = draw_profiles(200, 16, "specialist", seed=1)
    assert all(len(x["specialty"]) == 3 for x in p)
    assert all(x["model"] in LADDER and x["tool"] in tools.NAMES for x in p)


def test_python_is_gated_by_size_and_withheld_off_specialties():
    """`calculator` measured at or below the no-tool arm, so the draw collapsed to python. python
    itself is gated at `tool_min_b`: MEASURED, it makes smaller models worse (the 1.5B scored 0.30
    with it on chain_sum against 0.65 without). Above the gate `signature` still withholds it on
    handicapped families -- that is SPEC §1's "family tool removed"."""
    cfg = ladder()
    size = {m["id"]: m["params_b"] for m in cfg["models"]}
    assert TOOL == "python" and cfg["tool_min_b"] == 3.0
    for p in draw_profiles(200, 16, "specialist", seed=1):
        want = "python" if size[p["model"]] >= cfg["tool_min_b"] else "none"
        assert p["tool"] == want
        for f in range(16):
            assert signature(p, f, 512, 512)[2] == (want if f in p["specialty"] else "none")


def test_signature_count_is_two_per_model():
    sigs = {signature(p, f, 512, 512) for p in draw_profiles(300, 16, "specialist", 1)
            for f in range(16)}
    assert len(sigs) == 2 * len(LADDER)      # one specialty + one handicapped per model


def test_profile_draw_is_seeded():
    assert draw_profiles(50, 16, "specialist", 7) == draw_profiles(50, 16, "specialist", 7)
    assert draw_profiles(50, 16, "specialist", 7) != draw_profiles(50, 16, "specialist", 8)


def test_heavy_tail_and_bimodal_shapes():
    ht = draw_profiles(2000, 16, "heavy_tail", 3)
    experts = [x for x in ht if x["specialty"]]
    assert 0.06 < len(experts) / len(ht) < 0.15
    # heavy_tail experts are drawn from `big`, all of which clear the tool gate
    assert all(x["tool"] == "python" for x in ht if x["specialty"])

    bm = draw_profiles(2000, 16, "bimodal", 5)
    good = [x for x in bm if x["specialty"]]
    assert 0.15 < len(good) / len(bm) < 0.26
    assert len({x["model"] for x in bm}) == 2


def test_correlated_specialty_is_group_level():
    for p in draw_profiles(100, 16, "correlated", 2):
        for g in {f % 4 for f in p["specialty"]}:
            assert all(f in p["specialty"] for f in range(16) if f % 4 == g)


def test_iid_uniform_is_unstructured():
    sizes = np.array([len(p["specialty"]) for p in draw_profiles(500, 16, "iid_uniform", 4)])
    assert 6 < sizes.mean() < 10 and sizes.std() > 1.0


def test_unknown_dist_raises():
    with pytest.raises(ValueError):
        draw_profiles(10, 4, "nope", 0)


def test_model_ids_come_only_from_config():
    cfg = ladder()
    assert {m["id"] for m in cfg["models"]} == set(LADDER)
    assert all(m["gpu_share"] > 0 and m["params_b"] > 0 for m in cfg["models"])


# --------------------------------------------------------------------------- prompts / handicap
def test_handicap_withholds_exemplar_description_and_tool():
    p = {"id": 0, "model": LADDER[0], "specialty": [0], "tool": "python"}
    assert signature(p, 0, 512, 512) == (LADDER[0], False, "python", 512)
    assert signature(p, 1, 512, 512) == (LADDER[0], True, "none", 512)

    ok = prompts.build("gcd", "Q?", handicapped=False, tool="python")[0]["content"]
    hand = prompts.build("gcd", "Q?", handicapped=True, tool="none")[0]["content"]
    assert "Worked example" in ok and "Worked example" not in hand
    assert "gcd" in ok and "gcd" not in hand
    assert "Python sandbox" in ok and "Python sandbox" not in hand
    assert "<answer>" in ok and "<answer>" in hand      # format instruction is NOT the handicap


def test_token_budget_cap_is_opt_in():
    """It measured NON-MONOTONE (helped a 0.5B on syllogism), so it must be off unless asked."""
    kw = dict(K=2, dist="specialist", seed=1, families=["gcd", "basic_arithmetic"])
    assert LLMBackend(n=1, **kw).handicap_max_tokens == 512
    assert LLMBackend(n=1, handicap_max_tokens=160, **kw).handicap_max_tokens == 160


def test_equal_signatures_give_identical_prompts():
    """The memo's correctness condition."""
    p1 = {"id": 0, "model": LADDER[2], "specialty": [0, 1, 2], "tool": "calculator"}
    p2 = {"id": 9, "model": LADDER[2], "specialty": [2, 5, 7], "tool": "calculator"}
    assert signature(p1, 2, 512, 512) == signature(p2, 2, 512, 512)
    assert signature(p1, 3, 512, 512) != signature(p1, 2, 512, 512)


@pytest.mark.parametrize("text,want", [
    ("blah <answer>42</answer>", "42"),
    ("<answer>1</answer> then <answer>2</answer>", "2"),
    ("<ANSWER> 7 </ANSWER>", "7"),
    ("no tags here\nfinal line", "final line"),
    ("", ""),
])
def test_extract_answer(text, want):
    assert prompts.extract_answer(text) == want


def test_find_tool_call_skips_when_already_answered():
    assert prompts.find_tool_call("<calc>1+1</calc>", "calculator") == "1+1"
    assert prompts.find_tool_call("<calc>1+1</calc><answer>2</answer>", "calculator") is None
    assert prompts.find_tool_call("```python\nprint(1)\n```", "python").strip() == "print(1)"
    assert prompts.find_tool_call("<calc>1+1</calc>", "none") is None


@pytest.mark.parametrize("text,want", [("<answer>0.8</answer>", 0.8), ("70%", 0.7), ("no idea", 0.5)])
def test_parse_rating(text, want):
    assert prompts.parse_rating(text) == pytest.approx(want)


# --------------------------------------------------------------------------- tools
def test_calculator_evaluates_and_refuses_non_arithmetic():
    assert tools.calculator("2 + 3 * 4") == "14"
    assert tools.calculator("2^10") == "1024"
    for bad in ("__import__('os').system('echo hi')", "open('/etc/passwd')", "9 ** 9999"):
        assert tools.calculator(bad).startswith("ERROR")


def test_python_tool_runs_and_contains_failures():
    assert tools.python("print(6*7)") == "42"
    assert tools.python("while True: pass").startswith("ERROR: timed out")
    assert "network disabled" in tools.python(
        "import socket\ntry:\n socket.socket()\nexcept OSError as e:\n print(e)")


# --------------------------------------------------------------------------- mock fleet
class _Mock(BaseHTTPRequestHandler):
    """Answers `<answer>42</answer>`; a calculator agent's FIRST turn gets a <calc> block back so
    the tool round is exercised; the supervisor gets whatever `supervisor_reply` says."""
    calls = 0
    supervisor_reply = None

    def log_message(self, *a):
        pass

    def do_GET(self):
        self._send({"ok": True})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        msgs = req.get("messages", [])
        system = msgs[0]["content"] if msgs else ""
        tooled = any("Tool result:" in (m.get("content") or "") for m in msgs)
        if "routing supervisor" in system and type(self).supervisor_reply is not None:
            content = type(self).supervisor_reply
        elif "<calc>" in system and not tooled:
            content = "Let me compute. <calc>21*2</calc>"
        else:
            content = "<answer>42</answer>"
        type(self).calls += 1
        self._send({"choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": content}}]})

    def _send(self, body):
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture()
def fleet(tmp_path, monkeypatch):
    from rte import llm_client
    from rte.methods.frameworks import _common
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Mock)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    (tmp_path / "endpoints.json").write_text(json.dumps({m: url for m in LADDER}))
    monkeypatch.setattr(llm_client, "ENDPOINTS_PATH", tmp_path / "endpoints.json")
    monkeypatch.setattr(llm_client, "ENDPOINT_DIR", tmp_path / "endpoints.d")
    monkeypatch.setattr(llm_client, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(llm_client, "_mem", None)
    monkeypatch.setattr(llm_client, "_shard", None)
    monkeypatch.setattr(llm_client, "_clients", {})
    monkeypatch.setattr(_common, "RTE_DATA", str(tmp_path))       # its own _endpoint() lookup
    llm_client.reset_stats()
    _Mock.calls, _Mock.supervisor_reply = 0, None
    yield llm_client
    srv.shutdown()


def test_memo_returns_the_identical_answer_and_counts_a_hit(fleet):
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    a = fleet.complete(LADDER[0], msgs, 32, cache_key="k1")
    b = fleet.complete(LADDER[0], msgs, 32, cache_key="k1")
    assert a == b == "<answer>42</answer>"
    s = fleet.stats()
    assert s["generations"] == 1 and s["hits"] == 1 and s["cache_hit_rate"] == 0.5


def test_complete_batch_deduplicates_repeated_keys(fleet):
    """40 requests, 2 distinct keys -> 2 generations. The whole measurement budget rests on this."""
    msgs = [[{"role": "user", "content": f"u{i % 2}"}] for i in range(40)]
    out = fleet.complete_batch(LADDER[0], msgs, [f"key{i % 2}" for i in range(40)], 32)
    assert out == ["<answer>42</answer>"] * 40
    assert fleet.stats()["generations"] == 2 and _Mock.calls == 2


def test_backend_end_to_end_and_tool_round(fleet):
    from rte.world import Task
    b = LLMBackend(n=4, K=2, dist="specialist", seed=1, families=["basic_arithmetic", "gcd"])
    assert current_backend() is b
    assert b.execute(0, Task(0, 0, 999)) in (0, 1)
    out = b.execute_many(np.arange(4)[:, None], np.zeros((4, 1), int), np.arange(2)[None, :] + 10 * np.arange(4)[:, None])  # (np.random.default_rng(0))
    assert out.shape == (4, 2) and set(np.unique(out)) <= {0, 1}

    # a calculator agent: one <calc> reply, we evaluate it, one follow-up turn -> 2 server calls
    b.profiles[0] = {"id": 0, "model": LADDER[0], "specialty": [0], "tool": "calculator"}
    calls, tool_calls = _Mock.calls, b.stats()["llm_tool_calls"]
    b.execute(0, Task(0, 0, 555))
    assert b.stats()["llm_tool_calls"] == tool_calls + 1 and _Mock.calls == calls + 2


def test_text_accessors(fleet, tmp_path):
    from rte.world import Task
    b = LLMBackend(n=3, K=2, dist="specialist", seed=1, families=["basic_arithmetic", "gcd"],
                   population_dir=str(tmp_path / "pop"))
    d = b.descriptions()
    assert len(d) == 3 and all(isinstance(x, str) and x for x in d)
    assert b.descriptions() is d                                  # cached
    assert len(b.family_descriptions()) == 2
    assert b.task_text(Task(0, 1, 7)) == families.question("gcd", 7)


def test_no_endpoints_raises_a_clear_error(tmp_path, monkeypatch):
    from rte import llm_client
    monkeypatch.setattr(llm_client, "ENDPOINTS_PATH", tmp_path / "nope.json")
    monkeypatch.setattr(llm_client, "ENDPOINT_DIR", tmp_path / "nope.d")
    with pytest.raises(llm_client.NoEndpointsError, match="no vLLM endpoints"):
        llm_client.endpoints()


# --------------------------------------------------------------------------- llm_supervisor
def test_llm_supervisor_end_to_end_and_ledger(fleet, tmp_path):
    from rte.budget import Budget
    from rte.methods.llm_supervisor import LLMSupervisor
    from rte.world import World

    w = World(n=6, K=2, dist="specialist", beta=0.0, seed=1, backend="llm",
              backend_kwargs={"families": ["basic_arithmetic", "gcd"], "measure_probes": 2,
                              "measure_probes_large": 2, "population_dir": str(tmp_path / "p"),
                              "concurrency": 4})
    view = w.view({"declared"})
    m = LLMSupervisor(k=4)
    before = view.ledger.snapshot()
    m.build(view, Budget())
    assert view.ledger.diff(before)["messages"] == w.n      # n declarations collected at build

    before = view.ledger.snapshot()
    a = m.fetch(w.tasks(1)[0])
    d = view.ledger.diff(before)
    assert 0 <= a < w.n
    assert (d["comparisons"], d["hops"], d["messages"]) == (4, 1, 6)   # k, 1, k + 2
    assert d["probes"] == d["reports"] == 0                 # declared-channel method


def test_llm_supervisor_picks_the_index_the_model_names(fleet, tmp_path):
    from rte.budget import Budget
    from rte.methods.llm_supervisor import LLMSupervisor
    from rte.world import World

    _Mock.supervisor_reply = "2"
    w = World(n=6, K=2, dist="specialist", beta=0.0, seed=4, backend="llm",
              backend_kwargs={"families": ["basic_arithmetic", "gcd"], "measure_probes": 1,
                              "measure_probes_large": 1, "population_dir": str(tmp_path / "p2"),
                              "concurrency": 4})
    view = w.view({"declared"})
    m = LLMSupervisor(k=5)
    m.build(view, Budget())
    task = w.tasks(1)[0]
    assert m.fetch(task) == int(m.retrieve(task)[2])
    assert m.stats["picks"] == 1


# --------------------------------------------------------------------------- live (skipped w/o fleet)
@pytest.fixture()
def live_ladder(monkeypatch):
    """Pin the ladder to what is actually served, so these work against a one-model smoke fleet."""
    from rte.backends import llm, population
    served = _served()
    if not served:
        pytest.skip("fleet went away between collection and this test")   # servers register/deregister
    cfg = population.pinned_cfg(served)
    for mod in (population, llm):            # llm.py imported the name, so patch both bindings
        monkeypatch.setattr(mod, "ladder", lambda cfg=cfg: cfg)
    return served


@needs_endpoints
def test_live_execution(live_ladder):
    from rte.world import Task
    b = LLMBackend(n=2, K=2, dist="specialist", seed=1, families=["basic_arithmetic", "gcd"])
    assert b.execute(0, Task(0, 0, 12345)) in (0, 1)
    assert b.stats()["llm_executions"] == 1


@needs_endpoints
def test_live_execute_many_shape(live_ladder):
    b = LLMBackend(n=3, K=2, dist="specialist", seed=1, families=["basic_arithmetic", "gcd"])
    out = b.execute_many(np.array([0, 1, 2])[:, None], np.zeros((3, 1), int), np.arange(2)[None, :] + 7 * np.arange(3)[:, None])  # (np.random.default_rng(0))
    assert out.shape == (3, 2) and set(np.unique(out)) <= {0, 1}


def test_measurement_instances_do_not_depend_on_the_population_seed(tmp_path, monkeypatch):
    """S is a property of the prompt SIGNATURE, so the probe set is fixed project-wide: the ~86k
    measurement generations are produced once and every population at every grid seed and every n
    is served from the memo."""
    seen = {}

    def record(self, items):
        seen.setdefault(self.seed, []).extend(i for _, _, i in items)
        return np.zeros(len(items), dtype=np.int8)

    monkeypatch.setattr(LLMBackend, "_outcomes", record)
    for seed in (1, 7):
        LLMBackend(n=2, K=2, dist="specialist", seed=seed, families=["gcd", "basic_arithmetic"],
                   measure_probes=5, measure_probes_large=5,
                   population_dir=str(tmp_path / f"s{seed}")).true_skill()
    assert seen[1] == seen[7] and len(set(seen[1])) == 2 * 5      # 2 families x 5 probes, shared


def test_memo_shards_are_per_process_and_read_by_later_processes(tmp_path):
    """The live grid runs one process per method against the same cache directory. Each writes
    only its own shard; a process reads every shard that exists when it starts."""
    import subprocess
    import sys
    write = ("import sys; sys.path.insert(0, %r)\n"
             "from rte import llm_client as c\n"
             "c.CACHE_DIR = __import__('pathlib').Path(%r)\n"
             "c._mem = c._shard = None\n"
             "mem, shard = c._memo()\n"
             "shard.execute('INSERT OR REPLACE INTO memo VALUES (?,?)', (sys.argv[1], sys.argv[2]))\n"
             "shard.commit()\n"
             "print(len(list(__import__('pathlib').Path(%r).glob('*.sqlite'))))\n")
    root = str(pathlib.Path(__file__).resolve().parents[1])
    cache = str(tmp_path / "cache")
    for k, v in (("k_a", "from_A"), ("k_b", "from_B")):
        r = subprocess.run([sys.executable, "-c", write % (root, cache, cache), k, v],
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr

    shards = sorted(p.name for p in (tmp_path / "cache").glob("*.sqlite"))
    assert len(shards) == 2 and all(n.startswith("memo_") for n in shards)

    from rte import llm_client
    llm_client.CACHE_DIR = tmp_path / "cache"
    llm_client._mem = llm_client._shard = None
    try:
        mem, shard = llm_client._memo()
        assert mem["k_a"] == "from_A" and mem["k_b"] == "from_B"   # third process reads both
        assert len(list((tmp_path / "cache").glob("*.sqlite"))) == 3   # and added its own
        assert llm_client.compact() == 2
        left = list((tmp_path / "cache").glob("*.sqlite"))
        assert [p.name for p in left] == ["memo_compact.sqlite"]
    finally:
        llm_client.CACHE_DIR = llm_client.RTE_DATA / "cache"
        llm_client._mem = llm_client._shard = None
