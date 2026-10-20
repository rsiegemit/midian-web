"""De-cloned live populations (LLMBackend declone, 2026-10-16): prompt variants make same-model agents differ in skill,
per-agent seeded sampling makes them differ in answers; a plain population (declone=0) is unchanged."""
from __future__ import annotations

import numpy as np
import pytest

from midian import llm_client
from midian.backends import prompts
from midian.backends.llm import LLMBackend
from midian.backends.population import draw_profiles, signature


@pytest.fixture()
def gen(tmp_path, monkeypatch):
    """No fleet: _generate records (model, temperature, seed, system prompt) and answers 42."""
    calls = []

    def fake(model, messages, max_tokens, temperature=0.0, seed=0):
        calls.append((model, temperature, seed, messages[0]["content"]))
        return "<answer>42</answer>"

    monkeypatch.setattr(llm_client, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(llm_client, "_mem", None)
    monkeypatch.setattr(llm_client, "_shard", None)
    monkeypatch.setattr(llm_client, "_generate", fake)
    return calls


def test_plain_population_and_prompts_are_unchanged():
    plain, dc = draw_profiles(300, 16, "specialist", 3), draw_profiles(300, 16, "specialist", 3, variants=36)
    assert all("variant" not in p for p in plain)
    assert [{k: v for k, v in p.items() if k != "variant"} for p in dc] == plain   # same models / specialties
    assert {p["variant"] for p in dc} == set(range(36))
    assert len(signature(plain[0], 0, 512)) == 4 and signature(dc[0], 0, 512)[4] == dc[0]["variant"]
    assert prompts.build("gcd", "q", False, "python") == prompts.build("gcd", "q", False, "python", None)
    assert prompts.describe_self("m", "python", "gcd") == prompts.describe_self("m", "python", "gcd", None)


def test_variants_change_the_prompt():
    msgs = {prompts.build("gcd", "q", False, "python", v)[0]["content"] for v in prompts.VARIANTS}
    assert len(msgs) == len(prompts.VARIANTS) // len(prompts.TEMPERATURES)    # temperature is not in the text


def test_greedy_memo_keys_are_unchanged(gen):
    m = [{"role": "user", "content": "u"}]
    k0 = llm_client.content_key("x", m, 8)
    llm_client.complete_batch("x", [m], None, 8)
    assert k0 in llm_client._memo()[0] and gen[-1][1:3] == (0.0, 0)


def test_seeded_samples_dedupe_in_batch_and_skip_the_memo(gen):
    m = [{"role": "user", "content": "u"}]
    llm_client.complete_batch("x", [m, m, m], None, 8, sampling=[(0.7, 1), (0.7, 2), (0.7, 1)])
    assert sorted(c[2] for c in gen) == [1, 2]                       # seed 1 twice in the batch: one generation
    assert not llm_client._memo()[0]                                  # the text memo stays clean


def test_declone_backend_samples_per_agent(gen, tmp_path):
    fams = ["basic_arithmetic", "gcd"]
    b = LLMBackend(n=40, K=2, dist="specialist", seed=1, families=fams, declone=36,
                   population_dir=str(tmp_path / "pop"))
    plain = LLMBackend(n=40, K=2, dist="specialist", seed=1, families=fams)
    assert plain.dir.name == "specialist_n40_K2_seed1" and LLMBackend(
        n=40, K=2, dist="specialist", seed=1, families=fams, declone=36).dir.name.endswith("_dc36")
    b.execute_many(np.arange(40), np.zeros(40, int), np.full(40, 7))          # 40 agents, one instance
    assert len(gen) == 40 and len({c[2] for c in gen}) == 40                  # 40 generations, 40 seeds
    assert {c[1] for c in gen} <= set(prompts.TEMPERATURES)
    for a in range(5):                                                        # each at its own variant's temperature
        assert b._sample(a, 0, 7)[0] == prompts.VARIANTS[b.profiles[a]["variant"]]["temperature"]
    del gen[:]
    plain.execute_many(np.arange(40), np.zeros(40, int), np.full(40, 7))
    assert len(gen) <= 14 and {c[1:3] for c in gen} == {(0.0, 0)}            # plain: clones share generations


def test_declone_descriptions_and_ratings_are_per_agent(gen, tmp_path):
    b = LLMBackend(n=12, K=2, dist="bimodal", seed=1, families=["basic_arithmetic", "gcd"], declone=36,
                   population_dir=str(tmp_path / "pop"))
    b.descriptions()
    b.declared("self_described")
    assert len(gen) == 12 + 12 * 2 and len({c[2] for c in gen}) == len(gen)  # one sample per agent (and family)


def test_measurement_is_shared_across_populations(gen, tmp_path):
    """S is sampled per signature, not per agent or population: a second population re-measures nothing new."""
    fams = ["basic_arithmetic", "gcd"]
    kw = dict(n=30, K=2, dist="bimodal", families=fams, declone=4, measure_probes=5, measure_probes_large=5)
    b1 = LLMBackend(seed=1, population_dir=str(tmp_path / "p1"), **kw)
    b1.true_skill()
    first = len(gen)
    b2 = LLMBackend(seed=2, population_dir=str(tmp_path / "p2"), **kw)
    sigs = lambda b: {b._sig(a, f) for a in range(b.n) for f in range(b.K)}  # noqa: E731
    b2.true_skill()
    assert first > 0 and (len(gen) == first) == (sigs(b2) <= sigs(b1))      # nothing new unless a new signature


def test_outcome_store_serves_every_later_method(gen, tmp_path):
    """Outcomes land in the population's store; a fresh backend (a later job) reads them instead of re-sampling."""
    fams = ["basic_arithmetic", "gcd"]
    kw = dict(n=10, K=2, dist="specialist", seed=1, families=fams, declone=36, population_dir=str(tmp_path / "pop"))
    first = LLMBackend(**kw).execute_many(np.arange(10), np.zeros(10, int), np.full(10, 3))
    n = len(gen)
    again = LLMBackend(**kw).execute_many(np.arange(10), np.zeros(10, int), np.full(10, 3))
    assert n == 10 and len(gen) == n and (first == again).all()
    assert list((tmp_path / "pop" / "outcomes").glob("*.sqlite"))


def test_plain_population_still_uses_the_memo(gen, tmp_path):
    b = LLMBackend(n=10, K=2, dist="specialist", seed=1, families=["basic_arithmetic", "gcd"])
    b.execute_many(np.arange(10), np.zeros(10, int), np.full(10, 3))
    assert llm_client._memo()[0] and b._store is None


def test_declone_tool_round_skips_the_memo(gen, tmp_path, monkeypatch):
    from midian.backends import tools
    seen = []
    monkeypatch.setattr(tools, "_python", lambda code: seen.append(code) or "6")
    monkeypatch.setattr(llm_client, "memo_call", lambda *a, **k: (_ for _ in ()).throw(AssertionError("memo used")))
    replies = iter(["```python\nprint(6)\n```", "<answer>6</answer>"] * 50)
    monkeypatch.setattr(llm_client, "_generate", lambda *a, **k: next(replies))
    b = LLMBackend(n=1, K=1, dist="bimodal", seed=1, families=["basic_arithmetic"], declone=36,
                   population_dir=str(tmp_path / "pop"))
    b.profiles[0].update(specialty=[0], tool="python")
    b.execute_many(np.array([0]), np.array([0]), np.array([3]))
    assert seen == ["print(6)\n"] and b.stats()["llm_tool_calls"] == 1
