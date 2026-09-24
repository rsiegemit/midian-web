"""verbal_confidence: refuses non-LLM backends, charges 2k messages + k comparisons per fetch, parses verbal ratings,
routes to the most confident candidate (first maximum), and the liar's inflated confidence lives in the World.
The LLM is mocked by a `confidence` method on the bernoulli backend that rates each agent round(10 * S)."""
import numpy as np
import pytest

from rte.backends.prompts import CONF_MAX, build, rate_task
from rte.budget import Budget
from rte.methods import load_method
from rte.methods.verbal_confidence import parse_confidence
from rte.world import AccessError, World

K, B = 8, Budget(3)


def mocked(beta=0.0, lie_mode="inflate", n=60, garbage=()):
    w = World(n, K, "specialist", beta, liar_select="low_skill_first", seed=3, lie_mode=lie_mode)
    w.asked = []

    def confidence(agents, f, inst):
        w.asked += list(agents)
        return ["I cannot say" if a in garbage else f"<answer>{int(round(10 * w.S[a, f]))}</answer>" for a in agents]
    w.backend.confidence = confidence
    return w


def built(w, **p):
    m = load_method("verbal_confidence")(**p); m.build(w.view(m.needs), B)
    return m


def test_refuses_non_llm_backends():
    assert load_method("verbal_confidence").requires_llm
    w = World(60, K, "specialist", 0.0, seed=3)
    m = built(w)
    with pytest.raises(NotImplementedError):
        m.fetch(w.tasks(1)[0])


def test_ledger_build_n_messages_fetch_2k_messages_and_k_comparisons():
    w = mocked(); s = w.ledger.snapshot(); m = built(w, k=10)
    assert w.ledger.diff(s)["messages"] == w.n
    for t in w.tasks(20):
        s = w.ledger.snapshot(); m.fetch(t); d = w.ledger.diff(s)
        assert (d["messages"], d["comparisons"], d["probes"], d["reports"]) == (20, 10, 0, 0)
    assert m.stats["asked"] == 200


@pytest.mark.parametrize("text,want", [
    ("<answer>7</answer>", 0.7), ("<answer>10</answer>", 1.0), ("0", 0.0), ("8/10", 0.8), ("7 out of 10", 0.7),
    ("I'd say 85%", 0.85), ("<answer>0.6</answer>", 0.6), ("1.0", 1.0), ("<answer>8.5</answer>", 0.85),
    ("think 3 ... <answer>9</answer>", 0.9), (CONF_MAX, 1.0),
    ("On a scale of 0 to 10, I'd say 8", 0.8), ("85", 0.85), ("<10></answer>", 1.0), ("<answer>7</answer> (out of 10)", 0.7),
    ("", None), (None, None), ("no idea", None), ("<answer>eleven</answer>", None), ("-3", None), ("<answer>-3</answer>", None),
    ("5/0", None), ("150%", None), ("250", None)])
def test_parse_confidence(text, want):
    got = parse_confidence(text)
    assert got == pytest.approx(want) if want is not None else got is None


def test_routes_to_most_confident_first_maximum_and_counts_garbage():
    w = mocked(garbage={0, 1, 2}); m = built(w, k=10); D = w.D
    for t in w.tasks(30):
        cand = np.argsort(-D[:, t.family], kind="stable")[:10]
        c = np.array([0.0 if a in (0, 1, 2) else round(10 * w.S[a, t.family]) / 10 for a in cand])
        assert m.fetch(t) == cand[np.argmax(c)]
    assert m.stats["unparseable"] == sum(a in (0, 1, 2) for a in w.asked)


@pytest.mark.parametrize("mode", ["inflate", "squat", "max"])
def test_liar_confidence_is_max_under_every_lie_mode_and_honest_untouched(mode):
    w, h = mocked(beta=0.5, lie_mode=mode), mocked()
    agents, t = np.arange(w.n), w.tasks(1)[0]
    lied, honest = w.confidence(agents, t), h.confidence(agents, t)
    assert w.liars.any() and all(lied[a] == CONF_MAX for a in agents[w.liars])
    assert all(lied[a] == honest[a] for a in agents[~w.liars])
    assert set(w.asked) == set(agents[~w.liars].tolist())        # liars are never asked: no generation spent on them


def test_never_touches_beta_liars_or_S_and_uses_every_need():
    w = mocked(beta=0.5); m = load_method("verbal_confidence")(); v = w.view(m.needs)
    for attr in ("beta", "liars", "S"):
        with pytest.raises(AccessError): getattr(v, attr)
    m.build(v, B); [m.fetch(t) for t in w.tasks(20)]                # runs through the guarded View only
    for drop in m.needs:
        m2 = load_method("verbal_confidence")()
        with pytest.raises(AccessError):
            m2.build(w.view(m.needs - {drop}), B); m2.fetch(w.tasks(1)[0])


def test_deterministic():
    picks = [[m.fetch(t) for t in w.tasks(30)] for w in (mocked(beta=0.5), mocked(beta=0.5)) for m in [built(w, k=5)]]
    assert picks[0] == picks[1]


def test_rate_task_is_the_agents_own_solve_prompt():
    for hand, tool in ((False, "python"), (True, "none")):
        solve, ask = build("gcd", "Q?", hand, tool), rate_task("gcd", "Q?", hand, tool)
        assert ask[0] == solve[0] and ask[1]["content"].startswith("Q?") and "0 (certainly wrong) to 10" in ask[1]["content"]


def test_cartel_wins_whenever_a_liar_is_shortlisted():
    w = mocked(beta=0.5, lie_mode="max"); m = built(w, k=10)
    for t in w.tasks(30):
        cand = np.argsort(-w.D[:, t.family], kind="stable")[:10]
        if w.liars[cand].any(): assert m.fetch(t) == cand[w.liars[cand]][0]


# all_methods() computed at 0570629; equal at master 8c38981 (no method file added in between)
ALL_AT_0570629 = ["cluster_head_router", "cnp_self_bid", "declared_argmax", "declared_softmax", "disrouter_cascade", "flat_nsw_router",
                  "flat_probe_argmax", "gossip_reputation_greedy", "knn_router", "linucb_honest", "llm_supervisor", "midian",
                  "midian_llm_descent", "mlp_router", "random", "referral_network", "route_to_k_majority", "sequential_halving",
                  "thompson_per_family", "trueskill_per_family", "ucb_per_family", "verify_on_claim", "warm_start_bandit"]


@pytest.mark.parametrize("backend", ["llm", "bernoulli"])
def test_methods_all_expansion_unchanged(backend):
    """`methods: all` grids must not grow a verbal_confidence arm: their stored units stay complete."""
    from rte.run import all_methods
    want = [m for m in ALL_AT_0570629 if backend == "llm" or m not in ("llm_supervisor", "midian_llm_descent")]
    assert all_methods(backend) == want


def test_rejects_unknown_shortlist_and_params():
    cls = load_method("verbal_confidence")
    with pytest.raises(ValueError): cls(shortlist="tfidf")
    with pytest.raises(TypeError): cls(shortlst="embed")


def test_llm_backend_asks_each_agent_its_own_model_and_prompt_deduplicated(monkeypatch, tmp_path):
    """The real path below the World: LLMBackend.confidence -> llm_client.complete_batch, with only the network stubbed."""
    from rte import llm_client
    from rte.backends import families, llm as L
    monkeypatch.setattr(L, "_CURRENT", L._CURRENT)                     # restored after: other tests read current_backend()
    shard = type("Shard", (), {"execute": lambda *a: None, "executemany": lambda *a: None, "commit": lambda *a: None})()
    monkeypatch.setattr(llm_client, "_memo", lambda: ({}, shard)); monkeypatch.setattr(llm_client, "_refresh", lambda force=False: None)
    sent = []
    monkeypatch.setattr(llm_client, "_generate", lambda model, msgs, mt: sent.append((model, msgs, mt)) or "<answer>7</answer>")
    be = L.LLMBackend(n=40, K=16, dist="specialist", seed=1, population_dir=str(tmp_path))
    agents, f, inst = np.arange(40), 3, 7
    fam = be.families[f]; q = families.question(fam, inst)
    assert be.confidence(agents, f, inst) == ["<answer>7</answer>"] * 40
    want = {(s[0], str(rate_task(fam, q, s[1], s[2]))) for s in (be._sig(int(a), f) for a in agents)}
    assert {(m, str(msgs)) for m, msgs, _ in sent} == want and len(sent) == len(want) < 40   # one generation per signature
    assert {mt for *_, mt in sent} == {32}
