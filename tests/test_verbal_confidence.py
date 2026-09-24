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
    ("", None), (None, None), ("no idea", None), ("<answer>eleven</answer>", None), ("42", None), ("5/0", None), ("150%", None)])
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
