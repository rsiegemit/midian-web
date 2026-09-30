"""Endpoint choice across replicas: no herding onto one endpoint, failures count as latency."""

import random

from midian import llm_client as lc


def test_pick_url_spreads_over_replicas(monkeypatch):
    urls = ["http://a/v1", "http://b/v1", "http://c/v1"]
    monkeypatch.setattr(lc, "_lat", {"http://a/v1": 0.1, "http://b/v1": 5.0, "http://c/v1": 9.0})
    random.seed(0)
    picks = [lc.pick_url(urls) for _ in range(3000)]
    share = {u: picks.count(u) / len(picks) for u in urls}
    assert 0.6 < share["http://a/v1"] < 0.73  # the fastest wins whenever sampled (2 of 3 pairs), never always
    assert share["http://c/v1"] == 0.0        # the slowest never beats a sampled peer


def test_pick_url_single_endpoint():
    assert lc.pick_url(["http://only/v1"]) == "http://only/v1"


def test_observe_counts_slow_failures(monkeypatch):
    monkeypatch.setattr(lc, "_lat", {"http://a/v1": 0.5})
    lc._observe("http://a/v1", 600.0)  # a request that timed out
    assert lc._lat["http://a/v1"] > 100.0
