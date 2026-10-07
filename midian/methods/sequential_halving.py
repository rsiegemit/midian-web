"""Per-family fixed-budget best-arm identification (sequential halving) over all agents.

Mechanism: per family, ceil(log2 n) rounds on a budget of n*b probes: probe the surviving agents equally, keep the
better half. With peer_reported=True the router is not a trusted observer: each probe outcome reaches it only as the
reports of r-1 random other agents, aggregated by a per-reporter trimmed mean (MIDIAN's channel). Fetch is a cached
lookup. audit (peer_reported only, 2026-10-07 review ablation: halving + MIDIAN's audits, no tree): a uniform `audit`
rate (True = 5%, MIDIAN's) of probe instances is re-run by the auditor (`view.probe_at`, charged as a probe) and every
peer's report about it compared with the truth; a reporter with STRIKES mismatches is excluded from every later
aggregation (MIDIAN's rule). The tournament then gets (1 - rate) of the budget, so audits stay inside n*K*b.

Ledger: build <= n*K*b probes (+ r-1 reports per probe when peer_reported); fetch = 1 comparison; observe = 0.
Churn: churn_mode="rebuild" reruns the whole build (charged); "stale" keeps the old picks.
Params: peer_reported=False, r=10, delta=1/3, churn_mode="stale", audit=False."""
import math

import numpy as np

from ._est import CHUNK, lookup, trimmed_by_reporter
from .base import Method
from .midian import AUDIT_RATE, STRIKES


class Auditor:
    """MIDIAN's report audits over a whole build: `rate` of instances re-run, STRIKES mismatches exclude a reporter."""

    def __init__(self, n, rate):
        self.rate, self.hits, self.excluded = rate, np.zeros(n, np.int32), np.zeros(n, bool)

    def audit(self, view, a, f, k0, rep, reported):
        """rep[V, peers], reported[V, peers, pulls] about agents a[V] (pulls k0[V] + j); returns probes spent."""
        v, j = np.nonzero(view.rng.random((a.size, reported.shape[2])) < self.rate)
        if v.size:
            truth = view.probe_at(a[v], f, k0[v] + j)
            np.add.at(self.hits, rep[v][reported[v, :, j] != truth[:, None]], 1)
            self.excluded |= self.hits >= STRIKES
        return v.size


def halving(view, f, budget, peers=0, delta=1 / 3, auditor=None):
    """Sequential halving over all agents for family f. Returns (best agent, probes used).
    peers>0: the router is NOT a trusted observer — each probe outcome reaches it only as the reports of `peers`
    random other agents (liars may corrupt them), aggregated by a per-reporter trimmed mean, exactly the channel of
    MIDIAN w/o audits. `auditor` (peers>0 only): audit the reports, mask excluded reporters before trimming."""
    n = view.n
    alive, tot, cnt, used, audits = np.arange(n), np.zeros(n), np.zeros(n, np.int64), 0, 0
    rounds = max(1, math.ceil(math.log2(n)))
    for r in range(rounds):
        pulls = max(1, (budget - used) // (alive.size * (rounds - r)))
        if alive.size == 1 or pulls * alive.size > budget - used:
            break
        for lo in range(0, alive.size, CHUNK):
            a = alive[lo:lo + CHUNK]
            out = view.probe_many(a, f, pulls)
            if peers:
                # random peers, never self
                rep = (a[:, None] + view.rng.integers(1, n, (a.size, peers))) % n
                reported = view.report_many(rep[:, :, None], a[:, None, None], out[:, None, :], f)
                ex = None
                if auditor is not None:
                    audits += auditor.audit(view, a, f, cnt[a], rep, reported)
                    ex = auditor.excluded[rep]
                    ex &= ~ex.all(-1, keepdims=True)               # every peer excluded: all count (MIDIAN's rule)
                out = trimmed_by_reporter(reported, delta, peers + 1, ex)[:, None] * pulls
            tot[a] += out.sum(1)
            cnt[a] += pulls
        used += alive.size * pulls
        alive = alive[np.argsort(-tot[alive] / cnt[alive], kind="stable")[:max(1, alive.size // 2)]]
    return int(alive[np.argmax(tot[alive] / np.maximum(cnt[alive], 1))]), used + audits


class SequentialHalving(Method):
    name = "sequential_halving"
    needs = frozenset({"probe"})

    def __init__(self, peer_reported=False, r=10, delta=1 / 3, churn_mode="stale", audit=False, **p):
        super().__init__(peer_reported=peer_reported, r=r, delta=delta, churn_mode=churn_mode, audit=audit, **p)
        self.peers, self.delta, self.churn_mode = (r - 1) if peer_reported else 0, delta, churn_mode
        self.rate = (AUDIT_RATE if audit is True else float(audit)) if audit else 0.0
        if self.rate and not peer_reported:
            raise ValueError("audit needs peer_reported=True (a trusted observer has no reports to audit)")
        if peer_reported:
            self.needs = self.needs | {"reports"}                       # no trusted observer: MIDIAN's channel

    def build(self, view, budget):
        self.view, self.budget = view, budget
        aud = Auditor(view.n, self.rate) if self.rate else None
        per_f = int(view.n * budget.b * (1 - self.rate))                  # audits are paid from the same budget
        res = [halving(view, f, per_f, self.peers, self.delta, aud) for f in range(view.K)]
        self.stats = {"excluded": int(aud.excluded.sum())} if aud else {}
        self.best = np.array([a for a, _ in res])
        assert sum(u for _, u in res) <= budget.total_probes(view.n, view.K)

    def fetch(self, task):
        return lookup(self.view, self.best, task.family)

    def churn(self, departed, arrived):
        """rebuild: the full n*K*b halving again at every churn event (charged); stale: keep the old picks."""
        if self.churn_mode == "rebuild":
            self.build(self.view, self.budget)
