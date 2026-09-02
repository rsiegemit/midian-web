"""Shared estimation helpers. Every probing method spends the same build budget through here."""
from __future__ import annotations
import numpy as np
from .base import Method

CHUNK = 1_000_000


def probe_successes(view, b: int) -> np.ndarray:
    """Probe every agent b times per family (n*K*b probes, chunked). Returns successes float64[n, K]."""
    S = np.zeros((view.n, view.K))                 # float: exact-estimate mocks return fractions
    for f in range(view.K):
        for lo in range(0, view.n, CHUNK):
            hi = min(view.n, lo + CHUNK)
            S[lo:hi, f] = view.probe_many(np.arange(lo, hi), f, b).sum(1)
    return S


class BetaBandit(Method):
    """Thompson sampling over Beta(alpha, beta) per (agent, family); subclasses set the prior in `prior(view)`."""
    needs = frozenset({"probe"})

    def prior(self, view):                       # -> (alpha0, beta0) arrays or scalars
        return 1.0, 1.0

    def build(self, view, budget):
        self.view = view
        a0, b0 = self.prior(view)
        s = probe_successes(view, budget.b)
        self.alpha, self.beta = a0 + s, b0 + (budget.b - s)

    def fetch(self, task):
        self.view.ledger.compare(self.view.n)
        return int(np.argmax(self.view.rng.beta(self.alpha[:, task.family], self.beta[:, task.family])))

    def observe(self, task, agent, outcome):
        (self.alpha if outcome else self.beta)[agent, task.family] += 1


REPORT_ELEMS = 8_000_000       # cap on one report tensor: n*K*b*(r-1) is never materialized at once


def trim_k(delta: float, s: int, b: int) -> int:
    """Reports trimmed from each side for a cohort of size s (SPEC §5), clamped to leave one report."""
    return max(0, min(int(delta * (s - 1) + 1e-9), ((s - 1) * b - 1) // 2))


def peer_reported_estimates(view, b: int, cohorts: np.ndarray, delta: float) -> np.ndarray:
    """MIDIAN's level-0 estimates (SPEC §5): b probes per (agent, family), each outcome reported by
    every other member of the agent's cohort, aggregated by a trimmed mean. `cohorts` is int32[N, r]
    of agent ids with -1 padding, which (padding being contiguous) can only shorten the last cohort.
    Spends exactly n*K*b probes and sum_c size_c*(size_c-1)*K*b reports, in cohort-sized chunks."""
    K, r = view.K, cohorts.shape[1]
    est = np.zeros((view.n, K), np.float32)
    step = max(1, min(CHUNK, REPORT_ELEMS // (K * b * max(r - 1, 1))) // r)
    short = cohorts[-1, -1] < 0
    full = cohorts[:len(cohorts) - short]
    blocks = [full[lo:lo + step] for lo in range(0, len(full), step)]
    if short:
        blocks.append(cohorts[-1][cohorts[-1] >= 0][None, :])
    for ag in blocks:
        C, s = ag.shape
        out = view.probe_many(ag.reshape(-1, 1), np.arange(K)[None, :], b)              # (C*s, K, b)
        if s == 1:                                                                       # no peers to report
            est[ag.ravel()] = out.mean(2)
            continue
        peers = np.array([[j for j in range(s) if j != m] for m in range(s)], np.int32)   # (s, s-1)
        rep = view.report_many(ag[:, peers][:, :, None, :, None],                        # reporter j
                               ag[:, :, None, None, None],                               # about member m
                               out.reshape(C, s, K, 1, b)                                # what j saw
                               ).reshape(C * s, K, (s - 1) * b)
        rep.sort(-1)                                                                     # trimmed mean =
        t = trim_k(delta, s, b)
        est[ag.ravel()] = rep[:, :, t:rep.shape[2] - t].mean(-1)                         # sort, then slice
    return est


def one_observer_reports(view, b: int, pick):
    """Peer-reported estimation for the decentralized methods: probe every agent b times per family
    (n*K*b probes, the whole build budget) and have exactly ONE peer observe and report each outcome,
    so a lying peer corrupts its own picture of the network. `pick(agents, f) -> (reporters[n, b], tag)`
    chooses the observer; `tag` is handed back untouched. Yields (f, agents, reporters, tag, reported)
    once per family, with the whole family reported in one batch."""
    ag = np.arange(view.n)
    for f in range(view.K):
        out = np.concatenate([view.probe_many(ag[lo:lo + CHUNK], f, b)
                              for lo in range(0, view.n, CHUNK)])
        obs, tag = pick(ag, f)
        yield f, ag, obs, tag, view.report_many(obs, ag[:, None], out)


def greedy_walk(view, start: int, depth: int, step) -> int:
    """One decentralized route: from `start`, `depth` times, ask this node's neighbours for their
    scores (`step(cur) -> (neighbour ids, scores)`), move to the best, and return the best agent seen.
    Charges 1 hop, len(neighbours) comparisons and 2 messages per neighbour consulted (ask + answer).
    The walk always takes `depth` steps -- a self-loop is still paid for -- so the cost is exact."""
    cur, best, best_s = int(start), int(start), -np.inf
    for _ in range(depth):
        nb, sc = step(cur)
        view.bus.send_many(2 * len(nb)); view.ledger.compare(len(nb)); view.ledger.hop(1)
        i = int(np.argmax(sc))
        if sc[i] > best_s:
            best, best_s = int(nb[i]), float(sc[i])
        cur = int(nb[i])
    return best


def observed_reports(view, f: int, b: int, observers):
    """Decentralized estimation: probe every agent b times on family f; each outcome is reported by ONE peer chosen by
    `observers(agents, b) -> int[m, b]`. Yields (agents, reporters, reported) per chunk. n*b probes and n*b reports."""
    for lo in range(0, view.n, CHUNK):
        ag = np.arange(lo, min(view.n, lo + CHUNK))
        obs = observers(ag, b)
        yield ag, obs, view.report_many(obs, ag[:, None], view.probe_many(ag, f, b))


def greedy_walk(view, start: int, depth: int, neighbors, score) -> int:
    """Greedy search on a graph: at each hop ask the current node's neighbours (2 messages each), move to the best-scoring
    one; return the best-scoring agent seen. `neighbors(cur) -> ids`, `score(cur, ids) -> values`."""
    cur, best = start, (-np.inf, start)
    for _ in range(depth):
        nb = neighbors(cur); sc = score(cur, nb)
        view.bus.send_many(2 * len(nb)); view.ledger.compare(len(nb)); view.ledger.hop(1)
        i = int(np.argmax(sc)); best = max(best, (float(sc[i]), int(nb[i]))); cur = int(nb[i])
    return best[1]
