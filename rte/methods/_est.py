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


def trimmed_mean(x: np.ndarray, t: int) -> np.ndarray:
    """SPEC §5 trimmed mean along the last axis: sort, drop t from each end, average."""
    x = np.sort(x, axis=-1)
    return x[..., t:x.shape[-1] - t].mean(-1)


def trimmed_by_reporter(rep: np.ndarray, delta: float, s: int, exclude: np.ndarray | None = None):
    """rep[..., s-1, b] -> mean after dropping the floor(delta*(s-1)) highest- and lowest-reporting PEERS
    (a colluding peer corrupts all b of its reports at once, so trimming reports one by one under-trims).
    `exclude[..., s-1]` (bool) drops reporters before trimming (MIDIAN-A audits); the trim then adapts to how many remain."""
    per = rep.mean(-1)
    if exclude is None:
        return trimmed_mean(per, min(int(delta * (s - 1) + 1e-9), (s - 2) // 2))
    per = np.sort(np.where(exclude, np.nan, per), axis=-1)                       # NaN sorts last
    v = (~exclude).sum(-1, keepdims=True)
    t = np.clip(np.minimum(int(delta * (s - 1) + 1e-9), (v - 2) // 2), 0, None)
    cs = np.cumsum(np.nan_to_num(per), -1); hi = np.maximum(v - t, 1)
    tot = np.take_along_axis(cs, hi - 1, -1) - np.where(t > 0, np.take_along_axis(cs, np.maximum(t - 1, 0), -1), 0)
    return (tot / np.maximum(hi - t, 1))[..., 0]


def peer_estimate(view, agents, fams, reps: int, reporters: np.ndarray, delta: float, exclude=None):
    """Probe agents[i] on fams[i] `reps` times; each of reporters[i, :] reports EVERY outcome (one report per probe);
    trimmed per-reporter mean. Charges V*reps probes and V*k*reps reports. Returns (est[V], reports[V, k, reps])."""
    agents, fams = np.asarray(agents), np.asarray(fams)
    out = view.probe_many(agents, fams, reps)                                                 # (V, reps)
    per = view.report_many(reporters[:, :, None], agents[:, None, None], out[:, None, :])     # (V, k, reps)
    return trimmed_by_reporter(per, delta, per.shape[1] + 1, exclude), per


def probe_outcomes(view, b: int) -> np.ndarray:
    """Every agent b times per family: outcomes[n, K, b] (n*K*b probes, chunked)."""
    out = np.empty((view.n, view.K, b), np.float32)
    for lo in range(0, view.n, CHUNK):
        out[lo:lo + CHUNK] = view.probe_many(np.arange(lo, min(view.n, lo + CHUNK))[:, None], np.arange(view.K)[None, :], b)
    return out


def peer_reported_estimates(view, b: int, cohorts: np.ndarray, delta: float, by_reporter: bool = False,
                            observers: int | None = None, outcomes: np.ndarray | None = None) -> np.ndarray:
    """MIDIAN's level-0 estimates (SPEC §5): b probes per (agent, family), each outcome reported by
    every other member of the agent's cohort, aggregated by a trimmed mean. `cohorts` is int32[N, r]
    of agent ids with -1 padding, which (padding being contiguous) can only shorten the last cohort.
    Spends exactly n*K*b probes and sum_c size_c*(size_c-1)*K*b reports, in cohort-sized chunks.
    `by_reporter`: trim by peer (each peer's b reports averaged first), optionally a random `observers` peers.
    `outcomes[n, K, b]`: already-spent probes (stratified cohorts probe before grouping); reports are still charged."""
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
        out = outcomes[ag.ravel()] if outcomes is not None else view.probe_many(ag.reshape(-1, 1), np.arange(K)[None, :], b)   # (C*s, K, b)
        if s == 1:                                                                       # no peers to report
            est[ag.ravel()] = out.mean(2)
            continue
        peers = np.array([[j for j in range(s) if j != m] for m in range(s)], np.int32)   # (s, s-1)
        if by_reporter:                                                                  # every peer reports every probe,
            k = min(observers or s - 1, s - 1)                                           # trimmed by PEER (a random k of them)
            obs = ag[:, peers] if k == s - 1 else np.take_along_axis(
                ag[:, peers], np.argsort(view.rng.random((C, s, s - 1)), 2)[:, :, :k], 2)
            per = view.report_many(obs[:, :, None, :, None], ag[:, :, None, None, None],
                                   out.reshape(C, s, K, 1, b)).reshape(C * s, K, k, b)
            est[ag.ravel()] = trimmed_by_reporter(per, delta, k + 1)
            continue
        rep = view.report_many(ag[:, peers][:, :, None, :, None],                        # reporter j
                               ag[:, :, None, None, None],                               # about member m
                               out.reshape(C, s, K, 1, b)                                # what j saw
                               ).reshape(C * s, K, (s - 1) * b)
        est[ag.ravel()] = trimmed_mean(rep, trim_k(delta, s, b))                         # SPEC §5: trim single reports
    return est

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
