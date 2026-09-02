"""Plain MIDIAN (SPEC §5): a tree of cohorts routed by per-family max-summaries.

Leaves are agents grouped into cohorts of `r`.  Level-0 skill estimates come only
from verified probes (`b` per agent-family) reported by the `r-1` cohort peers and
aggregated with a trimmed mean (trim `floor(delta*(r-1))` each side).  Every node
carries, per family, the best estimated skill in its subtree and which child holds
it; `fetch` descends from the root, `ceil(log_r n)` decisions of `r` children each.

Engineering (SPEC §3): the tree is arrays per level, built by padding each level to
a multiple of `r` and reshaping; probes and reports are vectorized and processed in
cohort chunks so no array of size n*K*b*(r-1) is ever materialized.
"""
from __future__ import annotations

import numpy as np

from .base import Method

NEG = np.float32(-np.inf)
PROBE_CHUNK = 1_000_000      # SPEC §3: probe draws in chunks of 1e6 agents
REPORT_ELEMS = 8_000_000     # max elements in one report tensor
TREE_ELEMS = 4_000_000       # max elements in one node-reduction block


class Midian(Method):
    name = "midian"
    needs = frozenset({"probe", "reports"})

    def __init__(self, r=10, delta=1 / 3, online=True, stratify=False, **params):
        super().__init__(r=r, delta=delta, online=online, stratify=stratify, **params)
        self.r, self.delta = int(r), float(delta)
        self.online, self.stratify = bool(online), bool(stratify)
        if self.stratify:
            # Stratified-random cohorts read the declared means once, at build, to form the strata.
            # That is an extra channel, so widen `needs` on this instance (the class default stays
            # {"probe","reports"}); the runner builds the View from the instance's `needs`.
            self.needs = self.needs | {"declared"}
        self._cnt: dict = {}

    # ------------------------------------------------------------------ build
    def build(self, view, budget) -> None:
        self.view, self.n, self.K = view, view.n, view.K
        n, K, r, b = view.n, view.K, self.r, budget.b
        N0 = -(-n // r)
        if self.stratify:                                   # one agent per declared-mean stratum
            order = np.argsort(np.asarray(view.declared).mean(axis=1), kind="stable").astype(np.int32)
            arr = np.full((r, N0), -1, np.int32)
            for i, s in enumerate(np.array_split(order, r)):
                s = s.copy(); view.rng.shuffle(s); arr[i, :s.size] = s
            leaves = np.ascontiguousarray(arr.T)            # padding lands in the last cohort only
        else:
            leaves = np.full(N0 * r, -1, np.int32)
            leaves[:n] = view.rng.permutation(n).astype(np.int32)
            leaves = leaves.reshape(N0, r)
        self.leaves = leaves
        ok = leaves >= 0
        self.leaf_of = np.empty(n, np.int32)
        self.leaf_of[leaves[ok]] = np.broadcast_to(np.arange(N0, dtype=np.int32)[:, None], leaves.shape)[ok]
        self.est = self._estimate(view, budget, leaves)
        self._w0 = max(1, (r - 1) * b - 2 * self._trim(r, b))    # reports behind each build estimate
        self._build_tree(view)

    def _trim(self, s: int, b: int) -> int:
        """Reports trimmed from each side of a cohort of size s (SPEC §5), clamped to leave >=1 report."""
        R = (s - 1) * b
        return 0 if R <= 0 else max(0, min(int(self.delta * (s - 1) + 1e-9), (R - 1) // 2))

    def _estimate(self, view, budget, leaves) -> np.ndarray:
        """est[n,K] from peer-reported probes. Overridable: tests inject exact estimates here."""
        K, r, b = view.K, self.r, budget.b
        est = np.zeros((view.n, K), np.float32)
        short = 0 if leaves[-1, -1] >= 0 else 1              # at most one partly-filled cohort
        per_agent = K * b * max(r - 1, 1)
        step = max(1, max(1, min(PROBE_CHUNK, REPORT_ELEMS // per_agent)) // r)
        full = leaves[:len(leaves) - short]
        for lo in range(0, len(full), step):
            self._cohort_block(view, est, full[lo:lo + step], b)
        if short:
            last = leaves[-1]
            self._cohort_block(view, est, last[last >= 0][None, :], b)
        return est

    def _cohort_block(self, view, est, ag, b) -> None:
        """One chunk of whole cohorts: probe every member, have every peer report every outcome."""
        C, s = ag.shape
        K = view.K
        out = view.probe_many(ag.reshape(-1, 1), np.arange(K)[None, :], b)          # (C*s, K, b)
        if s == 1:                                          # no peers: fall back to the own-probe mean
            est[ag.ravel()] = out.mean(axis=2)
            return
        peers = np.array([[j for j in range(s) if j != m] for m in range(s)], np.int32)   # (s, s-1)
        rep = view.report_many(ag[:, peers][:, :, None, :, None],                  # reporters
                               ag[:, :, None, None, None],                         # subject
                               out.reshape(C, s, K, 1, b))                          # (C,s,K,s-1,b)
        rep = rep.reshape(C * s, K, (s - 1) * b)
        rep.sort(axis=-1)
        t = self._trim(s, b)
        est[ag.ravel()] = rep[:, :, t:rep.shape[2] - t].mean(axis=-1)

    def _tree_level(self, ch, src):
        """ch int32[M,r] child ids (-1 empty), src[*,K] child values -> (summary, best_child, leader slot)."""
        M, r, K = ch.shape[0], self.r, self.K
        summ = np.empty((M, K), np.float32)
        best = np.empty((M, K), np.int32)
        lead = np.empty(M, np.int32)
        step = max(1, TREE_ELEMS // (r * K))
        for lo in range(0, M, step):
            c = ch[lo:lo + step]
            v = np.where(c[:, :, None] >= 0, src[c], NEG)
            bi = v.argmax(axis=1).astype(np.int32)
            best[lo:lo + step] = bi
            summ[lo:lo + step] = np.take_along_axis(v, bi[:, None, :].astype(np.intp), axis=1)[:, 0, :]
            lead[lo:lo + step] = v.mean(axis=2).argmax(axis=1)
        return summ, best, lead

    def _build_tree(self, view) -> None:
        """Level 0 over agents, then random regrouping of nodes into groups of r using summaries only."""
        r = self.r
        self.children, self.summary, self.best_child, self.parent, self.leader = [self.leaves], [], [], [], []
        src, ch = self.est, self.leaves
        while True:
            summ, best, lead = self._tree_level(ch, src)
            rep = ch[np.arange(len(ch)), lead]                       # this node's representative child
            self.leader.append(rep if not self.leader else self.leader[-1][rep])
            self.summary.append(summ); self.best_child.append(best)
            m = len(summ)
            if m == 1:
                break
            N = -(-m // r)
            perm = view.rng.permutation(m).astype(np.int32)
            par = np.empty(m, np.int32); par[perm] = np.arange(m, dtype=np.int32) // r
            self.parent.append(par)
            nxt = np.full(N * r, -1, np.int32); nxt[:m] = perm
            ch = nxt.reshape(N, r)
            self.children.append(ch); src = summ
        self.depth = len(self.summary)

    # ------------------------------------------------------------------ route
    def fetch(self, task) -> int:
        f, node = int(task.family), 0
        for l in range(self.depth - 1, -1, -1):
            self.view.ledger.hop(1)
            self.view.ledger.compare(self.r)
            node = int(self.children[l][node, self.best_child[l][node, f]])
        return node

    def observe(self, task, agent, outcome) -> None:
        """Running-mean update of est[a,f], then recompute f's summary up a's path (log_r n nodes)."""
        if not self.online:
            return
        f, a = int(task.family), int(agent)
        k = self._cnt.get((a, f), self._w0) + 1
        self._cnt[(a, f)] = k
        self.est[a, f] += (float(outcome) - self.est[a, f]) / k
        node = int(self.leaf_of[a])
        for l in range(self.depth):
            ch = self.children[l][node]
            src = self.est if l == 0 else self.summary[l - 1]
            v = np.where(ch >= 0, src[ch, f], NEG)
            s = int(v.argmax())
            self.best_child[l][node, f] = s
            self.summary[l][node, f] = v[s]
            if l + 1 < self.depth:
                node = int(self.parent[l][node])
