"""Plain MIDIAN (SPEC §5): a tree of cohorts, routed by per-family max-summaries.

Leaves are agents in random cohorts of `r`, estimated from verified probes reported by the r-1
cohort peers and trimmed (`_est.peer_reported_estimates`). Every node carries, per family,
the best estimated skill in its subtree and which child holds it; upper levels are random
regroupings of nodes using summaries only. `verify=True` (MIDIAN-V): every candidate a child
forwards is re-probed at the parent by the sibling leaders before the comparison, paid for out of
the same n*K*b budget (level 0 keeps b-1 probes per cell), so the root's pick rests on ~b(r-1)
probes instead of b. `stratify=True`: level-0 cohorts take one random member from each of r strata of the
measured probe mean (the same n*K*b probes, spent before grouping), so every cohort spans the skill range.
`fetch` descends from the root: ceil(log_r n) decisions over r children. The tree is arrays per level, each
level padded to a multiple of r and reshaped, so nothing here loops over agents. Subclasses override `_level0`
(SH: successive halving inside cohorts; A: audited reports) and `_choose` (LLM descent).
"""
import numpy as np

from ._est import peer_estimate, peer_reported_estimates, probe_outcomes, trim_k
from .base import Method

NEG = np.float32(-np.inf)
CHUNK_ELEMS = 8_000_000


class Midian(Method):
    name = "midian"
    needs = frozenset({"probe", "reports"})

    def __init__(self, r=10, delta=1 / 3, online=True, verify=False, observers=None, b0=None, cached=False, top=1, stratify=False, **p):
        super().__init__(r=r, delta=delta, online=online, verify=verify, observers=observers, b0=b0, cached=cached, top=top, stratify=stratify, **p)
        self.stratify = bool(stratify)
        self.top = int(top)                                                   # candidates each node forwards per family (V only)
        self.r, self.delta, self.online, self.verify = int(r), float(delta), bool(online), bool(verify)
        self.observers = int(observers) if observers else self.r - 1          # peers observing each probe (V only)
        self.b0, self.cached = b0, bool(cached)                               # level-0 probes per cell (V only); cache root picks
        self.cnt = {}

    def _cohorts(self, view, key=None):
        """Random partition into cohorts of r; -1 pads the last one. With `key[n]` (measured probe mean): the last
        cohort is q random agents, the rest are r equal strata of key with one random member of each per cohort."""
        c = np.full(-(-view.n // self.r) * self.r, -1, np.int32)
        if key is None:
            c[:view.n] = view.rng.permutation(view.n)
            return c.reshape(-1, self.r)
        m = len(c) // self.r; q = view.n - (m - 1) * self.r; perm = view.rng.permutation(view.n)
        band = perm[q:][np.argsort(key[perm[q:]], kind="stable")].reshape(self.r, m - 1)      # stratum j = row j
        band = np.take_along_axis(band, np.argsort(view.rng.random(band.shape), 1), 1)          # random within stratum
        c[:(m - 1) * self.r] = band.T.ravel(); c[(m - 1) * self.r:(m - 1) * self.r + q] = perm[:q]
        return c.reshape(-1, self.r)

    def _level(self, ch, src):
        """Nodes with children `ch` int32[M,r] (-1 empty) holding values `src`: (summary, best_child)."""
        v = src[ch]
        v[ch < 0] = NEG
        best = v.argmax(1).astype(np.int32)
        return np.take_along_axis(v, best[:, None, :], 1)[:, 0], best

    def _structure(self, view, key=None):
        """children per level (level 0 = leaf cohorts), parent maps, and the number of nodes per level."""
        self.leaves = self._cohorts(view, key)
        self.children, self.parent, m = [self.leaves], [], len(self.leaves)
        while m > 1:
            perm = view.rng.permutation(m).astype(np.int32)              # regroup nodes at random
            par = np.empty(m, np.int32)
            par[perm] = np.arange(m, dtype=np.int32) // self.r
            nxt = np.full(-(-m // self.r) * self.r, -1, np.int32)
            nxt[:m] = perm
            self.parent.append(par); self.children.append(nxt.reshape(-1, self.r)); m = len(self.children[-1])
        self.depth = len(self.children)

    def _verify(self, view, ch, cand, lead, e, slot_child=None):
        """Verify at promotion: every candidate a child forwards is re-probed e times, each outcome reported by the
        r-1 OTHER children's representatives (trimmed by reporter as at level 0), and folded into its running estimate.
        cand int32[M,r,K] candidate agents (-1 empty), lead int32[M,r] the children's representatives."""
        r, k = self.r, min(self.observers, self.r - 1)
        node, slot, fam = np.nonzero(cand >= 0)                                             # valid candidates only
        child = slot if slot_child is None else slot_child[slot]                            # which child each slot came from
        peers = np.array([[j for j in range(r) if j != m] for m in range(r)], np.int32)
        L = np.where(ch >= 0, lead, -1)
        rep_of = L[node[:, None], peers[child]]                                             # (V, r-1) OTHER children's reps
        bad = rep_of < 0                                                                    # short (padded) node: cycle
        if bad.any():
            first = np.where(ch >= 0, lead, lead[:, :1])[node]                              # a valid reporter per node
            rep_of = np.where(bad, first[:, :1].repeat(r - 1, 1), rep_of)
        if k < r - 1:                                                                       # a random k of the r-1 peers
            rep_of = np.take_along_axis(rep_of, np.argsort(view.rng.random(rep_of.shape), 1)[:, :k], 1)
        agents, step = cand[node, slot, fam], max(1, CHUNK_ELEMS // (r * e))
        for lo in range(0, len(agents), step):
            a, f, rep = agents[lo:lo + step], fam[lo:lo + step], rep_of[lo:lo + step]
            m_new, _ = peer_estimate(view, a, f, e, rep, self.delta)
            self.est[a, f] = (self.est[a, f] * self.k[a, f] + m_new * e) / (self.k[a, f] + e)
            self.k[a, f] += e

    def _level0(self, view, cohorts, b, outcomes=None):
        """Level-0 estimates est[n, K]: b probes per cell, reported by the cohort peers, trimmed. Variants override this."""
        return peer_reported_estimates(view, b, cohorts, self.delta, by_reporter=self.verify, observers=self.observers, outcomes=outcomes)

    def build(self, view, budget):
        self.view, r, b, K = view, self.r, budget.b, view.K
        self.b = b0 = (max(1, min(b, self.b0 or b - 1))) if self.verify else b   # level 0 keeps b0 (default b-1); the rest buys promotions
        out = probe_outcomes(view, b0) if self.stratify else None            # stratify: probe first, group by measured mean
        self._structure(view, None if out is None else out.mean((1, 2)))
        ok = self.leaves >= 0
        self.leaf_of = np.empty(view.n, np.int32)
        self.leaf_of[self.leaves[ok]] = np.repeat(np.arange(len(self.leaves), dtype=np.int32), r)[ok.ravel()]
        C = self.top * sum((c >= 0).sum() for c in self.children[1:])      # candidates re-verified over all upper levels
        e = int((b - b0) * view.n // C) if self.verify and C else 0        # probes per promoted candidate; exact budget
        self.est = self._level0(view, self.leaves, b0, out)
        self.k = np.full((view.n, K), float(b0), np.float32)
        self.w0 = max(1, (r - 1) * b - 2 * trim_k(self.delta, r, b))     # reports behind a build estimate
        self.summary, self.best, self.cand, self.topc, self.lead, self.rep = [], [], [], [], [], []
        cand = self.leaves[:, :, None].repeat(K, 2)                        # level 0: candidates = the members
        for l, ch in enumerate(self.children):
            if l > 0:
                lead = self.lead[-1][ch]                                 # (M, r) the children's leaders
                cand = self.topc[-1][ch].reshape(len(ch), -1, K)        # (M, r*top, K) forwarded agents
                cand[np.repeat(ch < 0, self.top, 1)] = -1
                slot_child = np.repeat(np.arange(self.r), self.top)       # which child each slot came from
                if e:                                                    # reporters: RANDOM members of the sibling
                    self._verify(view, ch, cand, self.rep[-1][ch], e, slot_child)
                    valid = self.cand[-1] >= 0                                                     # children's summaries now
                    self.summary[-1] = np.where(valid, self.est[np.where(valid, self.cand[-1], 0), np.arange(K)], NEG)
            v = np.where(cand >= 0, self.est[np.where(cand >= 0, cand, 0), np.arange(K)], NEG)
            order = np.argsort(-v, axis=1, kind="stable")
            best = (order[:, 0] if l == 0 else slot_child[order[:, 0]]).astype(np.int32)         # best CHILD per family
            self.summary.append(np.take_along_axis(v, order[:, :1], 1)[:, 0]); self.best.append(best)
            self.cand.append(np.take_along_axis(cand, order[:, :1], 1)[:, 0])                     # (M,K) summary holder
            self.topc.append(np.take_along_axis(cand, order[:, :self.top], 1))                    # (M,top,K) forwarded
            mc = (v if l == 0 else v.reshape(len(ch), self.r, self.top, K).mean(2)).mean(2)      # per-child mean est
            mean = np.where(ch >= 0, mc, NEG)                                                      # leader = best mean
            self.lead.append((self.leaves if l == 0 else lead)[np.arange(len(ch)), mean.argmax(1)])
            pick = (view.rng.random(len(ch)) * (ch >= 0).sum(1)).astype(int)                  # one random subtree member per node
            self.rep.append((self.leaves if l == 0 else self.rep[-1][ch])[np.arange(len(ch)), pick])
        # messages: each member tells its leader (n - N0), each node but the root tells its parent
        view.ledger.message(view.n - len(self.leaves) + sum(len(s) for s in self.summary) - 1)

    def _values(self, l, node, f):
        """The r children's summaries for family f at one node (-inf where the slot is empty)."""
        ch = self.children[l][node]
        return np.where(ch >= 0, (self.est if l == 0 else self.summary[l - 1])[ch, f], NEG)

    def _choose(self, l, node, f):
        """Which child to descend into. The LLM ablation overrides exactly this."""
        return int(self.best[l][node, f])

    def fetch(self, task):
        if self.cached:                                                  # the root remembers its pick per family;
            self.view.ledger.compare(1); self.view.ledger.message(2)     # one lookup, root -> agent
            return int(self.cand[-1][0, task.family])
        f, node = int(task.family), 0
        for l in range(self.depth - 1, -1, -1):
            self.view.ledger.hop(1)
            self.view.ledger.compare(self.r)
            self.view.ledger.message(2)                                  # request down, answer up
            node = int(self.children[l][node, self._choose(l, node, f)])
        return node

    def _recompute(self, node, f):
        """Recompute best/summary (and cached candidates) for families `f` (int array) on the path from leaf `node` up."""
        for l in range(self.depth):
            ch = self.children[l][node]
            v = np.where(ch[:, None] >= 0, (self.est if l == 0 else self.summary[l - 1])[ch][:, f], NEG)   # (r, |f|)
            s = v.argmax(0)
            self.best[l][node, f] = s; self.summary[l][node, f] = v[s, np.arange(len(f))]
            if self.cached:
                self.cand[l][node, f] = ch[s] if l == 0 else self.cand[l - 1][ch[s], f]
            if l + 1 < self.depth:
                node = int(self.parent[l][node])

    def observe(self, task, agent, outcome):
        """Running mean on est[a,f], then recompute f's summary up a's path (log_r n nodes)."""
        if not self.online:
            return
        f, a = int(task.family), int(agent)
        k = self.cnt[a, f] = self.cnt.get((a, f), self.w0) + 1
        self.est[a, f] += (outcome - self.est[a, f]) / k
        self._recompute(int(self.leaf_of[a]), np.array([f]))

    def churn(self, departed, arrived):
        """Repair after replacement (ids reused: arrived ⊆ former ids): each arrived agent is re-probed b times per
        family, every outcome reported by its cohort peers (trimmed as at build), and its whole path is recomputed
        for all K families. Cost per arrived agent: K*b probes, K*b*(r-1) reports, (r-1) + depth messages.
        Departed ids that were not refilled keep no estimate (-inf)."""
        view, K = self.view, self.view.K
        self.est[np.setdiff1d(departed, arrived)] = NEG
        for a in np.asarray(arrived, dtype=int):
            leaf = int(self.leaf_of[a]); peers = self.leaves[leaf]; peers = peers[(peers >= 0) & (peers != a)]
            self.est[a] = (peer_estimate(view, np.full(K, a), np.arange(K), self.b, np.broadcast_to(peers, (K, len(peers))), self.delta)[0]
                           if len(peers) else view.probe_many(a, np.arange(K), self.b).mean(-1))
            self.k[a] = self.b; self.cnt = {kf: c for kf, c in self.cnt.items() if kf[0] != a}
            view.ledger.message(len(peers) + self.depth); self._recompute(leaf, np.arange(K))
