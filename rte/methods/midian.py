"""Plain MIDIAN (SPEC §5): a tree of cohorts routed by per-family max-summaries. Leaves are agents in random cohorts
of r, estimated from probes reported by the r-1 cohort peers and trimmed; each node holds per family the best estimate
in its subtree and which child has it; upper levels regroup nodes at random. fetch descends ceil(log_r n) levels.
verify=True = MIDIAN-V (see midian_v.py); stratify=True groups level 0 by measured probe mean. Arrays per level, padded
to a multiple of r. Subclasses override `_level0` (SH, A) and `_choose` (LLM descent)."""
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
        self.r, self.delta, self.online, self.verify, self.stratify = int(r), float(delta), bool(online), bool(verify), bool(stratify)
        self.observers = int(observers) if observers else self.r - 1          # peers observing each verification probe (V)
        self.b0, self.cached, self.top = b0, bool(cached), int(top)          # V: level-0 probes per cell; cached root pick; forwarded per family
        self.cnt = {}; self.stats = {"observe_charged": 1}                  # rows before 2026-09-03 15:20 lack observe-time charges (analyzer adds them)

    def _cohorts(self, view, key=None):
        """Cohorts of r (-1 pads the last). With key[n]: the last cohort is q random agents, the rest take one random
        member from each of r equal strata of key."""
        c = np.full(-(-view.n // self.r) * self.r, -1, np.int32)
        if key is None:
            c[:view.n] = view.rng.permutation(view.n)
            return c.reshape(-1, self.r)
        m = len(c) // self.r; q = view.n - (m - 1) * self.r; perm = view.rng.permutation(view.n)
        band = perm[q:][np.argsort(key[perm[q:]], kind="stable")].reshape(self.r, m - 1)      # stratum j = row j
        band = np.take_along_axis(band, np.argsort(view.rng.random(band.shape), 1), 1)          # random within stratum
        c[:(m - 1) * self.r] = band.T.ravel(); c[(m - 1) * self.r:(m - 1) * self.r + q] = perm[:q]
        return c.reshape(-1, self.r)

    def _structure(self, view, key=None):
        """children per level (level 0 = leaf cohorts), parent maps, depth."""
        self.leaves = self._cohorts(view, key)
        self.children, self.parent, m = [self.leaves], [], len(self.leaves)
        while m > 1:
            perm = view.rng.permutation(m).astype(np.int32)              # regroup nodes at random
            par = np.empty(m, np.int32); par[perm] = np.arange(m, dtype=np.int32) // self.r
            nxt = np.full(-(-m // self.r) * self.r, -1, np.int32); nxt[:m] = perm
            self.parent.append(par); self.children.append(nxt.reshape(-1, self.r)); m = len(self.children[-1])
        self.depth = len(self.children)

    def _verify(self, view, ch, cand, lead, e, slot_child):
        """Re-probe every forwarded candidate (cand int32[M,r*top,K], -1 empty) e times, reported by the r-1 OTHER
        children's representatives lead[M,r] (trimmed by reporter), folded into its running estimate."""
        r, k = self.r, min(self.observers, self.r - 1)
        node, slot, fam = np.nonzero(cand >= 0); child = slot_child[slot]                # valid candidates, their child
        peers = np.array([[j for j in range(r) if j != m] for m in range(r)], np.int32)
        rep_of = np.where(ch >= 0, lead, -1)[node[:, None], peers[child]]                # (V, r-1) OTHER children's reps
        if (bad := rep_of < 0).any():                                                    # short (padded) node: cycle
            rep_of = np.where(bad, np.where(ch >= 0, lead, lead[:, :1])[node][:, :1].repeat(r - 1, 1), rep_of)
        if k < r - 1:                                                                    # a random k of the r-1 peers
            rep_of = np.take_along_axis(rep_of, np.argsort(view.rng.random(rep_of.shape), 1)[:, :k], 1)
        agents, step = cand[node, slot, fam], max(1, CHUNK_ELEMS // (r * e))
        for lo in range(0, len(agents), step):
            a, f = agents[lo:lo + step], fam[lo:lo + step]
            ex = getattr(self, "excluded", None)                    # audited variants (midian_va) mask caught liars
            if ex is not None:
                ex = ex[rep_of[lo:lo + step]]; ex &= ~ex.all(-1, keepdims=True)
            m_new, _ = peer_estimate(view, a, f, e, rep_of[lo:lo + step], self.delta, exclude=ex)
            self.est[a, f] = (self.est[a, f] * self.k[a, f] + m_new * e) / (self.k[a, f] + e); self.k[a, f] += e

    def _level0(self, view, cohorts, b, outcomes=None):
        """Level-0 estimates est[n, K]: b probes per cell, reported by the cohort peers, trimmed. Variants override this."""
        return peer_reported_estimates(view, b, cohorts, self.delta, by_reporter=self.verify, observers=self.observers, outcomes=outcomes)

    def build(self, view, budget):
        self.view, r, b, K = view, self.r, budget.b, view.K
        self.b = b0 = (max(1, min(b, self.b0 or b - 1))) if self.verify else b   # level 0 keeps b0 (default b-1); the rest buys promotions
        out = probe_outcomes(view, b0) if self.stratify else None            # stratify: probe first, group by measured mean
        self._structure(view, None if out is None else out.mean((1, 2)))
        ok = self.leaves >= 0; self.leaf_of = np.empty(view.n, np.int32)
        self.leaf_of[self.leaves[ok]] = np.repeat(np.arange(len(self.leaves), dtype=np.int32), r)[ok.ravel()]
        C = self.top * sum((c >= 0).sum() for c in self.children[1:])      # V: candidates re-verified over all upper levels,
        e = int((b - b0) * view.n // C) if self.verify and C else 0        # e probes each (exact budget)
        self.est = self._level0(view, self.leaves, b0, out)
        self.k = np.full((view.n, K), float(b0), np.float32); self.w0 = max(1, (r - 1) * b - 2 * trim_k(self.delta, r, b))   # probes / reports behind a build estimate
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
            self.lead.append((self.leaves if l == 0 else lead)[np.arange(len(ch)), np.where(ch >= 0, mc, NEG).argmax(1)])
            pick = (view.rng.random(len(ch)) * (ch >= 0).sum(1)).astype(int)                  # one random subtree member per node
            self.rep.append((self.leaves if l == 0 else self.rep[-1][ch])[np.arange(len(ch)), pick])
        view.ledger.message(view.n - len(self.leaves) + sum(len(s) for s in self.summary) - 1)   # member->leader, node->parent

    def _values(self, l, node, f):
        """The r children's summaries at one node for family f (int or int array); -inf where the slot is empty."""
        ch = self.children[l][node]
        v = (self.est if l == 0 else self.summary[l - 1])[ch][:, f]
        return np.where((ch >= 0).reshape((-1,) + (1,) * (v.ndim - 1)), v, NEG)

    def _choose(self, l, node, f):
        """Which child to descend into. The LLM ablation overrides exactly this."""
        return int(self.best[l][node, f])

    def fetch(self, task):
        if self.cached:                                                  # the root remembers its pick per family:
            self.view.ledger.compare(1); self.view.ledger.message(2); return int(self.cand[-1][0, task.family])
        f, node = int(task.family), 0
        for l in range(self.depth - 1, -1, -1):
            self.view.ledger.hop(1); self.view.ledger.compare(self.r); self.view.ledger.message(2)   # request down, answer up
            node = int(self.children[l][node, self._choose(l, node, f)])
        return node

    def _recompute(self, node, f):
        """Recompute best/summary (and cached candidates) for families `f` (int array) on the path from leaf `node` up."""
        for l in range(self.depth):                                     # observe-time cost: r comparisons + 1 message (child->parent update) per level per family
            self.view.ledger.compare(self.r * len(f)); self.view.ledger.message(len(f))
            v = self._values(l, node, f); s = v.argmax(0)                                            # (r, |f|)
            self.best[l][node, f] = s; self.summary[l][node, f] = v[s, np.arange(len(f))]
            if self.cached:
                self.cand[l][node, f] = self.children[l][node][s] if l == 0 else self.cand[l - 1][self.children[l][node][s], f]
            node = int(self.parent[l][node]) if l + 1 < self.depth else node

    def observe(self, task, agent, outcome):
        """Running mean on est[a,f], then recompute f's summary up a's path (log_r n nodes)."""
        if self.online:
            f, a = int(task.family), int(agent); k = self.cnt[a, f] = self.cnt.get((a, f), self.w0) + 1
            self.est[a, f] += (outcome - self.est[a, f]) / k; self._recompute(int(self.leaf_of[a]), np.array([f]))

    def churn(self, departed, arrived):
        """Repair (ids reused): each arrived agent is re-probed b times per family, reported by its cohort peers (trimmed
        as at build), and its path recomputed. Per arrival: K*b probes, K*b*(r-1) reports, (r-1)+depth messages."""
        view, K = self.view, self.view.K
        self.est[np.setdiff1d(departed, arrived)] = NEG
        for a in np.asarray(arrived, dtype=int):
            leaf = int(self.leaf_of[a]); peers = self.leaves[leaf]; peers = peers[(peers >= 0) & (peers != a)]
            self.est[a] = (peer_estimate(view, np.full(K, a), np.arange(K), self.b, np.broadcast_to(peers, (K, len(peers))), self.delta)[0]
                           if len(peers) else view.probe_many(a, np.arange(K), self.b).mean(-1))
            self.k[a] = self.b; self.cnt = {kf: c for kf, c in self.cnt.items() if kf[0] != a}
            view.ledger.message(len(peers) + self.depth); self._recompute(leaf, np.arange(K))
