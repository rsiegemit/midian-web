"""Plain MIDIAN (SPEC §5): a tree of cohorts, routed by per-family max-summaries.

Leaves are agents in random cohorts of `r`, estimated from verified probes reported by the r-1
cohort peers and trimmed (`_est.peer_reported_estimates`). Every node carries, per family,
the best estimated skill in its subtree and which child holds it; upper levels are random
regroupings of nodes using summaries only. `verify=True` (MIDIAN-V): every candidate a child
forwards is re-probed at the parent by the sibling leaders before the comparison, paid for out of
the same n*K*b budget (level 0 keeps b-1 probes per cell), so the root's pick rests on ~b(r-1)
probes instead of b. `fetch` descends from the root: ceil(log_r n) decisions over r children. The tree is arrays per level, each level padded to a multiple
of r and reshaped, so nothing here loops over agents.
"""
import numpy as np

from ._est import peer_reported_estimates, trim_k, trimmed_by_reporter
from .base import Method

NEG = np.float32(-np.inf)
CHUNK_ELEMS = 8_000_000


class Midian(Method):
    name = "midian"
    needs = frozenset({"probe", "reports"})

    def __init__(self, r=10, delta=1 / 3, online=True, verify=False, **p):
        super().__init__(r=r, delta=delta, online=online, verify=verify, **p)
        self.r, self.delta, self.online, self.verify = int(r), float(delta), bool(online), bool(verify)
        self.cnt = {}

    def _cohorts(self, view):
        """Random partition into cohorts of r; -1 pads the last one."""
        c = np.full(-(-view.n // self.r) * self.r, -1, np.int32)
        c[:view.n] = view.rng.permutation(view.n)
        return c.reshape(-1, self.r)

    def _level(self, ch, src):
        """Nodes with children `ch` int32[M,r] (-1 empty) holding values `src`: (summary, best_child)."""
        v = src[ch]
        v[ch < 0] = NEG
        best = v.argmax(1).astype(np.int32)
        return np.take_along_axis(v, best[:, None, :], 1)[:, 0], best

    def _structure(self, view):
        """children per level (level 0 = leaf cohorts), parent maps, and the number of nodes per level."""
        self.leaves = self._cohorts(view)
        self.children, self.parent, m = [self.leaves], [], len(self.leaves)
        while m > 1:
            perm = view.rng.permutation(m).astype(np.int32)              # regroup nodes at random
            par = np.empty(m, np.int32)
            par[perm] = np.arange(m, dtype=np.int32) // self.r
            nxt = np.full(-(-m // self.r) * self.r, -1, np.int32)
            nxt[:m] = perm
            self.parent.append(par); self.children.append(nxt.reshape(-1, self.r)); m = len(self.children[-1])
        self.depth = len(self.children)

    def _verify(self, view, ch, cand, lead, e):
        """Verify at promotion: every candidate a child forwards is re-probed e times, observed and reported
        by the r-1 OTHER leaders of the node (trimmed as at level 0), and folded into its running estimate.
        cand int32[M,r,K] candidate agents (-1 empty), lead int32[M,r] the children's leaders."""
        r, K = self.r, view.K
        node, slot, fam = np.nonzero(cand >= 0)                                             # valid candidates only
        peers = np.array([[j for j in range(r) if j != m] for m in range(r)], np.int32)
        L = np.where(ch >= 0, lead, -1)
        rep_of = L[node[:, None], peers[slot]]                                              # (V, r-1) sibling leaders
        bad = rep_of < 0                                                                    # short (padded) node: cycle
        if bad.any():
            first = np.where(ch >= 0, lead, lead[:, :1])[node]                              # a valid leader per node
            rep_of = np.where(bad, first[:, :1].repeat(r - 1, 1), rep_of)
        agents = cand[node, slot, fam]
        for lo in range(0, len(agents), max(1, CHUNK_ELEMS // (r * e))):
            sl = slice(lo, lo + max(1, CHUNK_ELEMS // (r * e)))
            out = view.probe_many(agents[sl], fam[sl], e)                                    # (v, e)
            per = view.report_many(rep_of[sl], agents[sl][:, None], out.mean(-1)[:, None])           # one report per peer
            m_new = trimmed_by_reporter(per[..., None], self.delta, r)
            a, f = agents[sl], fam[sl]
            self.est[a, f] = (self.est[a, f] * self.k[a, f] + m_new * e) / (self.k[a, f] + e)
            self.k[a, f] += e

    def build(self, view, budget):
        self.view, r, b, K = view, self.r, budget.b, view.K
        self._structure(view)
        ok = self.leaves >= 0
        self.leaf_of = np.empty(view.n, np.int32)
        self.leaf_of[self.leaves[ok]] = np.repeat(np.arange(len(self.leaves), dtype=np.int32), r)[ok.ravel()]
        b0 = max(1, b - 1) if self.verify else b                          # level 0 keeps b-1; the rest buys promotions
        C = sum((c >= 0).sum() for c in self.children[1:])                 # candidates re-verified over all upper levels
        e = int((b - b0) * view.n // C) if self.verify and C else 0        # probes per promoted candidate; exact budget
        self.est = peer_reported_estimates(view, b0, self.leaves, self.delta, by_reporter=self.verify)
        self.k = np.full((view.n, K), float(b0), np.float32)
        self.w0 = max(1, (r - 1) * b - 2 * trim_k(self.delta, r, b))     # reports behind a build estimate
        self.summary, self.best = [], []
        cand = self.leaves[:, :, None].repeat(K, 2)                        # level 0: candidates = the members
        lead = None
        for l, ch in enumerate(self.children):
            if l > 0:
                cand, lead = self.cand[-1][ch], self.lead[-1][ch]        # (M,r,K) agents, (M,r) leaders
                cand[ch < 0] = -1
                if e:                                                    # reporters: RANDOM members of the sibling
                    self._verify(view, ch, cand, self.rep[-1][ch], e)   # subtrees, so liars are not enriched upward
                    valid = self.cand[-1] >= 0                                                     # children's summaries now
                    self.summary[-1] = np.where(valid, self.est[np.where(valid, self.cand[-1], 0), np.arange(K)], NEG)
            v = np.where(cand >= 0, self.est[np.where(cand >= 0, cand, 0), np.arange(K)], NEG)
            best = v.argmax(1).astype(np.int32)
            self.summary.append(np.take_along_axis(v, best[:, None, :], 1)[:, 0]); self.best.append(best)
            self.cand = getattr(self, "cand", []) if l else []
            self.cand.append(np.take_along_axis(cand, best[:, None, :], 1)[:, 0])                 # (M,K) summary holder
            self.lead = getattr(self, "lead", []) if l else []
            mean = np.where(cand[:, :, 0] >= 0, v.mean(2), NEG)                                   # leader = best mean
            self.lead.append((self.leaves if l == 0 else lead)[np.arange(len(ch)), mean.argmax(1)])
            self.rep = getattr(self, "rep", []) if l else []              # one random subtree member per node
            nvalid = (ch >= 0).sum(1)
            pick = (view.rng.random(len(ch)) * nvalid).astype(int)
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
        f, node = int(task.family), 0
        for l in range(self.depth - 1, -1, -1):
            self.view.ledger.hop(1)
            self.view.ledger.compare(self.r)
            self.view.ledger.message(2)                                  # request down, answer up
            node = int(self.children[l][node, self._choose(l, node, f)])
        return node

    def observe(self, task, agent, outcome):
        """Running mean on est[a,f], then recompute f's summary up a's path (log_r n nodes)."""
        if not self.online:
            return
        f, a = int(task.family), int(agent)
        k = self.cnt[a, f] = self.cnt.get((a, f), self.w0) + 1
        self.est[a, f] += (outcome - self.est[a, f]) / k
        node = int(self.leaf_of[a])
        for l in range(self.depth):
            v = self._values(l, node, f)
            self.best[l][node, f] = s = int(v.argmax())
            self.summary[l][node, f] = v[s]
            if l + 1 < self.depth:
                node = int(self.parent[l][node])
