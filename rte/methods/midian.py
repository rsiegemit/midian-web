"""Plain MIDIAN (SPEC §5): a tree of cohorts, routed by per-family max-summaries.

Leaves are agents in random cohorts of `r`, estimated from verified probes reported by the r-1
cohort peers and trimmed (`_est.peer_reported_estimates`). Every node carries, per family,
the best estimated skill in its subtree and which child holds it; upper levels are random
regroupings of nodes using summaries only. `fetch` descends from the root: ceil(log_r n)
decisions over r children. The tree is arrays per level, each level padded to a multiple
of r and reshaped, so nothing here loops over agents.
"""
import numpy as np

from ._est import peer_reported_estimates, trim_k
from .base import Method

NEG = np.float32(-np.inf)


class Midian(Method):
    name = "midian"
    needs = frozenset({"probe", "reports"})

    def __init__(self, r=10, delta=1 / 3, online=True, **p):
        super().__init__(r=r, delta=delta, online=online, **p)
        self.r, self.delta, self.online = int(r), float(delta), bool(online)
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

    def build(self, view, budget):
        self.view, r, b = view, self.r, budget.b
        self.leaves = self._cohorts(view)
        ok = self.leaves >= 0
        self.leaf_of = np.empty(view.n, np.int32)
        self.leaf_of[self.leaves[ok]] = np.repeat(np.arange(len(self.leaves), dtype=np.int32), r)[ok.ravel()]
        self.est = peer_reported_estimates(view, b, self.leaves, self.delta)
        self.w0 = max(1, (r - 1) * b - 2 * trim_k(self.delta, r, b))     # reports behind a build estimate
        self.children, self.summary, self.best, self.parent = [self.leaves], [], [], []
        src = self.est
        while True:
            summ, best = self._level(self.children[-1], src)
            self.summary.append(summ)
            self.best.append(best)
            if (m := len(summ)) == 1:
                break
            perm = view.rng.permutation(m).astype(np.int32)              # regroup nodes at random
            par = np.empty(m, np.int32)
            par[perm] = np.arange(m, dtype=np.int32) // r
            self.parent.append(par)
            nxt = np.full(-(-m // r) * r, -1, np.int32)
            nxt[:m] = perm
            self.children.append(nxt.reshape(-1, r))
            src = summ
        self.depth = len(self.summary)
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
