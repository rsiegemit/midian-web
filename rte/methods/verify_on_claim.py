"""Rank by declared D[:, f]; probe the top unverdicted candidate k times; accept if mean >= D - margin, else reject and
try the next (max_tries fresh verifications per fetch). Verdicts cached per (agent, family). The most dangerous baseline:
verification budget drains on liars. Ranking is O(n) once per family, then cached (compare(1))."""
import numpy as np
from .base import Method


class VerifyOnClaim(Method):
    name = "verify_on_claim"
    needs = frozenset({"declared", "probe"})

    def __init__(self, k=3, max_tries=5, margin=0.15, **p):
        super().__init__(k=k, max_tries=max_tries, margin=margin, **p)
        self.k, self.max_tries, self.margin = k, max_tries, margin

    def build(self, view, budget):
        self.view, self.D = view, view.declared
        self.order, self.verdict = {}, {}                   # family -> ranked agents ; (a, f) -> (accepted, mean)

    def fetch(self, task):
        f = task.family
        if f not in self.order:
            self.order[f] = np.argsort(-self.D[:, f], kind="stable"); self.view.ledger.compare(self.view.n)
        else:
            self.view.ledger.compare(1)
        tries, best = 0, (-1.0, int(self.order[f][0]))
        for a in map(int, self.order[f]):
            if (a, f) not in self.verdict:
                if tries == self.max_tries:
                    break
                m = self.view.probe_many([a], f, self.k).mean()
                self.verdict[a, f] = (m >= self.D[a, f] - self.margin, m); tries += 1
            ok, m = self.verdict[a, f]
            if ok:
                return a
            best = max(best, (m, a))
        return best[1]
