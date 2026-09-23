"""LinUCB-honest (labeled rival, 2026-09-03): a contextual bandit whose only context is the agent's OWN observed
outcome history -- never model identity, specialty, or the declared channel. Per family f the arm-context of agent a is
x = [1, mean_af, sqrt(count_af), mean_a-over-families]; the family's ridge model (A_f = I + sum x x^T, b_f = sum x y)
scores arms by x^T theta_f + alpha * sqrt(x^T A_f^-1 x). Warm-up = the shared n*K*b probe budget (b pulls per arm,
each fed to the model as an observation); online updates on. Fetch scans the n arms (compare(n))."""
import numpy as np

from ._est import probe_successes
from .base import Method


class LinUcbHonest(Method):
    name = "linucb_honest"
    needs = frozenset({"probe"})

    def __init__(self, alpha=1.0, bonus="context", **p):
        # bonus="own" (post-hoc fix, 2026-09-23 audit): the exploration bonus uses only the agent's OWN evidence for the
        # family ([1, sqrt(count)]), not the full context. With the full context, agents tied at the top estimate are
        # separated by how ATYPICAL their cross-family mean is -- i.e. weak agents that got lucky -- so LinUCB fell to
        # random at 10^4-10^5. Default "context" keeps the pre-registered behaviour (and every existing row).
        super().__init__(alpha=alpha, **({"bonus": bonus} if bonus != "context" else {}), **p)
        self.alpha, self.bonus = float(alpha), bonus

    def features(self, f):
        """Context of every agent for family f: a function of observed outcomes only. float64[n, 4]."""
        c = self.cnt[:, f]
        return np.stack([np.ones(self.view.n), self.mean[:, f], np.sqrt(c), self.mean.mean(1)], 1)

    def build(self, view, budget):
        self.view, d = view, 4
        self.cnt = np.full((view.n, view.K), budget.b, np.int64)
        self.mean = probe_successes(view, budget.b) / budget.b
        self.A = np.tile(np.eye(d), (view.K, 1, 1)); self.b = np.zeros((view.K, d))
        for f in range(view.K):                                  # the warm-up pulls train the model, one row per arm
            x = self.features(f)
            self.A[f] += budget.b * x.T @ x; self.b[f] += budget.b * x.T @ self.mean[:, f]

    def fetch(self, task):
        f = task.family
        x = self.features(f); Ainv = np.linalg.inv(self.A[f])
        self.view.ledger.compare(self.view.n)
        if self.bonus == "own":                                  # uncertainty of the agent's own (intercept, count) evidence
            xo = x[:, [0, 2]]; Ao = np.linalg.inv(self.A[f][np.ix_([0, 2], [0, 2])])
            bonus = np.sqrt(np.einsum("ij,jk,ik->i", xo, Ao, xo)) / np.sqrt(self.cnt[:, f])
        else:
            bonus = np.sqrt(np.einsum("ij,jk,ik->i", x, Ainv, x))
        return int(np.argmax(x @ (Ainv @ self.b[f]) + self.alpha * bonus))

    def observe(self, task, agent, outcome):
        f = task.family
        x = self.features(f)[agent]
        self.A[f] += np.outer(x, x); self.b[f] += x * outcome
        self.cnt[agent, f] += 1
        self.mean[agent, f] += (outcome - self.mean[agent, f]) / self.cnt[agent, f]
