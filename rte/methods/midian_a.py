"""MIDIAN-A (labeled variant, 2026-09-03): plain MIDIAN plus REPORT AUDITS. At build, a uniform 5% of level-0 probe
instances are re-run by the auditor (`view.probe_at`: the same index-seeded instance, charged as a probe) and every
peer's report about that instance is compared with the true outcome; a reporter with 2 mismatches is excluded from
every aggregation for the rest of the run (its later reports are still charged, never used). Online, 5% of routed
outcomes are put to the agent's cohort peers through the report channel and compared the same way; a new exclusion
re-aggregates that cohort's level-0 estimates from the stored per-peer report means and recomputes their paths.
Level 0 is MIDIAN-SH's engine with halving off: b probes per cell, one report per peer per probe, est = trimmed mean
over non-excluded peers of each peer's mean report. Build probes = n*K*b*(1 + audit) <= 1.05x plain."""
import numpy as np

from .midian_sh import MidianSH


STRIKES = 2                                            # mismatches before a reporter is excluded (work order 1.2)


class MidianA(MidianSH):
    name = "midian_a"

    def __init__(self, audit=0.05, halving=False, **p):
        super().__init__(halving=halving, audit=audit, **p)
        self.audit, self.rate = True, float(audit)

    def _strike(self, reporters, claims, truth):
        """Count claim != truth per reporter; exclude at STRIKES mismatches; return the ids newly excluded."""
        if not hasattr(self, "hits"):
            self.hits = np.zeros(self.excluded.size, np.int32)
        np.add.at(self.hits, reporters[claims != truth], 1)
        new = (self.hits >= STRIKES) & ~self.excluded
        self.excluded |= new
        return np.flatnonzero(new)

    def _audit(self, view, ag, fam, surv, k0, rep_ids, out, rep):
        """Re-run a uniform `rate` of this round's (member, family, probe) instances; compare all peers' claims."""
        C, K, sz, p = out.shape
        pick = np.nonzero(view.rng.random((C, K, sz, p)) < self.rate)                  # (c, f, slot, j)
        if not pick[0].size:
            return
        c, f, i, j = pick
        cidx = np.arange(C)[:, None, None]
        a = ag[cidx, surv][c, f, i]
        truth = view.probe_at(a, f, k0[c, f, surv[c, f, i]] + j)                       # same instance, charged as probes
        self._strike(rep_ids[cidx, surv][c, f, i], rep[c, f, i, :, j], truth[:, None])

    def observe(self, task, agent, outcome):
        super().observe(task, agent, outcome)
        if self.view.rng.random() >= self.rate:
            return
        peers = self.peer_of[agent]; peers = peers[peers >= 0]
        if not peers.size:
            return
        claims = self.view.report_many(peers, np.full(peers.shape, agent), np.full(peers.shape, outcome))
        for j in self._strike(peers, claims, outcome):                                  # newly excluded reporter j:
            members = self.leaves[self.leaf_of[j]]; members = members[members >= 0]     # re-aggregate its cohort
            K = self.view.K
            self.est[members] = self._estimates(members[:, None], np.arange(K)[None, :], len(members))
            self._recompute(int(self.leaf_of[members[0]]), np.arange(K))                  # one path: the cohort's own
