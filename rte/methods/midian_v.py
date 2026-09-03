"""MIDIAN-V: plain MIDIAN with verification at promotion and a cached root pick (labeled variant, added post-hoc on
2026-09-02 after the bernoulli mirrors; confirmatory replication = grid `midian_v_replication`, TARGETS_rte_v2 V2-8).

What is verified, when: at every level l >= 1, each candidate a child node forwards to its parent (its per-family best,
`top` per family) is re-probed at the parent BEFORE the parent compares children. How many: level 0 spends b0 = b-1
probes per (agent, family) instead of b; the saved n*K*(b-b0) probes are split budget-exactly over the
C = top * (number of child slots over all upper levels) forwarded candidates: e = floor((b-b0)*n / C) fresh probes per
(candidate, family), so total build probes = n*K*b0 + C*K*e <= n*K*b. Who reports: each of the e outcomes is reported
by up to `observers` (default r-1) representatives of the OTHER children of that node, one report per (reporter, probe);
the per-reporter means are trimmed exactly as at level 0 (drop floor(delta*(r-1)) from each end) and the result is
folded into the candidate's running estimate (weighted by probe count) and written back into the child's summary.
Charged: probes n*K*b0 + C*K*e; reports (r-1)*(that) i.e. one per (reporter, member, family, probe), so at the default
observers = r-1 build reports = build probes * (r-1); messages as plain MIDIAN ((r-1) per cohort + 1 per upper node).
Per fetch (cached=True): the root's per-family pick is remembered, 1 comparison + 2 messages (root -> agent), 0 hops;
observe() updates the estimate and rewrites the cached pick along the agent's path, so it still learns online.
Equivalent to `midian(verify=True, cached=True, r=r, ...)`, which the 2026-09-02 rows use.
"""
from .midian import Midian


class MidianV(Midian):
    name = "midian_v"

    def __init__(self, r=10, **p):
        p.pop("verify", None); p.pop("cached", None)
        super().__init__(r=r, verify=True, cached=True, **p)
