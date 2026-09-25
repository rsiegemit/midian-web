"""MIDIAN: a tree of cohorts routed by per-family max-summaries.

Mechanism: leaves are agents in random cohorts of r, estimated from probes reported by the r-1 cohort peers and trimmed;
each node holds per family the best estimate in its subtree and which child has it; upper levels regroup nodes at
random. Fetch descends ceil(log_r n) levels; online=True folds each routed outcome into a running mean and recomputes
the family's summaries up the agent's path.

Two defenses, both ON by default (the method the paper calls MIDIAN):
  audit=True   report audits with reporter exclusion. At build, a uniform `audit` rate (True = 5%) of level-0
               probe instances is re-run by the auditor (`view.probe_at`: the same index-seeded instance, charged as
               a probe) and every peer's report about it is compared with the truth; a reporter with STRIKES
               mismatches is excluded from every later aggregation. Online, the same rate of routed outcomes is put
               to the agent's cohort peers. Level 0 then estimates est = trimmed mean over non-excluded PEERS of each
               peer's mean report (one round of b0 probes, one report per peer per probe).
  verify=True  verification at promotion: level 0 keeps b0 = b-1 probes per cell and the saved n*K probes re-probe
               every candidate forwarded to a parent, reported by the OTHER children's representatives (trimmed by
               reporter, excluded reporters masked). `cached` (default = verify) remembers the root's pick per family.
Ablations switch them off: midian{"verify": false} (w/o verification), midian{"audit": false} (w/o audits; its level 0
trims by reporter), midian{"audit": false, "verify": false} (w/o defenses: level 0 trims by report). `cohort` selects
how level 0 is grouped (random | stratify | block | specialty | declared; stratify=True is cohort="stratify"). Arrays
per level, padded to a multiple of r. `_choose` is the LLM-descent hook.

Ledger: build = n*K*b probes (w/o defenses; with verify n*K*b0 at level 0 plus e per forwarded candidate, e =
floor((b-b0)*n / #candidates); with audit + the audited instances, n*K*b0*audit) + (r-1) reports per level-0 and
verification probe + (n - #leaves + #summaries - 1) messages; fetch = per level 1 hop, r comparisons, 2 messages
(cached: 1 comparison, 2 messages); observe (online) = per level r comparisons + 1 message.
Churn: per arrival K*b probes, K*b*(r-1) reports, (r-1) + depth messages.
Params: r=10, delta=1/3, online=True, audit=True, verify=True, cached=verify, observers=r-1, b0=b-1, top=1,
stratify=False, cohort="random"."""
import numpy as np

from ._est import (REPORT_ELEMS, cohort_blocks, others, peer_estimate, peer_reported_estimates, probe_outcomes, trim_k,
                   trimmed_by_reporter)
from .base import Method

NEG = np.float32(-np.inf)
CHUNK_ELEMS = 8_000_000
STRIKES = 2                                            # mismatches before a reporter is excluded
AUDIT_RATE = 0.05                                      # audit=True


class Midian(Method):
    name = "midian"
    needs = frozenset({"probe", "reports"})

    def __init__(self, r=10, delta=1 / 3, online=True, audit=True, verify=True, observers=None, b0=None, cached=None,
                 top=1, stratify=False, cohort=None, **p):
        cached = verify if cached is None else cached
        super().__init__(r=r, delta=delta, online=online, audit=audit, verify=verify, observers=observers, b0=b0,
                         cached=cached, top=top, stratify=stratify, **({"cohort": cohort} if cohort else {}), **p)
        self.r, self.delta, self.online = int(r), float(delta), bool(online)
        self.verify, self.stratify = bool(verify), bool(stratify)
        self.rate = (AUDIT_RATE if audit is True else float(audit)) if audit else 0.0
        self.audit = self.rate > 0
        # How level-0 cohorts are formed. "random" is the default; the rest are labeled variants and are
        # BUDGET-NEUTRAL: the key reuses the same probes `_level0` would have spent anyway (or, for "declared", none).
        #   stratify  one member per ability stratum  -> maximally DIVERSE cohorts (pre-existing, == stratify=True)
        #   block     contiguous ability blocks       -> maximally HOMOGENEOUS cohorts
        #   specialty grouped by argmax-family of the measured per-family profile -> a cohort owns a category
        #   declared  grouped by argmax-family of what agents CLAIM (reads the declaration channel; liars move it)
        self.cohort = str(cohort) if cohort else ("stratify" if self.stratify else "random")
        if self.cohort not in ("random", "stratify", "block", "specialty", "declared"):
            raise ValueError(f"unknown cohort mode {self.cohort!r}")
        if self.cohort == "declared":
            self.needs = frozenset(self.needs | {"declared"})
        # peers observing each verification probe (V)
        self.observers = int(observers) if observers else self.r - 1
        # V: level-0 probes per cell; cached root pick; forwarded per family
        self.b0, self.cached, self.top = b0, bool(cached), int(top)
        # older rows lack observe-time charges (the analyzer adds them)
        self.cnt = {}
        self.stats = {"observe_charged": 1}

    def _cohort_key(self, view, out):
        """Grouping signal per cohort mode; None means random. `out` is outcomes[n, K, b] when the mode needs probes."""
        if self.cohort in ("stratify", "block"):
            return out.mean((1, 2))                                  # scalar measured ability
        if self.cohort == "specialty":
            return out.mean(2)                                       # [n, K] measured per-family profile
        if self.cohort == "declared":
            return np.asarray(view.declared, np.float32)             # [n, K] as CLAIMED - liars move this
        return None

    def _cohorts(self, view, key=None):
        """Cohorts of r (-1 pads the last). With a 1-D key: stratified (one member per stratum) or, for cohort="block",
        contiguous blocks of similar key. With a 2-D key[n, K]: agents whose argmax category agrees sit together."""
        c = np.full(-(-view.n // self.r) * self.r, -1, np.int32)
        if key is None:
            c[:view.n] = view.rng.permutation(view.n)
            return c.reshape(-1, self.r)
        key = np.asarray(key)
        if key.ndim == 2:
            # category grouping: sort by best family, random within
            c[:view.n] = np.lexsort((view.rng.random(view.n), key.argmax(1)))
            return c.reshape(-1, self.r)
        if self.cohort == "block":                                   # homogeneous: sort by ability, random within ties
            c[:view.n] = np.lexsort((view.rng.random(view.n), key))
            return c.reshape(-1, self.r)
        m = len(c) // self.r
        q = view.n - (m - 1) * self.r
        perm = view.rng.permutation(view.n)
        band = perm[q:][np.argsort(key[perm[q:]], kind="stable")].reshape(self.r, m - 1)      # stratum j = row j
        band = np.take_along_axis(band, np.argsort(view.rng.random(band.shape), 1), 1)          # random within stratum
        c[:(m - 1) * self.r] = band.T.ravel()
        c[(m - 1) * self.r:(m - 1) * self.r + q] = perm[:q]
        return c.reshape(-1, self.r)

    def _structure(self, view, key=None):
        """children per level (level 0 = leaf cohorts), parent maps, depth."""
        self.leaves = self._cohorts(view, key)
        self.children, self.parent, m = [self.leaves], [], len(self.leaves)
        while m > 1:
            perm = view.rng.permutation(m).astype(np.int32)              # regroup nodes at random
            par = np.empty(m, np.int32)
            par[perm] = np.arange(m, dtype=np.int32) // self.r
            nxt = np.full(-(-m // self.r) * self.r, -1, np.int32)
            nxt[:m] = perm
            self.parent.append(par)
            self.children.append(nxt.reshape(-1, self.r))
            m = len(self.children[-1])
        self.depth = len(self.children)

    def _verify(self, view, ch, cand, lead, e, slot_child):
        """Re-probe every forwarded candidate (cand int32[M,r*top,K], -1 empty) e times, reported by the r-1 OTHER
        children's representatives lead[M,r] (trimmed by reporter), folded into its running estimate."""
        r, k = self.r, min(self.observers, self.r - 1)
        # valid candidates, their child
        node, slot, fam = np.nonzero(cand >= 0)
        child = slot_child[slot]
        peers = others(r)
        # (V, r-1) OTHER children's reps
        rep_of = np.where(ch >= 0, lead, -1)[node[:, None], peers[child]]
        if (bad := rep_of < 0).any():                                                    # short (padded) node: cycle
            rep_of = np.where(bad, np.where(ch >= 0, lead, lead[:, :1])[node][:, :1].repeat(r - 1, 1), rep_of)
        if k < r - 1:                                                                    # a random k of the r-1 peers
            rep_of = np.take_along_axis(rep_of, np.argsort(view.rng.random(rep_of.shape), 1)[:, :k], 1)
        agents, step = cand[node, slot, fam], max(1, CHUNK_ELEMS // (r * e))
        for lo in range(0, len(agents), step):
            a, f = agents[lo:lo + step], fam[lo:lo + step]
            ex = getattr(self, "excluded", None)                    # audits on: mask the reporters caught lying
            if ex is not None:
                ex = ex[rep_of[lo:lo + step]]
                ex &= ~ex.all(-1, keepdims=True)
            m_new, _ = peer_estimate(view, a, f, e, rep_of[lo:lo + step], self.delta, exclude=ex)
            self.est[a, f] = (self.est[a, f] * self.k[a, f] + m_new * e) / (self.k[a, f] + e)
            self.k[a, f] += e

    def _level0(self, view, cohorts, b, outcomes=None):
        """Level-0 estimates est[n, K]: b probes per cell, reported by the cohort peers, trimmed (the audited engine
        when audit)."""
        if self.audit:
            return self._level0_audited(view, cohorts, b, outcomes)
        return peer_reported_estimates(view, b, cohorts, self.delta, by_reporter=self.verify, observers=self.observers,
                                       outcomes=outcomes)

    def _level0_audited(self, view, cohorts, b, outcomes=None):
        """One round of b probes per (member, family); every probe outcome reported by the s-1 other members (one report
        per peer per probe); a uniform `rate` of instances re-run by the auditor; est = trimmed-over-peers mean of each
        non-excluded peer's mean report."""
        n, K, r = view.n, view.K, cohorts.shape[1]
        self.est = np.zeros((n, K), np.float32)
        self.rsum, self.rcnt = np.zeros((n, K, r - 1), np.float32), np.zeros((n, K, r - 1), np.int32)
        self.peer_of, self.excluded = np.full((n, r - 1), -1, np.int32), np.zeros(n, bool)
        step = max(1, REPORT_ELEMS // (K * r * r * max(b, 1)))
        fam = np.arange(K)[None, :, None]
        for ag in cohort_blocks(cohorts, step):
            C, s = ag.shape
            if s == 1:                                                              # nobody to report: own probes
                own = outcomes[ag[:, 0]] if outcomes is not None else view.probe_many(ag, fam[0], b)
                self.est[ag[:, 0]] = own.mean(-1)
                continue
            peers = others(s)
            rep_ids = ag[:, peers]                                                  # (C, s, s-1) reporter ids
            self.peer_of[ag.ravel(), :s - 1] = rep_ids.reshape(-1, s - 1)
            cidx = np.arange(C)[:, None, None]
            surv = np.broadcast_to(np.arange(s), (C, K, s)).copy()
            mem, f = ag[cidx, surv], np.broadcast_to(fam, (C, K, s))                 # (C, K, s)
            rep = rep_ids[cidx, surv].reshape(-1, s - 1)                             # (V, s-1) reporters of each probe
            if outcomes is not None:
                per = view.report_many(rep[:, :, None], mem.reshape(-1, 1, 1), outcomes[mem, f].reshape(-1, 1, b))
            else:
                _, per = peer_estimate(view, mem.ravel(), f.ravel(), b, rep, self.delta)
            self._audit(view, mem.ravel(), f.ravel(), np.zeros(mem.size, np.int32), rep, per)   # claims per[V, s-1, b]
            per = per.reshape(C, K, s, s - 1, b)
            np.add.at(self.rsum[:, :, :s - 1], (mem, f), per.sum(-1))
            np.add.at(self.rcnt[:, :, :s - 1], (mem, f), b)
            self.est[ag.ravel()] = self._estimates(ag[:, :, None], fam.reshape(1, 1, K), s).reshape(-1, K)
        return self.est

    def _estimates(self, mem, fam, s):
        """Trimmed-over-peers mean of each peer's mean report about mem (index arrays broadcast); excluded peers are
        masked unless every peer of a member is excluded (then all count)."""
        cnt, means = self.rcnt[mem, fam][..., :s - 1], self.rsum[mem, fam][..., :s - 1]
        ex = (cnt == 0) | self.excluded[self.peer_of[mem][..., :s - 1]]
        ex &= ~ex.all(-1, keepdims=True)
        return trimmed_by_reporter((means / np.maximum(cnt, 1))[..., None], self.delta, s, ex)

    def _strike(self, reporters, claims, truth):
        """Count claim != truth per reporter; exclude at STRIKES mismatches; return the ids newly excluded."""
        if not hasattr(self, "hits"):
            self.hits = np.zeros(self.excluded.size, np.int32)
        np.add.at(self.hits, reporters[claims != truth], 1)
        new = (self.hits >= STRIKES) & ~self.excluded
        self.excluded |= new
        return np.flatnonzero(new)

    def _audit(self, view, agents, fams, k, reporters, claims):
        """Re-run a uniform `rate` of the (probe v, pull j) instances; compare every peer's claim with the truth."""
        # claims[V, s-1, p] -> (V, p) draws
        v, j = np.nonzero(view.rng.random(claims[:, 0].shape) < self.rate)
        if v.size:
            # same instance, charged as probes
            truth = view.probe_at(agents[v], fams[v], k[v] + j)
            self._strike(reporters[v], claims[v, :, j], truth[:, None])

    def build(self, view, budget):
        self.view, r, b, K = view, self.r, budget.b, view.K
        # level 0 keeps b0 (default b-1); the rest buys promotions
        self.b = b0 = (max(1, min(b, self.b0 or b - 1))) if self.verify else b
        # these keys are read off probes taken up front
        pre = self.cohort in ("stratify", "block", "specialty")
        # reused by _level0, so the budget is unchanged
        out = probe_outcomes(view, b0) if pre else None
        self._structure(view, self._cohort_key(view, out))
        ok = self.leaves >= 0
        self.leaf_of = np.empty(view.n, np.int32)
        self.leaf_of[self.leaves[ok]] = np.repeat(np.arange(len(self.leaves), dtype=np.int32), r)[ok.ravel()]
        # V: candidates re-verified over all upper levels,
        C = self.top * sum((c >= 0).sum() for c in self.children[1:])
        e = int((b - b0) * view.n // C) if self.verify and C else 0        # e probes each (exact budget)
        self.est = self._level0(view, self.leaves, b0, out)
        # probes / reports behind a build estimate
        self.k = np.full((view.n, K), float(b0), np.float32)
        self.w0 = max(1, (r - 1) * b - 2 * trim_k(self.delta, r, b))
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
                    # children's summaries now
                    valid = self.cand[-1] >= 0
                    self.summary[-1] = np.where(valid, self.est[np.where(valid, self.cand[-1], 0), np.arange(K)], NEG)
            v = np.where(cand >= 0, self.est[np.where(cand >= 0, cand, 0), np.arange(K)], NEG)
            order = np.argsort(-v, axis=1, kind="stable")
            best = (order[:, 0] if l == 0 else slot_child[order[:, 0]]).astype(np.int32)         # best CHILD per family
            self.summary.append(np.take_along_axis(v, order[:, :1], 1)[:, 0])
            self.best.append(best)
            self.cand.append(np.take_along_axis(cand, order[:, :1], 1)[:, 0])                     # (M,K) summary holder
            self.topc.append(np.take_along_axis(cand, order[:, :self.top], 1))                    # (M,top,K) forwarded
            mc = (v if l == 0 else v.reshape(len(ch), self.r, self.top, K).mean(2)).mean(2)      # per-child mean est
            best_child = np.where(ch >= 0, mc, NEG).argmax(1)
            self.lead.append((self.leaves if l == 0 else lead)[np.arange(len(ch)), best_child])
            # one random subtree member per node
            pick = (view.rng.random(len(ch)) * (ch >= 0).sum(1)).astype(int)
            self.rep.append((self.leaves if l == 0 else self.rep[-1][ch])[np.arange(len(ch)), pick])
        # member->leader, node->parent
        view.ledger.message(view.n - len(self.leaves) + sum(len(s) for s in self.summary) - 1)

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
            self.view.ledger.compare(1)
            self.view.ledger.message(2)
            return int(self.cand[-1][0, task.family])
        f, node = int(task.family), 0
        for l in range(self.depth - 1, -1, -1):
            # request down, answer up
            self.view.ledger.hop(1)
            self.view.ledger.compare(self.r)
            self.view.ledger.message(2)
            node = int(self.children[l][node, self._choose(l, node, f)])
        return node

    def _recompute(self, node, f):
        """Recompute best/summary (and cached candidates) for families `f` (int array) on the path from leaf `node`
        up."""
        for l in range(self.depth):
            # observe-time cost: r comparisons + 1 message (child->parent update) per level per family
            self.view.ledger.compare(self.r * len(f))
            self.view.ledger.message(len(f))
            # (r, |f|)
            v = self._values(l, node, f)
            s = v.argmax(0)
            self.best[l][node, f] = s
            self.summary[l][node, f] = v[s, np.arange(len(f))]
            if self.cached:
                picked = self.children[l][node][s]
                self.cand[l][node, f] = picked if l == 0 else self.cand[l - 1][picked, f]
            node = int(self.parent[l][node]) if l + 1 < self.depth else node

    def observe(self, task, agent, outcome):
        """Running mean on est[a,f], then recompute f's summary up a's path (log_r n nodes); with audits, a `rate` of
        outcomes is put to the agent's cohort peers and a newly excluded reporter's cohort is re-aggregated."""
        if self.online:
            f, a = int(task.family), int(agent)
            k = self.cnt[a, f] = self.cnt.get((a, f), self.w0) + 1
            self.est[a, f] += (outcome - self.est[a, f]) / k
            self._recompute(int(self.leaf_of[a]), np.array([f]))
        if not self.audit or self.view.rng.random() >= self.rate:
            return
        peers = self.peer_of[agent]
        peers = peers[peers >= 0]
        if not peers.size:
            return
        claims = self.view.report_many(peers, np.full(peers.shape, agent), np.full(peers.shape, outcome))
        for j in self._strike(peers, claims, outcome):                                  # newly excluded reporter j:
            # re-aggregate its cohort
            members = self.leaves[self.leaf_of[j]]
            members = members[members >= 0]
            K = self.view.K
            self.est[members] = self._estimates(members[:, None], np.arange(K)[None, :], len(members))
            self._recompute(int(self.leaf_of[members[0]]), np.arange(K))                  # one path: the cohort's own

    def churn(self, departed, arrived):
        """Repair (ids reused): each arrived agent is re-probed b times per family, reported by its cohort peers
        (trimmed as at build), and its path recomputed. Per arrival: K*b probes, K*b*(r-1) reports, (r-1)+depth
        messages."""
        view, K = self.view, self.view.K
        self.est[np.setdiff1d(departed, arrived)] = NEG
        for a in np.asarray(arrived, dtype=int):
            leaf = int(self.leaf_of[a])
            peers = self.leaves[leaf]
            peers = peers[(peers >= 0) & (peers != a)]
            if len(peers):
                reporters = np.broadcast_to(peers, (K, len(peers)))
                self.est[a] = peer_estimate(view, np.full(K, a), np.arange(K), self.b, reporters, self.delta)[0]
            else:
                self.est[a] = view.probe_many(a, np.arange(K), self.b).mean(-1)
            self.k[a] = self.b
            self.cnt = {kf: c for kf, c in self.cnt.items() if kf[0] != a}
            view.ledger.message(len(peers) + self.depth)
            self._recompute(leaf, np.arange(K))
