"""MIDIAN-SH (labeled variant, 2026-09-03): plain MIDIAN whose level-0 estimation runs SUCCESSIVE HALVING inside every
cohort, per family, through the same trimmed peer-report channel. Round 1 probes all s members, keeps the top
ceil(s/2) by trimmed peer estimate, round 2 probes the survivors again, ... until one remains; the per-round pulls are
chosen so the cohort spends exactly the s*b probes plain MIDIAN spends (any remainder goes to the winner). Every probe
outcome is reported by the s-1 other members (one report per peer per probe); est[m, f] is the trimmed mean over
PEERS of each peer's mean report about m (`_est.trimmed_by_reporter`, with MIDIAN-A's exclusion mask). Tree,
descent, online updates: plain MIDIAN. `halving=False` is plain MIDIAN's single round of b probes through this same
engine (MIDIAN-A builds on it). Overrides only `Midian._level0`."""
import numpy as np

from ._est import REPORT_ELEMS, trimmed_by_reporter
from .midian import Midian


def _schedule(s, b, halving):
    """[(survivors, pulls each)] per round, spending exactly s*b probes (remainder -> the winner)."""
    sizes = [s]
    while halving and sizes[-1] > 1:
        sizes.append(-(-sizes[-1] // 2))
    rounds, rem = [], s * b
    for t, sz in enumerate(sizes[:-1] if halving else sizes):
        p = max(1, rem // (sz * (len(sizes) - 1 - t))) if halving else b
        rounds.append((sz, p)); rem -= sz * p
    return rounds + ([(1, rem)] if rem else [])


class MidianSH(Midian):
    name = "midian_sh"

    def __init__(self, halving=True, **p):
        super().__init__(halving=halving, **p)
        self.halving = bool(halving)

    def _audit(self, view, ag, fam, surv, k0, rep_ids, out, rep):
        """Report audits (MIDIAN-A overrides). Exclusions land in self.excluded."""

    def _level0(self, view, cohorts, b, outcomes=None):
        if outcomes is not None and self.halving:
            raise ValueError("midian_sh: stratify (pre-drawn probes) and halving cannot be combined")
        n, K, r = view.n, view.K, cohorts.shape[1]
        self.est = np.zeros((n, K), np.float32)
        self.rsum, self.rcnt = np.zeros((n, K, r - 1), np.float32), np.zeros((n, K, r - 1), np.int32)
        self.peer_of, self.excluded = np.full((n, r - 1), -1, np.int32), np.zeros(n, bool)
        short = cohorts[-1, -1] < 0
        full = cohorts[:len(cohorts) - short]
        step = max(1, REPORT_ELEMS // (K * r * r * max(b, 1)))
        blocks = [full[lo:lo + step] for lo in range(0, len(full), step)] + ([cohorts[-1][cohorts[-1] >= 0][None]] if short else [])
        fam = np.arange(K)[None, :, None]
        for ag in blocks:
            C, s = ag.shape
            if s == 1:                                                              # nobody to report: own probes
                self.est[ag[:, 0]] = (outcomes[ag[:, 0]] if outcomes is not None else view.probe_many(ag, fam[0], b)).mean(-1)
                continue
            peers = np.array([[j for j in range(s) if j != m] for m in range(s)], np.int32)
            rep_ids = ag[:, peers]                                                  # (C, s, s-1) reporter ids
            self.peer_of[ag.ravel(), :s - 1] = rep_ids.reshape(-1, s - 1)
            cidx = np.arange(C)[:, None, None]
            surv = np.broadcast_to(np.arange(s), (C, K, s)).copy()                  # member slots still in the race
            k0 = np.zeros((C, K, s), np.int32)                                      # probes taken so far per slot
            for sz, p in _schedule(s, b, self.halving):
                mem = ag[cidx, surv]                                                # (C, K, sz)
                out = outcomes[mem, fam] if outcomes is not None else view.probe_many(mem, fam, p)   # (C, K, sz, p)
                rep = view.report_many(rep_ids[cidx, surv][..., None], mem[..., None, None], out[..., None, :])
                self._audit(view, ag, fam, surv, k0, rep_ids, out, rep)              # rep: (C, K, sz, s-1, p)
                np.add.at(self.rsum[:, :, :s - 1], (mem, fam), rep.sum(-1)); np.add.at(self.rcnt[:, :, :s - 1], (mem, fam), p)
                np.add.at(k0, (cidx, fam, surv), p)
                if sz > 1:                                                          # keep the top half by estimate
                    e = self._estimates(mem, fam, s)
                    surv = np.take_along_axis(surv, np.argsort(-e, axis=-1, kind="stable")[..., :-(-sz // 2)], -1)
            self.est[ag.ravel()] = self._estimates(ag[:, :, None], fam.reshape(1, 1, K), s).reshape(-1, K)
        return self.est

    def _estimates(self, mem, fam, s):
        """Trimmed-over-peers mean of each peer's mean report about mem (index arrays broadcast); excluded peers are
        masked unless every peer of a member is excluded (then all count)."""
        cnt, means = self.rcnt[mem, fam][..., :s - 1], self.rsum[mem, fam][..., :s - 1]
        ex = (cnt == 0) | self.excluded[self.peer_of[mem][..., :s - 1]]
        ex &= ~ex.all(-1, keepdims=True)
        return trimmed_by_reporter((means / np.maximum(cnt, 1))[..., None], self.delta, s, ex)
