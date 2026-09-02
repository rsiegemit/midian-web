"""Replay backend: RouterBench's 11 real models x K real-eval categories,
pre-recorded outcomes. Agents are (model, per-category handicap mask) profiles
drawn per `dist`; execution is a table lookup, never a fresh generation.

Real-model outcomes at zero inference cost; used for n up to 1e6 by sharding
models into specialty profiles (SPEC.md sec3). See DEVIATIONS.md (2026-09-02
entries) for: why K=67 not 64 on the actual RouterBench pickle, the per-dist
profile-draw rule, and the handicap (masked-category) rule.

Cell table (`scripts/02_download_routerbench.py` builds it):
    model_names    (M,)   str
    category_names (K_full,) str
    offsets        (K_full+1,) int64   outcomes[offsets[c]:offsets[c+1]] is category c
    n_prompts      (K_full,) int64
    outcomes       (sum(n_prompts), M) int8
"""
from __future__ import annotations

import os

import numpy as np

from ..stable_hash import stable_seed_32

DEFAULT_CELLS_PATH = os.path.join(
    os.environ.get("RTE_DATA", os.path.expanduser("~/rte_data")), "data", "routerbench_cells.npz")


def _draw_profiles(dist: str, n: int, K: int, model_rank: np.ndarray, rng: np.random.Generator):
    """Vectorized (model_id[n] int8, mask[n,K] bool). model_rank = model indices sorted
    best -> worst by overall mean accuracy. See DEVIATIONS.md for the per-dist rule."""
    M = len(model_rank)
    model_id = np.zeros(n, dtype=np.int8)
    mask = np.zeros((n, K), dtype=bool)
    strongest = int(model_rank[0])
    half = (M + 1) // 2
    strong_half, weak_half = model_rank[:half], model_rank[half:]
    if weak_half.size == 0:            # degenerate M=1
        weak_half = model_rank

    if dist == "specialist":
        model_id[:] = rng.integers(0, M, size=n)
        k = min(3, K)
        keys = rng.random((n, K))
        unmask_idx = np.argsort(-keys, axis=1)[:, :k]
        mask[:, :] = True
        mask[np.arange(n)[:, None], unmask_idx] = False

    elif dist == "heavy_tail":
        strong = rng.random(n) < 0.1
        model_id[strong] = strongest
        model_id[~strong] = weak_half[rng.integers(0, weak_half.size, size=int((~strong).sum()))]
        mask[:, :] = True
        mask[strong, :] = False
        weak_rows = np.flatnonzero(~strong)
        unmask_idx = np.argmax(rng.random((n, K)), axis=1)     # 1 lucky unmasked category
        mask[weak_rows, unmask_idx[weak_rows]] = False

    elif dist == "bimodal":
        strong = rng.random(n) < 0.2
        model_id[strong] = strong_half[rng.integers(0, strong_half.size, size=int(strong.sum()))]
        model_id[~strong] = weak_half[rng.integers(0, weak_half.size, size=int((~strong).sum()))]
        # no masking: the good/bad split is realized entirely by model choice (see DEVIATIONS.md)

    elif dist == "correlated":
        model_id[:] = rng.integers(0, M, size=n)
        G = 4
        group_of_family = np.arange(K) % G
        group_mask = rng.random((n, G)) < 0.5
        mask[:, :] = group_mask[:, group_of_family]

    elif dist == "iid_uniform":
        model_id[:] = rng.integers(0, M, size=n)
        mask[:, :] = rng.random((n, K)) < 0.5

    else:
        raise ValueError(f"unknown dist {dist!r} for replay backend")

    return model_id, mask


class ReplayBackend:
    def __init__(self, n: int, K: int, dist: str, seed: int, rng: np.random.Generator,
                 cells_path: str | None = None, **_):
        self.n = int(n)
        self.seed = int(seed)
        self.dist = dist
        path = cells_path or DEFAULT_CELLS_PATH
        d = np.load(path)
        self.model_names = [str(x) for x in d["model_names"]]
        cat_names_full = [str(x) for x in d["category_names"]]
        offsets_full = d["offsets"].astype(np.int64)
        n_prompts_full = d["n_prompts"].astype(np.int64)
        self._outcomes = d["outcomes"].astype(np.int8)         # (total_rows, M), never copied per-agent
        M = len(self.model_names)
        K_full = len(cat_names_full)

        # deterministic category subset: most-prompts-first (documented in DEVIATIONS.md)
        order = np.argsort(-n_prompts_full, kind="stable")
        K_use = min(int(K), K_full)
        sel = np.sort(order[:K_use])
        self.K = int(K_use)
        self.families = [cat_names_full[i] for i in sel]
        self._row_start = offsets_full[sel]                    # (K,)
        self._n_prompts = n_prompts_full[sel]                   # (K,)

        model_cat_acc = np.empty((M, self.K), dtype=np.float64)
        for k in range(self.K):
            lo, hi = int(self._row_start[k]), int(self._row_start[k] + self._n_prompts[k])
            model_cat_acc[:, k] = self._outcomes[lo:hi].mean(axis=0)
        self._model_cat_acc = model_cat_acc.astype(np.float32)          # (M,K)
        self._weakest_model = np.argmin(model_cat_acc, axis=0).astype(np.int64)   # (K,) per-category weakest

        model_rank = np.argsort(-model_cat_acc.mean(axis=1))            # best -> worst, overall
        self.model_id, self.mask = _draw_profiles(dist, self.n, self.K, model_rank, rng)

        self.declared_noise = 0.05
        self._S = self._compute_true_skill()

    def _compute_true_skill(self) -> np.ndarray:
        own = self._model_cat_acc[self.model_id, :]                              # (n,K)
        weak = self._model_cat_acc[self._weakest_model, np.arange(self.K)][None, :]  # (1,K) broadcast
        return np.where(self.mask, weak, own).astype(np.float32)

    def true_skill(self) -> np.ndarray:
        return self._S

    def declared(self, source: str = "programmatic") -> np.ndarray:
        if source not in ("programmatic", "self_described"):
            raise ValueError(source)
        # no LLM in this backend to self-describe (same deviation as bernoulli.py)
        rng = np.random.default_rng(stable_seed_32(self.seed, "replay_declared"))
        return np.clip(self._S + rng.normal(0, self.declared_noise, size=self._S.shape), 0, 1).astype(np.float32)

    def execute(self, a: int, task) -> int:
        f = task.family
        n_prompts = int(self._n_prompts[f])
        idx = task.instance % n_prompts
        row = int(self._row_start[f]) + idx
        m = int(self._weakest_model[f]) if self.mask[a, f] else int(self.model_id[a])
        return int(self._outcomes[row, m])

    def execute_many(self, agents, families, reps: int, rng: np.random.Generator) -> np.ndarray:
        agents = np.asarray(agents, dtype=np.int64)
        families = np.asarray(families, dtype=np.int64)
        agents, families = np.broadcast_arrays(agents, families)
        reps = int(reps)
        n_prompts = self._n_prompts[families]                                       # shape = agents.shape
        u = rng.random(agents.shape + (reps,))
        idx = np.minimum((u * n_prompts[..., None]).astype(np.int64), n_prompts[..., None] - 1)
        rows = self._row_start[families][..., None] + idx
        own = self.model_id[agents][..., None].astype(np.int64)
        weak = self._weakest_model[families][..., None]
        use_weak = self.mask[agents, families][..., None]
        model_idx = np.broadcast_to(np.where(use_weak, weak, own), rows.shape)
        return self._outcomes[rows, model_idx].astype(np.int8)

    def stats(self) -> dict:
        return {"replay_K_used": self.K, "replay_n_models": len(self.model_names),
                "replay_dist": self.dist}
