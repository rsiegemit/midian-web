"""Replay backend: RouterBench's real models x K real-eval categories,
pre-recorded outcomes. A thin table lookup — agents are (model, per-category
handicap mask) profiles; execution never leaves the table. Swap datasets by
pointing `cells_path` at a different npz with the same layout, nothing else.

Cell table (`scripts/02_download_routerbench.py` builds it):
    model_names(M,) category_names(K_full,) str; offsets(K_full+1,) n_prompts(K_full,) int64;
    outcomes(sum(n_prompts), M) int8 -- outcomes[offsets[c]:offsets[c+1]] is category c.

See DEVIATIONS.md (2026-09-02) for: K=67 not 64 on the real pickle, the per-dist
profile draw, and the masked-category (handicap) rule.
"""
from __future__ import annotations

import os

import numpy as np

from . import noisy_declared
from ._profiles import pick_k_per_agent, group_mask

DEFAULT_CELLS_PATH = os.path.join(
    os.environ.get("RTE_DATA", os.path.expanduser("~/rte_data")), "data", "routerbench_cells.npz")


def _draw_profiles(dist: str, n: int, K: int, model_rank: np.ndarray, rng: np.random.Generator):
    """(model_id[n] int8, mask[n,K] bool). model_rank = models sorted best->worst overall."""
    M = len(model_rank)
    half = max(1, (M + 1) // 2)
    strong_half, weak_half = model_rank[:half], model_rank[half:] if M > half else model_rank
    model_id = rng.integers(0, M, size=n).astype(np.int8)
    mask = np.zeros((n, K), dtype=bool)

    if dist == "specialist":
        mask = ~pick_k_per_agent(n, K, 3, rng)
    elif dist == "heavy_tail":
        strong = rng.random(n) < 0.1
        model_id[strong] = model_rank[0]
        model_id[~strong] = weak_half[rng.integers(0, weak_half.size, size=int((~strong).sum()))]
        mask = ~pick_k_per_agent(n, K, 1, rng)          # weak: 1 lucky unmasked category
        mask[strong, :] = False                          # strong: fully unmasked
    elif dist == "bimodal":
        strong = rng.random(n) < 0.2
        model_id[strong] = strong_half[rng.integers(0, strong_half.size, size=int(strong.sum()))]
        model_id[~strong] = weak_half[rng.integers(0, weak_half.size, size=int((~strong).sum()))]
        # no masking: good/bad realized entirely by model choice (see DEVIATIONS.md)
    elif dist == "correlated":
        mask = group_mask(n, K, 4, 0.5, rng)
    elif dist == "iid_uniform":
        mask = rng.random((n, K)) < 0.5
    else:
        raise ValueError(f"unknown dist {dist!r} for replay backend")

    return model_id, mask


class ReplayBackend:
    def __init__(self, n: int, K: int, dist: str, seed: int, rng: np.random.Generator,
                 cells_path: str | None = None, **_):
        self.n, self.seed, self.dist = int(n), int(seed), dist
        d = np.load(cells_path or DEFAULT_CELLS_PATH)
        self.model_names = [str(x) for x in d["model_names"]]
        cat_names = [str(x) for x in d["category_names"]]
        offsets, n_prompts_full = d["offsets"].astype(np.int64), d["n_prompts"].astype(np.int64)
        self._outcomes = d["outcomes"].astype(np.int8)                 # (total_rows, M)

        sel = np.sort(np.argsort(-n_prompts_full, kind="stable")[:min(int(K), len(cat_names))])
        self.K = len(sel)
        self.families = [cat_names[i] for i in sel]
        self._row_start, self._n_prompts = offsets[sel], n_prompts_full[sel]

        model_cat_acc = np.stack([self._outcomes[s:s + c].mean(0)
                                  for s, c in zip(self._row_start, self._n_prompts)], axis=1)  # (M,K)
        self._model_cat_acc = model_cat_acc.astype(np.float32)
        self._weakest_model = np.argmin(model_cat_acc, axis=0)         # (K,) per-category weakest
        model_rank = np.argsort(-model_cat_acc.mean(axis=1))           # (M,) best -> worst overall

        self._model_rank = model_rank
        self.model_id, self.mask = _draw_profiles(dist, self.n, self.K, model_rank, rng)
        self._S = self._skill(self.model_id, self.mask)

    def _skill(self, model_id, mask):
        own = self._model_cat_acc[model_id, :]
        weak = self._model_cat_acc[self._weakest_model, np.arange(self.K)]
        return np.where(mask, weak, own).astype(np.float32)

    # ---- churn
    def snapshot(self): return (self.model_id.copy(), self.mask.copy())
    def restore(self, snap): self.model_id, self.mask = (x.copy() for x in snap); self._S = self._skill(self.model_id, self.mask)
    def redraw(self, ids, rng):
        self.model_id[ids], self.mask[ids] = _draw_profiles(self.dist, len(ids), self.K, self._model_rank, rng)
        self._S = self._skill(self.model_id, self.mask)

    def true_skill(self) -> np.ndarray:
        return self._S

    def declared(self, source: str = "programmatic") -> np.ndarray:
        return noisy_declared(self._S, self.seed)                       # no LLM here: both sources are the honest control

    def execute(self, a: int, task) -> int:
        f = task.family
        row = int(self._row_start[f]) + task.instance % int(self._n_prompts[f])
        m = int(self._weakest_model[f]) if self.mask[a, f] else int(self.model_id[a])
        return int(self._outcomes[row, m])

    def execute_many(self, agents, families, inst) -> np.ndarray:
        agents, families, inst = np.broadcast_arrays(np.asarray(agents), np.asarray(families), np.asarray(inst))
        rows = self._row_start[families] + inst % self._n_prompts[families]
        model_idx = np.where(self.mask[agents, families], self._weakest_model[families], self.model_id[agents])
        return self._outcomes[rows, model_idx].astype(np.int8)

    def stats(self) -> dict:
        return {"replay_K_used": self.K, "replay_n_models": len(self.model_names), "replay_dist": self.dist}
