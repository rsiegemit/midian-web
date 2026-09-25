"""Download RouterBench (withmartian/routerbench, 0-shot pickle) and normalize it into a compact cell table for
`rte.backends.replay`.   python scripts/data/02_download_routerbench.py      (login node: needs internet)

Category definition (as the reference implementation's `_normalize_routerbench`): category = the raw `eval_name`
(each MMLU subject is its own eval_name, e.g. "mmlu-abstract-algebra"); keep categories with >= MIN_CATEGORY_SAMPLES
(60) rows; a model's score counts as correct at >= 0.5.

At that threshold 67 categories survive for THIS pickle, not the 64 the CONTRACT prose names; main() prints the count
and the difference is recorded in DEVIATIONS.md. The code's rule is followed, not the prose number.

Output: $RTE_DATA/data/routerbench_cells.npz, a CSR-style compact table:
    model_names   (M,)  str            M=11 RouterBench models
    category_names(K,)  str            K categories (eval_name, filtered)
    offsets       (K+1,) int64         outcomes[offsets[k]:offsets[k+1]] is category k's rows
    n_prompts     (K,)  int64          = diff(offsets)
    outcomes      (sum(n_prompts), M) int8   binary correct/incorrect per (prompt, model)
"""

from __future__ import annotations

import os
import sys
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rte.config import RTE_DATA  # noqa: E402

DATA_DIR = os.path.join(RTE_DATA, "data")
PICKLE_URL = "https://huggingface.co/datasets/withmartian/routerbench/resolve/main/routerbench_0shot.pkl"
RAW_PATH = os.path.join(DATA_DIR, "routerbench_0shot.pkl")
OUT_PATH = os.path.join(DATA_DIR, "routerbench_cells.npz")
MIN_CATEGORY_SAMPLES = 60  # matches the old repo's DataConfig.min_category_samples


def ensure_raw_pickle() -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(RAW_PATH):
        return RAW_PATH
    part = RAW_PATH + ".part"
    print(f"[download] {PICKLE_URL} -> {RAW_PATH}")
    urllib.request.urlretrieve(PICKLE_URL, part)
    os.replace(part, RAW_PATH)  # atomic: never leave a truncated file at RAW_PATH
    return RAW_PATH


def normalize(raw_path: str) -> dict:
    df = pd.read_pickle(raw_path)
    cost_cols = [c for c in df.columns if c.endswith("|total_cost")]
    models = [c[: -len("|total_cost")] for c in cost_cols]
    models = sorted(models)  # deterministic order

    counts = df["eval_name"].value_counts()
    keep_categories = sorted(counts[counts >= MIN_CATEGORY_SAMPLES].index.tolist())

    offsets = [0]
    outcome_chunks = []
    for cat in keep_categories:
        sub = df[df["eval_name"] == cat]
        block = np.stack([(sub[m].to_numpy(dtype=float) >= 0.5).astype(np.int8) for m in models], axis=1)
        outcome_chunks.append(block)
        offsets.append(offsets[-1] + block.shape[0])

    outcomes = np.concatenate(outcome_chunks, axis=0)  # (total_prompts, M) int8
    offsets = np.array(offsets, dtype=np.int64)
    n_prompts = np.diff(offsets)

    return {
        "model_names": np.array(models),
        "category_names": np.array(keep_categories),
        "offsets": offsets,
        "n_prompts": n_prompts,
        "outcomes": outcomes,
    }


def main():
    raw_path = ensure_raw_pickle()
    cells = normalize(raw_path)
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(OUT_PATH, **cells)
    K = len(cells["category_names"])
    M = len(cells["model_names"])
    print(
        f"[normalize] wrote {OUT_PATH}: {M} models, {K} categories, "
        f"{cells['outcomes'].shape[0]} total (prompt, model) rows"
    )
    if K != 64:
        print(
            f"[normalize] NOTE: K={K} categories survive at min_category_samples="
            f"{MIN_CATEGORY_SAMPLES}, not the 64 named in the task prose. "
            f"Following the old code's actual behavior; see DEVIATIONS.md."
        )


if __name__ == "__main__":
    main()
