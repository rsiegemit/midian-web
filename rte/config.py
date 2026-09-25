"""Paths and environment switches, in one place.

Every module reads its data root and its optional switches from here, never from `os.environ` directly.

    RTE_DATA            data root: populations/, data/, models/, cache/, results/, endpoints.json.
                        Default ~/rte_data. Set it explicitly on a cluster.
    RTE_WORKERS         processes for non-LLM grids (rte.run --workers overrides).
    RTE_FW_PARALLEL     concurrent framework requests per unit (identical picks; speed only).
    RTE_TEXT_PROCS      processes that render prompt texts for the learned routers (speed only).
    RTE_EMBED_BATCH     embed prompts in batches (speed only; embeddings differ by <= 2.4e-7 from the per-prompt path).
    RTE_MINILM_CUDA     run MiniLM on a GPU when one is visible (speed only).
    RTE_OUTCOME_CACHE   cache scored outcomes within a process (speed only; identical outcomes).
    RTE_RG_CACHE        cache Reasoning-Gym data files within a process (speed only; identical tasks).

Path-valued switches, read where they are used (defaults under RTE_DATA or the repo):
    RTE_ENDPOINTS          the merged model -> URL registry (default $RTE_DATA/endpoints.json).
    RTE_ENDPOINT_DIR       per-replica endpoint files merged into it (default endpoints.d/ beside RTE_ENDPOINTS).
    RTE_LLM_CACHE          the LLM reply memo, sqlite shards (default $RTE_DATA/cache).
    RTE_LLM_CACHE_NOLOCK   open the memo shards with sqlite nolock=1 (default 1: NFS; 0 uses normal file locking).
    RTE_POPULATIONS        the live-backend agent populations (default $RTE_DATA/populations).
    RTE_MODELS_YAML        the model ladder (default configs/models.yaml).
    RTE_EMBED_CACHE_DIR    opt-in embedding/shortlist cache for non-live backends (unset: no cache).

The speed-only switches are off by default. Stored rows of three kNN grids were produced with RTE_EMBED_BATCH=1
(docs/errata.md), so reproducing those rows bit-for-bit needs the same setting.
"""
import os
from pathlib import Path

RTE_DATA = Path(os.environ.get("RTE_DATA", Path.home() / "rte_data"))


def flag(name: str) -> bool:
    """An on/off switch: on only when the variable is exactly "1"."""
    return os.environ.get(name) == "1"


def count(name: str, default: int = 1) -> int:
    """A process or concurrency count."""
    return int(os.environ.get(name, str(default)))
