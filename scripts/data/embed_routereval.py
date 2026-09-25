"""Pre-warm the content-keyed embedding / SOTA-shortlist caches for framework grids on a NON-live backend (RouterEval),
so their routing units run on CPU nodes. For every (cell, seed) world the grids define and every distinct retrieval
setting they use, it calls FrameworkMethod._index -- the exact code build() runs -- with RTE_EMBED_CACHE_DIR set.
    python scripts/data/embed_routereval.py re_sl_embed_small re_sl_embed_1k re_sl_embed_5k            # GPU job: fill
    CUDA_VISIBLE_DEVICES= python scripts/data/embed_routereval.py --check re_sl_embed_small ...   # CPU: every file hit
--check forbids the embedder and the reranker, so it passes only if a routing unit would never touch a GPU."""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rte.methods import load_method
from rte.methods.frameworks import _common as C
from rte.run import CELL, blocks, cells, expand, jkey, load_config, method_specs, seeds
from rte.world import World

ap = argparse.ArgumentParser()
ap.add_argument("grids", nargs="+")
ap.add_argument("--check", action="store_true")
a = ap.parse_args()
assert os.environ.get("RTE_EMBED_CACHE_DIR"), "set RTE_EMBED_CACHE_DIR (the routing units must read the same root)"
if a.check:

    def _no(*_a, **_k):
        raise RuntimeError("cache miss: the reranker would run in a routing unit")

    C.FrameworkMethod._rerank = _no
cfg = load_config()
rr, t0, done = None, time.time(), 0
units = []
for g in a.grids:
    for blk in blocks(cfg, g):
        specs = {}
        for s in method_specs(blk):  # the nine frameworks share texts and retrieval: one per setting
            if s["params"].get("retrieval") in C.DENSE:
                specs.setdefault(jkey(s["params"]), s)
        units += [(g, cell, seed, list(specs.values())) for cell in cells(blk) for seed in seeds(blk["seeds"])]
sota = C.FrameworkMethod._sota_table
# two passes so a 20 GB GPU slice suffices: the embedder alone (shortlist tables skipped), freed, then the reranker
# alone
for stage in ("embed", "rerank") if not a.check else ("check",):
    C.FrameworkMethod._sota_table = (lambda self, view: None) if stage == "embed" else sota
    for g, cell, seed, specs in units:
        world = World(
            **{k: cell[k] for k in CELL if k not in ("b", "Q")},
            seed=seed,
            backend_kwargs=expand(cell["backend_kwargs"]) or None,
        )
        for s in specs:
            m = load_method(s["name"])(**s["params"])
            m._rr = rr
            m._index(world.view(m.needs))
            rr = m._rr or rr  # one reranker for the whole pass
            done += 1
        print(
            f"[{stage} {time.time() - t0:.0f}s] {g} n={cell['n']} beta={cell['beta']} seed={seed}: {len(specs)} "
            f"settings "
            f"-> {m._popdir(world.view(m.needs))}",
            flush=True,
        )
    if stage == "embed":
        import torch

        from rte.methods._learned import _models

        _models.clear()
        torch.cuda.empty_cache()
print(f"EMBED_ROUTEREVAL_{'CHECK_' if a.check else ''}OK {done} (world, setting) pairs")
