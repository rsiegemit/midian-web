"""dedup=True offers one agent per DISTINCT description text; dedup=False is the pre-registered top-k, untouched.
Descriptions are forced into a few templates so clones dominate the top-k, as they do on the live backend at scale."""
from __future__ import annotations

import numpy as np

from rte.budget import Budget
from rte.methods.frameworks._common import FrameworkMethod
from rte.world import World

N, K, Q = 300, 16, 12


class _Fw(FrameworkMethod):
    name, env, worker = "fw_test", "rte", "echo_worker.py"

    def _texts(self, view):                       # 3 templates per family -> ~6 clones of every text
        D, fams = view.declared, list(view.families)
        desc = [f"expert in {fams[int(np.argmax(D[a]))]} level {int(D[a].max() * 3)}" for a in range(view.n)]
        return desc, [f"Tasks of family {f}" for f in fams], (lambda task: f"task {task.family}")


def _built(**kw):
    m = _Fw(base_url="http://127.0.0.1:1/v1", **kw)
    m.build(World(N, K, "specialist", 0.0, seed=1).view(m.needs), Budget(1))
    m.bridge.select = lambda *a, **k: {"choice": None, "error": None, "raw": None}
    return m


def test_dedup_offers_k_distinct_descriptions():
    m = _built(dedup=True)
    for task in World(N, K, "specialist", 0.0, seed=1).tasks(Q):
        top = m.retrieve(task)
        texts = [m.desc[a] for a in top]
        assert len(set(texts)) == len(texts) == min(m.k, len(set(m.desc)))
        sims = m._Xa @ m._Xf[task.family]
        ref = list(dict.fromkeys(m.desc[a] for a in np.argsort(-sims, kind="stable")))[:len(top)]
        assert texts == ref                                    # same ranking as the plain top-k, clones collapsed
        assert all(a == min(np.flatnonzero(np.array(m.desc) == m.desc[a])) for a in top)   # lowest id represents


def test_plain_topk_is_unchanged_and_clone_filled():
    m = _built()
    for task in World(N, K, "specialist", 0.0, seed=1).tasks(Q):
        top = m.retrieve(task)
        sims = m._Xa @ m._Xf[task.family]
        assert list(top) == list(np.argsort(-sims, kind="stable")[:m.k])
        assert len({m.desc[a] for a in top}) < len(top)      # the pre-registered adapter does return clones here
