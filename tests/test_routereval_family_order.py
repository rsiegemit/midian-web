"""RouterEval MMLU families must not depend on PYTHONHASHSEED ('high school biology' and 'philosophy' tie at 248 train
prompts)."""
import os
import subprocess
import sys

import pytest

from midian.backends.routereval import DATA

CODE = (
    "import sys; sys.path.insert(0, '.'); from midian.world import World; "
    "w = World(backend='routereval', n=10, K=16, dist='strong_to_weak', beta=0.0, liar_select='random', collude=True, "
    "declared_source='programmatic', lie_mode='inflate', demand='uniform', seed=1, "
    "backend_kwargs={'dataset': 'mmlu'}); "
    "print('|'.join(w.backend.families))"
)


@pytest.mark.skipif(not os.path.exists(f"{DATA}/mmlu_router_dataset.pkl"), reason="RouterEval data not staged")
def test_family_order_is_hash_seed_independent():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outs = {subprocess.run([sys.executable, "-c", CODE], cwd=root, capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": str(h)}).stdout for h in (0, 6, 8, 9, 11)}
    assert len(outs) == 1, outs
