"""lie_mode='max': liars declare perfect skill in every family; honest agents and the other modes are untouched."""
import numpy as np
from rte.world import apply_lying


def test_max_lie_sets_every_liar_claim_to_one_and_leaves_the_rest():
    rng = np.random.default_rng(0); D = rng.uniform(0, 1, (50, 16)).astype(np.float32); liars = np.zeros(50, bool); liars[::3] = True
    L = apply_lying(D, liars, "max")
    assert (L[liars] == 1.0).all() and np.array_equal(L[~liars], D[~liars]) and not np.shares_memory(L, D)
    assert np.array_equal(apply_lying(D, liars, "inflate"), np.clip(np.where(liars[:, None], D + 0.4, D), 0, 1))
