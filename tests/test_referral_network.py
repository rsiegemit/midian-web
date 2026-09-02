"""referral_network's belief store is only meaningful if the neighbour relation is exactly
d-regular and symmetric under the partner slot `s ^ 1`; the per-edge coverage is b/d by design."""
import numpy as np

from rte.budget import Budget
from rte.methods.referral_network import ReferralNetwork, regular_graph
from rte.world import World


def test_graph_is_d_regular_and_beliefs_are_sparse_by_construction():
    w = World(200, 4, "specialist", 0.25, seed=3)
    m = ReferralNetwork()
    w.ledger.reset()
    m.build(w.view(m.needs), Budget(3))
    n, d, s = w.n, m.d, np.arange(m.d)
    assert w.ledger.probes == Budget(3).total_probes(n, w.K) == w.ledger.reports  # one observer each
    assert m.nbr.shape == (n, d) and m.nbr.min() >= 0 and m.nbr.max() < n
    assert np.array_equal(m.nbr[m.nbr, s ^ 1][np.arange(n)[:, None], s],
                          np.repeat(np.arange(n)[:, None], d, axis=1))
    assert m.belief.shape == (n, d, w.K)
    assert 0.0 < (m.belief > 0).any(axis=2).mean() < 1.0
