"""LinUCB-honest: needs only probes, its context is a function of observed outcomes alone, budget and validity."""
import numpy as np

from rte.budget import Budget
from rte.methods.linucb_honest import LinUcbHonest
from rte.world import AccessError, View, World


def test_linucb_no_oracle_features():
    w = World(100, 16, "specialist", 0.25, seed=1)
    m = LinUcbHonest()
    assert m.needs == frozenset({"probe"})
    view = w.view(m.needs)
    before = w.ledger.snapshot(); m.build(view, Budget(3)); build = w.ledger.diff(before)
    assert build["probes"] == 100 * 16 * 3 and build["reports"] == 0
    try:
        view.declared; raised = False
    except AccessError:
        raised = True
    assert raised, "the view must refuse the declared channel to this method"
    x = m.features(0)
    assert x.shape == (100, 4)
    m2 = LinUcbHonest(); m2.cnt, m2.mean, m2.view = m.cnt.copy(), m.mean.copy(), view      # same histories ->
    assert np.array_equal(m2.features(0), x)                                                # same features
    m2.mean[:, 0] = m2.mean[::-1, 0]; m2.cnt[:, 0] = m2.cnt[::-1, 0]                        # different histories ->
    assert not np.array_equal(m2.features(0)[:, 1], x[:, 1])                                # different features
    s = np.mean([w.execute(m.fetch(t), t) for t in w.tasks(200)])
    assert s > 0.5
