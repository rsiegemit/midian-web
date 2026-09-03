"""knn_router with an exact family embedding and k = b reduces to flat_probe_argmax: same probes, same picks."""
import numpy as np, re
import rte.methods._learned as L
from rte.budget import Budget
from rte.methods.flat_probe_argmax import FlatProbeArgmax
from rte.methods.knn_router import KNNRouter
from rte.world import World


def test_knn_router_equals_flat_with_family_embedding(monkeypatch):
    w = World(60, 8, "specialist", 0.0, seed=3); names = list(w.families)
    def fam_onehot(texts):
        E = np.zeros((len(texts), len(names)), np.float32)
        for i, t in enumerate(texts): E[i, names.index(re.search(r"family (.+?) \(", t).group(1))] = 1
        return E
    monkeypatch.setattr(L, "embed", fam_onehot); monkeypatch.setattr("rte.methods.knn_router.embed", fam_onehot)
    flat, knn = FlatProbeArgmax(), KNNRouter()
    w.reset(); flat.build(w.view(flat.needs), Budget(3)); w.reset(); knn.build(w.view(knn.needs), Budget(3))
    tasks = w.tasks(50)
    assert [flat.fetch(t) for t in tasks] == [knn.fetch(t) for t in tasks]
