"""RouterBench's MLP predictive router (Hu et al. 2024) on our terms: MIDIAN's probe budget, no report channel.
One regressor (prompt embedding ⊕ agent one-hot) -> success, fit on the n*K*b probes; at route time every agent is
scored on the task's text and the argmax is picked (n comparisons per task). Offline only (sklearn has no masked
multi-output partial fit); the online learned router is knn_router(online=True)."""
import numpy as np
from sklearn.neural_network import MLPRegressor
from .base import Method
from ._learned import probe_set, task_vec


class MLPRouter(Method):
    name = "mlp_router"
    needs = frozenset({"probe"})

    def __init__(self, hidden=128, epochs=30, **p):
        super().__init__(hidden=hidden, epochs=epochs, **p)
        self.hidden, self.epochs = hidden, epochs

    def build(self, view, budget):
        self.view = view
        E, Y, _ = probe_set(view, budget.b)                                # (n, m, d), (n, m)
        n, m, d = E.shape; self.eye = np.eye(n, dtype=np.float32)
        X = np.concatenate([np.repeat(self.eye, m, 0), E.reshape(n * m, d)], 1)
        self.model = MLPRegressor(hidden_layer_sizes=(self.hidden,), max_iter=self.epochs, random_state=0).fit(X, Y.reshape(-1))

    def fetch(self, task):
        self.view.ledger.compare(self.view.n)
        q = task_vec(self.view, task)
        return int(np.argmax(self.model.predict(np.concatenate([self.eye, np.broadcast_to(q, (self.view.n, q.size))], 1))))
