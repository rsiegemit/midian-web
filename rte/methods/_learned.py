"""Shared pieces of the learned (RouterBench-style) routers: a local text embedder and the probe set they train on.

The routers train on exactly MIDIAN's probe budget (b probes per agent per family), but keep the prompt text of every
probe: a router "on our terms" learns a map (prompt, agent) -> outcome from the probes it paid for, then scores every
agent on the incoming task's text. Embeddings are all-MiniLM-L6-v2 on CPU (the same model scripts/routerbench_terms.py
uses for RouterBench's own routers); the embedding arithmetic is not in the ledger, like the frameworks' TF-IDF shortlist."""
import os, numpy as np
from ._est import CHUNK

os.environ.setdefault("HF_HOME", os.path.join(os.environ.get("RTE_DATA", os.path.expanduser("~/rte_data")), "hf_cache"))
_model = None


def embed(texts) -> np.ndarray:
    """(len(texts), 384) unit-norm float32 embeddings."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _model.encode(list(texts), batch_size=256, show_progress_bar=False, normalize_embeddings=True).astype(np.float32)


def probe_set(view, b: int):
    """Probe every agent b times per family, keeping the prompts: E[n, K*b, d] embeddings, Y[n, K*b] outcomes, F[K*b] family."""
    Y, I = np.zeros((view.n, view.K, b)), np.zeros((view.n, view.K, b), np.int64)
    for f in range(view.K):
        for lo in range(0, view.n, CHUNK):
            Y[lo:lo + CHUNK, f], I[lo:lo + CHUNK, f] = view.probe_text(np.arange(lo, min(view.n, lo + CHUNK)), f, b)
    E = np.stack([vec(view, f, i, True) for a in range(view.n) for f in range(view.K) for i in I[a, f]]).reshape(view.n, view.K * b, -1)
    return E, Y.reshape(view.n, view.K * b), np.repeat(np.arange(view.K), b)


def vec(view, f, inst, probe=False) -> np.ndarray:
    """The prompt's embedding: the backend's own vectors when it has them (routereval), else MiniLM over its text."""
    e = view.embedding(f, inst, probe)
    return e if e is not None else embed([view.text(f, inst, probe)])[0]


def task_vec(view, task) -> np.ndarray:
    return vec(view, task.family, task.instance)
