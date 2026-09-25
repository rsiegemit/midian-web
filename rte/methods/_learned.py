"""Shared pieces of the learned (RouterBench-style) routers: a local text embedder and the probe set they train on.

The routers train on exactly MIDIAN's probe budget (b probes per agent per family), but keep the prompt text of every
probe: a router "on our terms" learns a map (prompt, agent) -> outcome from the probes it paid for, then scores every
agent on the incoming task's text. Embeddings are all-MiniLM-L6-v2 on CPU (the same model
scripts/routerbench_terms.py uses for RouterBench's own routers); the embedding arithmetic is not in the ledger, like
the frameworks' TF-IDF shortlist.
`embed(texts, model=...)` also serves the frameworks' SOTA retrieval stack, which passes a strong Qwen3 embedder."""
import os, numpy as np
from ..config import RTE_DATA, count, flag
from ._est import CHUNK

os.environ.setdefault("HF_HOME", os.path.join(RTE_DATA, "hf_cache"))
MINILM = "all-MiniLM-L6-v2"
_models: dict[str, object] = {}


def resolve(model: str) -> str:
    """Prefer weights staged at $RTE_DATA/models/<name>: compute nodes have no route to the hub."""
    p = os.path.join(RTE_DATA, "models", model.split("/")[-1])
    return p if os.path.isdir(p) else model


def embed(texts, model: str = MINILM, prompt_name: str | None = None, prompt: str | None = None) -> np.ndarray:
    """Unit-norm float32 embeddings. Default all-MiniLM-L6-v2 on CPU (384-d, unchanged); a larger `model` (the SOTA
    retrieval stack passes Qwen3-Embedding) runs on GPU when one is visible and in bf16 to fit. `prompt` is a raw
    prefix (an asymmetric model's query instruction); `prompt_name` selects one the model ships."""
    if model not in _models:
        import torch
        from sentence_transformers import SentenceTransformer
        big = model != MINILM
        # opt-in: MiniLM on GPU (fp32)
        dev = "cuda" if (big or flag("RTE_MINILM_CUDA")) and torch.cuda.is_available() else "cpu"
        kw = {"model_kwargs": {"dtype": torch.bfloat16}} if dev == "cuda" else {}
        _models[model] = SentenceTransformer(resolve(model), device=dev, **kw)
    m = _models[model]
    bs = 256 if model == MINILM else 32
    kw = {"prompt": prompt} if prompt else ({"prompt_name": prompt_name} if prompt_name else {})
    E = m.encode(list(texts), batch_size=bs, show_progress_bar=False, normalize_embeddings=True, **kw)
    return E.astype(np.float32)


def probe_set(view, b: int):
    """Probe every agent b times per family, keeping the prompts.
    Returns E[n, K*b, d] embeddings, Y[n, K*b] outcomes, F[K*b] family."""
    Y, I = np.zeros((view.n, view.K, b)), np.zeros((view.n, view.K, b), np.int64)
    for f in range(view.K):
        for lo in range(0, view.n, CHUNK):
            Y[lo:lo + CHUNK, f], I[lo:lo + CHUNK, f] = view.probe_text(np.arange(lo, min(view.n, lo + CHUNK)), f, b)
    if flag("RTE_EMBED_BATCH") and view.embedding(0, int(I[0, 0, 0]), True) is None:   # opt-in: one batched encode
        # exact reuse: the same probe instances give the same texts
        import hashlib
        key = (view.n, view.K, b, view.text(0, int(I[0, 0, 0]), True), hashlib.sha1(I.tobytes()).hexdigest())
        if _E_CACHE.get("key") != key:
            _E_CACHE.clear()
            items = [(f, int(i)) for a in range(view.n) for f in range(view.K) for i in I[a, f]]
            _E_CACHE.update(key=key, E=embed(_texts(view, items)).reshape(view.n, view.K * b, -1))
            # shared by every method of the process: never written in place
            _E_CACHE["E"].flags.writeable = False
        E = _E_CACHE["E"]
    else:
        # default: one encode per prompt (float-level differences only)
        rows = [vec(view, f, i, True) for a in range(view.n) for f in range(view.K) for i in I[a, f]]
        E = np.stack(rows).reshape(view.n, view.K * b, -1)
    return E, Y.reshape(view.n, view.K * b), np.repeat(np.arange(view.K), b)


_V = None
_E_CACHE = {}                                                   # the last probe-set embedding (opt-in batch path only)


def _text1(fi):
    return _V.text(fi[0], fi[1], True)


def _texts(view, items):
    """Probe prompt texts; RTE_TEXT_PROCS > 1 (opt-in) generates them in forked worker processes (deterministic, same
    texts)."""
    global _V
    n = count("RTE_TEXT_PROCS")
    if n <= 1:
        return [view.text(f, i, True) for f, i in items]
    import multiprocessing as mp
    _V = view
    with mp.get_context("fork").Pool(n) as pool:
        return pool.map(_text1, items, chunksize=5000)


def vec(view, f, inst, probe=False) -> np.ndarray:
    """The prompt's embedding: the backend's own vectors when it has them (routereval), else MiniLM over its text."""
    e = view.embedding(f, inst, probe)
    return e if e is not None else embed([view.text(f, inst, probe)])[0]


def task_vec(view, task) -> np.ndarray:
    return vec(view, task.family, task.instance)
