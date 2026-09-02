"""The practitioner default, run for real (SPEC §9, "Two LLM-native additions").

At n=1,000 the agent descriptions do not fit in a context window, so what real systems do is
retrieve-then-pick: TF-IDF over the agents' natural-language self-descriptions returns the top
20 for the task family, and a supervisor model (Qwen2.5-7B-Instruct) reads those 20 descriptions
plus their declared scores for that family and names one.

Descriptions live on the LLM backend, and a Method only ever sees a `View`, which exposes no
backend handle. So the backend registers itself in a module global on construction and this
method reads it lazily at build time via `rte.backends.llm.current_descriptions()`.
Charges `compare(20)` and `hop(1)` per fetch, matching the "max over r children" convention.
"""
from __future__ import annotations

import re

import numpy as np

from .base import Method

TOP_K = 20
_IDX_RE = re.compile(r"\d+")


class LLMSupervisor(Method):
    name = "llm_supervisor"
    needs = frozenset({"declared"})
    requires_llm = True

    def __init__(self, top_k: int = TOP_K, model: str | None = None, max_tokens: int = 64, **params):
        super().__init__(top_k=top_k, model=model, max_tokens=max_tokens, **params)
        self.top_k, self.model, self.max_tokens = int(top_k), model, int(max_tokens)
        self._desc: list[str] | None = None
        self._vec = None
        self._mat = None
        self._fam_desc: dict[int, str] = {}

    # ---- build ---------------------------------------------------------
    def build(self, view, budget) -> None:
        self.view = view
        self.D = np.asarray(view.declared)
        try:
            from ..backends import llm as llm_backend
        except ImportError as e:                              # pragma: no cover
            raise ImportError("llm_supervisor needs the llm backend (reasoning-gym, openai)") from e
        if self.model is None:
            self.model = llm_backend.SUPERVISOR_MODEL
        self._desc = llm_backend.current_descriptions()
        if not self._desc or len(self._desc) != view.n:
            raise RuntimeError(
                "llm_supervisor needs per-agent descriptions from the llm backend; got "
                f"{0 if not self._desc else len(self._desc)} for n={view.n}. Build the World with "
                "backend='llm' (the backend registers itself in rte.backends.llm) and run "
                "`python -m rte.backends.llm --measure ...` first so descriptions.json exists.")
        self._fam_desc = {f: llm_backend.family_description(name)
                          for f, name in enumerate(view.families)}
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        self._mat = self._vec.fit_transform(self._desc)

    # ---- retrieve then pick --------------------------------------------
    def _retrieve(self, f: int) -> np.ndarray:
        q = self._vec.transform([self._fam_desc[f]])
        sims = np.asarray((self._mat @ q.T).todense()).ravel()
        k = min(self.top_k, sims.size)
        # tie-break on declared score so the shortlist is deterministic
        order = np.lexsort((-self.D[:, f], -sims))
        return order[:k]

    def _ask(self, f: int, cand: np.ndarray) -> int:
        from .. import llm_client
        lines = [f"[{i}] {self._desc[a]} Declared competence on this family: {self.D[a, f]:.2f}"
                 for i, a in enumerate(cand)]
        msgs = [{"role": "system", "content":
                 "You are a routing supervisor. Pick the ONE candidate most likely to solve the "
                 "task correctly. Reply with only the bracketed index number, nothing else."},
                {"role": "user", "content":
                 f"Task family: {self.view.families[f]}\n{self._fam_desc[f]}\n\n"
                 f"Candidates:\n" + "\n".join(lines) + "\n\nIndex of your pick:"}]
        key = f"supervisor:{self.view.n}:{f}:" + ",".join(str(int(a)) for a in cand)
        try:
            txt = llm_client.complete(self.model, msgs, max_tokens=self.max_tokens, cache_key=key)
        except Exception:                                     # noqa: BLE001 -- no fleet, or a dead server
            return -1
        m = _IDX_RE.search(txt or "")
        if not m:
            return -1
        i = int(m.group(0))
        return i if 0 <= i < len(cand) else -1

    def fetch(self, task) -> int:
        f = int(task.family)
        cand = self._retrieve(f)
        self.view.ledger.compare(len(cand))
        self.view.ledger.hop(1)
        i = self._ask(f, cand)
        if i < 0:                                             # fallback: declared argmax on the shortlist
            i = int(np.argmax(self.D[cand, f]))
        return int(cand[i])
