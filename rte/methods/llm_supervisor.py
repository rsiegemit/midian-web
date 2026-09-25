"""The practitioner default, run for real: retrieve top-k agents by description, an LLM supervisor picks one.

Mechanism: at n = 1,000 the descriptions do not fit a context window, so retrieve-then-pick is what real systems do.
Retrieval, name/id mapping, ledger accounting and the declared-argmax fallback are FrameworkMethod's; this file is only
the difference: one direct `llm_client` call instead of a framework's subprocess bridge. It is not a framework.

Ledger: build = n messages; fetch = k comparisons, 1 hop, k + 2 messages; observe = 0.
Params: k=20, max_tokens=64, plus FrameworkMethod's (supervisor, retrieval, ...)."""
from __future__ import annotations

import re

from .frameworks._common import FrameworkMethod

_IDX = re.compile(r"\d+")


class LLMSupervisor(FrameworkMethod):
    name = "llm_supervisor"
    env = worker = ""                    # no bridge: we call the supervisor ourselves

    def __init__(self, k: int = 20, max_tokens: int = 64, **params):
        super().__init__(k=k, max_tokens=max_tokens, **params)
        self.max_tokens = int(max_tokens)

    def fetch(self, task) -> int:
        from .. import llm_client
        D, cand = self.view.declared, self.retrieve(task)
        self.view.ledger.compare(len(cand))
        self.view.ledger.hop(1)
        self.view.ledger.message(len(cand) + 2)     # k descriptions read + supervisor request/reply
        listing = "\n".join(f"[{i}] {self.desc[a]} Declared competence on this family: "
                            f"{D[a, task.family]:.2f}" for i, a in enumerate(cand))
        msgs = [{"role": "system", "content":
                 "You are a routing supervisor. Pick the ONE candidate most likely to solve the "
                 "task correctly. Reply with only the bracketed index number, nothing else."},
                {"role": "user", "content": f"Task:\n{self._task_text(task)}\n\nCandidates:\n"
                                            f"{listing}\n\nIndex of your pick:"}]
        # "supervisor" prefixes the memo key; llm_client hashes the full request behind it.
        m = _IDX.search(llm_client.complete(self.supervisor, msgs, self.max_tokens,
                                            cache_key="supervisor"))
        i = int(m.group(0)) if m else -1
        self.stats["picks" if 0 <= i < len(cand) else "fallbacks"] += 1
        return int(cand[i]) if 0 <= i < len(cand) else int(cand[D[cand, task.family].argmax()])
