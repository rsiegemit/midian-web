"""MIDIAN with an LLM making the descent decisions (SPEC §9). Same tree, same estimates, same
ledger as `midian`; only the choice among a node's r children changes, so the ablation is
quality-only. An unparseable or out-of-range answer falls back to the argmax, counted in `stats`."""
import re

import numpy as np

from ..stable_hash import stable_seed_32
from .midian import Midian

SYSTEM = ("You route a task to the most capable group of agents. You are shown each candidate group's "
          "best available skill on this task family, in [0,1]. Answer with one index and nothing else.")


class MidianLLMDescent(Midian):
    name = "midian_llm_descent"
    requires_llm = True

    def __init__(self, model="Qwen/Qwen2.5-7B-Instruct", **p):
        super().__init__(**p)
        self.params["model"] = self.model = model
        self.stats = {"calls": 0, "fallbacks": 0}

    def _choose(self, l, node, f):
        from ..llm_client import complete            # lazy: importing this file must not need an endpoint
        v = self._values(l, node, f)
        ok = np.flatnonzero(np.isfinite(v))
        if ok.size < 2:
            return int(ok[0]) if ok.size else 0
        vals = [round(float(v[i]), 4) for i in ok]
        self.stats["calls"] += 1
        answer = complete(self.model, [{"role": "system", "content": SYSTEM},
                                       {"role": "user", "content": f"Task family {f}.\n"
                                        + "\n".join(f"{i}: {x:.3f}" for i, x in zip(ok, vals))
                                        + f"\n\nReply with one index from {ok.tolist()}."}],
                          max_tokens=8,
                          cache_key=f"{self.name}:{stable_seed_32(self.model, f, tuple(vals)):08x}")
        m = re.search(r"\d+", answer or "")
        if m and int(m.group()) in ok:
            return int(m.group())
        self.stats["fallbacks"] += 1
        return super()._choose(l, node, f)
