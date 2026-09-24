"""Verbalized-confidence routing (Tian et al. 2023 style self-knowledge): shortlist k candidates, ask each how confident
it is that it solves THIS task, route to the most confident. LLM backend only (the agents must be able to answer).

Shortlist: `declared` = top-k by the declared claim D[:, f]; `embed` = top-k by MiniLM cosine between the agents' self-
descriptions and the family description (the frameworks' retrieval="embed", cached with the population). The question is
asked in each agent's own solve prompt by its own model (rte.backends.prompts.rate_task); the reply is a 0-10 rating,
parsed here (also x/y, x%, a 0-1 decimal). Unparseable = 0, counted in stats. Ties go to the earliest shortlist position
(declared: the higher claim; embed: the closer description). A liar claims the maximum -- that lie lives in the World.
Ledger: build n messages (declarations/descriptions to the registry); fetch 2k messages (k queries + k replies, charged
by view.ask_confidence) and compare(k) for the argmax."""
import re
import numpy as np
from .base import Method
from .frameworks._common import FrameworkMethod
from ..backends.prompts import ANSWER_RE

_NUM = re.compile(r"(\d+(?:\.\d+)?)\s*(%|(?:/|out of)\s*(\d+))?")


def parse_confidence(text):
    """A verbal confidence in [0, 1], or None when nothing parseable (or out of range) is said."""
    m = _NUM.search((ANSWER_RE.findall(text or "") or [text or ""])[-1])
    if not m: return None
    x, num, den = float(m.group(1)), m.group(1), m.group(3)
    if den: v = x / float(den) if float(den) else -1.0
    else: v = x / 100 if m.group(2) == "%" else x if "." in num and x <= 1 else x / 10
    return v if 0 <= v <= 1 else None


class VerbalConfidence(Method):
    name = "verbal_confidence"
    needs = frozenset({"declared", "bus"})
    requires_llm = True

    def __init__(self, k=10, shortlist="declared", **p):
        super().__init__(k=k, shortlist=shortlist, **p)
        self.k, self.shortlist = int(k), shortlist
        self.stats = {"asked": 0, "unparseable": 0, "ties": 0}

    def build(self, view, budget):
        self.view = view
        view.ledger.message(view.n)                                    # declarations / descriptions to the registry
        self.sl = FrameworkMethod(k=self.k, retrieval=self.shortlist)   # its retrieval only: no supervisor, no bridge
        self.sl.view = view; self.sl._index(view)

    def fetch(self, task):
        cand = self.sl.retrieve(task)
        conf = [parse_confidence(t) for t in self.view.ask_confidence(cand, task)]
        self.stats["asked"] += len(cand); self.stats["unparseable"] += conf.count(None)
        c = np.array([0.0 if x is None else x for x in conf])
        self.view.ledger.compare(len(cand)); self.stats["ties"] += int((c == c.max()).sum() > 1)
        return int(cand[np.argmax(c)])                                # first maximum: earliest shortlist position
