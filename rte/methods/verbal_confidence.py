"""Verbalized-confidence routing (Tian et al. 2023 style self-knowledge): shortlist k candidates, ask each how confident
it is that it solves THIS task, route to the most confident. LLM backend only (the agents must be able to answer).

Shortlist: `declared` = top-k by the declared claim D[:, f]; `embed` = top-k by MiniLM cosine between the agents' self-
descriptions and the family description (the frameworks' retrieval="embed", cached with the population). The question is
asked in each agent's own solve prompt by its own model (rte.backends.prompts.rate_task); the reply is a 0-10 rating,
parsed here (see parse_confidence). Unparseable = 0; stats count unparseable, untagged (no <answer> tag) and ties (ties
_distinct: tied by different reply texts, not by clones). Ties go to the earliest shortlist position (declared: the
higher claim; embed: the closer description). A liar claims the maximum -- that lie lives in the World.
Ledger: build n messages (declarations/descriptions to the registry); fetch 2k messages (k queries + k replies, charged
by view.ask_confidence) and compare(k) for the argmax."""
import re
import numpy as np
from .base import Method
from .frameworks._common import FrameworkMethod
from ..backends.prompts import ANSWER_RE

_NUM = re.compile(r"(?<![\d.\-])(\d+(?:\.\d+)?)\s*(%|(?:/|out of)\s*(\d+))?")    # no sign: "-3" is not a rating


def parse_confidence(text):
    """A verbal confidence in [0, 1], or None when nothing parseable (or out of range) is said. The number read is the
    first inside the last <answer> tag, else the LAST in the reply ("on a scale of 0 to 10, I'd say 8" -> 0.8).
    x/y and "x out of y" -> x/y; x% -> x/100; a decimal <= 1 (0.7, 1.0) is already a fraction; any other x <= 10 is on the
    asked 0-10 scale (x/10); 10 < x <= 100 reads as a percentage (a bare "85" -> 0.85); anything else -> None."""
    tag = ANSWER_RE.findall(text or "")
    ms = list(_NUM.finditer(tag[-1] if tag else text or ""))
    if not ms: return None
    m = ms[0] if tag else ms[-1]
    x, num, den = float(m.group(1)), m.group(1), m.group(3)
    if den: v = x / float(den) if float(den) else -1.0
    elif m.group(2) == "%" or x > 10: v = x / 100
    else: v = x if "." in num and x <= 1 else x / 10
    return v if 0 <= v <= 1 else None


class VerbalConfidence(Method):
    name = "verbal_confidence"
    needs = frozenset({"declared", "bus"})
    requires_llm = True

    def __init__(self, k=10, shortlist="declared"):              # no **params: a typo must not enter the row id silently
        if shortlist not in ("declared", "embed"): raise ValueError(f"shortlist must be declared|embed, got {shortlist!r}")
        super().__init__(k=k, shortlist=shortlist)
        self.k, self.shortlist = int(k), shortlist
        self.stats = {"asked": 0, "unparseable": 0, "untagged": 0, "ties": 0, "ties_distinct": 0}

    def build(self, view, budget):
        self.view = view
        view.ledger.message(view.n)                                    # declarations / descriptions to the registry
        self.sl = FrameworkMethod(k=self.k, retrieval=self.shortlist)   # its retrieval only: no supervisor, no bridge
        self.sl.view = view; self.sl._index(view)

    def fetch(self, task):
        cand = self.sl.retrieve(task)
        said = self.view.ask_confidence(cand, task); conf = [parse_confidence(t) for t in said]
        self.stats["asked"] += len(cand); self.stats["unparseable"] += conf.count(None)
        self.stats["untagged"] += sum(not ANSWER_RE.search(t or "") for t in said)
        c = np.array([0.0 if x is None else x for x in conf]); top = np.flatnonzero(c == c.max())
        self.view.ledger.compare(len(cand))
        self.stats["ties"] += int(top.size > 1); self.stats["ties_distinct"] += int(len({said[i] for i in top}) > 1)
        return int(cand[top[0]])                                       # first maximum: earliest shortlist position
