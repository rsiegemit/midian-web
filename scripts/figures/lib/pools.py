"""The arm pools and shortlist rules the figures aggregate over (A / B, C, the max-lie panels, E-H, figure_status).

    ARMS            the arms A / B read from the rows, in legend order (figspec.ORDER without oracle / best framework)
    LEARNED, BANDIT the learned-router and bandit arms; POOLS: the members each cross-fitted pooled bar picks from
    CLAIM_KEY       families whose declared claims are an answer key (the pre-registered warm-start bandit counts there
                    only on the calibrated-claims reruns, erratum 30)
    B_INVARIANT     arms that never probe, so b does not move them
    not_runnable(fam, n)  the pool members that cannot run in a cell (configs/grids pool_fill_*)
    MIN_FW          a shortlist bar needs this many frameworks, each with the full seed count
    source(params)  a framework row's params -> its shortlist source key (None: not a drawn shortlist)
"""

from __future__ import annotations

import json

from scripts.figures.lib.figspec import ORDER

ARMS = [k for k in ORDER if k not in ("oracle", "best_framework")]
LEARNED = [
    "knn_router",
    "knn_router_online",
    "mlp_router",
    "flat_nsw_router",
    "cluster_head_router",
    "disrouter_cascade",
]
BANDIT = ["ucb_per_family", "thompson_per_family", "warm_start_bandit", "linucb_honest", "trueskill_per_family"]
# The bandit pool takes the TUNED warm-start bandit (n0 = 0.5) and swaps linucb_honest for the fixed bonus (post hoc,
# audit 2026-09-23: the context bonus picks the WEAKEST tied agent).
POOLS = {
    "best_learned": LEARNED,
    "best_bandit": [a for a in BANDIT if a != "linucb_honest"]
    + ["linucb_honest[bonus=own]", "warm_start_bandit[n0=0.5]"],
}
# The pre-registered warm-start bandit (n0 = 5) trusts the declared claims; on the non-live backends those are true
# skill + 5 % noise (an answer key), so it counts there only on the calibrated-claims reruns (erratum 30).
CLAIM_KEY = {"bernoulli", "replay", "routereval", "llmrouterbench"}
B_INVARIANT = {"declared_argmax", "random", "cluster_head_router", "disrouter_cascade"}  # no probes, no reports


def not_runnable(fam, n):
    """The pool members that cannot run in a (family, n) cell."""
    return (
        ({"trueskill_per_family"} if n >= 10**5 else set())
        | ({"mlp_router"} if n >= 5000 else set())
        | ({"knn_router", "knn_router_online", "mlp_router"} if fam in ("bernoulli", "replay") else set())
    )


MIN_FW = 6


def source(params):
    """params JSON of a framework row -> shortlist source key, or None when the row is not drawn (not a shortlist arm,
    or dropped: dedup TF-IDF, fusion without the reranker, the cohort of MIDIAN w/o audits, the shuffled MIDIAN cohort,
    r != 10)."""
    p = json.loads(params) if isinstance(params, str) and params.startswith("{") else {}
    if "supervisor" in p or "lie_text" in p:
        return None
    ret = p.get("retrieval")
    ins = str(p.get("embed_instruct", ""))
    tag = "_icomp" if "competent" in ins else "_idemo" if "demonstrated" in ins else ""
    if ret is None:
        return None if p.get("dedup") else "tfidf"
    if ret == "midian":
        return "va_cohort" if p.get("r") == 10 and not p.get("shuffle") else None
    if ret == "embed":
        return ("dense" + tag) if "Qwen" in str(p.get("embed_model", "")) else "embed"
    if ret == "sota":
        return "sota" + tag
    return ret if ret in ("bm25", "declared") else None
