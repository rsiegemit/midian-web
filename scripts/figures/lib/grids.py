"""Every grid registry the figure and number scripts read, and the grid config every script in scripts/ and cluster/
uses.

    config()          the grid config: midian.run.load_config() (configs/grids/)
    LIVE_GRIDS        live bars, per n (bar_figs, seed_tables b = 3)
    REF_GRIDS         where a shortlist condition's oracle / MIDIAN / random come from (shortlist_figs)
    H30_FW, H30_REF   erratum 30: RouterEval shortlists on the no-repeat + calibrated-claims reruns
    ERRATUM30         per non-live family, the rerun grid its B cell switches to once complete
    SIZE              live n -> grid-name size tag (va_b_<tag>, rivals_b_<tag>, lie_max_<tag>, ...)
    BUDGET_GRIDS      the b = 1 / 5 cells of A / B: (family, group, tag of va_b_* / rivals_b_*)
    MATRICES          scale sweeps read from their matrix_success{,_new}.csv (bar_figs; A / B b = 1)
    VA_GRIDS          exact MIDIAN build ledgers by (n, b) (efficiency D)
    S4_10, with_s4_10 seeds 4-10 of Figure 1's live 10^4 / 10^5 arms, pooled with their sources only under RTE_SEEDS_4_10=1
    S_TO_10, TEN      the other seed mirrors (every non-framework arm of A-I at 10 seeds); TEN = S4_10 + S_TO_10
    source(grid)      the grid a seed mirror mirrors
    fixes()           the MIDIAN-fix mirrors <g>_fix (configs/grids/midian_fix.yaml: midian{reagg: keep}, beta > 0
                      only); with_s4_10 adds them too, and the readers draw them as MIDIAN (rows.swap_fixed)
    SHORTLIST_SOURCES, NINE, DOC_REF, DOC_REF_CARTEL, VARIANTS
                      the framework-shortlist tables (doc_tables, fw_variant_numbers)

REF_GRIDS["live"][100] leaves out live_core_n100 (pre-09-02 probe instances), as LIVE_GRIDS[100] does, so the
reference lines of the n = 10^2 shortlist figures and the bars read the same instances (docs/errata.md, erratum 31).
"""

from __future__ import annotations

import re
from functools import cache


def config():
    """The grid config (midian.run.load_config: configs/grids/, named sets expanded)."""
    from midian.run import load_config

    return load_config()


LIVE_GRIDS = {
    100: [
        "fw_live_n100",
        "learned_n100",
        "fw_live_n100_lowskill",
    ],  # live_core_n100: pre-09-02 probe instances, dropped
    1000: ["fw_live_n1000", "live_f1_n1000", "variants_f1", "learned_f1", "fw_live_n1000_lowskill"],
    10000: ["learned_n10k", "live_n10k_v2", "fw_live_n10k_cartel", "live_n10k_cartel_random"],
    100000: ["live_n100k"],
}
REF_GRIDS = {
    "live": {
        100: ["fw_live_n100", "learned_n100", "fw_live_n100_lowskill"],   # live_core_n100: pre-09-02 instances, dropped
        1000: ["fw_live_n1000", "live_f1_n1000", "variants_f1", "learned_f1", "fw_live_n1000_lowskill"],
        10000: ["learned_n10k", "live_n10k_v2", "fw_live_n10k_cartel", "learned_n10k_beta01"],
        100000: ["live_n100k", "live_n100k_fill", "live_n100k_beta01"],
    },
    "routereval": {
        10: ["routereval_mmlu"],
        100: ["routereval_mmlu"],
        1000: ["routereval_mmlu"],
        5000: ["routereval_mmlu5k"],
    },
}
H30_FW = [
    f"{g}_norep_cal"
    for g in (
        "fw_routereval_small",
        "fw_routereval_1k",
        "fw_routereval_5k",
        "fw_routereval_small_em",
        "fw_routereval_1k_em",
        "fw_routereval_5k_em",
        "fw_routereval_small_va",
        "fw_routereval_1k_va",
        "fw_routereval_5k_va",
        "re_sl_declared_small",
        "re_sl_declared_1k",
        "re_sl_declared_5k",
        "re_sl_embed_small",
        "re_sl_embed_1k",
        "re_sl_embed_5k",
    )
]
H30_REF = {
    10: ["routereval_mmlu_norep_cal"],
    100: ["routereval_mmlu_norep_cal"],
    1000: ["routereval_mmlu_norep_cal"],
    5000: ["routereval5k_norep_cal"],
}
ERRATUM30 = {
    "bernoulli": "bernoulli_1e7_cal",
    "replay": "replay_1e6_split_cal",
    "routereval": "routereval5k_norep_cal",
    "llmrouterbench": "llmrouterbench_norep_cal",
}
SIZE = {100: "n100", 1000: "n1000", 10000: "n10k", 100000: "n100k"}
BUDGET_GRIDS = [
    ("live", "specialist", "n100"),
    ("live", "specialist", "n1000"),
    ("live", "specialist", "n10k"),
    ("live", "specialist", "n100k"),
    ("routereval", "strong_to_weak", "routereval5k"),
    ("llmrouterbench", "20 models", "llmrouterbench"),
    ("bernoulli", "specialist", "bernoulli_1e7"),
    ("replay", "all shapes pooled", "replay_1e6"),
]
MATRICES = {"bernoulli": "bernoulli_scale_v5", "replay": "replay_scale_v5"}
VA_GRIDS = ["va_b_bernoulli_1e7", "va_b_n1000", "va_b_n100k", "fw_live_n1000", "live_n100k"]


def b_grids(tag):
    """The grids that carry b = 1 / 5 (and pool fills) for one size tag."""
    return [f"va_b_{tag}", f"rivals_b_{tag}", f"pool_fill_{tag}", f"linucb_fix_{tag}", f"trueskill_fix_{tag}"]


# Seeds 4-10 of Figure 1's non-framework arms at live 10^4 / 10^5: <source>_s4_10 (configs/grids/seeds_4_10.yaml).
# They join their sources only under RTE_SEEDS_4_10=1 (the *_new versions of A / B; lib.NEW), and then only once EVERY
# one is complete (seed_tables.complete), so no bar ever mixes 3 and 10 seeds and the 3-seed figures never change.
S4_10 = [
    f"{g}_s4_10"
    for g in (
        "learned_n10k", "live_n10k_v2", "live_n10k_cartel_random", "va_b_n10k", "rivals_b_n10k", "pool_fill_n10k",
        "tuned_wsb_n10k", "linucb_fix_n10k", "trueskill_fix_n10k",
        "live_n100k", "va_b_n100k", "rivals_b_n100k", "pool_fill_n100k", "tuned_wsb_n100k", "linucb_fix_n100k",
        "lie_max_arms_n10k", "lie_max_arms_n100k",
    )
]


# Seeds up to 10 for every other non-framework arm a figure draws: <source>_s<lo>_10 (configs/grids/seeds_to_10.yaml):
# B's RouterEval 5k / LLMRouterBench cells and their max-lie cells, H's reference lines, I_max_lie. Same rule as S4_10.
S_TO_10 = [
    "routereval5k_norep_cal_s4_10", "llmrouterbench_norep_cal_s6_10", "lie_max_routereval5k_norep_cal_s4_10",
    "lie_max_llmrouterbench_norep_cal_s6_10", "routereval_mmlu_norep_cal_s6_10",
    "lie_max_n1000_s4_10", "lie_max_n10k_s4_10", "lie_max_n100k_s4_10",
]
TEN = S4_10 + S_TO_10


def source(grid):
    """The grid a seeds mirror (<source>_s<lo>_10) mirrors; any other grid is its own source."""
    return re.sub(r"_s\d+_10$", "", grid) if grid in TEN else grid


# The fixed MIDIAN (midian{reagg: keep}, commit 1663638) on every beta > 0 MIDIAN cell a figure draws: <g>_fix mirrors
# g, sources and seed mirrors alike (configs/grids/midian_fix.yaml). Same rule as TEN: they join only under
# RTE_SEEDS_4_10=1, once every one is complete (its beta > 0 rows: beta = 0 is never run), and the readers draw them as
# MIDIAN in place of midian{} (rows.swap_fixed).
@cache
def fixes():
    """The MIDIAN-fix mirrors: every grid <g>_fix of the config that mirrors g."""
    gs = config()["grids"]
    return [g for g in gs if g.endswith("_fix") and gs[g].get("mirror_of") == g[:-4]]


def with_s4_10(gs):
    """`gs` plus the seed mirrors of each (TEN), and the MIDIAN-fix mirror of each of those (fixes()), under
    RTE_SEEDS_4_10=1 (which fails until every one has landed)."""
    from scripts.figures.lib import SEEDS_4_10
    from scripts.figures.lib.seed_tables import complete

    if not SEEDS_4_10:
        return list(gs)
    if missing := [g for g in TEN + fixes() if not complete(g)]:
        raise SystemExit(f"RTE_SEEDS_4_10=1 but {len(missing)} seed-mirror / MIDIAN-fix grids are incomplete: "
                         f"{missing}")
    out = list(gs) + [m for g in gs for m in TEN if source(m) == g]
    return out + [f"{g}_fix" for g in out if f"{g}_fix" in fixes()]


def shortlist_grid(family, grid, h30):
    """True when `grid` holds framework shortlist rows of `family` (routereval: the erratum-30 set once `h30`)."""
    if family == "live":
        return grid.startswith("fw_live_n") and "lietext" not in grid or grid in ("live_n10k_v2", "live_n100k")
    return grid.startswith(("fw_routereval_", "re_sl_")) and grid.endswith("_norep_cal") == h30


# ---- framework-shortlist tables (scripts/analysis/doc_tables.py, fw_variant_numbers.py)
# --------------------------------
SHORTLIST_SOURCES = [
    (
        "TF-IDF (pre-registered)",
        "plain",
        {100: "fw_live_n100", 1000: "fw_live_n1000", 10000: "live_n10k_v2", 100000: "live_n100k"},
        {
            100: "fw_live_n100_lowskill",
            1000: "fw_live_n1000_lowskill",
            10000: "fw_live_n10k_cartel",
            100000: "live_n100k",
        },
    ),
    (
        "dedup",
        "dedup",
        {100: "fw_live_n100_dd", 1000: "fw_live_n1000_dd", 10000: "fw_live_n10k_dd", 100000: "fw_live_n100k_dd"},
        {
            100: "fw_live_n100_lowskill_dd",
            1000: "fw_live_n1000_lowskill_dd",
            10000: "fw_live_n10k_cartel_dd",
            100000: "fw_live_n100k_dd",
        },
    ),
    (
        "MiniLM",
        "embed",
        {100: "fw_live_n100_em", 1000: "fw_live_n1000_em", 10000: "fw_live_n10k_em", 100000: "fw_live_n100k_em"},
        {
            100: "fw_live_n100_lowskill_em",
            1000: "fw_live_n1000_lowskill_em",
            10000: "fw_live_n10k_cartel_em",
            100000: "fw_live_n100k_em",
        },
    ),
    ("MIDIAN w/o audits cohort", "midian_wo_audit", {100: "fw_live_n100_verified", 1000: "fw_live_n1000_verified"}, {}),
    (
        "MIDIAN cohort",
        "midian",
        {
            100: "fw_live_n100_verified_va",
            1000: "fw_live_n1000_verified_va",
            10000: "fw_live_n10k_verified_va",
            100000: "fw_live_n100k_verified_va",
        },
        {
            100: "fw_live_n100_verified_va_lowskill",
            1000: "fw_live_n1000_verified_va_lowskill",
            10000: "fw_live_n10k_cartel_verified_va",
            100000: "fw_live_n100k_verified_va",
        },
    ),
]
NINE = {
    "fw_live_n10k_cartel",
    "fw_live_n10k_cartel_dd",
    "fw_live_n10k_cartel_em",
    "fw_live_n10k_cartel_verified_va",
}  # no Magentic-One by design
DOC_REF = {100: "fw_live_n100", 1000: "fw_live_n1000", 10000: "live_n10k_v2", 100000: "live_n100k"}
DOC_REF_CARTEL = {
    100: "fw_live_n100_lowskill",
    1000: "fw_live_n1000_lowskill",
    10000: "learned_n10k",
    100000: "live_n100k",
}
VARIANTS = {  # variant -> [(grid, filter on params)]; pre-registered = the tfidf rows of the source grids
    "tfidf": [
        ("fw_live_n100", "plain"),
        ("fw_live_n1000", "plain"),
        ("fw_live_n100_lowskill", "plain"),
        ("fw_live_n1000_lowskill", "plain"),
        ("live_n10k_v2", "plain"),
        ("fw_live_n10k_cartel", "plain"),
        ("live_n100k", "plain"),
    ],
    "dedup": [
        (g, "dedup")
        for g in (
            "fw_live_n100_dd",
            "fw_live_n1000_dd",
            "fw_live_n100_lowskill_dd",
            "fw_live_n1000_lowskill_dd",
            "fw_live_n10k_dd",
            "fw_live_n10k_cartel_dd",
            "fw_live_n100k_dd",
        )
    ],
    "embed": [
        (g, "embed")
        for g in (
            "fw_live_n100_em",
            "fw_live_n1000_em",
            "fw_live_n100_lowskill_em",
            "fw_live_n1000_lowskill_em",
            "fw_live_n10k_em",
            "fw_live_n10k_cartel_em",
            "fw_live_n100k_em",
        )
    ],
    "va_cohort": [
        (g, "midian")
        for g in (
            "fw_live_n100_verified_va",
            "fw_live_n100_verified_va_lowskill",
            "fw_live_n1000_verified_va",
            "fw_live_n1000_verified_va_lowskill",
            "fw_live_n10k_verified_va",
            "fw_live_n10k_cartel_verified_va",
            "fw_live_n100k_verified_va",
        )
    ],
    "v_cohort": [(g, "midian_wo_audit") for g in ("fw_live_n100_verified", "fw_live_n1000_verified")],
}
