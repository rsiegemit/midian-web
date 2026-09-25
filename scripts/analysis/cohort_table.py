"""Cross-pool table of the v4 cohort modes, each paired against the SAME base with cohort=random (MIDIAN's default).

    python scripts/analysis/cohort_table.py                 # every pool, both regimes
    python scripts/analysis/cohort_table.py --regime cartel # one regime

Deltas are paired per (shape, seed) cell and bootstrapped over seeds, the convention used everywhere else in the repo;
`*` marks a 95% interval excluding zero. Regimes follow RESULTS_rte_v4: beta = 0 is liar-free, so its two
liar-selection cells are bit-identical and only one is kept; `cartel` is beta = 0.5 with liar_select=low_skill_first.
The n = 100,000 pool keeps its cohort arms in live_n100k_cohort and its cohort=random baselines in live_n100k, so the
two grids are loaded together."""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rte.analyze import load
from scripts.figures.lib.rows import label
from scripts.figures.lib.stats import ci

POOLS = [
    ("RouterEval m=1,000", ["cohort_routereval"]),
    ("RouterEval m=5,000", ["cohort_routereval5k"]),
    ("LLMRouterBench m=20", ["cohort_llmrouterbench"]),
    ("RTE n=1,000", ["cohort_rte"]),
    ("RTE n=100,000", ["live_n100k_cohort", "live_n100k"]),
]
MODES = [
    ("stratify", {"stratify": True}),
    ("block", {"cohort": "block"}),
    ("specialty", {"cohort": "specialty"}),
    ("declared", {"cohort": "declared"}),
]
BASES = [
    ("MIDIAN w/o defenses", {"audit": False, "verify": False}),
    ("MIDIAN w/o verification", {"verify": False}),
    ("MIDIAN", {}),
]
arm = lambda *p: label(
    "midian", json.dumps({k: v for d in p for k, v in d.items()})
)  # the arm's label, as rte.analyze builds it


def delta(df, base, lab, b, beta, ls):
    q = df[df.label.isin([base, lab]) & np.isclose(df.beta, beta) & (df.liar_select == ls) & (df.b == b)]
    w = q.pivot_table(index=["dist", "seed"], columns="label", values="success")
    if base not in w or lab not in w:
        return None
    d = (w[lab] - w[base]).dropna()
    if d.empty:
        return None
    lo, hi = ci(d)
    return d.mean(), lo, hi, len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["beta0", "cartel", "both"], default="both")
    a = ap.parse_args()
    regimes = (
        [("beta = 0 (no liars)", 0.0, "random")]
        if a.regime == "beta0"
        else [("cartel (beta = 0.5, low-skill-first)", 0.5, "low_skill_first")]
        if a.regime == "cartel"
        else [("beta = 0 (no liars)", 0.0, "random"), ("cartel (beta = 0.5, low-skill-first)", 0.5, "low_skill_first")]
    )
    frames = {}
    for name, grids in POOLS:
        try:
            frames[name] = load(grids)
        except SystemExit:
            print(f"[cohort_table] {name}: no rows, skipped", file=sys.stderr)
    for title, beta, ls in regimes:
        print(f"\n## {title}\n")
        print("| pool | b | base | " + " | ".join(m for m, _ in MODES) + " |")
        print("|---|---|---|" + "---|" * len(MODES))
        for name, _ in POOLS:
            df = frames.get(name)
            if df is None:
                continue
            for b in sorted(df.b.dropna().unique()):
                for blabel, base in BASES:
                    cells = []
                    for _, mode in MODES:
                        r = delta(df, arm(base), arm(base, mode), b, beta, ls)
                        cells.append("--" if r is None else f"{r[0]:+.3f}{'*' if (r[1] > 0 or r[2] < 0) else ''}")
                    if set(cells) == {"--"}:
                        continue
                    print(f"| {name} | {int(b)} | {blabel} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
