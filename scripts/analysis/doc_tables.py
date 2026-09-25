"""The framework-shortlist tables the write-ups quote, generated from the rows so the docs can be verified against them.
    python scripts/analysis/doc_tables.py             # print every table (markdown)
    python scripts/analysis/doc_tables.py --verify    # every generated table row must appear verbatim in RESULTS.md; exit 1 otherwise
    python scripts/analysis/doc_tables.py --sync      # rewrite every <!-- doc_tables:NAME --> ... <!-- /doc_tables --> block in RESULTS.md
Sources: the pre-registered TF-IDF rows (source grids), dedup (_dd), MiniLM (_em), MIDIAN w/o audits cohort (_verified), MIDIAN
cohort (_verified_va*). Cells: frameworks pooled (mean of per-framework seed means) and the best single framework. Asterisks mark
cells whose framework set is incomplete (a framework with fewer seeds than the cell has) or that still has an erratum-28 rerun outstanding.
Nothing is read at import; the grid registry is scripts/figures/lib/grids.py (SHORTLIST_SOURCES, NINE, DOC_REF*)."""
from __future__ import annotations

import argparse
import functools
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.analysis.fw_variant_numbers import load, pending_reruns, regime, select                          # noqa: E402
from scripts.figures.lib import ROOT                                                                           # noqa: E402
from scripts.figures.lib.figspec import FW_NAME as NAMES                                                       # noqa: E402
from scripts.figures.lib.grids import DOC_REF as REF, DOC_REF_CARTEL as REF_CARTEL, NINE                       # noqa: E402
from scripts.figures.lib.grids import SHORTLIST_SOURCES as SOURCES                                             # noqa: E402

DOC = os.path.join(ROOT, "RESULTS.md")
_cache = {}
PENDING = functools.cache(pending_reruns)                   # read on first use, never at import


def fw_rows(grid, kind):
    if (grid, kind) not in _cache:
        df = load(grid); _cache[(grid, kind)] = pd.DataFrame() if df.empty else select(df, kind).assign(regime=lambda d: [regime(b, l) for b, l in zip(d.beta, d.liar_select)])
    return _cache[(grid, kind)]


def cell(fw, dist, reg, expected=10):
    q = fw[(fw.dist == dist) & (fw.regime == reg)]
    if reg == "beta0" and q.liar_select.nunique() > 1: q = q[q.liar_select == "random"]   # liar-free: the random cell, once
    if q.empty: return None
    per = q.groupby(["method", "seed"]).success.mean().unstack(0)
    pend = any((g, m, dist, reg) in PENDING() for g, m in zip(q.grid, q.method))
    star = "*" if per.isna().any().any() or per.shape[1] < expected or pend else ""
    return per.mean(axis=1).mean(), per.mean().max(), NAMES.get(per.mean().idxmax(), per.mean().idxmax()), star


def ref(n, reg, dist, arm):
    g = REF_CARTEL[n] if reg == "cartel" else REF[n]
    df = load(g)
    if df.empty: return float("nan")
    q = df[(df.method == arm) & (df.params == "{}") & (df.dist == dist)].assign(regime=lambda d: [regime(b, l) for b, l in zip(d.beta, d.liar_select)])
    q = q[q.regime == reg]
    return q.success.mean() if len(q) else float("nan")


def shortlist_table(dist="specialist"):
    """Frameworks pooled by shortlist source, n x regime; MIDIAN itself and the oracle beside. Cell = mean (best framework)."""
    lines = [f"| n | regime | " + " | ".join(s[0] for s in SOURCES) + " | MIDIAN itself | oracle |", "|---|---|" + "---|" * (len(SOURCES) + 2)]
    for n in (100, 1000, 10000, 100000):
        for reg in ("beta0", "cartel"):
            cells = []
            for name, kind, honest, cartel in SOURCES:
                g = (cartel if reg == "cartel" else honest).get(n)
                c = cell(fw_rows(g, kind), dist, reg, 9 if g in NINE else 10) if g else None
                cells.append("--" if c is None else f"{c[0]:.3f} ({c[1]:.3f}){c[3]}")
            if all(x == "--" for x in cells): continue
            lines.append(f"| 10^{int(np.log10(n))} | {'honest' if reg == 'beta0' else 'cartel'} | " + " | ".join(cells) + f" | {ref(n, reg, dist, 'midian'):.3f} | {ref(n, reg, dist, 'oracle'):.3f} |")
    return "\n".join(lines)


def per_framework_table(n=100000, dist="specialist"):
    """One row per framework, columns = source x (honest, cartel)."""
    cols = []
    for name, kind, honest, cartel in SOURCES:
        if n not in honest: continue
        for reg, grids in (("honest", honest), ("cartel", cartel)):
            g = grids.get(n)
            if not g: continue
            fw = fw_rows(g, kind); q = fw[(fw.dist == dist) & (fw.regime == ("beta0" if reg == "honest" else "cartel"))]
            if reg == "honest" and q.liar_select.nunique() > 1: q = q[q.liar_select == "random"]
            if q.empty: continue
            r = "beta0" if reg == "honest" else "cartel"
            cols.append((f"{name}, {reg}", q.groupby("method").success.mean(), q.groupby("method").seed.nunique(), {m for m in q.method.unique() if (g, m, dist, r) in PENDING()}))
    frameworks = sorted({m for _, s, _, _ in cols for m in s.index}, key=lambda m: -cols[-1][1].get(m, 0))
    lines = ["| framework | " + " | ".join(c[0] for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for m in frameworks:
        vals = []
        for _, s, k, pend in cols:
            if m not in s: vals.append("--"); continue
            star = "*" if k[m] < k.max() or m in pend else ""
            vals.append(f"{s[m]:.3f}{star}")
        lines.append(f"| {NAMES.get(m, m)} | " + " | ".join(vals) + " |")
    means = [f"{s.mean():.3f}" + ("*" if pend else "") for _, s, _, pend in cols]
    lines.append("| **mean of ten** | " + " | ".join(means) + " |")
    return "\n".join(lines)


def halving_table():
    """Live 10^5 peer-reported halving, per cell: seeds landed, success per seed, oracle per seed; * while a cell is partial."""
    live = load("live_n100k")
    h = live[(live.method == "sequential_halving") & live.params.str.contains("peer_reported")]
    o = live[live.method == "oracle"].set_index(["beta", "liar_select", "seed"]).success
    lines = ["| cell | seeds | halving per seed | oracle per seed | routes to a liar |", "|---|---|---|---|---|"]
    for (beta, ls), q in h.groupby(["beta", "liar_select"]):
        q = q.sort_values("seed"); star = "*" if len(q) < 3 else ""
        lines.append(f"| β = {float(beta):g}, {'low-skill cartel' if ls == 'low_skill_first' else 'random liars'} | {len(q)} / 3{star} | "
                     + " / ".join(f"{v:.3f}" for v in q.success) + " | " + " / ".join(f"{o.get((b, l, int(s)), float('nan')):.3f}" for b, l, s in zip(q.beta, q.liar_select, q.seed))
                     + " | " + " / ".join(f"{v:.2f}" for v in q.misroute_to_liar) + " |")
    return "\n".join(lines)


TABLES = {"shortlist_specialist": lambda: shortlist_table("specialist"), "shortlist_heavy_tail": lambda: shortlist_table("heavy_tail"),
          "shortlist_bimodal": lambda: shortlist_table("bimodal"), "per_framework_1e5": lambda: per_framework_table(100000),
          "per_framework_1e3": lambda: per_framework_table(1000), "per_framework_1e2": lambda: per_framework_table(100), "halving_live": halving_table}


def sync():
    doc = open(DOC).read(); n = 0
    for name, fn in TABLES.items():
        pat = re.compile(rf"(<!-- doc_tables:{name} -->\n).*?(\n<!-- /doc_tables -->)", re.S)
        doc, k = pat.subn(lambda m: m.group(1) + fn() + m.group(2), doc); n += k
    open(DOC, "w").write(doc); print(f"synced {n} table blocks in RESULTS.md")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--verify", action="store_true", help="every generated table row must appear verbatim in RESULTS.md")
    g.add_argument("--sync", action="store_true", help="rewrite the doc_tables blocks in RESULTS.md")
    a = p.parse_args(argv)
    if a.sync:
        return sync()
    doc = open(DOC).read() if a.verify else ""
    bad = 0
    for name, fn in TABLES.items():
        t = fn()
        print(f"\n### {name}\n{t}")
        if a.verify:
            for row in t.splitlines()[2:]:
                if row not in doc:
                    print(f"  MISSING IN RESULTS.md: {row[:90]}")
                    bad += 1
    if a.verify:
        print(f"\n{'OK: every table row is in RESULTS.md' if not bad else f'{bad} rows differ from RESULTS.md'}")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
