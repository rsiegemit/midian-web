"""The framework-shortlist tables the write-ups quote, generated from the rows so the docs can be verified against them.
    python scripts/doc_tables.py             # print every table (markdown)
    python scripts/doc_tables.py --verify    # every generated table row must appear verbatim in RESULTS.md; exit 1 otherwise
    python scripts/doc_tables.py --sync      # rewrite every <!-- doc_tables:NAME --> ... <!-- /doc_tables --> block in RESULTS.md
Sources: the pre-registered TF-IDF rows (source grids), dedup (_dd), MiniLM (_em), MIDIAN-V cohort (_verified), MIDIAN-VA cohort
(_verified_va*). Cells: frameworks pooled (mean of per-framework seed means) and the best single framework. Asterisks mark
cells whose framework set is incomplete (a framework with fewer seeds than the cell has)."""
from __future__ import annotations
import os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fw_variant_numbers import load, select, regime, RTE_DATA

R = f"{RTE_DATA}/results"
NAMES = {"fw_google_adk": "Google ADK", "fw_crewai": "CrewAI", "fw_magentic_one": "Magentic-One", "fw_openai_agents": "OpenAI Agents",
         "fw_llamaindex": "LlamaIndex", "fw_langgraph": "LangGraph", "fw_autogen": "AutoGen", "fw_camel_workforce": "CAMEL", "fw_maf": "MAF", "fw_smolagents": "smolagents"}
SOURCES = [("TF-IDF (pre-registered)", "plain", {100: "fw_live_n100", 1000: "fw_live_n1000", 10000: "live_n10k_v2", 100000: "live_n100k"},
            {100: "fw_live_n100_lowskill", 1000: "fw_live_n1000_lowskill", 10000: "fw_live_n10k_cartel", 100000: "live_n100k"}),
           ("dedup", "dedup", {100: "fw_live_n100_dd", 1000: "fw_live_n1000_dd", 10000: "fw_live_n10k_dd", 100000: "fw_live_n100k_dd"},
            {100: "fw_live_n100_lowskill_dd", 1000: "fw_live_n1000_lowskill_dd", 10000: "fw_live_n10k_cartel_dd", 100000: "fw_live_n100k_dd"}),
           ("MiniLM", "embed", {100: "fw_live_n100_em", 1000: "fw_live_n1000_em", 10000: "fw_live_n10k_em", 100000: "fw_live_n100k_em"},
            {100: "fw_live_n100_lowskill_em", 1000: "fw_live_n1000_lowskill_em", 10000: "fw_live_n10k_cartel_em", 100000: "fw_live_n100k_em"}),
           ("MIDIAN-V cohort", "midian", {100: "fw_live_n100_verified", 1000: "fw_live_n1000_verified"}, {}),
           ("MIDIAN-VA cohort", "midian_va", {100: "fw_live_n100_verified_va", 1000: "fw_live_n1000_verified_va", 10000: "fw_live_n10k_verified_va", 100000: "fw_live_n100k_verified_va"},
            {100: "fw_live_n100_verified_va_lowskill", 1000: "fw_live_n1000_verified_va_lowskill", 10000: "fw_live_n10k_cartel_verified_va", 100000: "fw_live_n100k_verified_va"})]
NINE = {"fw_live_n10k_cartel", "fw_live_n10k_cartel_dd", "fw_live_n10k_cartel_em", "fw_live_n10k_cartel_verified_va"}   # Magentic-One excluded there by design
REF = {100: "fw_live_n100", 1000: "fw_live_n1000", 10000: "live_n10k_v2", 100000: "live_n100k"}
REF_CARTEL = {100: "fw_live_n100_lowskill", 1000: "fw_live_n1000_lowskill", 10000: "learned_n10k", 100000: "live_n100k"}
_cache = {}


def fw_rows(grid, kind):
    if (grid, kind) not in _cache:
        df = load(grid); _cache[(grid, kind)] = pd.DataFrame() if df.empty else select(df, kind).assign(regime=lambda d: [regime(b, l) for b, l in zip(d.beta, d.liar_select)])
    return _cache[(grid, kind)]


def cell(fw, dist, reg, expected=10):
    q = fw[(fw.dist == dist) & (fw.regime == reg)]
    if reg == "beta0" and q.liar_select.nunique() > 1: q = q[q.liar_select == "random"]   # liar-free: the random cell, once
    if q.empty: return None
    per = q.groupby(["method", "seed"]).success.mean().unstack(0)
    star = "*" if per.isna().any().any() or per.shape[1] < expected else ""
    return per.mean(axis=1).mean(), per.mean().max(), NAMES.get(per.mean().idxmax(), per.mean().idxmax()), star


def ref(n, reg, dist, arm):
    g = REF_CARTEL[n] if reg == "cartel" else REF[n]
    df = load(g)
    if df.empty: return float("nan")
    q = df[(df.method == arm) & (df.params == "{}") & (df.dist == dist)].assign(regime=lambda d: [regime(b, l) for b, l in zip(d.beta, d.liar_select)])
    q = q[q.regime == reg]
    return q.success.mean() if len(q) else float("nan")


def shortlist_table(dist="specialist"):
    """Frameworks pooled by shortlist source, n x regime; VA itself and the oracle beside. Cell = mean (best framework)."""
    lines = [f"| n | regime | " + " | ".join(s[0] for s in SOURCES) + " | MIDIAN-VA itself | oracle |", "|---|---|" + "---|" * (len(SOURCES) + 2)]
    for n in (100, 1000, 10000, 100000):
        for reg in ("beta0", "cartel"):
            cells = []
            for name, kind, honest, cartel in SOURCES:
                g = (cartel if reg == "cartel" else honest).get(n)
                c = cell(fw_rows(g, kind), dist, reg, 9 if g in NINE else 10) if g else None
                cells.append("--" if c is None else f"{c[0]:.3f} ({c[1]:.3f}){c[3]}")
            if all(x == "--" for x in cells): continue
            lines.append(f"| 10^{int(np.log10(n))} | {'honest' if reg == 'beta0' else 'cartel'} | " + " | ".join(cells) + f" | {ref(n, reg, dist, 'midian_va'):.3f} | {ref(n, reg, dist, 'oracle'):.3f} |")
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
            cols.append((f"{name}, {reg}", q.groupby("method").success.mean(), q.groupby("method").seed.nunique()))
    frameworks = sorted({m for _, s, _ in cols for m in s.index}, key=lambda m: -cols[-1][1].get(m, 0))
    lines = ["| framework | " + " | ".join(c[0] for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for m in frameworks:
        vals = []
        for _, s, k in cols:
            if m not in s: vals.append("--"); continue
            star = "*" if k[m] < k.max() else ""
            vals.append(f"{s[m]:.3f}{star}")
        lines.append(f"| {NAMES.get(m, m)} | " + " | ".join(vals) + " |")
    means = [f"{s.mean():.3f}" for _, s, _ in cols]
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

DOC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "RESULTS.md")


def sync():
    doc = open(DOC).read(); n = 0
    for name, fn in TABLES.items():
        pat = re.compile(rf"(<!-- doc_tables:{name} -->\n).*?(\n<!-- /doc_tables -->)", re.S)
        doc, k = pat.subn(lambda m: m.group(1) + fn() + m.group(2), doc); n += k
    open(DOC, "w").write(doc); print(f"synced {n} table blocks in RESULTS.md")


if __name__ == "__main__":
    if "--sync" in sys.argv: sync(); sys.exit(0)
    verify = "--verify" in sys.argv
    doc = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "RESULTS.md")).read() if verify else ""
    bad = 0
    for name, fn in TABLES.items():
        t = fn(); print(f"\n### {name}\n{t}")
        if verify:
            for row in t.splitlines()[2:]:
                if row not in doc: print(f"  MISSING IN RESULTS.md: {row[:90]}"); bad += 1
    if verify: print(f"\n{'OK: every table row is in RESULTS.md' if not bad else f'{bad} rows differ from RESULTS.md'}"); sys.exit(1 if bad else 0)
