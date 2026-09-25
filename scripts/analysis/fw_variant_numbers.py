"""v6.* entries: the framework shortlist variants (pre-registered TF-IDF, dedup, MiniLM embed, MIDIAN cohort) and the
live 10^5 peer-halving cells. Read straight from each grid's rows.csv / rows.d (<= 1,500 rows per grid; never through
midian.analyze.load).
    python scripts/analysis/fw_variant_numbers.py   # standalone: prints the entries
    from fw_variant_numbers import collect; collect(N)   # from paper_numbers.py
Keys: v6.<variant>.n<n>.<dist>.<regime>.<framework|frameworks_mean|frameworks_best>  and
v6.halving_live.n100000.<regime>.seed<k> Regimes follow RESULTS: beta0 (liar-free, one cell), beta<x>_random,
beta<x>_cartel (low_skill_first); the cartel at beta = 0.5 is 'cartel'."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.figures.lib.grids import VARIANTS  # noqa: E402
from scripts.figures.lib.regimes import tag as regime  # noqa: E402
from scripts.figures.lib.rows import load_fw as load  # noqa: E402
from scripts.figures.lib.rows import pending_reruns
from scripts.figures.lib.stats import bootstrap_ci as ci  # noqa: E402

PENDING: set = set()


def select(df, kind):
    fw = df[
        df.method.astype(str).str.startswith("fw_") & ~df.params.str.contains("supervisor")
    ]  # the 14B Magentic arm is its own entry elsewhere
    if kind == "plain":
        return fw[~fw.params.str.contains("retrieval|dedup")]
    if kind == "dedup":
        return fw[fw.params.str.contains('"dedup"') & ~fw.params.str.contains("retrieval")]
    if kind == "embed":
        return fw[fw.params.str.contains('"embed"')]
    if kind in ("midian", "midian_wo_audit"):
        return fw[fw.params.str.contains(f'"retrieval":"{kind}"') & fw.params.str.contains('"r":10')]  # a MIDIAN cohort
    raise ValueError(kind)


def entry(per_seed, grid, note=None):
    lo, hi = ci(per_seed)
    e = dict(
        value=round(float(np.mean(per_seed)), 4), grid=grid, units=int(len(per_seed)), ci=[round(lo, 4), round(hi, 4)]
    )
    return e | ({"note": note} if note else {})


def collect(N):
    global PENDING
    PENDING = pending_reruns()
    for variant, grids in VARIANTS.items():
        for grid, kind in grids:
            df = load(grid)
            if df.empty:
                continue
            fw = select(df, kind)
            if fw.empty:
                continue
            fw = fw.assign(regime=[regime(b, l) for b, l in zip(fw.beta, fw.liar_select)])
            for (n, dist, reg), q in fw.groupby(["n", "dist", "regime"]):
                if reg == "beta0" and q.liar_select.nunique() > 1:
                    q = q[q.liar_select == "random"]  # liar-free: one cell
                base = f"v6.{variant}.n{int(n)}.{dist}.{reg}"
                per_fw = q.groupby(["method", "seed"]).success.mean().unstack(0)  # seeds x frameworks
                star = {m for m in per_fw if (grid, m, dist, reg) in PENDING}
                pend = lambda ms: "; * erratum-28 rerun outstanding" if ms & star else ""
                N[f"{base}.frameworks_mean"] = entry(
                    per_fw.mean(axis=1), grid, f"{per_fw.shape[1]} frameworks, seed means" + pend(set(per_fw))
                )
                best = per_fw.mean().idxmax()
                N[f"{base}.frameworks_best"] = entry(per_fw[best], grid, f"best = {best}" + pend({best}))
                for m in per_fw:
                    N[f"{base}.{m}"] = entry(per_fw[m].dropna(), grid, pend({m})[2:] or None)
            # the identical-framework (clone) signature: cells where every framework has the same success
            same = fw.groupby(["n", "dist", "regime", "seed"]).success.nunique().eq(1).groupby(level=[0, 1, 2]).mean()
            for (n, dist, reg), v in same.items():
                N[f"v6.{variant}.n{int(n)}.{dist}.{reg}.identical_cells_frac"] = dict(
                    value=round(float(v), 4), grid=grid, units=None, ci=None
                )
    # live 10^5 peer-reported halving, per cell and seed (asterisked while the liar cells are partial)
    live = load("live_n100k")
    if not live.empty:
        h = live[(live.method == "sequential_halving") & live.params.str.contains("peer_reported")]
        o = live[live.method == "oracle"].set_index(["beta", "liar_select", "seed"]).success
        for _, r in h.iterrows():
            key = f"v6.halving_live.n100000.{regime(r.beta, r.liar_select)}.seed{int(r.seed)}"
            N[key] = dict(
                value=round(float(r.success), 4),
                grid="live_n100k",
                units=1,
                ci=None,
                note=f"oracle same cell {float(o.get((r.beta, r.liar_select, int(r.seed)), float('nan'))):.3f}; "
                     f"misroute_to_liar {float(r.get('misroute_to_liar', float('nan'))):.3f}",
            )
        done = h.groupby(["beta", "liar_select"]).seed.nunique().to_dict()
        N["v6.halving_live.n100000.cells_done"] = dict(
            value={f"{regime(b, l)}": int(k) for (b, l), k in done.items()},
            grid="live_n100k",
            units=None,
            ci=None,
            note="seeds landed per cell out of 3; partial cells carry an asterisk in RESULTS",
        )
    return N


if __name__ == "__main__":
    N = collect({})
    for k in sorted(N):
        if k.endswith(("frameworks_mean", "frameworks_best")) or "halving_live" in k:
            print(k, N[k]["value"], N[k].get("ci"), N[k].get("note", ""))
    print(len(N), "entries")
