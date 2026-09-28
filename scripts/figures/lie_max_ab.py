"""Appendix panels A_max_lie / B_max_lie: Figures 1 / 2 in their CARTEL cells (beta = 0.5 low-skill-first) under the max
lie (lie_mode max: every cartel member claims perfect skill in every family) against the standard lie (live: true skill
+ 0.4, clipped; non-live: the calibrated claims). One hatched bar per arm at b = 3 (+/- 1 s.e. over seeds), a black
tick = the same arm in the standard pipeline's cartel cell; the oracle dotted (B: every value / the honest oracle, as
Figure 2). Honest cells are unchanged by the lie and are not drawn; neither is the best framework (its max-lie values
are F_max_lie and I_max_lie).
How the max-lie cell is built: the standard pipeline's cell (condensed_figs.build: b = 3 bars, per-seed tables) with
ONLY the rows of the arms that read declared claims -- seed_tables.CLAIM_READERS: cluster_head_router,
disrouter_cascade, warm_start_bandit, warm_start_bandit[n0=0.5], declared_argmax -- replaced by their rows from the
max-lie grid of that cell: lie_max_arms_n{100,1000,10k,100k} (live), lie_max_bernoulli_1e7_cal,
lie_max_replay_1e6_split_cal, lie_max_routereval5k_norep_cal, lie_max_llmrouterbench_norep_cal. Every other arm
(MIDIAN, MIDIAN w/o defenses, flat probe argmax, the probe-based pool members, random, oracle) keeps its standard-cartel
rows: none of them reads a claim, so the lie cannot move them (same world, liars and seeds). The pooled arms (best
learned/declared router, best bandit) are then re-cross-fitted per seed exactly as in condensed_figs (arms_at).
A cell is drawn only when its max-lie grid has EVERY claim reader on every seed of the standard cell at b = 3;
otherwise it prints MISSING and is skipped (rerun this script once the grid lands).
    python scripts/figures/lie_max_ab.py [--out DIR] [--from-csv]
      -> <out>/{A,B}_max_lie.{png,pdf,csv} + results/aggregates/figures/{A,B}_max_lie.csv;
         --from-csv redraws from the latter alone (no $RTE_DATA) and skips a figure whose csv is not there."""

import argparse
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from scripts.figures.lib import AGG, fig_out  # noqa: E402
from scripts.figures.lib import figspec as S  # noqa: E402

DRAW = f"{AGG}/figures"
KEYS = [k for k in S.ORDER if k not in ("oracle", "best_framework")]
FIGS = {"A_max_lie": False, "B_max_lie": True}  # name -> normalised by the honest oracle


def max_tables(key):
    """{(regime, b): seed x arm} of the max-lie grid of one A / B cell (family, group, n), read as seed_tables reads
    the standard grids (live / bernoulli: specialist; replay: shapes pooled per seed)."""
    from scripts.figures.lib.grids import ERRATUM30, SIZE
    from scripts.figures.lib.seed_tables import _table, rows

    fam, _, n = key
    grid = f"lie_max_arms_{SIZE[n]}" if fam == "live" else f"lie_max_{ERRATUM30[fam]}"
    df = rows(grid, n)
    if fam in ("live", "bernoulli"):
        df = df[df.dist == "specialist"]
    return grid, _table(df, pooled_shapes=fam == "replay")


def substitute(cell, Tm):
    """The cartel cell at b = 3 with the claim readers' per-seed rows (and declared argmax's bar) from the max-lie
    table Tm, or the reason it cannot be built yet."""
    from scripts.figures.lib.seed_tables import CLAIM_READERS
    from scripts.figures.lib.stats import se

    Ts, T3 = cell["seeds"][3], Tm.get(("cartel", 3))
    if T3 is None:
        return "no b = 3 rows yet"
    readers = [a for a in sorted(CLAIM_READERS) if a in Ts.columns]
    part = {a: int(T3[a].reindex(Ts.index).notna().sum()) if a in T3 else 0 for a in readers}
    if any(v < len(Ts) for v in part.values()):
        return "partial: " + ", ".join(f"{a} {v}/{len(Ts)} seeds" for a, v in part.items() if v < len(Ts))
    T = Ts.copy()
    T[readers] = T3[readers].reindex(Ts.index)
    raw = dict(cell["raw"][3])
    if "declared_argmax" in raw:
        raw["declared_argmax"] = (float(T.declared_argmax.mean()), *map(float, se(T.declared_argmax.values)))
    if "oracle" in T3:  # the same instances: the max-lie grid's oracle should match the standard one
        d = (T3.oracle.reindex(Ts.index) - Ts.oracle).abs().max()
        print(f"    oracle per seed, max-lie vs standard grid: max |diff| = {d:.4f}")
    return {**cell, "seeds": {3: T}, "raw": {3: raw}, "fw_seeds": None}


def compute(C, name):
    """One row per (cell, arm) at b = 3: max lie (bar, +/- 1 s.e.) and standard lie (tick)."""
    from scripts.figures.condensed_figs import arms_at, groups_A, groups_B

    norm = FIGS[name]
    rec = []
    for tick, key in (groups_B if norm else groups_A)(C):
        cell = C.get(key + ("cartel",))
        if cell is None or 3 not in cell.get("seeds", {}):
            continue
        grid, Tm = max_tables(key)
        print(f"[{name}] {key}: {grid}")
        cm = substitute(cell, Tm)
        if isinstance(cm, str):
            print(f"[{name}] MISSING {key} ({grid}): {cm} -- cell skipped")
            continue
        std = arms_at({**cell, "seeds": {3: cell["seeds"][3]}, "raw": {3: cell["raw"][3]}, "fw_seeds": None})
        got = arms_at(cm)
        o = C[key + ("beta0",)]["oracle"][0]
        z = o if norm else 1.0
        for k in KEYS:
            if 3 not in got.get(k, {}):
                continue
            m, lo, hi, chosen = got[k][3]
            s = std.get(k, {}).get(3)
            rec.append(
                dict(
                    group=tick.replace("\n", " "),
                    tick=tick,
                    family=key[0],
                    n=key[2],
                    arm=S.NAME[k],
                    key=k,
                    max_lie=m / z,
                    ci_lo=lo / z,
                    ci_hi=hi / z,
                    standard_lie=s[0] / z if s else None,
                    oracle=o / z,
                    chosen=chosen,
                    chosen_standard=s[3] if s else None,
                    grid=grid,
                )
            )
    return pd.DataFrame(rec)


def render(rec, name, out):
    fig, ax = S.figure("appendix")
    groups = list(dict.fromkeys(rec.tick))
    w = 0.86 / len(KEYS)
    for i, g in enumerate(groups):
        q = rec[rec.tick == g]
        for r in q.itertuples():
            x = i + (KEYS.index(r.key) - (len(KEYS) - 1) / 2) * w
            S.bar(ax, x, r.max_lie, w * 0.95, r.key, cartel=True)
            S.whisker(ax, x, r.max_lie, r.ci_lo, r.ci_hi)
            if pd.notna(r.standard_lie):
                ax.hlines(r.standard_lie, x - w * 0.45, x + w * 0.45, colors="black", lw=1.2, zorder=8)
        S.oracle(ax, i - 0.46, i + 0.46, q.oracle.iloc[0])
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    S.finish(ax, ylabel=S.AXIS["rel" if FIGS[name] else "success"], ylim=(0.0, 1.05 if FIGS[name] else 0.95))
    from matplotlib.lines import Line2D

    S.legend(
        ax,
        ["oracle"] + KEYS,
        ncol=5,
        columnspacing=0.5,
        handlelength=1.0,
        extra=([Line2D([], [], color="black", lw=1.2)], ["standard lie"]),
    )
    S.save(fig, name, out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", help="output directory (default: $RTE_FIG_OUT or figures/paper)")
    p.add_argument("--from-csv", action="store_true", help="redraw from results/aggregates/figures only")
    a = p.parse_args(argv)
    out = fig_out(a.out)
    if not a.from_csv:
        from scripts.figures.condensed_figs import build

        C = build()
        os.makedirs(DRAW, exist_ok=True)
    for name in FIGS:
        if a.from_csv:
            if not os.path.exists(f"{DRAW}/{name}.csv"):
                print(f"[{name}] no {DRAW}/{name}.csv shipped: skipped")
                continue
            rec = pd.read_csv(f"{DRAW}/{name}.csv")
            shutil.copyfile(f"{DRAW}/{name}.csv", f"{out}/{name}.csv")
        else:
            rec = compute(C, name)
            rec.to_csv(f"{out}/{name}.csv", index=False)
            rec.to_csv(f"{DRAW}/{name}.csv", index=False)
        render(rec, name, out)


if __name__ == "__main__":
    main()
