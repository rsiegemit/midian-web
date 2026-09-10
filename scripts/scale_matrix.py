"""The big matrix for a scale sweep: every arm x every n, one table per regime, mean success with a 95% seed-bootstrap CI.

    python scripts/scale_matrix.py bernoulli_scale_v5            # markdown to stdout, CSV beside the grid
    python scripts/scale_matrix.py replay_scale_v5 --metric comparisons_per_task
    python scripts/scale_matrix.py replay_scale_v5 --dist specialist   # one shape (replay has three; default pools them)

Reads rows.csv directly (NOT rte.analyze.load, whose consolidate is a no-op on a grid with a .merge_owner) and runs it
through rte.analyze.prepare for the same labels/aliases as every other table. Regimes follow RESULTS_rte_v4: beta = 0
once (liar-free, so liar-selection is degenerate) and each beta > 0 per liar-selection cell."""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rte.analyze import RTE_DATA, prepare
from extra_figs import ci


def regimes(df):
    for beta in sorted(df.beta.dropna().unique()):
        if np.isclose(beta, 0.0): yield "beta=0 (no liars)", beta, "random"
        else:
            for ls in sorted(df.liar_select.dropna().unique()):
                yield f"beta={beta:g} {'CARTEL (low-skill-first)' if ls == 'low_skill_first' else 'random liars'}", beta, ls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid"); ap.add_argument("--metric", default="success"); ap.add_argument("--dist")
    a = ap.parse_args()
    d = f"{RTE_DATA}/results/{a.grid}"
    df = prepare(pd.read_csv(f"{d}/rows.csv", low_memory=False))
    if a.dist: df = df[df.dist == a.dist]
    ns = sorted(df.n.unique()); out = []
    print(f"# {a.grid}: {a.metric}, mean [95% seed-bootstrap CI]; seeds per cell in the last column\n")
    for title, beta, ls in regimes(df):
        q = df[np.isclose(df.beta, beta) & (df.liar_select == ls)]
        piv = {}
        for (lab, n), g in q.groupby(["label", "n"]):
            s = g.set_index(["dist", "seed"])[a.metric].dropna() if "dist" in g else g.set_index("seed")[a.metric].dropna()
            lo, hi = ci(s); piv[(lab, n)] = (s.mean(), lo, hi, s.index.get_level_values("seed").nunique())
            out.append(dict(regime=title, beta=beta, liar_select=ls, label=lab, n=n, metric=a.metric,
                            mean=s.mean(), ci_lo=lo, ci_hi=hi, seeds=piv[(lab, n)][3], units=len(s)))
        labs = sorted({l for l, _ in piv}, key=lambda l: -piv.get((l, ns[-1]), piv.get((l, ns[0]), (0,)))[0])
        print(f"\n## {title}\n")
        print("| arm | " + " | ".join(f"n={n:,}" for n in ns) + " | seeds |")
        print("|---|" + "---|" * (len(ns) + 1))
        for l in labs:
            cells = []
            for n in ns:
                v = piv.get((l, n))
                cells.append("--" if v is None else f"{v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}]")
            seeds = max(v[3] for (ll, _), v in piv.items() if ll == l)
            print(f"| {l} | " + " | ".join(cells) + f" | {seeds} |")
    pd.DataFrame(out).to_csv(f"{d}/matrix_{a.metric}.csv", index=False)
    print(f"\n(csv: {d}/matrix_{a.metric}.csv)", file=sys.stderr)


if __name__ == "__main__":
    main()
