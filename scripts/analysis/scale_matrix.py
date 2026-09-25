"""The big matrix for a scale sweep: every arm x every n, one table per regime, mean success with a 95% seed-bootstrap CI.

    python scripts/analysis/scale_matrix.py bernoulli_scale_v5            # markdown to stdout, CSV beside the grid
    python scripts/analysis/scale_matrix.py replay_scale_v5 --metric comparisons_per_task
    python scripts/analysis/scale_matrix.py replay_scale_v5 --dist specialist   # one shape (replay has three; default pools them)

Reads rows.csv directly (NOT rte.analyze.load, whose consolidate is a no-op on a grid with a .merge_owner) and runs it
through rte.analyze.prepare for the same labels/aliases as every other table. Regimes follow RESULTS_rte_v4: beta = 0
once (liar-free, so liar-selection is degenerate) and each beta > 0 per liar-selection cell."""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rte.analyze import RTE_DATA, prepare
from scripts.figures.lib.regimes import matrix_title
from scripts.figures.lib.stats import ci


def regimes(df):
    for beta in sorted(df.beta.dropna().unique()):
        if np.isclose(beta, 0.0): yield "beta=0 (no liars)", beta, "random"
        else:
            for ls in sorted(df.liar_select.dropna().unique()):
                yield matrix_title(beta, ls), beta, ls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid"); ap.add_argument("--metric", default="success"); ap.add_argument("--dist")
    a = ap.parse_args()
    d = f"{RTE_DATA}/results/{a.grid}"
    df = prepare(pd.read_csv(f"{d}/rows.csv", low_memory=False))
    if a.dist: df = df[df.dist == a.dist]
    multi_b = df.b.nunique() > 1                                    # top rungs carry b=1 (control) and b=3
    df["col"] = [f"{n:,}" + (f" (b={b})" if multi_b else "") for n, b in zip(df.n, df.b)]
    ns = [c for _, _, c in sorted({(n, b, c) for n, b, c in zip(df.n, df.b, df.col)}, key=lambda t: (t[0], t[1]))]
    ns = list(dict.fromkeys(ns)); out = []
    print(f"# {a.grid}: {a.metric}, mean [95% seed-bootstrap CI]; seeds per cell in the last column\n")
    for title, beta, ls in regimes(df):
        q = df[np.isclose(df.beta, beta) & (df.liar_select == ls)]
        piv = {}
        for (lab, col), g in q.groupby(["label", "col"]):
            if lab == "sequential_halving": continue          # trusted-observer halving: never reported (erratum 26)
            n = int(g.n.iloc[0])                                        # numeric n and b go to the CSV; `col` is display only
            s = g.set_index(["dist", "seed"])[a.metric].dropna() if "dist" in g else g.set_index("seed")[a.metric].dropna()
            per_seed = s.groupby(level="seed").mean()                      # one value per seed (shapes averaged)
            lo, hi = ci(s)
            st = dict(mean=s.mean(), ci_lo=lo, ci_hi=hi, seeds=per_seed.size, units=len(s),
                      std_seed=per_seed.std(ddof=1), var_seed=per_seed.var(ddof=1),   # spread OVER SEEDS (the CI's basis)
                      std_unit=s.std(ddof=1), min=s.min(), max=s.max(),               # spread over every (shape, seed) unit
                      sem=per_seed.std(ddof=1) / np.sqrt(per_seed.size))
            piv[(lab, col)] = st
            out.append(dict(regime=title, beta=beta, liar_select=ls, label=lab, n=n, b=int(g.b.iloc[0]), metric=a.metric, **st))
        key_n = ns[-1] if any((l, ns[-1]) in piv for l, _ in piv) else ns[0]
        labs = sorted({l for l, _ in piv}, key=lambda l: -piv.get((l, key_n), piv.get((l, ns[0]), {"mean": 0}))["mean"])
        print(f"\n## {title}\n")
        print("cell = mean ± std over seeds [95% CI] (seeds)  — std is across per-seed means; CSV also has var, unit-level std, min, max, sem\n")
        print("| arm | " + " | ".join(f"n={n}" for n in ns) + " |")
        print("|---|" + "---|" * len(ns))
        for l in labs:
            cells = []
            for n in ns:
                v = piv.get((l, n))
                cells.append("--" if v is None else f"{v['mean']:.3f} ± {v['std_seed']:.3f} [{v['ci_lo']:.3f},{v['ci_hi']:.3f}] ({v['seeds']})")
            print(f"| {l} | " + " | ".join(cells) + " |")
    csv = f"{d}/matrix_{a.metric}" + (f"_{a.dist}" if a.dist else "") + ".csv"   # --dist must not clobber the pooled CSV
    pd.DataFrame(out).to_csv(csv, index=False)
    print(f"\n(csv: {csv})", file=sys.stderr)


if __name__ == "__main__":
    main()
