"""Paired statistics: bootstrap CIs, sign tests, paired-by-seed deltas against the reference arm, cost exponents."""
import numpy as np, pandas as pd
from .load import FLOOR, REF, cells

B_BOOT = 2000


def boot(x, B=B_BOOT, seed=12345):
    """(mean, lo, hi) percentile bootstrap of the mean."""
    x = np.asarray([v for v in np.asarray(x, float) if np.isfinite(v)])
    if x.size < 2:
        return (float(x[0]),) * 3 if x.size else (np.nan,) * 3
    m = x[np.random.default_rng(seed).integers(0, x.size, (B, x.size))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
def sign_test(d):
    """(n_positive, n_nonzero, two-sided p) of the paired sign test."""
    nz = np.asarray([v for v in np.asarray(d, float) if np.isfinite(v) and v != 0])
    if nz.size == 0:
        return 0, 0, 1.0
    from scipy.stats import binomtest
    return int((nz > 0).sum()), nz.size, float(binomtest(int((nz > 0).sum()), nz.size, 0.5).pvalue)
def aggregate(df, metric="success"):
    """Per (cell, group, label): seeds, mean, 95% bootstrap CI, seed envelope."""
    keys = cells(df) + ["group", "label"]
    return pd.DataFrame([
        {**dict(zip(keys, k)), "metric": metric, "n_seeds": int(np.isfinite(v).sum()),
         **dict(zip(("mean", "ci_lo", "ci_hi"), boot(v))),
         "envelope": float(v.max() - v.min()) if v.size > 1 else 0.0}
        for k, v in ((k, g[metric].to_numpy(float)) for k, g in df.groupby(keys, dropna=False))])
def paired(df, metric="success", ref=REF):
    """Paired-by-seed delta of `ref` against every other label, per cell."""
    keys, rows = cells(df), []
    for k, g in df.groupby(keys, dropna=False):
        piv = g.pivot_table(index="seed", columns="label", values=metric, aggfunc="mean")
        if ref not in piv:
            continue
        env = float(piv[ref].max() - piv[ref].min()) if piv[ref].notna().sum() > 1 else 0.0
        for lab in piv.columns.drop(ref):
            both = piv[[ref, lab]].dropna()
            if both.empty:
                continue
            d = (both[ref] - both[lab]).to_numpy(float)
            m, lo, hi = boot(d)
            pos, nz, p = sign_test(d)
            rows.append({**dict(zip(keys, k)), "ref": ref, "rival": lab,
                         "group": g.loc[g.label == lab, "group"].iloc[0], "n_pairs": d.size,
                         "delta_mean": m, "delta_lo": lo, "delta_hi": hi, "sign_pos": pos,
                         "sign_nonzero": nz, "sign_p": p, "seed_envelope": env,
                         "verdict": FLOOR if abs(m) <= env else ("midian_better" if m > 0 else "rival_better")})
    return pd.DataFrame(rows)
def fit(n, y, seeds, B=500, seed=7):
    """log10(y) = a + k log10(n), exponent k bootstrapped over seeds. None when unfittable."""
    n, y, seeds = map(np.asarray, (n, y, seeds))
    ok = np.isfinite(n) & np.isfinite(y.astype(float)) & (n > 0) & (y > 0)
    n, y, seeds = n[ok].astype(float), y[ok].astype(float), seeds[ok]
    if np.unique(n).size < 2:
        return None
    k, a = np.polyfit(np.log10(n), np.log10(y), 1)
    uniq, rng, ks = np.unique(seeds), np.random.default_rng(seed), []
    for _ in range(B if uniq.size > 1 else 0):
        m = np.concatenate([np.flatnonzero(seeds == s) for s in rng.choice(uniq, uniq.size, True)])
        if np.unique(n[m]).size >= 2:
            ks.append(np.polyfit(np.log10(n[m]), np.log10(y[m]), 1)[0])
    lo, hi = np.percentile(ks, [2.5, 97.5]) if ks else (np.nan, np.nan)
    return float(k), float(lo), float(hi), float(a)
def exponents(df, metrics):
    """Fitted cost/n exponents per (metric, label), plus rows flagging identically-zero costs."""
    out = []
    keys = ["label", "b"] if "b" in df.columns and df.b.nunique() > 1 else ["label"]   # never fit ACROSS budgets: probe
    for metric in [m for m in metrics if m in df.columns]:                              # counts are n*K*b, so a rung at
        for key, g in df.groupby(keys):                                                 # another b bends the n-slope
            lab, b = (key if len(keys) == 2 else (key, None))
            zero = not (g[metric] > 0).any()
            f = (0.0, 0.0, 0.0) if zero else (fit(g.n, g[metric], g.seed) or (None,))
            if f[0] is not None:
                note = ("identically zero at every n" if zero else
                        f"{g.n.nunique()} values of n" + (f" at b={int(b)}" if b is not None else ""))
                out.append({"metric": metric, "label": lab, **({"b": int(b)} if b is not None else {}),
                            "exponent": f[0], "exp_lo": f[1], "exp_hi": f[2], "note": note})
    return pd.DataFrame(out)


def pair(df, a, b, metric="success"):
    """Methods `a` and `b` paired on (cell, seed); empty when either is missing from `df`."""
    if df.empty:
        return pd.DataFrame()
    piv = df.pivot_table(index=cells(df) + ["seed"], columns="label", values=metric)   # label = method + params
    return piv[[a, b]].dropna() if {a, b} <= set(piv.columns) else pd.DataFrame()
def delta(df, a, b, metric="success"):
    """Mean paired (a - b) and pair count over (cell, seed); (nan, 0) without overlap."""
    d = pair(df, a, b, metric)
    return (float((d[a] - d[b]).mean()), len(d)) if not d.empty else (np.nan, 0)
def envelope(df):
    """Mean over cells of MIDIAN w/o defenses' seed envelope (max-min success across seeds)."""
    m = df[df.label == REF]
    return float(m.groupby(cells(m)).success.agg(lambda v: v.max() - v.min()).mean()) if len(m) else np.nan
