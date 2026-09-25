"""Interval and selection statistics shared by the figure and number scripts.

    ci(x)            95 % bootstrap CI of the mean (seed-resampling when x is indexed by seed); RNG seeded 0 per call
    bootstrap_ci(x)  the plain 95 % bootstrap of fw_variant_numbers / NUMBERS.json (no NaN filtering, (m, m) below 2)
    se(x)            mean -/+ 1 standard error over seeds (or over the units of x)
    crossfit(T, pool)  winner's-curse-free best of a pool in a seed x arm table

ci and bootstrap_ci are kept apart on purpose: they draw the same resamples when x is finite with >= 2 values, but
differ
on NaN and on a single value, and stored numbers depend on each.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ci(x, B=2000):
    """95% bootstrap CI of the mean. If `x` is indexed by seed (cells x seeds), resample SEEDS: each seed's mean over
    the
    fixed cells is the unit, so the bar is sampling error over seeds, not the between-shape spread of the pooled
    cells."""
    rng = np.random.default_rng(0)
    if isinstance(x, pd.Series) and "seed" in (x.index.names or []):
        x = x.dropna()
        per_seed = x.groupby(level="seed").mean().to_numpy()
        if not len(per_seed):
            return (np.nan, np.nan)
        return np.percentile(
            [per_seed[rng.integers(0, len(per_seed), len(per_seed))].mean() for _ in range(B)], [2.5, 97.5]
        )
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return np.percentile([rng.choice(x, len(x)).mean() for _ in range(B)], [2.5, 97.5]) if len(x) else (np.nan, np.nan)


def bootstrap_ci(x, B=2000, seed=0):
    """(lo, hi) of the plain 95% bootstrap of the mean; (mean, mean) below two values."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    if len(x) < 2:
        return float(x.mean()), float(x.mean())
    m = np.array([rng.choice(x, len(x)).mean() for _ in range(B)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def se(x):
    """mean -/+ 1 standard error over the units of `x` (seeds, or frameworks): the narrowest standard whisker, a ~68 %
    interval -- captions must say "+/- 1 s.e.", never "95 % CI". Per-seed values if `x` is indexed by seed, as ci()."""
    if isinstance(x, pd.Series) and "seed" in (x.index.names or []):
        x = x.dropna().groupby(level="seed").mean()
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return (np.nan, np.nan)
    h = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return (x.mean() - h, x.mean() + h)


def crossfit(T, pool):
    """Cross-fitted best of `pool` in a seed x arm table. For each seed s: the arm with the highest mean over the OTHER
    seeds (among arms that ran on s) is picked, and its success ON s is s's score. Returns (per-seed scores, picks)."""
    T = T[[a for a in pool if a in T.columns]].dropna(how="all")
    vals, picks = {}, {}
    for s in T.index:
        rest = T.drop(s).mean()  # skipna: an arm is judged on the other seeds it has
        rest = rest[T.loc[s].notna() & rest.notna()]
        if rest.empty:
            continue
        c = rest.idxmax()
        vals[s], picks[s] = float(T.at[s, c]), c
    return pd.Series(vals, dtype=float), pd.Series(picks, dtype=object)
