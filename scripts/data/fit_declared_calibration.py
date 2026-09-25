"""Fit and validate the `calibrated` declared source (erratum 30): the live self-rating error pattern, for the non-live
backends.   $RTE_DATA/env/rte/bin/python scripts/data/fit_declared_calibration.py

Model: D | S ~ the empirical distribution of the live self-rating D_self_described given S's decile (10 equal-width S
bins x the 11 rating values the live models emit), i.i.d. per (agent, family). Fitted on the pooled live specialist n =
100 and n = 1,000 populations (all seeds). Prints the table to paste into midian/backends/__init__.py (CAL_P), then
validates the IN-CODE table (midian.backends.calibrated_declared) against live: mean, spread, corr(S, D), mass at 0
and 1, and the one number declared argmax depends on, E[S of argmax_a D[a, f]] in random sub-populations of
n = 100 / 1,000. The rejected alternative, a censored-normal (Tobit) linear model D = clip(a + c S + N(0, s)), is fitted
and validated beside it, and both are applied to the heavy_tail / bimodal populations as an out-of-sample transfer
check."""

import glob
import os
import sys

import numpy as np
from scipy import optimize, stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from midian.backends import CAL_P, CAL_VALUES, cal_bin, calibrated_declared
from midian.config import RTE_DATA  # noqa: E402

POP = os.path.join(RTE_DATA, "populations")


def load(shape):
    out = [
        (np.load(f"{d}/S.npy"), np.load(f"{d}/D_self_described.npy"))
        for n in (100, 1000)
        for d in sorted(glob.glob(f"{POP}/{shape}_n{n}_K16_seed*"))
        if os.path.exists(f"{d}/D_self_described.npy")
    ]
    return np.concatenate([s for s, _ in out]), np.concatenate([d for _, d in out]), len(out)


def summary(S, D, rng):
    """mean, sd, corr(S, D), P(D = 0), P(D = 1), E[S at declared argmax] / E[max S] at n = 100 and 1,000."""
    pick = []
    for n in (100, 1000):
        v = []
        for _ in range(200):
            i = rng.choice(S.shape[0], n, replace=False)
            v.append((S[i][np.argmax(D[i], 0), np.arange(S.shape[1])].mean(), S[i].max(0).mean()))
        pick += list(np.mean(v, 0))
    return [D.mean(), D.std(), np.corrcoef(S.ravel(), D.ravel())[0, 1], (D == 0).mean(), (D == 1).mean(), *pick]


def show(tag, row):
    print(f"  {tag:34s} " + " ".join(f"{x:6.3f}" for x in row))


S, D, npop = load("specialist")
S64, D64 = S.astype(np.float64).ravel(), D.astype(np.float64).ravel()
print(
    f"live specialist n = 100 + 1,000: {npop} populations, {S.size:,} (agent, family) pairs; rating values "
    f"{np.unique(D).tolist()}"
)

# ---- the fit: P(D = v | S bin)
b, vals = cal_bin(S).ravel(), np.unique(D)
assert np.allclose(vals, CAL_VALUES), "live rating values changed: update CAL_VALUES"
counts = np.array([[(D.ravel()[b == k] == v).sum() for v in vals] for k in range(10)])
P = counts / counts.sum(1, keepdims=True)
print("pairs per S decile:", counts.sum(1).tolist())
print("CAL_P = np.array([          # P(D = CAL_VALUES[j] | S in decile k), rows k = 0..9")
for k in range(10):
    print("    [" + ", ".join(f"{p:.4f}" for p in P[k]) + "],")
print("])")
if not np.allclose(P, CAL_P, atol=5e-5):
    print("!! the table in midian/backends/__init__.py differs from this fit: paste the one above")


# ---- the rejected alternative: censored-normal linear model, fitted by maximum likelihood
def nll(th):
    a, c, ls = th
    s = np.exp(ls)
    mu = a + c * S64
    lo, hi = D64 <= 0, D64 >= 1
    mid = ~(lo | hi)
    return -(
        stats.norm.logcdf(-mu[lo] / s).sum()
        + stats.norm.logsf((1 - mu[hi]) / s).sum()
        + (stats.norm.logpdf((D64[mid] - mu[mid]) / s) - ls).sum()
    )


a, c, ls = optimize.minimize(
    nll, [0.5, 0.3, np.log(0.3)], method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-3}
).x
ols = np.polyfit(S64, D64, 1)
print(f"OLS: D = {ols[1]:.3f} + {ols[0]:.3f} S, residual sd {np.std(D64 - np.polyval(ols, S64)):.3f}")
print(f"Tobit (rejected alternative): D = clip({a:.3f} + {c:.3f} S + N(0, {np.exp(ls):.3f}))")
r = D64 - np.polyval(ols, S64)
agent = np.repeat(np.arange(S.shape[0]), S.shape[1])
print(
    f"between-agent share of the OLS residual variance (NOT modelled; draws are i.i.d.): "
    f"{np.var(np.bincount(agent, r)[agent] / S.shape[1]) / r.var():.3f}"
)

# ---- validation
print("\n  columns: mean sd corr P(D=0) P(D=1) | E[S@argmaxD] E[maxS] at n=100 | same at n=1000")
tobit = lambda S_, rng: np.clip(a + c * S_ + rng.normal(0, np.exp(ls), S_.shape), 0, 1)
for shape in ("specialist", "heavy_tail", "bimodal"):
    S_, D_, k = (S, D, npop) if shape == "specialist" else load(shape)
    print(
        f"{shape} ({k} populations){'' if shape == 'specialist' else ' -- out of sample: table fitted on specialist'}"
    )
    show("live self-described", summary(S_, D_, np.random.default_rng(0)))
    for seed in (1, 2):
        show(
            f"calibrated (in-code table), seed {seed}",
            summary(S_, calibrated_declared(S_, seed), np.random.default_rng(0)),
        )
    show("Tobit linear (rejected)", summary(S_, tobit(S_, np.random.default_rng(1)), np.random.default_rng(0)))
    show(
        "programmatic clip(S + N(0, .05))",
        summary(S_, np.clip(S_ + np.random.default_rng(1).normal(0, 0.05, S_.shape), 0, 1), np.random.default_rng(0)),
    )
