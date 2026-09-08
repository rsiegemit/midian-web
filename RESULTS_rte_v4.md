# RESULTS_rte_v4.md — non-random MIDIAN cohorts (IN PROGRESS, 2026-09-08)

Pre-registration: `TARGETS_rte_v4.md`, committed in `8ed4f47` **before** any v4 run. Plain MIDIAN is unchanged; the
three modes are labeled variants of one new parameter, `cohort`, on `Midian` (inherited by MIDIAN-A and MIDIAN-VA).

| mode | key | cohorts become |
|---|---|---|
| `random` | none | plain MIDIAN (default) |
| `stratify` | measured ability, one per stratum | maximally DIVERSE (pre-existing, V2-4) |
| `block` | measured ability, contiguous | maximally HOMOGENEOUS |
| `specialty` | measured per-family profile, argmax family | a cohort OWNS a category |
| `declared` | DECLARED per-family profile, argmax family | a cohort owns a CLAIMED category (reads `declared`) |

**Budget neutrality (measured).** `probe_outcomes` probes every agent b times per family before cohorts exist and hands
the outcomes to `_level0`; the key reuses them. At n = 200, K = 8, b = 3 every mode spends exactly 4,800 probes.

**Key sharpness rises with budget** (n = 200, K = 8, seed 1; computed from true S for instrumentation only — no method
sees S). This is why the b axis is part of the design and why V2-4's b = 3 null was not the end of the story:

| b | within-cohort ability sd (random 0.0236) | same-best-family frac (random 0.30) |
|---|---|---|
| 3 | block 0.0220 | specialty 0.43, declared 0.71 |
| 10 | block 0.0197 | specialty 0.56, declared 0.71 |
| 30 | block 0.0145 | specialty 0.66, declared 0.71 |

## Status

| grid | pool | rows | state |
|---|---|---|---|
| `cohort_routereval` | RouterEval, m = 1,000 real LLMs | 1,800 | COMPLETE |
| `cohort_routereval5k` | RouterEval, all 5,000 leaderboard LLMs | 360 | COMPLETE |
| `cohort_llmrouterbench` | LLMRouterBench, 20 models | 600 | COMPLETE |
| `cohort_rte` | our live benchmark, n = 1,000 | 28 / 1,680 | RUNNING (queued behind the 10^5 stage-2 run) |

All deltas below are means of the paired difference against the SAME base variant with `cohort=random`, on identical
cells. **No seed-bootstrap intervals yet** — several effects are in the 0.005-0.03 range where 3-5 seeds may not
separate them from zero. Do not quote the small ones until the intervals are computed.

## A. RouterEval, m = 1,000 (5 seeds x 3 pool types x 2 beta x 2 b)

| base | mode | b=3 beta=0 | b=3 cartel | b=10 beta=0 | b=10 cartel |
|---|---|---|---|---|---|
| MIDIAN | random | 0.5865 | 0.5323 | 0.6531 | 0.5261 |
| MIDIAN | block | −0.005 | −0.036 | −0.004 | **−0.054** |
| MIDIAN | specialty | +0.006 | +0.001 | +0.006 | −0.006 |
| MIDIAN | declared | +0.007 | −0.016 | −0.001 | **−0.041** |
| MIDIAN-A | random | 0.5865 | 0.5859 | 0.6508 | 0.6222 |
| MIDIAN-A | block | −0.005 | −0.064 | −0.001 | **−0.144** |
| MIDIAN-A | specialty | +0.006 | −0.005 | **+0.011** | +0.001 |
| MIDIAN-A | declared | +0.007 | −0.029 | +0.001 | **−0.110** |
| MIDIAN-VA | random | 0.6139 | 0.6069 | **0.6847** | **0.6837** |
| MIDIAN-VA | block | −0.000 | −0.014 | −0.025 | −0.044 |
| MIDIAN-VA | specialty | +0.012 | +0.005 | −0.012 | −0.012 |
| MIDIAN-VA | declared | +0.008 | +0.005 | −0.002 | −0.026 |

## B. RouterEval, all 5,000 leaderboard LLMs (3 seeds)

| base | mode | b=3 beta=0 | b=3 cartel | b=10 beta=0 | b=10 cartel |
|---|---|---|---|---|---|
| MIDIAN | random | 0.6433 | 0.5989 | 0.7567 | 0.6300 |
| MIDIAN | block | −0.016 | −0.031 | −0.019 | **−0.178** |
| MIDIAN | specialty | −0.004 | −0.013 | +0.001 | **−0.122** |
| MIDIAN | declared | **+0.042** | **−0.103** | +0.022 | **−0.150** |
| MIDIAN-A | random | 0.6433 | 0.6433 | 0.7567 | 0.7567 |
| MIDIAN-A | block | −0.016 | −0.070 | −0.019 | **−0.288** |
| MIDIAN-A | specialty | −0.004 | −0.012 | +0.001 | **−0.142** |
| MIDIAN-A | declared | **+0.042** | −0.096 | +0.022 | **−0.254** |
| MIDIAN-VA | random | 0.7056 | 0.7100 | 0.8222 | 0.8167 |
| MIDIAN-VA | block | −0.001 | −0.008 | −0.013 | −0.036 |
| MIDIAN-VA | specialty | **+0.029** | **+0.026** | +0.007 | −0.004 |
| MIDIAN-VA | declared | **+0.031** | −0.014 | **+0.029** | **+0.018** |

MIDIAN-VA + `declared` at b = 10 reaches **0.8511**, the highest value in the v4 slate.

## C. LLMRouterBench, 20 models — NULL, as predicted

Almost every delta is within +/-0.008 at beta = 0. With n = 20 and r = 10 there are two cohorts and a depth-1 tree, so
cohort STRUCTURE has essentially nothing to act on. The only movement is under the cartel for the unaudited arms, which
is the liar effect, not the cohort effect. Included for completeness; it cannot discriminate between the modes.

## Verdicts so far (RTE grid still outstanding)

- **T4-1 (block <= random everywhere) — HIT.** Negative in essentially every cell of all three pools, and the damage
  grows with b exactly as the mechanism predicts: worst at −0.288 (MIDIAN-A, 5k pool, b = 10, cartel).
- **T4-4 (declared up at beta = 0, down under the cartel) — HIT, and larger than expected.** On the 5k pool it is the
  best mode at beta = 0 (+0.042 for MIDIAN and MIDIAN-A) and the worst under collusion (−0.103 to −0.254).
- **T4-5 (declared degrades steeply) — HIT** on both RouterEval pools.
- **T4-6 (nothing beats VA at the cartel) — HIT on m = 1,000** (VA-random 0.6837 is the ceiling) but **MISS on the 5k
  pool**, where VA + `declared` reaches 0.8344 under the cartel against VA-random's 0.8167.
- **T4-2, T4-3 — UNTESTED.** They are shape-dependent (specialist vs bimodal) and no RouterEval pool has that axis;
  they need `cohort_rte`.

**Unregistered finding.** Whether the modes help MIDIAN-VA depends on pool size: at m = 1,000 every mode hurts VA,
while at m = 5,000 `specialty` and `declared` both help it, `declared` in BOTH regimes. With 5,000 candidates there is
enough structure for cohort organisation to add something on top of verification; at 1,000 there is not. This was not
predicted and should be treated as a hypothesis for the RTE grid, not a result.
