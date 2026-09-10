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
| `cohort_rte` | our live benchmark, n = 1,000 | 1,680 | COMPLETE |
| `live_n100k_cohort` | our live benchmark, n = 100,000 | 198 | COMPLETE |

All deltas below are means of the paired difference against the SAME base variant with `cohort=random`, on identical
cells. **Intervals are now computed** (95% bootstrap over seeds within fixed cells, the convention used everywhere else
in this repo); see *Intervals* below for which deltas actually clear zero, and `paper/NUMBERS.json` under
`v4.<pool>.<base>_b<b>_<regime>` for every one of them. Three claims in the first draft of this file did not survive
their own intervals and have been corrected in place; the point estimates in the tables are unchanged and correct.

**Column convention (sections A and B).** `beta=0` is the liar-free regime; with no liars the two liar-selection cells
are bit-identical, so the column is a single set of cells (3 shapes x seeds). `cartel` is **beta = 0.5 with
`liar_select = low_skill_first`** — the colluding-liar world the pre-registration targets. The beta = 0.5 cells with
randomly chosen liars are a separate, milder world and are NOT folded into the cartel column; pooling the two roughly
halves every cartel effect. Both are in `paper/NUMBERS.json` under `v4.<pool>.<base>_b<b>_<regime>`.

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

## D. Our own benchmark, n = 1,000 (3 shapes x 2 beta x 2 liar-selection x 2 b x 5 seeds)

This is the only pool with a SHAPE axis, so it is where T4-2 and T4-3 are decided. Deltas are paired per (shape, seed)
against the same base with `cohort=random`; `*` marks a 95% seed-bootstrap interval excluding zero.

| b | base | stratify | block | specialty | declared |
|---|---|---|---|---|---|
| 3 | MIDIAN | −0.002 | −0.020* | −0.001 | −0.011* |
| 3 | MIDIAN-A | -- | −0.019* | −0.000 | −0.010* |
| 3 | MIDIAN-VA | -- | **+0.015*** | +0.000 | +0.002 |
| 10 | MIDIAN | −0.001 | −0.003 | −0.001 | −0.001 |
| 10 | MIDIAN-A | -- | −0.002 | −0.000 | −0.001 |
| 10 | MIDIAN-VA | -- | −0.002* | +0.001 | +0.002 |

At beta = 0 the modes do essentially nothing: every delta is within 0.02 and most do not clear zero.

**Under the cartel they do real damage, and it grows with b.**

| b | base | stratify | block | specialty | declared |
|---|---|---|---|---|---|
| 3 | MIDIAN | −0.034* | −0.172* | −0.027 | −0.071* |
| 3 | MIDIAN-A | -- | −0.232* | −0.031* | −0.147* |
| 3 | MIDIAN-VA | -- | **+0.012*** | −0.003 | −0.002 |
| 10 | MIDIAN | −0.059* | −0.325* | −0.072* | −0.241* |
| 10 | MIDIAN-A | -- | **−0.413*** | −0.104* | −0.291* |
| 10 | MIDIAN-VA | -- | −0.011* | +0.000 | −0.008 |

`block` at b = 10 costs MIDIAN-A **0.413** — the largest effect anywhere in the v4 slate, and it is a loss. MIDIAN-VA
is close to immune in every cell (|delta| <= 0.015), which is the same pattern the RouterEval pools show.

**By shape (b = 10, cartel), the one mode that is not uniformly bad is `stratify`, and it flips sign:**
+0.045 [+0.037, +0.054] on specialist against −0.170 [−0.250, −0.065] on heavy-tail. Same knob, opposite directions.

## E. Our own benchmark at n = 100,000 (grid `live_n100k_cohort`, b = 3, 3 seeds)

Cohort arms on `live_n100k`'s exact cells; the `cohort=random` baselines are `live_n100k`'s own MIDIAN / -A / -VA rows,
so the two grids are loaded together. Only b = 3 exists at this scale.

| base | stratify | block | specialty | declared |
|---|---|---|---|---|
| MIDIAN, beta = 0 | −0.003 | +0.028 | +0.012 | +0.019 |
| MIDIAN-A, beta = 0 | -- | +0.028 | +0.012 | +0.019 |
| MIDIAN-VA, beta = 0 | -- | −0.014 | −0.016 | **−0.032*** |
| MIDIAN, cartel | +0.020 | **−0.127*** | −0.012 | −0.056 |
| MIDIAN-A, cartel | -- | **−0.170*** | −0.009 | **−0.064*** |
| MIDIAN-VA, cartel | -- | −0.030 | −0.001 | **−0.037*** |

Three seeds, so only the starred cells carry weight. The pattern matches every other pool: nothing clears zero
positively anywhere, `block` is the worst mode under the cartel (−0.127 / −0.170), and `declared` is significantly
negative for MIDIAN-VA in BOTH regimes. The positive-looking beta = 0 cells for MIDIAN and MIDIAN-A (+0.028, +0.019)
all span zero at 3 seeds and must not be quoted.

**Scale does not rescue the mechanism.** Two decades up from n = 1,000 the modes are still neutral-to-harmful, so the
v4 result is not an artifact of small pools.

## Intervals: which deltas clear zero

95% bootstrap over seeds, paired per (shape, seed) cell against the same base with `cohort=random`. **Seed counts are
small**: m = 1,000 and LLMRouterBench have 5 seeds, the 5,000-model pool has 3. A 3-seed interval is a three-point
resample — it is weak evidence in either direction, and every 5,000-pool statement below inherits that weakness.

**Significantly NEGATIVE (the main effect).** `block` and `declared` under the cartel at b = 10 clear zero in all three
pools, and the magnitude grows with b and with pool size exactly as the mechanism predicts. Extremes: MIDIAN-A on the
5,000 pool, b = 10, cartel — block −0.288 [−0.297, −0.273] and declared −0.254 [−0.467, −0.107]; on m = 1,000, block
−0.144 [−0.177, −0.116] and declared −0.110 [−0.131, −0.087]. At beta = 0 with b = 3, block is indistinguishable from
random in every pool: the damage is a collusion effect that needs budget to express, not a generic cost of homogeneity.

**Significantly POSITIVE — the complete list (12 of 216 cells).**

| pool | base | mode | b | regime | delta [95% CI] |
|---|---|---|---|---|---|
| m=1,000 | MIDIAN | specialty | 10 | cartel, random liars | +0.009 [+0.001, +0.014] |
| m=1,000 | MIDIAN-A | specialty | 10 | beta = 0 | +0.011 [+0.005, +0.016] |
| m=1,000 | MIDIAN-VA | block | 3 | cartel, random liars | +0.013 [+0.004, +0.022] |
| m=1,000 | MIDIAN-VA | declared | 3 | cartel, random liars | +0.014 [+0.003, +0.024] |
| m=5,000 | MIDIAN | stratify | 3 | beta = 0 | +0.027 [+0.007, +0.053] |
| m=5,000 | MIDIAN | declared | 3 | beta = 0 | +0.042 [+0.027, +0.060] |
| m=5,000 | MIDIAN-A | declared | 3 | beta = 0 | +0.042 [+0.027, +0.060] |
| m=5,000 | MIDIAN-VA | specialty | 3 | beta = 0 | +0.029 [+0.010, +0.047] |
| m=5,000 | MIDIAN-VA | specialty | 3 | cartel | +0.026 [+0.007, +0.037] |
| m=5,000 | MIDIAN-VA | specialty | 3 | cartel, random liars | +0.029 [+0.027, +0.030] |
| LLMRouterBench | MIDIAN-VA | block | 3 | cartel | +0.008 [+0.000, +0.015] |
| LLMRouterBench | MIDIAN-VA | block | 10 | cartel | +0.018 [+0.004, +0.032] |

**Quoted values that do NOT clear zero** (they stay in the tables as point estimates; do not present them as effects):

| claim | delta [95% CI] |
|---|---|
| VA + `declared` beats VA-random under the cartel on the 5k pool (0.8344 vs 0.8167) | +0.018 [−0.017, +0.043] |
| VA + `declared` reaches 0.8511 at b = 10, beta = 0, "highest in the v4 slate" | +0.029 [+0.000, +0.043] |
| `declared` helps VA at b = 3, beta = 0 on the 5k pool | +0.031 [−0.010, +0.067] |
| `declared` helps VA at b = 3 under the cartel on the 5k pool | −0.014 [−0.037, +0.010] |

The 0.8511 cell is the highest raw number in the slate and that is a fact about the table, but its interval touches zero
exactly, so it is not evidence that `declared` beat `random` there.

## Verdicts so far (RTE grid still outstanding)

- **T4-1 (block <= random everywhere) — MISS on the literal quantifier, HIT on the pattern.** Where the mechanism has
  room to act — b = 10, under the cartel, on the two RouterEval pools — block is negative and the interval clears zero
  in every case, down to −0.288. But "everywhere" is falsified: block is significantly POSITIVE in three cells, all
  MIDIAN-VA (LLMRouterBench b = 10 cartel +0.018 [+0.004, +0.032] and b = 3 cartel +0.008 [+0.000, +0.015]; m = 1,000
  b = 3 with randomly chosen liars +0.013 [+0.004, +0.022]). Two of the three are in the 20-model pool that section C
  already declares unable to discriminate, but the m = 1,000 cell is not, so the target is recorded as missed rather
  than rescued by excluding the pool after the fact. Registered as an "everywhere" claim, tested as one.
- **T4-4 (declared up at beta = 0, down under the cartel) — SPLIT.** The "down under the cartel" half is solid and
  clears zero across pools (to −0.254). The "up at beta = 0" half holds only on the 5,000-model pool (+0.042
  [+0.027, +0.060] for MIDIAN and MIDIAN-A at b = 3, on 3 seeds); on m = 1,000 the same comparison is +0.007 and spans
  zero. The first draft called this a HIT "larger than expected" on the strength of the 5k point estimate alone.
- **T4-5 (declared degrades steeply with the cartel) — HIT.** The b = 3 -> b = 10 progression under the cartel is
  monotone and separated on both RouterEval pools (m = 1,000 MIDIAN-A −0.029 -> −0.110; 5k MIDIAN −0.150, MIDIAN-A
  −0.254).
- **T4-6 (nothing beats VA at the cartel) — NOT CONTRADICTED.** The first draft recorded a MISS on the 5k pool because
  VA + `declared` reached 0.8344 against VA-random's 0.8167. That delta is +0.018 [−0.017, +0.043] on 3 seeds and spans
  zero, so it does not support a MISS. On m = 1,000 the target holds outright: every mode is negative for VA at b = 10
  and all three intervals clear zero.
- **T4-2 (specialty > random on specialist at b = 10 by >= +0.010) — MISS, on all six comparisons.** MIDIAN −0.0034
  [−0.0090, +0.0004], MIDIAN-A −0.0032 [−0.0088, +0.0008], MIDIAN-VA +0.0012 [−0.0038, +0.0076] at beta = 0; under the
  cartel −0.1032, −0.1386 and +0.0006. It never approaches the threshold and is sharply negative under collusion. This
  was the central hypothesis of the slate — that a cohort which OWNS a category routes a specialist workload better —
  and it is falsified on our own benchmark at the registered budget.
- **T4-3 (specialty null on bimodal at b = 10) — HIT, on all six.** Every |delta| <= 0.003 (largest +0.0006
  [+0.0000, +0.0014]). Predicted null, observed null.

**Unregistered finding (narrowed by the intervals).** Whether the modes help MIDIAN-VA depends on pool size, but only
`specialty` survives its interval: on the 5,000-model pool it is positive and clears zero in all three regimes at b = 3
(+0.029, +0.026, +0.029), while on m = 1,000 every mode is negative for VA at b = 10. `declared` was included in this
claim in the first draft; both of its VA cells span zero (+0.031 [−0.010, +0.067] and −0.014 [−0.037, +0.010]) and it is
dropped from the finding. The remaining reading — with 5,000 candidates there is enough structure for cohort
organisation to add something on top of verification, and at 1,000 there is not — was not predicted, rests on 3 seeds,
and should be treated as a hypothesis for `cohort_rte`, not a result.
