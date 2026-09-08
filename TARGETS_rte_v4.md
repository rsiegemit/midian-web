# TARGETS_rte_v4.md — non-random MIDIAN cohorts (pre-registered 2026-09-07, BEFORE any v4 run)

Plain MIDIAN is unchanged and its parameters are frozen as always. These are labeled variants selected by one new
parameter, `cohort`, added to `Midian`:

| mode | cohort key | cohorts become | channel |
|---|---|---|---|
| `random` | none | plain MIDIAN (default, unchanged) | probe + reports |
| `stratify` | measured ability, one member per stratum | maximally DIVERSE | probe + reports (pre-existing, V2-4) |
| `block` | measured ability, contiguous blocks | maximally HOMOGENEOUS | probe + reports |
| `specialty` | measured per-family profile, grouped by argmax family | a cohort OWNS a category | probe + reports |
| `declared` | DECLARED per-family profile, grouped by argmax family | a cohort owns a CLAIMED category | + declared |

**Budget neutrality (verified, not assumed).** `probe_outcomes` already probes every agent b times per family before
cohorts exist and hands the outcomes to `_level0`; the key reuses those probes. Measured on a bernoulli smoke at
n = 200, K = 8, b = 3: every mode spends exactly 4,800 probes, identical to `random`. `declared` needs no probes for
its key at all and additionally reads the declaration channel (its `needs` is widened; the View enforces this).

**Why this is not a re-run of V2-4.** V2-4 tested `stratify` at b = 3 only and reported −0.004 (8/20 cells). Diagnostics
at n = 200, K = 8, seed 1 (computed from true S, for instrumentation only — no method sees S) show the probe-derived
keys are simply too noisy at b = 3 to group anything, and sharpen with budget:

| b | within-cohort ability sd (random 0.0236) | same-best-family frac (random 0.30) |
|---|---|---|
| 3 | block 0.0220 | specialty 0.43, declared 0.71 |
| 10 | block 0.0197 | specialty 0.56, declared 0.71 |
| 30 | block 0.0145 | specialty 0.66, declared 0.71 |

So budget is an axis of the experiment, not a nuisance parameter. `declared` is flat in b because declarations carry no
sampling noise.

## Pre-registered expectations

- **T4-1.** `block` <= `random` at every b and shape: homogeneous cohorts make the within-cohort max uninformative, so
  level 0 discriminates less. Expect <= 0 everywhere, and the gap to WIDEN with b (as the sort gets sharper).
- **T4-2.** `specialty` > `random` on **specialist** at b = 10 by >= +0.01, and within +/-0.01 at b = 3 (too noisy to help).
- **T4-3.** `specialty` shows no effect on **bimodal** (|delta| <= 0.01 at every b): two skill levels means there is no
  per-family structure to group by.
- **T4-4.** `declared` > `random` at beta = 0 by >= +0.01 (its key is the sharpest available), and < `random` by
  beta = 0.5 low-skill-first: it is the only one of the three that reads a channel liars control.
- **T4-5.** `declared`'s degradation across beta is MONOTONE and steeper than plain MIDIAN's, because its cohort
  structure — not just its estimates — is corrupted by lying.
- **T4-6.** None of the three beats MIDIAN-VA at beta = 0.5 low-skill-first, since none of them audits.

Reported as measured, misses included. Verdicts go in RESULTS_rte_v4.md.
