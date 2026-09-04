# RTE / MIDIAN — Consolidated Results Dossier

**Snapshot:** 2026-09-04 15:30 (all grids complete) · **Purpose:** the single ordered source for the
manuscript. Detailed protocols, per-grid tables and figure captions are in the linked documents; this file carries every
number a paper needs, each tagged with its status.

---

## How to read this document

| tag | meaning |
|---|---|
| **FINAL** | grid complete; number will not change |
| **PROVISIONAL (x% done)** | grid ≥ 88% complete; the row states what is missing and the expected movement |
| **TODO** | not yet computable; the row states what produces it and when |

- Every comparison is **paired on identical (cell, seed) units**; intervals are **95% bootstrap over seeds**.
- **n** = number of agents (or real LLMs); **β** = fraction of agents that lie about observed outcomes; **low-skill** =
  the liars are the weakest agents (a colluding cartel); **random** = liars drawn uniformly.
- **Success** = fraction of routed tasks solved by the chosen agent. **Oracle** = always route to the agent with the
  highest true skill for the task's family. **Random** = uniform pick.
- Pre-registered expectations are cited by id (T1–T6, V2-1…V2-11, T3-1…T3-17) with their verdict; misses are reported as
  misses. Verdict rules and full texts: `TARGETS_rte.md`, `TARGETS_rte_v2.md`, `TARGETS_rte_v3.md`.
- Source documents: `RESULTS_rte.md` (phase 1), `RESULTS_rte_v2.md` (phase 2, incl. figure guide H1–H11),
  `RESULTS_rte_v3.md` (external comparisons, X1–X5), `RESULTS_energy.md`, `DEVIATIONS.md` (every as-run deviation, dated).

---

## Part I — Setup

### I.1 The benchmark in one paragraph

A population of *n* LLM-backed agents (Qwen 0.5B–14B and Gemma 2B/9B behind personas; n ∈ {100, 1,000, 10,000}) offers
services across K = 16 task families whose outcomes are programmatically checkable. A router must pick one agent for each
task of a stream of Q tasks (300–1,000). A method sees only what it pays for: agents' **self-descriptions** (the declared
channel, inflated by liars), **probes** it spends (b = 3 per agent per family), **peer reports** about probes (a fraction
β of agents corrupts them), and a **message bus**. True skill S and the liar set are never exposed to any method. Three
skill shapes: **specialist** (every agent is good at 3 families), **heavy_tail**, **bimodal** (20% strong everywhere).

### I.2 Methods compared

| group | methods |
|---|---|
| **MIDIAN family** | plain MIDIAN (cohorts of r = 10 build a max-tree by probing; trimmed peer reports; a route descends ⌈log_r n⌉ levels); **V** verified promotion + cached root pick; **A** 5% instance audits, two strikes exclude a reporter; **VA** both; SH in-cohort successive halving; SH+A; stratified cohorts; r = 5 / 20. Plain MIDIAN's parameters never changed; every mechanism is a labelled variant. |
| **Agent frameworks (10)** | AutoGen, LangGraph, CrewAI, Magentic-One (7B and 14B supervisor), Microsoft Agent Framework, OpenAI Agents SDK, Google ADK, LlamaIndex, smolagents, CAMEL workforce — each as a supervisor choosing from a TF-IDF top-10 shortlist of self-descriptions; strict accounting (`success_strict`, `fallback_rate`). |
| **Self-contained rivals (18)** | flat probe argmax (frozen / online), sequential halving (trusted observer / peer-reported), warm-start bandit, UCB, Thompson, LinUCB-honest, declared argmax / softmax, verify-on-claim, gossip reputation, referral network, NSW router, cluster-head router, CNP bidding, route-to-k majority, cascade, LLM supervisor over the whole roster. |
| **Published routers as methods** | RouterBench's KNN and MLP predictive routers on MIDIAN's probe budget (`knn_router`, `mlp_router`). |
| **External benchmarks / routers (Part III)** | RouterBench, RouteLLM (released BERT router, own harness), RouterEval (pools of 10 / 100 / 1,000 real LLMs; all 5,000 leaderboard LLMs), Avengers (AAAI 2026), EmbedLLM (ICLR 2025), GraphRouter (ICLR 2025, authors' library). |

### I.3 Cost accounting

Counted per method: probes, reports, messages, comparisons, hops (build and per task). Derived: GPU-seconds and joules
(measured 7B call = 0.0294 GPU-s = 20.6 J at 700 W; probe cost by model mix; message 10⁻³ J; comparison 10⁻⁸ J) and
latency (1 ms per hop, 10 ns per comparison, measured supervisor medians). Per-task costs include MIDIAN's observe-time
path update (commit 3415f03). Wall-clock is reported only in the supervisor-latency table.

---

## Part II — Results on our benchmark

### II.1 Headline: MIDIAN vs the ten agent frameworks — **FINAL**

*Table 1. n = 1,000, self-described channel, 3 shapes × 4 β × 10 seeds, Q = 1,000, fixed adapters (grid fw_live_n1000).*

| arm | success | strict success | fallback rate | Δ vs MIDIAN (paired) |
|---|---|---|---|---|
| oracle | 0.723 | | | |
| **MIDIAN-VA** | **0.675** | | | +0.021 [+0.016, +0.026] |
| MIDIAN-A | 0.668 | | | +0.014 |
| MIDIAN-V | 0.663 | | | +0.009 |
| MIDIAN | 0.654 | | | — |
| flat probe argmax (frozen) | 0.612 | | | −0.042 |
| declared argmax | 0.575 | | | |
| LLM supervisor (7B, whole roster) | 0.568 | 0.568 | 0.000 | |
| Magentic-One (7B) | 0.546 | 0.172 | 0.604 | −0.108 [−0.140, −0.076] (vs VA −0.129) |
| Google ADK | 0.542 | 0.380 | 0.285 | −0.112 |
| Magentic-One (14B orchestrator, asymmetric arm) | 0.533 | 0.287 | 0.432 | −0.121 |
| LangGraph | 0.531 | 0.427 | 0.187 | −0.123 [−0.157, −0.090] |
| LlamaIndex | 0.531 | 0.443 | 0.161 | −0.123 |
| OpenAI Agents | 0.528 | 0.444 | 0.153 | −0.125 |
| AutoGen | 0.528 | 0.497 | 0.060 | −0.126 [−0.160, −0.093] (vs VA −0.147) |
| CrewAI | 0.528 | 0.463 | 0.118 | −0.126 |
| CAMEL workforce | 0.527 | 0.429 | 0.177 | −0.127 [−0.161, −0.094] |
| smolagents | 0.526 | 0.462 | 0.069 | −0.128 |
| MAF | 0.524 | 0.501 | 0.028 | −0.130 |
| random | 0.314 | | | |

- **n = 100** (same design, complete): MIDIAN-A 0.656, MIDIAN-VA 0.653, MIDIAN-V 0.642, MIDIAN 0.639, frameworks
  0.522–0.553, paired −0.09 to −0.12 vs MIDIAN and −0.10 to −0.13 vs MIDIAN-VA.
- **n = 10,000** (specialist, 3 seeds, **FINAL**): MIDIAN-V 0.813, MIDIAN 0.786, flat online 0.774, frameworks
  0.257–0.367 — **below random (0.417)**: at 10,000 self-descriptions the TF-IDF shortlist matches the task's words, not
  its family.
- Strict success counts only picks the supervisor actually named (0.17–0.50); Magentic-One's 60% "fallback" is two
  thirds its orchestrator answering the task itself and naming no speaker (counted as failure under strict accounting);
  the 14B orchestrator turns those into named fallbacks and is 0.013 below the 7B arm on lenient success.
- By shape (n = 1,000): specialist frameworks 0.390 vs MIDIAN 0.778 / VA 0.802 (0 of 400 pairs won); heavy_tail 0.630
  vs 0.643 / 0.679; bimodal 0.573 = oracle vs 0.540 / 0.544 (all 400 pairs won). Frameworks are flat in β (0.530–0.532).
- Phase-1 replication (Q = 300, 3 seeds, 60 cells, earlier adapters): frameworks 0.51–0.54 vs MIDIAN 0.637 / V 0.659.

### II.2 Why the frameworks lose, and what repairs them — **FINAL**

- The frameworks' only signal is the self-description, which over-claims by +0.4 (clipped) and correlates with true
  skill at Spearman **0.443** (specialist), **0.194** (heavy_tail), **0.295** (bimodal).
- On **bimodal** populations every framework sits on the oracle (strong agents are strong everywhere, so description ≈
  skill). On **specialist** populations they collapse to **0.39 vs MIDIAN's 0.75**. The framework − MIDIAN gap is
  monotone in within-family legibility (Fig. H2).
- **Verified shortlist (Fig. H7) — FINAL.** Given MIDIAN-V's verified leaf cohort (r = 10) instead of the TF-IDF
  top-10, nine of ten frameworks gain +0.03 to +0.09 at n = 1,000 (Magentic-One +0.087 [+0.069, +0.104] → 0.633;
  AutoGen +0.081 → 0.609; smolagents +0.069; ADK +0.064; CAMEL +0.050; LangGraph +0.049; CrewAI +0.049; OpenAI Agents
  +0.038; MAF +0.030; LlamaIndex +0.010, CI covers 0) and +0.03 to +0.06 at n = 100 — and none reaches MIDIAN-V
  (0.663 / 0.642): the supervisor's own pick among ten verified candidates still costs 0.03–0.12. Full table:
  RESULTS_rte_v2.md §1b.
- **MIDIAN-VA's audited shortlist (grid fw_live_n1000_verified_va, 1,320 rows) — FINAL.** With VA's cohort the ten
  frameworks reach 0.539–0.622 (lifts +0.008 to +0.084 over plain), within ±0.011 of their V-shortlisted arm on average
  (VA − V: −0.003 / −0.017 / −0.006 at β = 0 / 0.1 / 0.25) and +0.021 above it at β = 0.5, where V's unaudited cohort
  admits liars (MIDIAN-V itself 0.685 → 0.607 across β, MIDIAN-VA 0.684 → 0.680). **T3-19 SPLIT**: the β ≤ 0.25
  clause holds, the β = 0.5 clause (≥ +0.03) holds for LangGraph, AutoGen and smolagents only. The audited cohort
  protects the supervisor less than MIDIAN-VA's own pick because the frameworks re-rank the ten candidates by their
  inflated self-descriptions.

### II.3 The MIDIAN family: add audits first, then verification — **FINAL**

*Table 2. 240 paired units (grid variants_f1: 3 shapes × 4 β × 2 liar selections × 10 seeds), n = 1,000, self-described.*

| success | β=0 | β=0.1 | β=0.25 | β=0.5 random | β=0.5 low-skill | mean |
|---|---|---|---|---|---|---|
| oracle | 0.723 | 0.723 | 0.723 | 0.723 | 0.723 | |
| peer halving | 0.722 | 0.722 | 0.718 | 0.678 | **0.402** | |
| **MIDIAN-VA** | 0.684 | 0.671 | 0.667 | 0.680 | **0.679** | **0.675** |
| MIDIAN-V | 0.684 | 0.684 | 0.676 | 0.608 | 0.531 | 0.653 |
| MIDIAN-A | 0.668 | 0.668 | 0.667 | 0.668 | 0.666 | 0.667 |
| flat probe argmax (online) | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 |
| MIDIAN | 0.668 | 0.664 | 0.648 | 0.627 | 0.569 | 0.645 |
| MIDIAN-SH | 0.670 | 0.670 | 0.648 | 0.522 | 0.426 | 0.616 |
| LinUCB-honest | 0.642 | 0.641 | 0.642 | 0.642 | 0.642 | 0.642 |

*Table 3. Paired deltas by β = 0 / 0.1 / 0.25 / 0.5 (60 pairs each); the two orders of adding mechanisms.*

| step | β=0 | β=0.1 | β=0.25 | β=0.5 | cost change |
|---|---|---|---|---|---|
| **A − MIDIAN** (audits first) | +0.000 | +0.004 | +0.019 | +0.069 (+0.097 low-skill, 30/30 cells) | +5% build probes, per-task unchanged |
| **VA − A** (then verification) | +0.016 | +0.003 | −0.000 | +0.013 | per-task 60 → 31.6 comparisons, 9 → 5 messages |
| V − MIDIAN (verification first) | +0.016 | +0.020 | +0.028 | **−0.029** | per-task halved |
| VA − V (then audits) | +0.000 | **−0.013** | **−0.009** | **+0.110** | +3% probes |

- Audits first is monotone at every β; verification first is a regression (V trusts corrupted reports) followed by a
  repair that gives back 0.01 of V's low-β edge. Once audits are in place, verification buys the halved per-task cost,
  +0.016 at β = 0 and +0.013 at β = 0.5, not accuracy.
- **Replication on ten fresh seeds** (360 units, n ∈ {100, 1,000}): V − MIDIAN +0.021 / +0.029 at β = 0 / 0.25 (V2-8
  HIT); VA − V **+0.020** overall, **+0.074** at β = 0.5 (n = 1,000), +0.058 (n = 100), +0.057 (n = 10,000).
- Negative results: SH −0.060 vs peer halving and −0.124 vs MIDIAN under collusion (V2-1 MISS); SH+A adds nothing over A
  (V2-3 MISS); LinUCB-honest 0.025–0.05 below flat online (V2-5 MISS); stratified cohorts ±0.01 (V2-4 reported);
  trimming δ = 1/3 hurts plain MIDIAN by 0.038 under collusion and MIDIAN-A by 0.000 (V2-10 HIT); r = 20 vs 10: −0.013 /
  +0.023. V2-11 (VA ≥ max(V, A) − 0.01 at every β) MISS within the seed floor: −0.013 at β = 0.1.

### II.4 MIDIAN vs the strongest self-contained rival, by scale — **FINAL**

*Table 4. Peer-reported sequential halving is the best honest-regime rival; MIDIAN-VA is the robust one.*

| n | oracle | peer halving, β ≤ 0.25 | MIDIAN-VA, β ≤ 0.25 | peer halving, β = 0.5 low-skill | MIDIAN-VA, β = 0.5 low-skill |
|---|---|---|---|---|---|
| 100 | 0.716 | 0.693 | 0.661 | 0.494 | **0.658** |
| 1,000 | 0.723 | 0.722 | 0.684 | 0.400 | **0.679** |
| 10,000 | 0.859 | 0.863 | 0.811 | 0.730 | **0.810** |

- Adaptive allocation is worth 0.03–0.05 over any static three-probe tree while liars are few or random, and collapses
  under low-skill collusion at every n; VA is flat in β at every n.
- Floors: flat frozen 0.588 / 0.615 / 0.691, flat online 0.661 / 0.667 / 0.774 (n = 100 / 1k / 10k). Warm-start bandit
  reads declarations and loses them under lies. LinUCB-honest is flat in β and below flat online.
- **Budget** (n = 1,000, β = 0.25; Fig. H8): b = 1 — MIDIAN 0.585 is the best probe-only arm (+0.10 over flat and
  halving); b = 3 — halving 0.722, MIDIAN-V 0.675, MIDIAN-VA 0.662, MIDIAN 0.650, flat 0.620; b = 10 — every probe method
  within 0.03 of the oracle, MIDIAN-VA 0.715 the best MIDIAN variant (VA − V +0.017; at b ≤ 3 with random liars VA is
  0.006–0.013 below V: T3-21 MISS/HIT); the bimodal framework gap closes to +0.002 while heavy_tail opens to −0.097 (V2-9 HIT).


### II.4b Scale to 10,000 and 100,000 agents — **FINAL** (calibrated synthetic backend; RESULTS_rte_v3.md part E, Fig. X6)

| n | oracle | peer halving β=0 → β=.5 low-skill | **MIDIAN-VA** β=0 → β=.5 low-skill | MIDIAN-V | MIDIAN | flat online | comparisons / task: VA vs flat |
|---|---|---|---|---|---|---|---|
| 10,000 | 0.847 | 0.847 → 0.654 | **0.798 → 0.792** | 0.798 → 0.684 | 0.772 → 0.727 | 0.757 | 42 vs 10,000 |
| 100,000 | 0.849 | 0.846 → 0.682 | **0.805 → 0.793** | 0.805 → 0.651 | 0.765 → 0.710 | 0.767 | 53 vs 100,000 |

VA − V under low-skill collusion: +0.108 (10k), **+0.143** (100k); VA − peer halving there +0.138 / +0.112; in the
honest regime VA is 0.04–0.05 below peer halving (T3-18 HIT). Frameworks have no arm at these n (no LLM calls); a live
100k run would cost 4.8M probe calls per seed and is not launched.

### II.5 Robustness axes

**Liars (FINAL, Tables 2–4).** The collusion regime is what MIDIAN-A / VA exist for.

**Low-skill-first framework grids — FINAL** (β = 0.5, the weakest half lie and collude; 3 shapes × 10 seeds). n = 1,000:
**MIDIAN-VA 0.679**, MIDIAN-A 0.665, flat frozen 0.612, MIDIAN 0.569, LLM supervisor 0.552, Magentic-One 0.538, Google
ADK 0.536, MIDIAN-V 0.531, the other seven frameworks 0.520–0.528, declared 0.515, random 0.315 (n = 100: VA 0.657, A
0.656, frameworks 0.52–0.54). Paired: VA − AutoGen **+0.154** [+0.082, +0.225], − Magentic-One +0.141, − ADK +0.144;
even a frozen probe scan beats AutoGen by +0.086. The frameworks never read reports, so collusion leaves them at their
§II.1 level; it is the report-trusting arms without audits (MIDIAN, V) that fall, and A / VA that hold.

**Churn (FINAL; grid churn_n1000, heavy_tail, 10% / 30% of agents replaced every 200 tasks, 20 units).** MIDIAN −0.008 /
−0.014 vs no churn; MIDIAN-VA 0.687 / 0.675 is 0.02–0.03 *below* plain MIDIAN (its cached root pick goes stale like V's; T3-20 MISS); repair = 10% / 30% of the build per event; MIDIAN-V −0.06; halving with stale scores −0.13 at 30%;
halving that rebuilds reaches 0.794 (+0.07) by re-spending the whole budget each event (9.4× MIDIAN's repair). V2-6:
quality HIT, cost MISS (the pre-registered 3% was mis-derived).

**Scale (FINAL).** Build cost grows as n¹ for every probe method (48k → 480k probes). MIDIAN's per-task comparisons grow as
n^0.14 (60 → 80 from 1k to 10k) against n¹ for flat (1,000 → 10,000). Success by n: Table 4.

### II.6 Cost, energy, latency — **FINAL** under stated assumptions (Figs. H4, H5, H10, H11)

*Table 5. Per-task and build costs at n = 1,000.*

| arm | comparisons / task | messages / task | hops | build probes | build reports | J / task (steady state) | latency / task |
|---|---|---|---|---|---|---|---|
| MIDIAN | 60 | 9 | 3 | 48,000 | 432,000 | ~0 (0.009 J messages) | 6 ms |
| MIDIAN-V / VA | 31.6 | 5 | 0 | 47,840 / 49,420 | 430,560 | ~0 | 2 ms |
| flat probe argmax | 1,000 | 0 | 0 | 48,000 | 0 | ~0 | 5 ms |
| learned KNN / MLP router | 1,000 | 0 | 0 | 48,000 (+2.8 min embedding) | 0 | ~0 | 9 ms (54 ms at 1k incl. scoring; 49 ms at 10k) |
| AutoGen (one 7B call) | 10 | 12 | 1 | 0 | 0 | 20.6 J | 1.92 s |
| Magentic-One (7B) | 10 | 12 | 1 | 0 | 0 | ≈ 200 J | 18 s |

- MIDIAN's build costs 277 GPU-s on specialist populations; cumulative cost crosses below AutoGen's after **9,416 tasks**
  in GPU-seconds and **9,415** in joules (the joules crossing is the one labelled in Fig. 1a and in `figures/F1_energy_crossings.csv`)
  and below Magentic-One's after **987** (CrewAI / LlamaIndex / CAMEL: 1,900–2,300; the framework supervisor costs are
  the final-grid latency ratios to AutoGen, re-measured 2026-09-04).
- Cost exponents (probes and comparisons vs n): MIDIAN route n^0.14, flat and declared scans n^1.00; MIDIAN-V ≈ n^0.

---

## Part III — External comparisons (RESULTS_rte_v3.md; Figs. X1–X5)

**Reading rule.** With truthful labels every MIDIAN variant reduces to the probe-family table (the max-tree is an argmax;
there is nothing to audit; V re-probes one cell per promotion). So "MIDIAN on their terms" is the probe table's number.
MIDIAN's own mechanisms act only where there are liars, costs and scale — Section III.5.

### III.1 RouterBench on its own protocol (11 models, AIQ = area under quality-vs-cost) — **FINAL**

| router | AIQ | labelled outcomes | build cost |
|---|---|---|---|
| RouterBench KNN, full train split | 0.713 | 277,915 | $229 |
| **probe table, 50 probes / family** | **0.707** | 36,850 | $30 |
| probe table, 20 probes / family | 0.688 | 14,740 | $12 |
| RouterBench MLP | 0.667 | 277,915 | $229 |
| best single model (zero router) | 0.661 | 0 | — |
| oracle (per prompt) | quality 0.915 | | |

T3-1 MISS (0.026 below KNN at b = 20, not 0.02); T3-2 HIT (predicting the family costs 0.007 AIQ); T3-3 held.

### III.2 RouteLLM's released router — **FINAL**

- On RouterBench outcomes for its own pair (gpt-4 vs mixtral), RouteLLM's BERT router scores **below random**: APGR
  **0.479** vs random 0.507; our 20-probe table 0.594; fully supervised KNN 0.650; optimal 0.993. T3-4 HIT.
- On its own benchmarks through its own harness (metrics recomputed from its per-threshold prints because it crashes at
  its metrics step): APGR 0.536 (MMLU), 0.531 (GSM8K), 0.751 (MT-Bench). The gap to RouterBench is distribution shift
  (Arena-trained router, benchmark items), not a broken router. Our router NOT RUN inside their harness (T3-5).

### III.3 Their routers inside our benchmark (grids learned_f1 / n100 / n10k) — **FINAL**

| finding | n = 100 | n = 1,000 | n = 10,000 |
|---|---|---|---|
| KNN router − flat frozen | −0.001 | −0.002 … +0.003 | −0.006 |
| MLP router − KNN router | +0.060 | +0.034 | does not fit (one-hot × 480k rows) |
| **MIDIAN-VA − MLP router** (β = 0 / 0.25 / 0.5 low-skill) | +0.014 / +0.001 / +0.010 | +0.036 / +0.020 / +0.032 | — |
| **MIDIAN-VA − KNN router** (β = 0 / 0.5 low-skill) | +0.074 / +0.070 | +0.070 / +0.066 | **+0.126 / +0.124** |

RouterBench's KNN router on MIDIAN's probe budget **is** flat probe argmax (T3-6 HIT). Its MLP router gains an agent
prior but stays below VA at every β and n (T3-7 mixed). Both are immune to liars and pay n comparisons per task. T3-9:
the VA − KNN gap grows with n (HIT); its "within ±0.02 at β = 0" clause MISSES because KNN = flat frozen at every n.
T3-10 HIT (latency).

### III.4 RouterEval on its own terms (pools of 10 / 100 / 1,000 real LLMs, 12 datasets) — **FINAL** (defaults and validation-tuned)

*Table 6. m = 1,000 candidates; μ = mean test score of the routed model; mean over 12 datasets × 3 pool types. "tuned" =
hyperparameters selected on RouterEval's own validation split (k for PRKnn, K for cluster tables, α for ridge, width for
MLPR, dim/epochs for EmbedLLM, b and K for the probe table), so no router is reported at a failure-mode default.*

| router | μ, their defaults | μ, validation-tuned | labelled outcomes |
|---|---|---|---|
| oracle (per prompt) | 0.986 | | |
| LinearR (theirs) | 0.661 | **0.667** (α mostly 10) | 3.35M |
| EmbedLLM MF (ICLR 2025) | 0.658 | 0.661 | 3.35M |
| MLPR (theirs) | 0.645 | 0.654 | 3.35M |
| cluster table, full labels (C-RoBERTa-cluster K = 3 / Avengers top-1) | 0.643 (K = 3) / 0.638 (K = 16) / 0.597 (K = 64, Avengers' default) | 0.647 (K = 3 or 16) | 3.35M |
| **probe table** (ours; MIDIAN's estimate) | 0.615 (16 clusters × 30 probes) | **0.630** (mostly 16 × 30; 0.26M labels = 8%) | 0.43M / 0.26M |
| best single model | 0.629 | | |
| PRKnn (theirs) | 0.489 (k = 5) | 0.627 (k = 50) | 3.35M |
| random | 0.461 | | |

- Tuning moves their kNN from below-random-plus-a-bit to the best-single-model level (k = 50 instead of 5), lifts the
  others by 0.003–0.009, and lifts the probe table by 0.015; the ordering is unchanged: a linear probe of the prompt
  embedding is the strongest published router (+0.038 over the best single model), the probe table sits at the best
  single model with 8% of the labels, and everything is 0.32 below the per-prompt oracle.
- Prompt-level routers beat the best single model by up to +0.14 on single-task datasets (gpqa, musr, bbh): the
  headroom is per-prompt, which no family table sees (T3-13 MISS). T3-11 HIT (tuned probe table within 0.003 of tuned
  PRKnn); T3-12 MISS (−0.017 to the tuned cluster table with 13× the labels); T3-14 HIT; T3-16 MISS (EmbedLLM +0.031).
- Per-dataset tuned μ at m = 1,000 (LinearR / EmbedLLM / cluster / probe): mmlu 0.755 / 0.741 / 0.745 / 0.696; bbh 0.727 /
  0.701 / 0.718 / 0.682; hellaswag 0.761 / 0.755 / 0.754 / 0.721; winogrande 0.811 / 0.816 / 0.790 / 0.756; gsm8k 0.725 /
  0.745 / 0.725 / 0.722; math 0.463 / 0.449 / 0.463 / 0.414; gpqa 0.422 / 0.408 / 0.381 / 0.369.

### III.5 MIDIAN and every rival with liars on RouterEval's real LLM pools — **FINAL**

*Table 7. Success on MMLU test prompts (16 subjects, b = 3). n ≤ 1,000: their pools (3 types × 5 seeds, Q = 1,000);
n = 5,000: all leaderboard LLMs (3 seeds, Q = 300).*

| arm | n=1000 β=0 | n=1000 β=.5 low-skill | n=5000 β=0 | n=5000 β=.5 low-skill |
|---|---|---|---|---|
| oracle | 0.747 | 0.748 | 0.902 | 0.902 |
| trusted-observer halving | 0.608 | 0.608 | 0.882 | 0.882 |
| peer halving | 0.608 | **0.483** | 0.882 | **0.564** |
| declared argmax (honest, noisy) | 0.710 | 0.470 | 0.864 | 0.614 |
| warm-start bandit | 0.646 | 0.516 | 0.822 | 0.634 |
| **MIDIAN-VA** | 0.617 | **0.607** | 0.706 | **0.710** |
| MIDIAN-V | 0.617 | 0.525 | 0.706 | 0.542 |
| MIDIAN-A | 0.586 | 0.585 | 0.643 | 0.643 |
| MIDIAN | 0.587 | 0.531 | 0.643 | 0.599 |
| LinUCB-honest | 0.618 | 0.616 | 0.609 | 0.609 |
| their MLP router | 0.620 | 0.620 | — | — |
| flat probe argmax (online) | 0.503 | 0.503 | 0.621 | 0.621 |
| their KNN router (= flat frozen) | 0.380 | 0.380 | 0.609 | 0.609 |
| random | 0.505 | 0.505 | 0.550 | 0.550 |

- With real LLMs and three probes per family a flat argmax is a lottery (hundreds tie at 3/3): their KNN router is below
  random at 1,000; the tree turns the same probes into 0.59–0.62 (1,000) and 0.64–0.71 (5,000).
- Under collusion only VA and A hold; the declaration reader, warm-start bandit, V and peer halving lose 0.15–0.32.
- In the honest regime at 5,000, adaptive halving (0.88 ≈ oracle) and the honest declaration (0.86) beat VA by 0.15:
  **MIDIAN-VA buys robustness at fixed cost, not honest-regime accuracy.**
- Their MLP router (an agent prior learned from the same probes) is the best method at 10 / 100 real LLMs (0.649 /
  0.627 vs VA 0.629 / 0.595) and **matches VA at 1,000** (0.620 vs 0.617; VA − MLP −0.003 [−0.026, +0.023] at β = 0,
  −0.028 [−0.051, −0.004] at β = 0.25, −0.019 at β = 0.5 random, −0.007 at β = 0.5 low-skill) while being immune to
  liars by construction; it does not fit at 5,000 (one-hot over 240k probes) or 10,000 and pays n comparisons per task.
- Paired at n = 1,000: VA − KNN +0.22 at every β; VA − flat online +0.09…+0.11; MLP − flat online +0.117.
  At n = 5,000: VA − flat online +0.084 / +0.090 / +0.089; VA − V +0.000 / +0.020 / **+0.168**; trusted halving − oracle
  −0.020. T3-15 HIT; T3-17 four of six clauses HIT (misses: VA − flat +0.09 vs pre-registered +0.10; LinUCB equals flat
  frozen rather than falling below random).

### III.6 GraphRouter (ICLR 2025) through the authors' library — **FINAL**

Their defaults and a tuned run (edge-performance label instead of their tie-broken one-hot argmax, description
embeddings for LLM nodes, hyperparameters selected on their validation split) both converge to **the best-single-model
router**: on RouterBench 0.784 = always GPT-4 at GPT-4's price (probe table 0.781, KNN 0.760); on RouterEval pools μ =
best single model with one distinct pick in all eight configurations. This reproduces LLMRouterBench's 2026 finding that
recent routers fail to beat a simple baseline. RouterDC (DeBERTa fine-tune, GPU) not run. T3-16 MISS via EmbedLLM
(+0.043 over the probe table at m = 1,000).

### III.7 LLMRouterBench (Findings@ACL'26) on its own terms — **FINAL** (RESULTS_rte_v3.md part F)

15 datasets × 20 lightweight models, their 70/30 × 5-seed protocol and metrics, hyperparameters tuned on a train slice.
Our reproduction matches their Table 11 where checkable (oracle 0.916 / 0.916, best single 0.688 / 0.680, random 0.476 /
0.488, Avengers 0.709 / 0.719, EmbedLLM 0.702 / 0.712). **The probe table with 30 probes per dataset per model (5.6% of the
labels) scores 0.704**: within 0.005 of Avengers and EmbedLLM, +0.016 over the best single model, 0.02 below the full-label
dataset table (0.718); their published leaders (Model-SAT 0.719, GraphRouter 0.703, RouterDC 0.613) bracket it. Every
router is 0.22–0.25 below the per-instance oracle. T3-22 HIT / MISS (within 0.02 of the best learned router; +0.016 vs a
pre-registered +0.02 over the best single model); T3-23 MISS (Gap@O just under 0.25). The 20-model liar pool (T3-24) is
complete: at 20 candidates the online flat scan and their MLP router tie for best (0.687), MIDIAN-A is the best
MIDIAN variant (0.677, flat in β) and VA is 0.02 below; under low-skill collusion V loses 0.07, peer halving 0.20 and the
declaration reader 0.15 while A / VA hold (T3-24 HIT). With one cohort per level the tree compresses nothing at n = 20.

---

## Part IV — Where MIDIAN loses, and the caveat that frames the paper

1. **Honest regime at scale.** Peer-reported halving (0.72 / 0.86 / 0.88 at n = 1k / 10k / 5k-real) and, on real pools,
   the honest declaration (0.86) and the warm-start bandit (0.82) beat MIDIAN-VA (0.71) when nobody lies. MIDIAN-VA's
   claim is flatness under collusion at O(log n) route cost.
2. **Learned skill priors on real pools.** At 10–100 LLMs their MLP router beats the probe table and VA by 0.02–0.03,
   and at 1,000 it matches VA (0.620 vs 0.617) with no report channel to corrupt — at n comparisons per task and with
   no path to 5,000+ agents. Where it fits, a learned prior over probe outcomes is as good a robust router as VA.
3. **Per-prompt headroom.** Every method, ours and theirs, is 0.2–0.4 below the per-prompt oracle on real data;
   family-level routing cannot see within-family variation, which is what the routing literature is after.
4. **Why nobody does this.** The benchmark supplies what probing needs and most deployments lack: a cheap probe with a
   checkable outcome on the task distribution. Flat probe argmax is an offline eval with a lookup table; the frameworks'
   loss is the loss of routing on self-description alone. What is not standard practice — decentralised, lie-robust,
   log-cost routing over thousands of untrusted agents — is what MIDIAN adds, and it matters only in that regime
   (RESULTS_rte_v2.md §11, last bullet).

---

## Part V — Pre-registration scoreboard

| phase | expectations | hits | misses | other |
|---|---|---|---|---|
| v1 (TARGETS_rte.md) | 6 | 0 | 5 | 1 split (T2) — each miss explained in RESULTS_rte.md §8 |
| v2 (TARGETS_rte_v2.md) | 11 | 6 (V2-2, 7, 8, 9, 10, V2-6 quality) | 4 (V2-1, 3, 5, 11 — two within the seed floor) | 1 cost-split (V2-6), 1 reported (V2-4) |
| v3 (TARGETS_rte_v3.md) | 24 | T3-2, 3, 4, 10, 11 (letter), 14, 15, 18, 24; most clauses of 6, 7, 9, 17; first clause of 19 and of 22 | T3-1 (by 0.006), 12, 13, 16, 20, 23; β = 0 clause of 9; two of six clauses of 17; β = 0.5 clause of 19 (+0.021 vs ≥ +0.03; 3/10 frameworks hit); second clause of 22 (by 0.004) | T3-5 ours NOT RUN; T3-8 stated; T3-21 split by budget |

---

## Part VI — Outstanding work (15:30)

Nothing is outstanding: every grid cited in this dossier is complete and every section is tagged FINAL (the energy and
latency figures remain estimates under the stated per-call cost model). Table 1, Appendix E of RESULTS_rte_v2.md, the
verified- and VA-shortlist tables, the H1 / H7 / H10 / H11 figures and the README headline were re-synced from the
completed grids on 2026-09-04.

Not planned without a decision: RouterDC (GPU fine-tune), RouteLLM causal_llm (gated Llama-3), mf / sw_ranking (OpenAI
embeddings), MODEL-SAT (LLM fine-tune), Avengers voting (needs generations).

---

## Part VII — Figure and document index

| id | content | status |
|---|---|---|
| H1 | headline by shape: frameworks vs MIDIAN (β panels; low-skill panel) | final |
| H2 | legibility: Spearman(description, skill) vs framework − MIDIAN | final |
| H3 | consistency vs robustness (β = 0 vs β = 0.5 low-skill) | final |
| H4, H5 | cost–quality Pareto with break-even; cost scaling 10² → 10⁷ | final |
| H6 | MIDIAN → +A → +VA vs halving, by β and liar selection; replay twin | final |
| H7 | frameworks given MIDIAN's verified shortlist | final |
| H8 | budget by channel | final |
| H9 | churn | final |
| H10, H11 | cumulative GPU-s / Wh / messages / comparisons; joules + latency with crossings | final (estimate under the stated cost model) |
| X1 | RouterBench protocol: quality vs cost, their routers vs the probe table | final |
| X2 | RouteLLM protocol on RouterBench outcomes | final |
| X3 | their KNN/MLP routers inside our benchmark vs MIDIAN variants, n = 100 / 1k / 10k | final |
| X4 | RouterEval on its terms by pool size, defaults and tuned | final |
| X5 | every arm with liars on real pools, n = 10 / 100 / 1,000 / 5,000 | final |
| X6 | every arm at n = 10,000 and 100,000 (synthetic, calibrated) | final |

Errata and change log against the earlier drafts: `CHANGES_AND_ERRATA.md`.

Scripts: `rte.analyze`, `scripts/extra_figs.py`, `scripts/v3_figs.py`, `scripts/energy.py`, `scripts/routerbench_terms.py`,
`scripts/rivals_routellm.py`, `scripts/routereval_terms.py`, `scripts/rivals_llmrouter.py`. Data and results under
`$RTE_DATA` (`results/<grid>/`, `results/{routerbench_terms,rivals_routellm,routereval_terms,rivals_llmrouter}/`).
