# CHANGES_AND_ERRATA.md — what the earlier results had wrong or incomplete, and what is final (2026-09-04)

Method names in the sections below are those of their date; §8g gives the 2026-09-24 rename (MIDIAN-VA → MIDIAN, …).

Three earlier states are compared with the final one:

| label | document | state |
|---|---|---|
| **phase 1** | `RESULTS_rte.md` (2026-09-02 23:30) | Q = 300, 3 seeds, first adapters; frozen, superseded on every number below |
| **v2 draft** | `RESULTS_rte_v2.md` (2026-09-03 19:05) | framework grids 89.7% / 89.8% (framework rows marked \*), verified grids 84% / 67% |
| **dossier 00:50 / 04:20** | `RESULTS.md` (2026-09-04) | framework grids 97%, tuned RouterEval 9 / 12 datasets, VA-shortlist grid 79% |
| **final** | commits c5d16ea, 945040e and this one | every grid complete; no PROVISIONAL / TODO tag left |

Sections: 1 coverage, 2 headline numbers, 3 minor numbers, 4 errata (claims that were wrong, not merely provisional),
5 new experiments since the v2 draft, 6 figures, 7 pre-registration verdicts, 8 what did not move, 9 still open.

---

## 1. Coverage — what was incomplete and is now complete

| grid / run | earlier state | final | unlocked |
|---|---|---|---|
| fw_live_n100 | 2,153 / 2,400 rows (89.7%, 09-03 13:40); 97% at 00:50 | 2,640 / 2,640 (the grid gained the MIDIAN-VA / A arms) | Table 1 n = 100, App. E, README |
| fw_live_n1000 | 2,156 / 2,400 (89.8%); 97% at 00:50 | 2,640 / 2,640 | Table 1, by-shape split, H1, energy |
| fw_live_n100_verified | 84% (09-03 19:05); 96.5% (00:50) | 2,520 / 2,520 | §1b n = 100, H7 |
| fw_live_n1000_verified | 67%; 94.2%; 99.7% (8 Magentic-One rows) | 2,520 / 2,520 | §1b n = 1,000, H7 |
| fw_live_n1000_verified_va | did not exist (launched 02:32); 79% at 04:20 | 1,320 / 1,320 | VA-shortlist columns, T3-19 |
| fw_live_n{100,1000}_lowskill | 80% (19:05); 89% (00:50; 60 Magentic-One rows missing) | complete (05:10) | low-skill block FINAL, H1 low-skill panel |
| churn_n1000 | 6 halving-rebuild rows missing | complete (20 / 20 units) | §5 churn |
| live_n10k_v2 | halving row TODO | complete | n = 10k table |
| live_f1_n1000 VA / A rows | absent | 240 rows each (the grid's "8,400 expected" is inflated by newer method files — ignore) | §4 MIDIAN → A → VA |
| learned_n10k (their KNN / MLP in our benchmark) | n = 10k tail running | complete | part C, X3 |
| RouterEval validation-tuned rerun | 9 / 12 datasets | 12 / 12 | Table 6 tuned rows, X4 |
| their MLP router on real pools, n = 1,000 | 5 rows | complete | Table 7 |
| their KNN router on the 5,000-LLM pool | 16 rows | complete | Table 7, X5 |
| routereval_mmlu5k (all 5,000 leaderboard LLMs) | running | complete | D2, T3-17 |
| llmrouterbench_pool (20-model pool with liars) | running | complete | F2, T3-24 |
| scale_100k (n = 10k / 100k, every arm) | did not exist | complete | II.4b, X6, T3-18 |
| RouteLLM's own harness | running | done (metrics recomputed from its per-threshold prints; it crashes at its metrics step) | part B, T3-5 |
| GraphRouter via the LLMRouter library | NOT RUN | run with defaults and with a validation-selected sweep | D3, T3-16 |
| `bernoulli_scale_v5`, `replay_scale_v5` (1000-seed sweeps, 10..10^7) | did not exist | complete, b = 3 everywhere, halving at 10^7 refilled (uint32) | II.4e, matrices, erratum 22-24 |
| `bernoulli_b_sweep` b ∈ {1,2,3,5,10,20,30} | did not exist | complete (1.69 M rows) | II.4f, the cartel capture above b = 10 |
| framework shortlist reruns: `*_dd` (11 grids), `*_em` (7), VA-cohort at 10^2-10^5 (7) | did not exist | complete except Magentic-One at 10^2 with the VA cohort (*) | II.2 five-source table, II.4c 10^5 row, erratum 25 |
| live 10^5 peer-reported halving | absent | honest cells complete (= oracle, 3 seeds); liar cells partial (*) | II.4c |

Not run, by decision (unchanged): RouterDC (GPU fine-tune), RouteLLM causal_llm (gated Llama-3), mf / sw_ranking
(OpenAI embeddings), MODEL-SAT (LLM fine-tune), Avengers voting (needs generations), RouterArena (paid API inference),
a live-LLM run at n = 100k (4.8M fresh probe calls per seed). AgentsNet is not applicable (no routing decision).

---

**Added 2026-09-10/13 (post-deadline):** the many-seed scale sweeps — `bernoulli_scale_v5` (n = 10..10^7, 24 arms, five
liar regimes, b = 3 at every rung plus b = 1 controls at 10^6/10^7, 1000 seeds to 10^4, 100 at 10^7; 2.16 M rows) and
`replay_scale_v5` (n = 10..10^6, three shapes, same design; 1.58 M rows); the probe-budget sweep `bernoulli_b_sweep`
(b in {1, 2, 3, 5, 10} x n in {10^3, 10^4, 10^5}, 22 arms, 1.21 M rows) and its control `bernoulli_b_probe`; and
`knn_router[online]` at live n = 10^5. RESULTS II.4e / II.4f. Only `sequential_halving` at live 10^5 remains open
(second attempt running).

## 2. Headline numbers — before → after

*n = 1,000, self-described channel, mean success; paired Δ vs MIDIAN over the 120 cell × seed pairs.*

| arm | phase 1 (Q = 300, 3 seeds) | v2 draft (89.7%) | dossier (97%) | **final** |
|---|---|---|---|---|
| oracle | 0.72 | 0.723 | 0.723 | **0.723** |
| MIDIAN-VA | — | 0.675 (added 03:56) | 0.675 | **0.675** (+0.021 [+0.015, +0.026] vs MIDIAN) |
| MIDIAN-A | — | 0.668 | 0.668 | **0.668** |
| MIDIAN-V | 0.659 | 0.663 | 0.663 | **0.663** |
| MIDIAN | 0.637 | 0.654 | 0.654 | **0.654** |
| flat probe argmax (frozen) | 0.612 | 0.612 | 0.612 | **0.612** |
| declared argmax | — | 0.575 | 0.575 | **0.575** |
| LLM supervisor (7B) | — | 0.568 | 0.568 | **0.568** |
| Magentic-One (7B) | 0.51–0.54 band | 0.550 | 0.552 | **0.546** (−0.108 [−0.140, −0.076]) |
| Magentic-One (14B orchestrator) | — | 0.546 | — | **0.533** (−0.121) |
| Google ADK | band | 0.544 | 0.542 | **0.542** (−0.112) |
| LangGraph | band | 0.533 | 0.537 | **0.531** (−0.123) |
| LlamaIndex | band | 0.528 | 0.533 | **0.531** (−0.123) |
| OpenAI Agents | band | 0.523 | 0.528 | **0.528** (−0.125) |
| AutoGen | band | 0.531 | 0.532 | **0.528** (−0.126 [−0.160, −0.093]) |
| CrewAI | 0.535 | 0.525 | 0.528 | **0.528** (−0.126) |
| smolagents | band | 0.527 | 0.526 | **0.526** (−0.128) |
| CAMEL workforce | band | 0.516 | 0.534 | **0.527** (−0.127) |
| MAF | band | 0.524 | 0.524 | **0.524** (−0.130) |
| random | 0.31 | 0.314 | 0.314 | **0.314** |

- **n = 100, final:** MIDIAN-A 0.656, MIDIAN-VA 0.653, MIDIAN-V 0.642 (phase 1: 0.623), MIDIAN 0.639 (phase 1: 0.626),
  frameworks 0.522–0.553; paired −0.09 to −0.12 vs MIDIAN (dossier said −0.08 to −0.11) and −0.10 to −0.13 vs MIDIAN-VA.
- **Headline sentence (README):** was "MIDIAN 0.64, MIDIAN-V 0.66 vs frameworks 0.51–0.54; specialist 0.39 vs 0.75" →
  now "MIDIAN 0.65, MIDIAN-V 0.66, MIDIAN-VA 0.68 vs frameworks 0.52–0.55; specialist 0.39 vs 0.78".
- **Paired framework deltas** moved by ≤ 0.010 from the draft (Magentic-One −0.107 → −0.108, AutoGen −0.122 → −0.126,
  LangGraph −0.118 → −0.123, CAMEL −0.133 → −0.127); VA − MIDIAN (+0.021) did not move.

---

**Scale, 1000 seeds (II.4e), supersedes II.4b's 3-seed synthetic points and II.4d entirely.** MIDIAN-VA under the
low-skill cartel at b = 3: bernoulli 0.783 / 0.788 / 0.789 / 0.792 / 0.794 at 10^3..10^7 (II.4b had 0.798 / 0.805 at
10^4 / 10^5 on 3 seeds); replay 0.742 / 0.754 / 0.759 / 0.760 at 10^3..10^6. Declared argmax 0.719-0.724 (bernoulli),
0.672 (replay) at every n. "Declared beats MIDIAN at 10^6-10^7" and "flat collapses at 10^7" (II.4d, 2026-09-10) were
b = 1 artefacts: erratum 22. Seed std under the cartel: VA 0.013-0.017, plain MIDIAN 0.03-0.07, MIDIAN-V 0.05-0.12.

## 3. Minor numbers that changed (v2 draft / dossier → final)

**Frameworks, fallback accounting (Appendix E).** The missing rows were the slow, high-fallback cells, so several rates rose:

| framework, n = 1,000 | fallback rate | failures / fallbacks | strict success |
|---|---|---|---|
| Magentic-One (7B) | 64.5% → **60.4%** | 36.9 / 27.6 → **40.3 / 20.1** | 0.153 → **0.172** |
| Magentic-One (14B) | 60.6% → **43.2%** | 1.6 / 59.1 → **2.3 / 40.9** | 0.203 → **0.287** |
| Google ADK | 28.8% → 28.5% | 10.6 / 18.2 → 10.6 / 17.9 | 0.380 → 0.380 |
| LangGraph | 19.9% → 18.7% | | 0.422 → 0.427 |
| LlamaIndex | 9.4% → **16.1%** | | 0.476 → **0.443** |
| CAMEL workforce | 9.2% → **17.7%** | | 0.465 → **0.429** |
| CrewAI | 5.3% → **11.8%** | | 0.497 → **0.463** |
| OpenAI Agents | 16.3% → 15.3% | | 0.433 → 0.444 |
| AutoGen | 6.4% → 6.0% | | 0.497 → 0.497 |
| smolagents | 6.7% → 6.9% | | 0.463 → 0.462 |
| MAF | 2.8% → 2.8% | | 0.501 → 0.501 |

n = 100 moved the same way (Magentic-One 7B 63.5% → 61.2%, 14B 60.7% → 45.3%, CAMEL 12.3% → 19.6%, CrewAI 9.5% → 11.2%,
LlamaIndex 12.1% → 14.2%). The sentence "CrewAI: 5% fallback, none of it failures" is now "12% fallback, none of it failures".

**By shape (n = 1,000).** Frameworks on specialist 0.391 (0.367–0.444) → 0.390 (0.367–0.434); heavy_tail 0.629
(0.624–0.631) → 0.630, all ten frameworks identical; bimodal 0.573 unchanged. Pairs framework > MIDIAN: 0 / 335,
106 / 334, 324 / 324 → 0 / 400, 130 / 400, 400 / 400. n = 100: frameworks 0.471 / 0.555 / 0.563 → 0.479 / 0.557 / 0.561;
pairs 0 / 317, 22 / 329, 283 / 340 → 0 / 400, 29 / 400, 320 / 400. MIDIAN-VA by shape added: 0.802 / 0.679 / 0.544.

**Legibility axis (H2).** Frameworks − MIDIAN −0.388 / −0.012 / +0.034 → −0.388 / −0.013 / +0.033 (n = 100:
−0.288 / −0.070 / +0.037 → −0.283 / −0.070 / +0.034). "Frameworks flat in β (±0.02)" → 0.530–0.532 from β = 0 to 0.5.

**Verified shortlist (§1b, n = 1,000; lift over the plain arm, r = 10).** Magentic-One +0.085 → **+0.087** (0.632 → 0.633),
AutoGen +0.077 → **+0.081**, smolagents +0.069, ADK +0.064, CAMEL +0.043 (dossier +0.040) → **+0.050**, LangGraph +0.047 →
+0.049, CrewAI +0.049, OpenAI Agents +0.038, MAF +0.030, LlamaIndex +0.007 → +0.010 (CI still covers 0). The plain
columns follow Table 1. "r = 5 vs r = 10 within ±0.02" → within ±0.03 (MAF +0.030 vs +0.056). n = 100 lifts unchanged.

**Energy / latency (§6b, H10, H11, RESULTS.md Table 5).** The framework supervisor cost is the measured latency ratio to
AutoGen times the 7B call cost. Re-measured on the final rows (lighter fleet load than 09-03) the ratios roughly halved:

| per-task GPU-s | before → after | crossing vs MIDIAN (tasks) | before → after |
|---|---|---|---|
| AutoGen | 0.0294 → 0.0294 (anchor) | AutoGen | 9,416 → 9,416 |
| MAF | 0.0611 → 0.0353 | MAF | 4,531 → 7,836 |
| Google ADK | 0.0737 → 0.0473 | ADK | 3,756 → 5,852 |
| smolagents | 0.0744 → 0.0440 | smolagents | 3,720 → 6,298 |
| OpenAI Agents | 0.0824 → 0.0617 | OpenAI Agents | 3,362 → 4,489 |
| LangGraph | 0.0835 → 0.0568 | LangGraph | 3,315 → 4,875 |
| CrewAI | 0.1721 → 0.1188 | CrewAI | 1,609 → 2,331 |
| LlamaIndex | 0.2058 → 0.1453 | LlamaIndex | 1,345 → 1,906 |
| CAMEL workforce | 0.2561 → 0.1484 | CAMEL | 1,081 → 1,867 |
| Magentic-One (7B) | 0.5050 → 0.2806 | Magentic-One | 548 → 987 |
| Magentic-One (14B) | 0.8020 → 0.4721 | 14B arm | 345 → 587 |
| LLM supervisor | 0.0138 → 0.0080 | | |

The AutoGen crossing is 9,416 tasks in GPU-seconds and 9,415 in joules (messages and comparisons add a hair to MIDIAN's
per-task cost). Fig. 1a, `figures/F1_energy_crossings.csv` and this document use the joules value, 9,415;
RESULTS_energy.md's GPU-second table keeps 9,416 and its joules sentence 9,415.
Joules per task: Magentic-One 353 → 196 J (14B 561 → 330 J); AutoGen 20.6 J unchanged, its latency 1.11 → 1.92 s.
The MIDIAN-side numbers did not move (build 277 GPU-s, ~0 per task). Caveat carried into the docs: framework compute
is an estimate whose scale depends on fleet load at measurement time; the ordering and the "linear in tasks vs flat"
shape are what the figures support.

---

## 4. Errata — statements that were wrong, not merely provisional

1. **Phase 1, CrewAI fallback 79–84% ("all fallbacks").** Was the corrupted shared task store, not CrewAI. Final: 11–12%
   fallback, 0% failures, success 0.528–0.529.
2. **Phase 1 headline (Q = 300, 3 seeds): MIDIAN 0.637, MIDIAN-V 0.659, frameworks 0.51–0.54, specialist 0.39 vs 0.75.**
   At Q = 1,000 and 10 seeds with the fixed adapters: 0.654, 0.663, 0.524–0.546, specialist 0.39 vs 0.78. The README
   carried the phase-1 numbers until 2026-09-04.
3. **Phase 1 verified-shortlist lifts "+0.04 to +0.12 for every framework".** At 10 seeds: +0.01 to +0.09 at n = 1,000
   and +0.01 to +0.06 at n = 100; LlamaIndex's lift is not distinguishable from 0.
4. **v2 draft: "the 14B orchestrator converts failures into fallbacks without changing lenient success (−0.006 vs the 7B,
   10 / 26 cells)".** Final: 14B fallback rate 43% (not 61%), lenient success 0.013 *below* the 7B arm, below it in
   the 4 specialist cells and identical in the other 8; strict 0.29 (not 0.20). Its missing rows were the slowest of the
   whole programme (6–14 h each).
5. **v2 draft / dossier: Magentic-One (7B) 0.550 / 0.552 with 37% failures.** Final 0.546 with 40% failures. Still the
   best framework under lenient accounting and the worst under strict.
6. **Dossier Table 5 / §II.6: AutoGen 1.11 s per task; MIDIAN crosses Magentic-One after 568 tasks, the multi-call
   frameworks after 1,000–1,600.** Final: 1.92 s; 987; 1,900–2,300 (section 3 above).
7. **v2 draft: "MIDIAN-VA everywhere" was not true** — VA / A had no rows in the framework headline, churn, budget, n10k,
   r20, b10-shapes or scale grids; all added 02:28–03:56 and complete.
8. **Dossier 00:50, RouterEval real pools: "their MLP router 0.598 (5 rows)" at n = 1,000 β = 0.** Final 0.620 (0.607
   at β = 0.5 low-skill) — it matches MIDIAN-VA (0.617 / 0.607) and is immune to liars. "Their KNN at 5,000:
   PROVISIONAL" → 0.609 = flat frozen exactly.
9. **Dossier 00:50, RouterEval tuned rows (9 / 12 datasets).** Final: probe table 0.630 = best single model; their kNN
   k = 50 0.627; LinearR 0.667; Table 6 and X4 redrawn.
10. **Dossier 00:50 low-skill block: "other frameworks 0.520–0.531; 60 Magentic-One rows missing".** Final: Magentic-One
    0.538, ADK 0.536, the other seven 0.520–0.528; MIDIAN-VA 0.679 and the +0.15 paired lifts did not move.
11. **Dossier scoreboard "v3: 17 expectations".** 24, with verdicts for T3-18…24 (section 7).
12. **live_f1_n1000 "8,400 rows expected"** in the progress counter was an artefact of newer method files; the grid's
    VA / A arms are complete at 240 rows each.
13. **Pre-registration T3-19** was listed as TODO; it is a SPLIT (section 7).
14. **Their MLP router on RouterEval's 1,000-LLM pool under the cartel was printed as 0.607** (RESULTS.md §III.5,
    RESULTS_rte_v3.md D2) — that is MIDIAN-VA's value from the row above. The MLP router reads neither declarations nor
    reports, so it is flat in β: **0.620** at every β and both liar selections. Corrected 2026-09-04. The surrounding
    text ("at 1,000 it matches VA, 0.620 vs 0.617") was already right.
15. **The 14B Magentic-One arm was described as below the 7B arm "in 12/12 cells"** (RESULTS_rte_v2.md §1, written
    earlier on 2026-09-04). It is strictly below in the 4 specialist cells and **identical in the other 8**: on bimodal
    and heavy_tail both orchestrators pick the same agent from the same shortlist. The paired mean (−0.013) and the
    "never above" claim are unchanged. Corrected 2026-09-04.
16. **"MIDIAN-VA is +0.15 over every framework under the cartel"** (this document's section 5, and the 2026-09-04 00:50
    dossier) rounds up: the paired lifts run **+0.141 (Magentic-One) to +0.159 (LlamaIndex)**, so "+0.14 … +0.16" is the
    accurate phrasing. RESULTS.md §II.5 and RESULTS_rte_v2.md already print the per-framework values.
17. **Two cost exponents are in circulation for MIDIAN's per-task comparisons.** The live cross-n fit
    (`combined_scale`, 6 values of n) gives **n^0.136 [0.131, 0.152]**, quoted as n^0.14; the calibrated-synthetic fit
    (`bernoulli_scale`, 10²–10⁷) gives **n^0.112 [0.106, 0.118]**. Both are correct for their grid; any text quoting
    n^0.14 should name `combined_scale`. Flat and declared scans are n^1.000 in both.

18. **v4 first draft (2026-09-08): every cohort delta was a bare point estimate with no interval.** All are now paired
    seed bootstraps in `paper/NUMBERS.json` (`v4.<pool>.<base>_b<b>_<regime>`). Three of the conclusions do not survive
    them and are corrected in `RESULTS_rte_v4.md` (items 19-20 and section 7). The point estimates themselves were and
    remain correct.
19. **v4 first draft: T4-6 recorded a MISS on the 5,000-model pool** because MIDIAN-VA + `declared` reached 0.8344 under
    the cartel against VA-random's 0.8167. That delta is **+0.018 [−0.017, +0.043]** on 3 seeds and spans zero, so it
    never supported a miss. Withdrawn. The related "VA + declared 0.8511, the highest value in the v4 slate" is a true
    statement about the table but its delta is +0.029 [+0.000, +0.043], which touches zero and is not an effect.
20. **v4 first draft: "whether the modes help MIDIAN-VA depends on pool size — `specialty` and `declared` both help at
    m = 5,000".** Only `specialty` survives (+0.029, +0.026, +0.029 at b = 3, all three regimes, all clearing zero).
    Both `declared` cells span zero (+0.031 [−0.010, +0.067]; −0.014 [−0.037, +0.010]) and it is dropped from the claim.
21. **v4 first draft never defined its `cartel` column.** It is beta = 0.5 with `liar_select = low_skill_first`; the
    beta = 0.5 cells with randomly chosen liars are a separate, milder world. Pooling the two roughly halves every
    cartel effect (MIDIAN block at b = 10 reads −0.028 pooled against the correct −0.054). Now stated in the doc.

22. **II.4d (2026-09-10) and the first v5 matrices: "declared argmax beats MIDIAN at 10^6-10^7" and "flat collapses at
    10^7" were presented as scale effects. They are BUDGET artefacts.** Those rungs run b = 1 (inherited from
    bernoulli_scale), every rung below runs b = 3, and at b = 1 verification gets zero probes, so the V and VA columns
    there are plain MIDIAN and MIDIAN-A under other names (identical to three decimals). A same-n control at 10^5
    (`bernoulli_b_probe`) reproduces the entire drop by changing b alone: VA 0.796 -> 0.679, flat 0.778 -> 0.671,
    declaration arms 0.000. Withdrawn as scale claims; the rungs are rerunning at b = 3. The reader who caught it was
    right: MIDIAN-VA had not "collapsed", it had been switched off.

23. **`world._probe_idx` was a uint16 counter.** It seeds each (agent, family)'s next probe instance and capped at 65,535;
    sequential_halving at n = 10^7, b = 3 hands its winner 146,452 pulls and overflowed, so all 500 units at that rung
    lacked the halving arm (and MIDIAN-SH's "-10 out of bounds for uint16" at b = 1 was the same counter). uint32 since
    `ed9671a`; no existing result changes (the instance sequence is identical). Halving at 10^7 rerun 2026-09-13.
24. **Erratum 17's exponents are superseded.** On 1000 seeds (`bernoulli_scale_v5`, b = 3) MIDIAN's comparisons per task fit
    **n^0.121 [0.120, 0.122]** on the old grid's range 10^2..10^7 (old: n^0.112 [0.106, 0.118], 3 seeds) and
    **n^0.158 [0.157, 0.160]** on the full 10..10^7. The law is 2r log_r n (20, 40, ..., 140 comparisons per decade), so
    the fitted exponent depends on the range; every quotation must name it. Replay: n^0.178 over 10..10^6. Flat and
    declared: n^1.000 [1.000, 1.000] everywhere; the earlier n^0.958 for build probes was a fit across two budgets (fixed:
    `analyze.exponents` fits per b, `268e25d`).

25. **The framework shortlist is clone-filled, and two headline sentences explained it wrongly.** II.2 said frameworks
    sit on the oracle on bimodal "because strong agents are strong everywhere", and section 2 noted heavy_tail's ten
    frameworks "identical" without a cause. The cause: an agent's self-description is generated from its prompt and
    memoized, and its answers are memoized per prompt signature, so agents sharing a signature are indistinguishable
    clones; the adapter's TF-IDF top-10 (stable sort, lowest ids) fills with copies of the best-matching text once the
    population has more copies than shortlist slots. Exact recomputation from the population files: bimodal has 2
    distinct descriptions and heavy_tail 5 at ANY n, so every 10^3 top-10 there is ONE signature (all 100 heavy_tail /
    bimodal headline cells: ten identical frameworks); specialist has 3,920 distinct prompts, so the top-10 holds ~4.4
    signatures at 10^3, ~2.6 at 10^4 and exactly ONE at 10^5 (the withheld II.4c row, 16/16 families, 3 seeds). The
    framework's choice among clones cannot matter; those cells measured the retriever. RouterEval's rendered
    descriptions are all unique (1,000/1,000; 5,000/5,000) -- `fw_routereval_*` are unaffected. Fix (2026-09-14,
    `ffa2a31`): labeled variant `dedup: true` offers one agent per distinct text; every affected live framework grid is
    rerun into a separate `_dd` grid (11 grids, 3,389 units, ~3.5M supervisor calls), so pre-registered and deduplicated
    rows never share a directory or a row id. **Landed 2026-09-15:** the ten frameworks differ in every cell; by shape at
    10^3 they move 0.390 -> 0.381 (specialist), 0.630 -> 0.614, 0.573 -> 0.558; at 10^5 the row becomes 0.363 honest /
    0.344 cartel (best ADK = CrewAI 0.561 / 0.504) against MIDIAN-VA 0.836 / 0.828. The below-random frameworks are a
    RETRIEVER effect (the top-10's mean true skill is 0.31 vs population 0.43); a MiniLM retriever (`*_em` grids) lifts
    the 10^5 row to 0.493 / 0.459. RESULTS II.4c has the table; the pre-registered numbers stay as quoted.

26. **The trusted-observer `sequential_halving` arm is withdrawn from every table and sentence (2026-09-14).** It was
    quoted in III (RouterEval 1k/5k: 0.608 / 0.882, "trusted halving − oracle −0.020"), in README's summary paragraph and,
    for a few hours, in II.4e / II.4f. Its tournament is scored by a trusted observer the setting never provides; only
    the peer-reported variant is a legitimate rival. The rows stay in the raw grids; `scale_matrix.py` and
    `paper_numbers.py` drop the label, and NUMBERS.json is regenerated without its 194 entries.

---

## 5. New experiments since the v2 draft (2026-09-03 19:05), one line each

| part | what | result |
|---|---|---|
| SOTA retrieval | `*_sota` grids: BM25, Qwen3-Embedding-8B, RRF hybrid, Qwen3-Reranker-4B in front of the ten frameworks; 327 units, 3,267 jobs | at 10^5 beta = 0 the stack reaches 0.419-0.431 against TF-IDF 0.377 and MiniLM 0.493 -- **fusion and reranking make it worse**, dense alone is the best text arm |
| instruction | `*_sota_instruct`, `*_dense_instruct`: Qwen3-Embedding is instruction-tuned, the campaign used its STOCK web-search prefix | a task-appropriate instruction moves sota 0.431 -> 0.465 and dense to 0.523 (flat across regimes); the main SOTA arms measure a handicapped configuration |
| shortlist skill | `jobs/shortlist_skill.py`, `jobs/cohort_skill.py`, `jobs/position1_skill.py`: true skill of what each source offers (S read for MEASUREMENT only) | **every text retriever's POSITION-1 agent is at or below random (0.434)**: SOTA+rerank 0.247, hybrid 0.32, TF-IDF 0.384, MiniLM 0.385, best text arm 0.428. MIDIAN-VA's pick is 0.832 against a population best of 0.845 |
| framework pick | recovery = (routing - shortlist mean) / (shortlist best - shortlist mean) | frameworks recover only 15-35 %: they are far closer to random-within-list than to competent. MiniLM routing (0.493) still beats its own position-1 (0.385) and top-10 mean (0.455), so a small genuine description signal (+0.04) exists |
| position control | `fw_live_n100k_verified_va_shuffled`: same cohort members, MIDIAN's pick no longer first | 0.593 -> **0.504**, i.e. recovery falls from 37 % to 15 %, exactly the text-shortlist rate. Frameworks largely consume position 1. Predicted "~0.50" in the commit before the data landed |
| declared shortlist | `retrieval="declared"`: top-k by the declared claim, the cheap baseline the text retrievers were never measured against | shortlist mean 0.615 honest (highest of any source) but position-1 0.627 and ceiling 0.66-0.75; it concentrates liars (74 % of its top-10 at beta = 0.5 random vs a 50 % base rate). Grid written, not yet run |
| dedup | `*_dd` grids: framework shortlist without clones (erratum 25), 11 grids, 3,389 units | ten frameworks now distinct in every cell; 10^3 means move <= 0.02 (heavy_tail 10^2 +0.05); 10^5 row 0.363 / 0.344 vs VA 0.836 / 0.828; seven of ten below random because the TF-IDF top-10's true skill averages 0.31 |
| embed | `*_em` grids: MiniLM cosine retriever over the deduped descriptions, 7 grids, 3,267 units (complete 2026-09-16) | 10^5 row 0.493 / 0.459 (best 0.633 / 0.503); 10^4 0.475 / 0.434; 10^3 specialist 0.530 (+0.14), cartel 0.515; heavy_tail / bimodal within 0.03 of before; frameworks sit at their shortlist's mean true skill; VA +0.20 to +0.32 over the best framework at every n |
| coverage fill | five rivals never scheduled on the sweeps (LinUCB, cascade, referral, gossip, TrueSkill) on bernoulli 10..10^7 and replay 10..10^6, 5 regimes | gossip 0.671 → 0.415 and referral 0.556 → 0.431 under the cartel (= random 0.419) while MIDIAN-VA holds 0.789: decentralised report reading without trimming is owned by a colluding majority. LinUCB-honest 0.726 → 0.261 from 10^3 to 10^7, identical in all five regimes: a pure scale failure (RESULTS II.4e) |
| VA-cohort at scale | frameworks holding MIDIAN-VA's leaf cohort at 10^3 cartel, 10^4 (+cartel), 10^5 (all cells); 4 grids, 567 units | best shortlist at every n (10^5: 0.593 / 0.567 vs MiniLM 0.493 / 0.459) and still 0.22-0.26 below VA itself everywhere: orchestration, not retrieval, caps the frameworks (RESULTS II.2, II.4c) |
| b20/30 | `bernoulli_b_sweep` extended to b = 20, 30 (same cells/seeds, 2026-09-14) | VA plateaus 0.005 under the oracle from b = 20; the O(n) own-probe scans reach VA at b = 30 (0.838); plain MIDIAN / A / V fall above b = 10 under the cartel (A 0.777 -> 0.689, liar-misroute 0.15 -> 0.40); per-task cost flat in b |
| A | RouterBench on its own protocol (11 models, AIQ) | probe table b = 50 AIQ 0.707 vs their KNN 0.713 with 7× fewer labels ($30 vs $229); T3-1 MISS by 0.006, T3-2 HIT |
| B | RouteLLM's released bert router on RouterBench outcomes, their APGR / CPT | bert APGR 0.48 < random 0.51; probe table 0.59; their own harness rerun for their routers (ours NOT RUN inside it, T3-5) |
| C | their KNN / MLP routers as methods in our benchmark, n = 100 / 1k / 10k | KNN = flat frozen exactly; MLP +0.03–0.06 over KNN, below MIDIAN-VA at every β and n; at 10k VA − KNN +0.11…+0.13 |
| D1 | RouterEval on its terms (12 datasets × pools 10 / 100 / 1,000 real LLMs), defaults and validation-tuned | tuned: LinearR 0.667 > EmbedLLM 0.658 > … > probe table 0.630 (= best single) at 13% of the labels; T3-11 HIT, T3-12 / 13 MISS |
| D2 | every arm with liars on RouterEval's real pools, n = 10 / 100 / 1,000 / 5,000 | n = 1,000 VA 0.617 / 0.607 vs KNN 0.38; 5,000: VA 0.706 / 0.710 flat, peer halving 0.882 → 0.564, declared 0.864 → 0.614; T3-15 HIT, T3-17 4 / 6 |
| D3 | GraphRouter through the authors' library, defaults and tuned | a constant best-single-model router in every configuration; T3-16 |
| E | scale_100k: every variant and rival at n = 10k / 100k (calibrated bernoulli) | 100k VA 0.805 → 0.793 under collusion, V → 0.651, peer halving → 0.682, at 53 comparisons / task vs 100,000; T3-18 HIT |
| F | LLMRouterBench on its terms (15 datasets × 20 models, their protocol / metrics) | reproduction matches their Table 11; probe table 0.704 (5.6% of labels) vs Avengers 0.709, EmbedLLM 0.702, BSM 0.688; T3-22 split, T3-23 MISS |
| F2 | the 20-model pool with liars | MLP / flat online 0.687, MIDIAN-A 0.677 flat in β, VA 0.669; V / peer halving / declared collapse; T3-24 HIT |
| v2 | MIDIAN-VA / A rows everywhere (live_f1, framework headline, churn, budget, n10k, r20, b10 shapes) | headline VA 0.675 (+0.13…+0.14 over AutoGen / Magentic-One); churn VA 0.02–0.03 below MIDIAN (T3-20 MISS); budget VA best at b = 10, below V at b ≤ 3 (T3-21 split) |
| v2 | frameworks with MIDIAN-VA's shortlist (fw_live_n1000_verified_va) | within ±0.02 of the V shortlist at β ≤ 0.25, +0.021 at β = 0.5 (3 / 10 frameworks ≥ +0.03); T3-19 SPLIT |
| v2 | low-skill-first collusion with frameworks (fw_live_n{100,1000}_lowskill) | VA 0.679, +0.14 … +0.16 over every framework; MIDIAN 0.569, V 0.531 |
| v2 | §4 restructured as MIDIAN → +A → +VA (audits first) | monotone; V first gives a regression then a fix |

---

**Post-deadline (2026-09-10/13):**

| part | what | result |
|---|---|---|
| v5 sweeps | 24 arms x 5 regimes x n = 10..10^7 (bernoulli) / 10..10^6 (replay), up to 1000 seeds, b = 3 | VA best non-oracle arm under the cartel at every n >= 10^3, flat across decades; V worse than plain under collusion; exponents n^0.121 (old range) / n^0.158 (full) |
| b = 1 controls | same n, b in {1, 3} (`bernoulli_b_probe`) | the 10^6/10^7 "drop" reproduces at 10^5 from b alone; VA − A = V − plain = 0 at b = 1 |
| b sweep | b in {1, 2, 3, 5, 10} x 3 n x 5 regimes, 22 arms | verification switches on at b = 2; VA − A grows 0 -> +0.063 with b; V − plain falls 0 -> −0.113 |
| live 10^5 knn online | resubmitted at 256 GB after OOM at 64 GB | 26 / 27 arms at live 10^5 |
| live 10^5 halving, attempt 2 | fleet 46327112, concurrency honoured, 48 h warm-ups | running |

## 6. Figures

| figure | status | what changed |
|---|---|---|
| H1 headline by shape | redrawn 05:22 and 11:07 | VA / A bars, framework band with extreme names, low-skill panel; final Magentic-One bars |
| H2 legibility, H3 consistency / robustness, H4 break-even, H5 cost scaling | redrawn 11:07 | final framework points; H4 crossings per section 3 |
| H6 MIDIAN → A → VA vs halving | new (09-04) | — |
| H7 shortlist lift | new 01:47, redrawn 11:07 | final lifts for all ten frameworks at both n |
| H8 budget, H9 churn | unchanged | — |
| H10 runtime / energy, H11 joules + latency | redrawn 11:07 | new framework per-task costs and crossings |
| A_fallback_table | redrawn | final fallback rates |
| X1–X6 | new (09-03 18:24 – 09-04 03:20) | RouterBench, RouteLLM, learned routers in our benchmark, RouterEval tuned, real pools with liars, 10k / 100k scale |
| A_* (others), F1–F6 | unchanged | phase-1 / v2 MIDIAN-side figures |

---

## 7. Pre-registration verdicts that changed

| target | earlier | final |
|---|---|---|
| T3-18 (VA at 10k / 100k) | not run | HIT, all three clauses |
| T3-19 (VA shortlist for frameworks) | TODO | SPLIT (β ≤ 0.25 clause holds; β = 0.5 clause +0.021 vs ≥ +0.03, 3 / 10 frameworks hit) |
| T3-20 (VA under churn) | not run | MISS (VA 0.02–0.03 below plain MIDIAN) |
| T3-21 (VA ≥ V at every budget) | not run | split (VA ≥ V at b = 10, below V at b ≤ 3) |
| T3-22 / 23 (LLMRouterBench on its terms) | not run | 22 split (within 0.02 of the best learned router; +0.016 vs a pre-registered +0.02 over the best single), 23 MISS (Gap@O just under 0.25) |
| T3-24 (20-model liar pool) | running | HIT, all three clauses |
| T4-1 (block ≤ random everywhere) | HIT | MISS on the quantifier — block is significantly positive in 3 cells, all MIDIAN-VA (m = 1,000 b = 3 random-liars +0.013 [+0.004, +0.022]; LLMRouterBench b = 10 cartel +0.018 [+0.004, +0.032], b = 3 cartel +0.008 [+0.000, +0.015]). The b = 10 cartel pattern still holds to −0.288 |
| T4-4 (declared up at β = 0, down at the cartel) | HIT, "larger than expected" | SPLIT — the cartel half is solid; the β = 0 half holds only on the 5,000-model pool (+0.042 [+0.027, +0.060], 3 seeds), spans zero at m = 1,000 |
| T4-6 (nothing beats VA at the cartel) | MISS on the 5k pool | NOT CONTRADICTED (the delta spans zero; errata 19) |
| T4-2 (specialty > random on specialist, b = 10) | untested | **MISS on all six comparisons** — never reaches the +0.010 threshold (−0.003 at β = 0) and is −0.103 / −0.139 under the cartel. The slate's central hypothesis, falsified on our own benchmark |
| T4-3 (specialty null on bimodal, b = 10) | untested | **HIT on all six** — every abs(delta) <= 0.003 |
| v1 targets (6), v2 targets (11) | — | unchanged (0 hits / 5 misses / 1 split; 6 hits / 4 misses / 1 cost-split) |

---

## 8. What did not move

Oracle, random, every MIDIAN-family number, flat, declared and supervisor arms in Table 1; VA − MIDIAN +0.021; the
low-skill collusion block; the by-shape story (frameworks on the oracle where skill is legible, 0.39 vs 0.78 on
specialist); every external comparison in RESULTS_rte_v3.md; H8 / H9; the MIDIAN-side energy numbers.

---

## 8b. Erratum 27 -- the lie model does not touch the self-description TEXT (2026-09-19)

`World.apply_lying` inflates the declared MATRIX (`D[liars] += 0.4`). The self-description text is generated once per
population and cached in `<population>/descriptions.json`, whose path carries no beta -- so a liar's TEXT is identical
to its honest text, and **every text-retrieval shortlist (TF-IDF, dedup, BM25, MiniLM, dense, hybrid, sota) is
bit-identical across all five liar regimes**. Their only beta-dependence is the declared-argmax fallback taken when a
framework fails to pick, which is what moves MiniLM 0.495 -> 0.459 from honest to the beta = 0.5 cartel.

Consequence for reporting: the apparent liar-robustness of the text arms -- most starkly `dense + I-competent` at
0.523 in BOTH the honest and cartel cells -- is an artefact of the threat model, NOT a result. A real adversary would
lie in its description too. Every cross-regime claim about a text-retrieval arm must carry this caveat, and the
declared-channel arms (declared_argmax, `retrieval="declared"`, the frameworks' fallback) are the only ones the
current lie model actually attacks.

---

## 8c. Erratum 28 -- framework rows that silently measured declared argmax (2026-09-22)

A supervisor/worker error returned `choice=None` and the adapter routed by declared argmax, writing a normal-looking row
that measured declared argmax under the framework's name. Root cause: 7 framework conda envs (autogen, crewai,
google_adk, langgraph, llamaindex, maf, openai_agents) had 240 dangling library symlinks each -- the targets of libffi,
libbz2, libreadline, libsqlite3 and ~100 more had been deleted, and the extracted package cache lost the same files.
Workers crashed only on nodes whose `/lib64` lacked a compatible copy, so the failure was bimodal per unit (0% or ~100%
fallback) and read as framework behaviour. CrewAI and ADK were hit hardest (~2/3 of units each).

Effect on reported numbers: declared argmax scores 0.619 honest, above most frameworks, so contamination biased CrewAI
and ADK UPWARD -- it is why they topped the ten-framework table (0.633 on MiniLM at 10^5, identical to each other in
every shortlist condition) -- and inflated every "mean of ten frameworks". Correcting it can only lower those numbers.

Fix: envs restored byte-identically from the package archives (`scripts/ops/repair_fw_envs.sh`; `scripts/check_envs.sh`
passes on all 14 envs); all ten frameworks verified live (3/3 correct delegations, 0 infra errors each);
`FrameworkMethod.fetch` now fails a unit instead of writing a row once infra errors pass 2% of calls; 4,269 contaminated
rows moved to `results/<grid>/quarantine/` (not deleted) and their 2,904 units rerun. Rules: OPS_RULES.md D1-D6.

---

## 8e. Erratum 29 -- the supervisor's own invalid actions were failing units as "infrastructure" (2026-09-23)

The erratum-28 rule counted EVERY framework-side exception as an infrastructure error, so units stopped writing rows once
max(3, 2%) of calls failed. The only exceptions that ever tripped it were the supervisor LLM's own invalid actions:
- ADK: 58 failed unit logs, "Tool 'agent_000030' not found" (a tool named after the agent, not `transfer_to_agent`);
- OpenAI Agents: 1, "Tool transfer_to_agent_00128 not found" (ModelBehaviorError);
- MAF: 1, "next_speaker must be provided"; and (found 2026-09-23 22:45, 14 units at 10^2-10^4) "KeyError: 'agent_005044'":
  the group-chat orchestrator named a participant that is not among its candidates, so the worker's name lookup fails.
ADK's honest 10^2 / 10^3 units therefore never landed, and ADK was absent from those shortlist bars with no mark.

These are the framework failing to route, not our infrastructure. `FrameworkMethod.fetch` now treats them as a non-pick
exactly like an unparseable reply: declared argmax inside the shortlist, counted as `invalid_action` in `fallback_rate`,
0 under `success_strict`, and not retried (no other non-pick gets a second sample). The classes are listed in
`_common.INVALID_ACTION`; every other error still fails the unit. Rows already written are unchanged (their few
tolerated errors took the old retry-then-fallback path); the failed units are rerun.

Also 2026-09-23: RouterEval's MMLU family order depended on PYTHONHASHSEED ("high school biology" and "philosophy" tie at
248 train prompts, family index 9 / 10). Each unit was internally consistent, but rows of one (cell, seed) run in
different processes could come from the two orders (two distinct oracle values in 240 of 270 `routereval_mmlu` units,
spread <= 0.013). Ties are now broken by name (`rte/backends/routereval.py`); means are unaffected, cross-grid pairing on
RouterEval m <= 1,000 is noisier for older rows.

---

## 8d. Rival note -- the warm-start bandit scores HIGHER under the cartel than honest (2026-09-22, post hoc)

Live, specialist, b = 3: warm_start_bandit is +0.04 under the β = 0.5 low-skill cartel at 10^4 and 10^5 (0.743 -> 0.783,
0.738 -> 0.778), paired per seed in all 3 seeds at both sizes, while routing 42-47 % of its tasks to liars (10^3: +0.004).
Not a lie benefit. Its prior is Beta(n0 D, n0 (1 - D)), n0 = 5, on the self-described claims; the lie clips liars' D near 1,
so their prior failure count is ~0 and every probe failure dominates -- for them the bandit becomes "trust the b probes".
An offline Thompson simulation on the real 10^4 populations (Bernoulli outcomes at true S; diagnostic only) reproduces it:
honest D 0.752 / 0.750 / 0.734, cartel D 0.758 / 0.789 / 0.769, EVERYONE +0.4 with no liars 0.755 / 0.780 / 0.766, flat
prior 0.689 / 0.697 / 0.700. So the honest warm-start rival is under-tuned (its prior is too pessimistic on self-described
claims); a tuned honest version would sit near its cartel number. Reported numbers are unchanged (the rival is as
pre-registered); state it beside any honest-vs-cartel comparison of this arm.

---

## 8f. Erratum 30 -- three leaks on the non-live backends, and their opt-in fixes (2026-09-23, post hoc)

Audit (paper/audit/leakage.md, rivals.md, protocol.md). Fixes 1-3 and the agent shuffle are opt-in: each is a new cell
value, so no row id moves and default cells stay bit-identical (tests/test_erratum30.py pins fingerprints taken with
the pre-fix code). The probe-index fix (last bullet) is NOT opt-in; it changes the default rows of `trueskill_per_family`
and of no other arm.
- **Declarations were an answer key.** On bernoulli, replay and RouterEval / LLMRouterBench both declared sources were
  `clip(S + N(0, 0.05))` (corr(S, D) 0.99); live self-ratings are over-confident and weak (mean 0.688 vs S 0.425, corr
  0.36). Fix: `declared_source: calibrated` draws D from the live self-rating's empirical distribution given S's decile
  (10 bins x the 11 values live models emit; `rte/backends/__init__.py` `CAL_P`, fitted on the 40 live specialist
  n = 100 / 1,000 populations, 352,000 pairs, by `scripts/fit_declared_calibration.py`). It reproduces live mean 0.688,
  sd 0.283, corr 0.354-0.357 (live 0.360), P(D = 0) 0.072, P(D = 1) 0.21. The declared-argmax pick has mean S 0.58
  (live 0.61, programmatic 0.83, max 0.84). A linear censored-normal fit, clip(0.523 + 0.485 S + N(0, 0.348)), picks
  worse (0.52) and was rejected.
  The draws are i.i.d. per (agent, family), but 25 % of the live residual variance is per agent. On heavy_tail / bimodal
  the calibrated claims are therefore less informative than live: pick 0.39 / 0.42 vs 0.62 / 0.57. Lying is unchanged
  and applies on top of D.
- **Calibrated ties broke toward the weakest RouterEval agent** (independent review). Calibrated D takes only 11 values,
  so many agents tie at the maximum. declared_argmax, declared top-k (stable argsort) and cluster_head_router break
  ties by the lowest agent index. RouterEval's mmlu pools are stored weak to strong (Spearman 1.0 between index and
  skill), so on mmlu strong_to_weak n = 1,000 the declared-argmax pick had S 0.25; with random ties it is 0.61-0.63.
  The same order effect is slight on LLMRouterBench (rho -0.11) and the leaderboard pool (rho -0.05). Bernoulli and
  replay agents are in random order, so they are unaffected. Fix: routereval `backend_kwargs: {shuffle: true}`
  permutes the pool per seed (`stable_seed_32(seed, "agent_order")`) before S is computed. It is set on every
  RouterEval / LLMRouterBench erratum-30 grid. Unshuffled `_cal` / `_norep_cal` RouterEval rows that already landed are
  superseded.
- **Replay probed and routed on the same prompts**, and S (oracle, liar selection, declarations) was the accuracy over
  all of them. Fix: replay `backend_kwargs: {split: true}` splits each category once (rng 0, 70 / 30 as LLMRouterBench)
  into probe rows (probes, S) and task rows (routed tasks). Under split the oracle picks by probe-row S and executes on
  task rows, as on RouterEval. It is therefore not an exact ceiling: an arm can beat it on the task rows. This is not
  changed.
- **RouterEval / LLMRouterBench streams repeated test prompts** (instance % pool; repeat share 0.48 on mmlu, 0.20 on
  LLMRouterBench), so online learners that see the prompt could memorise answers. Fix: `backend_kwargs: {no_repeat: true}`
  makes `World.tasks` visit each (family, test prompt) at most once. Families are drawn by the demand vector over the
  families that still have prompts, so demand stays uniform until the smallest family runs out, and Q above the pool
  raises. Test pools in the K used families: mmlu 790 prompts (27-154 per subject), leaderboard 1,570 (53-307),
  LLMRouterBench 3,445 (26-376). The new grids use Q = 300 everywhere; over 30 seeds the last 0.8-0.9 % of a stream
  lacks one family on average (at most 15 %).
- **Repeated cells in one probe call shared one instance** (protocol audit F4). `World._probe` read and bumped the probe
  index with fancy indexing, so an (agent, family) that appeared m times in ONE call got the same k-th instance m times
  and the index advanced once. Fixed in `rte/world.py` (`_probe`, `_occurrence`), and the fix CHANGES DEFAULT ROWS:
  each occurrence now takes the next `reps` instances, as sequential calls would. It went live at 2026-09-23 16:49, so
  `trueskill_per_family` rows written before then are pre-fix. A small test world shows the size of the change:
  success 0.85 -> 0.555 at b = 1 on bernoulli (tests pin both fingerprints). The audit saw live 10^3 fall from 0.745 in
  old rows to 0.607 in new rows. Calls without a repeat take a fast path with the old per-call cost. Calls without a repeated cell are byte-identical, which the tests check. A spy over all 69
  non-framework arms in grid.yaml (bernoulli and LLMRouterBench, b = 1 / 3 / 5, honest and cartel) found repeats
  in `trueskill_per_family` only. Its random pairings draw agents with replacement, so about half its probes were
  duplicates. Every `trueskill_per_family` row predating this fix needs a rerun; no other arm's rows change.
- **Grids**: `bernoulli_1e7_cal` (claim-reading arms only), `replay_1e6_split_cal` (every B arm, 30 seeds),
  `routereval5k_norep_cal`, `llmrouterbench_norep_cal` and `routereval_mmlu_norep_cal` ran 2026-09-23 / 24 and B reads
  them (`seed_tables.ERRATUM30`, switched per family once every planned row landed). `routereval5k_cal` and
  `llmrouterbench_cal` (calibrated claims, repeated prompts) are superseded by the norep grids and not drawn. H's
  framework mirrors `fw_routereval_{small,1k,5k}{,_em,_va}_norep_cal` and `re_sl_{declared,embed}_{small,1k,5k}_norep_cal`
  run after A / B; H switches only when all of them have landed.
- **Agent order on RouterEval / LLMRouterBench (`shuffle: true`, in every erratum-30 RouterEval grid).** Calibrated
  claims take 11 values, so many agents tie at the maximum and every claim reader broke ties by the lowest agent index;
  the RouterEval MMLU pools list agents weak to strong (Spearman 1.0 at m = 10 / 100 / 1,000), which sank declared argmax
  from 0.62 to 0.25 of skill and favoured MIDIAN-VA. A per-seed permutation of the pool makes the ties random. Unshuffled
  rows landed before the fix are in `results/_quarantine_unshuffled_2026-09-23` and are not read.
- **Speed-only opt-ins used for the kNN reruns (2026-09-23 / 24).** `RTE_EMBED_BATCH=1` embeds a probe set in one batched
  MiniLM call instead of one call per prompt; `RTE_MINILM_CUDA=1` runs MiniLM on a GPU (fp32); `RTE_TEXT_PROCS=N`
  generates the probe prompts in N forked processes (identical texts); with the batch path, the last probe-set embedding
  is cached in-process, keyed by the probe-instance matrix, so kNN / kNN-online and both regimes at one b share one encode
  (the cached array is read-only); `RTE_OUTCOME_CACHE=1` scores each (population, agent, family, instance) probe once per process and forks the scoring over `RTE_TEXT_PROCS` workers (outcomes identical; only the `llm_executions` diagnostic counts fewer calls); `RTE_RG_CACHE=1` caches reasoning-gym's word-corpus read and regex scan that letter_counting / word_sequence_reversal redo in every problem's constructor (identical problems and scores, ~27x). Used for `rivals_b_n10k` (cartel kNN), `rivals_b_n100k` (kNN) and the cartel
  `routereval5k_norep_cal` kNN units. Batched vs per-prompt embeddings differ by <= 2.4e-7 and gave identical top-10
  neighbour sets on 300 queries; defaults are unchanged.

## 8g. The MIDIAN rename (2026-09-24)

The full method is now called MIDIAN; the variants are its ablations. Nothing measured changed: every stored number is
reproduced, and the figure CSVs regenerated on the renamed rows match tag `submission-2026-09-24` value for value (737
rows matched, 0 differing).

| before | method key before | now | method key now | analysis label |
|---|---|---|---|---|
| MIDIAN-VA | `midian_va` | **MIDIAN** | `midian` (params `{}`) | `midian` |
| MIDIAN-A | `midian_a` | MIDIAN w/o verification | `midian{"verify": false}` | `midian_wo_verify` |
| MIDIAN-V | `midian_v`, `midian{"verify": true, "cached": true}` | MIDIAN w/o audits | `midian{"audit": false}` | `midian_wo_audit` |
| MIDIAN (plain, pre-registered) | `midian` | MIDIAN w/o defenses | `midian{"audit": false, "verify": false}` | `midian_wo_defenses` |
| MIDIAN-SH / MIDIAN-SHA | `midian_sh` / `midian_sha` | withdrawn | — | — |
| framework shortlist "VA cohort" | `retrieval: midian_va` | MIDIAN cohort | `retrieval: midian` | source key `va_cohort` (unchanged) |
| framework shortlist "V cohort" | `retrieval: midian` | cohort of MIDIAN w/o audits | `retrieval: midian_wo_audit` | not drawn |

Display strings follow `docs/FIGURE_SPEC.md` §2 (e.g. "MIDIAN-VA (whole population)" → "MIDIAN, no framework", "best
learned router" → "best learned/declared router", "flat probe argmax (online)" → "flat probe argmax"). Grid and
directory names (`va_b_*`, `*_verified_va*`, `midian_v_replication`, `paired_vs_midian.csv`, …) are identifiers and keep
their spelling.

- **One class, two flags** (`rte/methods/midian.py`, commit fc816f3). `Midian(audit=True, verify=True)` is MIDIAN;
  `audit` (report audits with reporter exclusion, rate 0.05, two strikes) and `verify` (verified promotion with
  b0 = b − 1; `cached` defaults to `verify`) switch the defenses off one at a time. `midian_a.py`, `midian_v.py`,
  `midian_va.py`, `midian_sh.py` and `midian_sha.py` are deleted. Level 0 keeps, per flag, the engine each variant had
  (with audits: the per-peer audited engine formerly in `MidianSH`; without audits: `peer_reported_estimates`, trimmed by
  reporter with verification and by report without), so every stored number is reproduced: the parity check reran 216
  stored rows under the new keys and all 216 are bit-identical.
- **`rte/methods/keys.py`**, the one place that knows the old keys. `to_new(method, params)` translates a stored or
  configured key (None for a withdrawn variant); `legacy(method, params)` is its inverse on the new keys; `normalize(df,
  dir)` translates the rows of an unmigrated directory on read and, for a migrated one, only checks that no old key is
  left (`assert_v2`). Every loader (`rte.analyze`, `seed_tables`, `fw_variant_numbers`, `efficiency_figs`, …) reads
  through it. The runner seeds the world with the **legacy** key (`rte/run.py`, `run_method`), so a rerun of any cell
  reproduces the rows stored under the old key.
- **One-time data migration** (`scripts/ops/migrate_method_keys.py`, dry run by default, `--apply` to write). Per results
  directory: every row of `rows.csv` and `rows.d/*.json` whose key changes under `to_new` gets the new key and a new rid
  (`rte.run.rid_of_row` on the rewritten row; rows.d files are renamed to it); withdrawn MIDIAN-SH / SHA rows are removed;
  every changed or removed original is first copied to `<dir>/_premigration_v2/` (`rows_changed.csv.gz` and the original
  rows.d files), so the step is reversible; unchanged values are written back byte-identical; the directory then gets the
  sentinel `.method_keys_v2` and is never migrated again (a second pass would turn the new full-method `midian` rows
  into the old undefended ones). It refuses, changing nothing, if a stored rid does not match the rid recomputed from its
  old key or two rows would share a new rid. The dry run over 199 directories found no rid mismatch and no collision;
  applied 2026-09-24 11:26–11:30, it re-keyed 1,049,487 CSV rows and 35,407 rows.d files and removed 179,040 SH / SHA CSV
  rows and 1,944 rows.d files (the sentinel files' counts).
- **`midian_v` and `midian{verify, cached}` are one arm now.** They were already pooled under one analysis label
  (`midian_v`) and both map to `midian{"audit": false}`. `legacy` returns `midian_v` for it, so a rerun is seeded with
  that key and reproduces the rows first written as `midian_v`.
- **The reference of every paired analysis is MIDIAN w/o defenses** (`rte.analyze.REF = "midian_wo_defenses"`), i.e.
  the same comparisons as before, when the plain `midian` was the reference; `paired_vs_midian.csv` keeps its name.
- **MIDIAN-SH and MIDIAN-SHA are withdrawn**: their code is deleted and their rows are only in the `_premigration_v2`
  backups. They were v2 negative controls on the do-not-add list and in no reported figure.
- **Grids and scripts** (commit 4f1f56b): `configs/grid.yaml`, the runner, the analysis, the framework adapter and every
  script use the new keys; the 193 grids plan the same arms as before. The figures follow `docs/FIGURE_SPEC.md` through
  `scripts/figspec.py` (commit 9fb00e5). Historical documents (RESULTS*.md, TARGETS_rte_v2–v4.md) keep the names of their
  date, with a note under the title.

## 9. Still open (not results)

- Coverage fill and expansions (COVERAGE.md §7-§9, 2026-09-17): the synthetic and live fills are complete and folded; three
  items remain and are marked * in RESULTS Part VI -- the live 10^5 peer-halving β = 0.5 cells (the cartel cell has no seed),
  the live 10^4 heavy_tail / bimodal campaign, and the RouterEval small-pool framework variants. `fill_fold_live` and
  `fill_fold_expand` fire when they drain; then re-run `scripts/doc_tables.py --sync --verify`, `scripts/paper_numbers.py`
  and `scripts/bar_figs.py`, and update COVERAGE §9.
- Live 10^5 peer-halving liar cells: β = 0.25 seeds 2-3 and β = 0.5 (both liar sets) seeds 1-3, rerunning with 3-day
  limits from the memo (the first follow-ons hit 24 h); II.4c carries * until they land. The halving row is otherwise
  complete (honest cells = the oracle, 3 seeds).
- 10^2 VA-cohort frameworks: Magentic-One units still running (`fw_live_n100_verified_va[_lowskill]`); the S1 / S6 cells
  carry * and `scripts/doc_tables.py --sync` refreshes them.
- Figures: M1-M4 and H6 / H7 regenerate after the two items above (10^5 live point; b = 20 / 30 panel; halving without
  the trusted-observer arm; the five-shortlist-source panel).
- `bernoulli_b_sweep` b = 20, 30: plain MIDIAN / A / V DECLINE above b = 10 under the cartel while their liar-misroute
  rate rises (RESULTS II.4f). The numbers are on 200-1000 seeds and the counters are unambiguous; the code path (which
  report-channel weight in `midian.py` grows with b, and why A's 5% audits stop keeping up) is not yet traced.
- MIDIAN-SH/SHA are absent at every b = 1 rung by construction; `units_todo` for those grids never reaches 0 for that reason.
- SOTA campaign refill (2026-09-19): 669 of the 10^3 four-arm jobs hit the 1-day walltime (4 specs x 1000 tasks = 4000
  supervisor calls) and the first launcher enumerated 5 regimes by hand where the grids carry 6 (beta = 0 x
  low_skill_first was dropped). `jobs/relaunch_gaps.sh` refills from the GRID LOADER at a 3-day limit; rte.run skips
  rids already on disk. Affected: n = 10^3 sota (2249/4920), n = 10^2 lowskill (31/630), and the three 10^5 controls.
- `retrieval="declared"` (`fw_live_n100k_declared`) is implemented and tested but NOT yet run. The diagnostics predict
  it beats the VA cohort as a framework shortlist in three of five regimes on shortlist MEAN, and loses badly on
  position-1 (0.627 vs 0.832); which matters depends on the framework's 15-35 % recovery, so it needs the data.
- The not-run rivals in section 1 stay not run without a decision.
