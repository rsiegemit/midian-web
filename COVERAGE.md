# COVERAGE.md — what exists, what is missing, and whether the gap is justified (audit of 2026-09-17)

Built from the bar-figure data (`figures/bars/<family>.csv`, every arm × n × regime × grouping) against the full method
inventory (`METHODS.md`). "Justified" = a documented cost, backend or scope reason; "not justified" = the arm would run
at trivial cost and simply was never scheduled. Do-not-add arms are listed for completeness but not counted as gaps.

## 1. Live RTE (llm backend; n = 10^2, 10^3, 10^4, 10^5)

Present at every n and regime: MIDIAN, V, A, VA, flat frozen / online, peer halving, warm-start bandit, KNN router
(+ online), declared argmax, random, oracle, the ten frameworks (pre-registered adapter; plus dedup / MiniLM / V-cohort /
VA-cohort variants where run). Full rival set (29 labels) at 10^3 only.

| gap | where | verdict |
|---|---|---|
| n = 10 | absent | justified: the SPEC's n set starts at 10^2; RouterEval carries n = 10 with real LLMs |
| heavy_tail, bimodal | only at 10^2, 10^3 | justified by cost (a 10^5 world is 4.8M generations per seed); documented in grid.yaml |
| β = 0.1 | only at 10^2, 10^3 | justified: the 10^4 / 10^5 grids pre-registered β ∈ {0, 0.25, 0.5} |
| cnp_self_bid, cluster_head_router, declared_softmax, disrouter_cascade | only at 10^3 | **not justified**: declared-channel arms cost no LLM calls beyond the routed execution (memo hits); absent at 10^2 (core4 grid) and 10^4 / 10^5 by grid design |
| ucb_per_family, thompson_per_family | only at 10^3 | **not justified**: same probe budget as flat; on the calibrated backend they sit ≈ flat frozen, so low value, but cheap |
| verify_on_claim, linucb_honest | verify_on_claim absent at 10^2; linucb absent at 10^2 | **not justified** (cheap); present at 10^3–10^5 |
| flat_nsw_router, referral_network, gossip_reputation_greedy | only at 10^3 | **not justified** at 10^2 and 10^4; at 10^5 the two decentralised arms are O(n·d) messages — feasible (bernoulli has NSW at 10^7) |
| trueskill_per_family | only at 10^3 | 10^5: justified (NotImplemented, pure-Python loop, DEVIATIONS); 10^2 and 10^4: not justified, just slow |
| mlp_router | 10^2, 10^3 only | justified (agent one-hot × 480k / 4.8M probes; DEVIATIONS) |
| llm_supervisor | 10^2, 10^3 only | **not justified**: the practitioner default is the frameworks' adapter with one direct call, 300 calls per unit at 10^4 / 10^5 |
| Magentic-One 14B orchestrator | 10^2, 10^3 only | justified: an asymmetric arm that answers one question (M3) |
| midian_llm_descent | 10^3 only | justified: SPEC §9 ablation, appendix |
| peer halving liar cells at 10^5 | 3 of 15 cells landed (honest cells complete) | in progress (3-day jobs, memo-resumed); asterisked |
| Magentic-One at 10^4 cartel | absent | justified and documented (18 s per task; the cartel grid ran nine frameworks) |

## 2. Calibrated bernoulli (`bernoulli_scale_v5`; n = 10 … 10^7, five regimes, 1000 seeds, b = 3)

Present: 23 arms at all seven rungs and all five regimes (MIDIAN family incl. r = 5 variants, SH, SHA; flat frozen / online;
NSW; UCB; Thompson; warm-start; verify-on-claim; peer halving; declared argmax / softmax; CNP; cluster-head; route-to-k;
random; oracle).

| gap | verdict |
|---|---|
| the ten frameworks, llm_supervisor, midian_llm_descent | justified: no LLM on this backend (no self-descriptions, no supervisor) |
| knn_router, mlp_router | justified at ≥ 10^6 (MiniLM over n·K·b synthetic prompts, 1.6e8 at 10^7); at ≤ 10^5 the synthetic prompt text carries only the family name, so a text-embedding router degenerates to flat — running it would measure nothing |
| linucb_honest | **not justified**: probe-only, cheap; simply not in `scale_arms` |
| trueskill_per_family | ≥ 10^5 justified (NotImplemented); 10 … 10^4 not justified (slow pure-Python, hours at most) |
| disrouter_cascade | **not justified**: declared-only, trivial cost |
| referral_network, gossip_reputation_greedy | **not justified** below 10^6 (reports channel exists here; O(n·d) messages); at 10^7 the EigenTrust power iteration over an n × n sparse matrix is the only real cost question |
| heavy_tail, bimodal shapes | partially justified: the v5 sweep's question was scale on the specialist shape; the v1 bernoulli mirrors carry the shapes at 10^3 |
| b ≠ 3 | b = 1 controls at 10^6 / 10^7 and the b sweep (1…30 at 10^3–10^5) exist; no b sweep at 10^6+ (justified by cost, and II.4f shows b-dependence identical across n up to b = 10) |

## 3. RouterBench replay (`replay_scale_v5`; n = 10 … 10^6)

Same 23 arms and five regimes at every rung; same arm gaps and verdicts as bernoulli. Additional:

| gap | verdict |
|---|---|
| n = 10^7 | justified: the backend's outcome matrix tops out at 10^6 |
| per-shape reporting | **reporting gap, not a data gap**: the three shapes are pooled in the matrix; per-shape matrices need one SLURM pass over the 1.6M-row file (`scale_matrix.py --dist`) |

## 4. RouterEval real LLM pools (`routereval_mmlu` 10 / 100 / 1,000 × 3 pool configs; `routereval_mmlu5k` 5,000; five regimes)

Present: 18 arms (MIDIAN family incl. SH / SHA; flat frozen / online; both halvings; warm-start; LinUCB; KNN (+online);
MLP; declared argmax; random; oracle) at 10 / 100 / 1,000; the nine frameworks at 1,000 and 5,000 (`fw_routereval_*`).

| gap | verdict |
|---|---|
| cnp_self_bid, cluster_head_router, declared_softmax, disrouter_cascade, route_to_k_majority | **not justified**: the grid is described as "every arm + liars"; the declared channel exists (noisy_declared(S)) and these cost nothing |
| ucb_per_family, thompson_per_family, verify_on_claim | **not justified**: the probe pool exists (that is how flat and halving run); cheap |
| flat_nsw_router, referral_network, gossip_reputation_greedy, trueskill_per_family | not justified at 10 / 100 / 1,000 (cheap); trueskill at 5,000 borderline |
| mlp_router at 5,000 | justified and documented (one-hot × 240k probes) |
| knn_router_online at 5,000 | **undocumented**: no reason recorded; likely memory/time, should be run or the reason written down |
| frameworks at 10 and 100 | not justified by cost (nine frameworks × few cells); justified by scope (the post-hoc framework grids targeted 1,000 and 5,000) |
| Magentic-One on RouterEval | justified and documented (18 s per task) |
| llm_supervisor, midian_llm_descent | justified: no self-descriptions on this backend (declarations are rendered vectors) |
| datasets other than MMLU in the liar grids | justified by pre-registration scope (D2 = MMLU); the 12-dataset comparison is the "on its terms" part without liars |

## 5. LLMRouterBench 20-model pool

The same 18-arm subset at n = 20, five regimes. Same verdicts as RouterEval for the missing declared-channel and bandit arms
(all cheap, all absent by grid design).

## 6. Summary: the cheap, unjustified gaps (one CPU campaign each)

1. Declared-channel rivals (CNP, cluster-head, softmax, cascade) on bernoulli (disrouter only), replay (disrouter only),
   RouterEval and LLMRouterBench (all five), and live 10^2 / 10^4 / 10^5.
2. UCB, Thompson, verify-on-claim, LinUCB on RouterEval / LLMRouterBench, and LinUCB on bernoulli / replay.
3. NSW, referral, gossip on bernoulli / replay (≤ 10^6), RouterEval, and live 10^2 / 10^4 / 10^5.
4. TrueSkill at ≤ 10^4 everywhere it is missing.
5. llm_supervisor at live 10^4 / 10^5 (300 supervisor calls per unit).
6. knn_router_online at RouterEval 5,000 (or write down why not).
7. Replay per-shape matrices (reporting only).
Everything else missing is justified by backend (no LLM / no descriptions), by documented cost (MLP one-hot, TrueSkill
≥ 10^5, KNN/MLP on synthetic prompts, 10^5 worlds per shape), by pre-registered scope (n sets, β sets, MMLU), or is an
ablation that belongs in an appendix.

## 7. Fill campaign (launched 2026-09-17 03:00; `logs/fill/launch.sh`)

Every item in §6 except the replay per-shape matrices: `fill_arms` (LinUCB, cascade, referral, gossip, TrueSkill ≤ 10^4) on
both v5 sweeps at every rung and regime (referral / gossip to 10^6); the twelve missing rivals on `routereval_mmlu`,
`routereval_mmlu5k` (+ KNN online) and `llmrouterbench_pool`; `fw_routereval_small` (nine frameworks at m = 10 / 100);
`learned_n100_fill`, `learned_n10k_fill` (+ `llm_supervisor`), `live_n100k_fill` (+ `llm_supervisor`). Chained folds
`fill_fold_synth` and `fill_fold_live`; the bar figures and matrices regenerate after them.

## 8. Expansions (launched 2026-09-17 02:00; `logs/fill/launch2.sh`)

- `fill_arms` on `bernoulli_b_sweep` (b = 1 … 30 at 10^3–10^5; TrueSkill ≤ 10^4).
- Live β = 0.1 at 10^4 (`learned_n10k_beta01`, `learned_n10k_fill_beta01`, `fw_live_n10k_beta01`) and 10^5 (`live_n100k_beta01`,
  `live_n100k_fill_beta01`): every arm the 10^4 / 10^5 tables carry, both liar sets.
- Live heavy_tail and bimodal at 10^4 (`learned_n10k_shapes`, `learned_n10k_fill_shapes`, `fw_live_n10k_shapes`,
  `fw_live_n10k_cartel_shapes`): six new populations (2 shapes × 3 seeds, ~480k generations each) warmed by one
  flat_probe_argmax job apiece (`warm10k_*`), the 950 other units launched by `shapes_gate` behind them.
- The nine frameworks on RouterEval (10 / 100, 1,000, 5,000) with the MiniLM (`_em`) and VA-cohort (`_va`) shortlists.
- Speed: every job of §7 and §8 is one unit (or one seed) and is submitted to `sapphire,serial_requeue`; §7's already-queued
  jobs were widened to both partitions by `logs/fill/spread.sh`. `kempner_requeue` refuses multi-partition submissions.
- Fold `fill_fold_expand`; then matrices, tables, NUMBERS.json, bar figures (M-figures gain the 10^4 shapes and β = 0.1 cells).

## 9. Status of the fill and expansion campaigns (2026-09-17 18:30)

| campaign | state |
|---|---|
| bernoulli fill (LinUCB, cascade, referral, gossip, TrueSkill ≤ 10^4), all 7 rungs × 5 regimes | **complete, folded**, in the matrices and RESULTS II.4e |
| replay fill, all 6 rungs × 5 regimes × 3 shapes | **complete, folded**; per-shape matrices generating |
| b-sweep fill (b = 1 … 30) | **complete** |
| real-pool arms (RouterEval 10/100/1,000 and 5,000, LLMRouterBench) | **complete** |
| live 10^2 fill (12 rivals), 10^4 fill (11), 10^5 fill (10 incl. llm_supervisor) | **complete** |
| live β = 0.1 at 10^4 and 10^5 (arms + frameworks + fill arms) | **complete** |
| RouterEval frameworks, MiniLM and VA-cohort shortlists, m = 1,000 and 5,000 | **complete** |
| RouterEval frameworks on m = 10 / 100 (three shortlists) | * ~60% |
| live 10^4 heavy_tail / bimodal (arms, fill arms, frameworks, cartel frameworks) | * ~40%; the six populations are built |
| live 10^5 peer halving, β = 0.5 | * random liars 2/3 seeds; cartel 0/3 |
| folds `fill_fold_live`, `fill_fold_expand` | waiting on the three * items |

Findings already in the docs from this campaign (RESULTS II.4e): gossip reputation and the referral network collapse to
random under the low-skill cartel (0.415 / 0.431 against random 0.419) while MIDIAN-VA holds 0.789 on the same channel;
LinUCB-honest falls below random from 10^5 up (0.437 → 0.261 at 10^7) with bit-identical numbers in all five regimes,
a pure scale failure rather than a robustness one.
