# TARGETS_rte_v3.md — pre-registered expectations for the external comparisons (written 2026-09-03 18:10, before any run)

Motivation (RESULTS_rte_v2.md §11, last bullet): the v1/v2 benchmark is built around a cheap, checkable probe, so
"probe then route" wins by construction. Two follow-ups answer the reviewer's question "why does nobody do this":

**A. Rival benchmark on its own terms — RouterBench (Hu et al. 2024).** Their data (36,497 prompts, 11 models,
per-prompt correctness and dollar cost, 67 eval_names with ≥ 60 prompts), their protocol (routers fit on a train split,
evaluated per prompt on a held-out split, cost–quality curve over a willingness-to-pay λ, summary = AIQ, the area under
the quality-vs-cost curve normalised by the cost range), their baselines (oracle = cheapest correct model per prompt;
zero router = convex hull of the single models; KNN and MLP predictive routers over prompt embeddings; a cascading
router). One deviation, applied to every router equally: prompt embeddings are a local sentence-transformer
(all-MiniLM-L6-v2), not OpenAI's, because no API key is used anywhere in this project. Our arm is the **probe-family
router**: b probe prompts per eval_name from the train split, every model run on them (outcomes looked up), estimate =
per-(model, family) accuracy; at test time the prompt's family is *predicted* (k-NN over the probe prompts' embeddings —
the router never sees eval_name), and the pick maximises estimate − λ·cost. Plain MIDIAN on 11 agents is a one-level
tree: its picks are identical to the probe-family router's (asserted in the script), so MIDIAN's tree contributes
nothing at this n and the comparison is honestly "eval-then-route vs learned routers". 5 stratified 70/30 splits.

- **T3-1.** Probe-family router at b = 20 probes per family (1,340 train prompts, 3.7% of the data) reaches AIQ within
  0.02 of the KNN router and above the zero-router hull. At b = 5 it is below KNN by ≥ 0.03.
- **T3-2.** With the true family given (oracle-family upper bound) the probe router at b = 20 is within 0.01 of the
  MLP router; the family classifier costs ≤ 0.02 AIQ (predicted-family minus oracle-family).
- **T3-3.** MIDIAN(n = 11, r = 10) picks == probe-family router picks on every test prompt (a check, not a result).

**B. Actual rivals — RouteLLM (Ong et al. 2024, lm-sys), released routers.** Their setting: route each prompt to a
strong or a weak model under a threshold; strong = gpt-4-1106-preview, weak = mixtral-8x7b-chat — both are RouterBench
models, so their routers can be scored on RouterBench's per-prompt outcomes with no new generations. Routers that run
without an API key: `bert` (BERT classifier, CPU) and `causal_llm` (LLM classifier, one GPU job). `mf` and
`sw_ranking` need OpenAI embeddings and are NOT run (stated as a miss of coverage, not silently dropped). Their
metrics: the quality-vs-%-strong-calls curve, CPT(50%) / CPT(80%) = the fraction of strong calls needed to reach 50% /
80% of the strong–weak performance gap, and APGR = average performance-gap recovered. Our arm: the probe-family router
restricted to the same two models (pick strong iff est_strong − est_weak > τ, τ swept), b = 20.

- **T3-4.** On RouterBench prompts (their pair, their metrics), the probe-family router's APGR ≥ `bert`'s and CPT(50%)
  ≤ `bert`'s; against `causal_llm` reported as measured (expected to lose on APGR by ≤ 0.05).
- **T3-5.** On RouteLLM's own benchmarks through their harness (MMLU, GSM8K, MT-Bench; their precomputed strong/weak
  outcomes): probe-family router APGR ≥ `bert` on MMLU (subjects are families) and < `bert` on MT-Bench (80 prompts, no
  family structure — the probe router degenerates to "always strong or always weak"). If the harness needs an API key
  for any benchmark, that benchmark is reported as NOT RUN.

Rules as before: every number paired on identical prompts/splits; misses reported as misses; no OpenAI/Anthropic keys;
all data and envs under $RTE_DATA; plain MIDIAN's parameters unchanged.

**C. Their routers on our terms (added 2026-09-03 18:20, before launch).** RouterBench's predictive routers as methods in
our benchmark (`knn_router`, `knn_router(online=True)`, `mlp_router`; grid `learned_f1` = the variants_f1 cells, n = 1000,
self-described, 3 shapes × 4 β × 2 liar selections × 10 seeds), on exactly MIDIAN's probe budget (b = 3 per agent per
family) and no report channel. They keep every probe's prompt text, embed it (MiniLM), and score each agent on the
incoming task's text: KNN = mean outcome of the agent's k = b nearest probes, MLP = one regressor on (prompt ⊕ agent).
RouteLLM's routers are two-way cost routers (strong vs weak) and cannot select among n agents; they are not run here
and this is stated as such. Paired against midian, midian_va, flat_probe_argmax_online and sequential_halving_peer on
identical cells.
- **T3-6.** knn_router = flat_probe_argmax within 0.01 at every β and flat in β (a learned router with no report channel
  is immune to liars, and instances are family-homogeneous by construction, so the prompt carries no signal beyond the
  family). knn_router(online) = flat_online within 0.01.
- **T3-7.** mlp_router within 0.02 of knn_router; both ≥ 0.03 below sequential_halving_peer at β ≤ 0.25 and ≥ 0.02 below
  MIDIAN-VA at β = 0.5 (the audits' edge is over the report channel the learned routers do not have, so at β = 0.5 the
  learned routers should sit at flat's level, above plain MIDIAN and below VA).
- **T3-8.** Per-task cost n comparisons, 0 messages (flat's), build = flat's probes; the embedding arithmetic is not in
  the ledger (stated).
- **Scale (added 18:45, before launch).** Grids `learned_n100` (variants_f1 cells at n = 100; every MIDIAN variant, flat
  frozen/online, halving trusted/peer, warm-start bandit, declared argmax, the three learned routers; frameworks pair
  from fw_live_n100) and `learned_n10k` (specialist, β ∈ {0, 0.25, 0.5}, both liar selections, seeds 1–3, Q = 300;
  same arms minus mlp_router, whose agent one-hot over 480k probes does not fit — stated as a coverage limit).
  **T3-9.** The learned routers track flat within 0.01 at every n; the MIDIAN-VA − knn_router gap at β = 0.5 low-skill
  grows with n (≥ +0.02 at n = 100, ≥ +0.05 at n = 10k), the β = 0 gap stays within ±0.02 at every n.
- **Speeds, measured 2026-09-03 on 2 CPU threads before launch, to be reported next to §6b:** MiniLM embedding
  3.5 ms/text (48k probe texts = 2.8 min build at n = 1000, 480k = 28 min at n = 10k); knn_router fetch 0.3 / 5.4 /
  45.5 ms at n = 100 / 1k / 10k plus 3.5 ms to embed the task (O(n·K·b·d) per task; MIDIAN's fetch is 6 ms at n = 1000
  and O(r·log n)); mlp_router fit 72 s at n = 1000 (48k × 1,384, 30 epochs), fetch 2.5 ms + embedding. Expectation
  **T3-10**: per-task latency of the learned routers exceeds MIDIAN-V's (2 ms) at every n and grows linearly in n.

**D. SOTA routing benchmarks at MIDIAN's scale (added 2026-09-03 19:00, before any run).** Research pass (2025–2026):
RouterEval (Huang et al., EMNLP 2025 Findings; 8,500 LLMs, 12 datasets, candidate pools of 10 / 100 / 1,000 real
LLMs — the only benchmark whose pools reach our n), LLMRouterBench (Li et al. 2026; 33 models, 21 datasets, 10 routers:
RouterDC, EmbedLLM, MODEL-SAT, Avengers, HybridLLM, FrugalGPT, RouteLLM, GraphRouter, Avengers-Pro, OpenRouter; finding:
"many routing methods exhibit similar performance … several recent approaches, including commercial routers, fail to
reliably outperform a simple baseline"), the LLMRouter library (ulab-uiuc; 16 routers incl. GraphRouter, RouterDC, MF,
Elo, HybridLLM, AutoMix), EmbedLLM (ICLR 2025; 112 models), RouterArena (ICLR 2026), and the June-2026 "market for
lemons" trust-layer paper (conceptual; the only work framing lying capability advertisements, no learnable rival).
- **D1 — RouterEval on its own terms** (`scripts/routereval_terms.py`): their hard setting, all 12 datasets × m ∈ {10,
  100, 1000} × 3 pool configs, their metrics (μ, V_B, V_R), their baselines re-implemented from their router/ scripts
  (PRKnn k = 5, C-RoBERTa-cluster K = 3, LinearR, MLPR — sklearn batch 32 instead of batch 1, stated —, random, oracle).
  Our arm: probe-family router with unsupervised KMeans families (K = 3, 16) and, on mmlu, the subject in the prompt.
  - **T3-11.** At m = 1000, probe_cluster16_b10 (10 probes × 16 families × m labels = 1.4% of their train labels) is
    within 0.02 μ of PRKnn averaged over the 12 datasets, and probe_cluster16_b30 within 0.01.
  - **T3-12.** probe_cluster3_b30 ≥ C-RoBERTa-cluster − 0.01 at every m (same partition; the label budget is the only
    difference).
  - **T3-13.** Model-level scaling: at m = 1000 PRKnn and the probe routers beat the best single model by ≥ 0.02 on
    mmlu (subjects) and by < 0.02 on the single-task datasets (arc, gsm8k, hellaswag, winogrande, ifeval, truthfulqa).
  - **T3-14.** MIDIAN's max-tree == argmax on every pool (check).
- **D2 — MIDIAN and every rival with liars on RouterEval's real 1,000-LLM pools** (`rte/backends/routereval.py`, grid
  `routereval_mmlu`): our World over their score matrix (agents = the pool's real LLMs, families = MMLU subjects,
  probes = index-seeded train prompts, tasks = test prompts), reporters and liars as in v1/v2. Expectation **T3-15**:
  the v2 ordering reproduces on real LLM outcomes — MIDIAN-A / VA flat in β, MIDIAN-V best at β = 0, peer halving
  best at β ≤ 0.25 and collapsing under low-skill-first collusion, learned routers (knn_router) = flat and immune.
- **D3 — LLMRouter's SOTA routers** (GraphRouter, RouterDC, MF, Elo, HybridLLM, KNN, MLP, SVM) via the released
  library, trained on RouterBench (part A protocol) and RouterEval pools; reported on their terms next to our arms.
  Expectation **T3-16** (from LLMRouterBench's own finding): none beats the probe-family table by more than 0.02 μ
  at m = 1000; RouterDC / GraphRouter are within 0.02 of PRKnn.
- **D2 at n = 5,000 (added 2026-09-03 20:50, before launch; grid `routereval_mmlu5k`).** RouterEval's leaderboard_old
  score table (5,000 real LLMs × 57 MMLU subjects, binary, complete) as one pool: the 16 largest subjects, a fixed 80/20
  split per subject, b = 3, Q = 300, β ∈ {0, 0.25, 0.5}, both liar selections, 3 seeds; every arm of routereval_mmlu
  except mlp_router (one-hot over 240k probes) — knn_router embeds the 5-shot prompts itself (no embeddings shipped).
  Smoke run (seed 1, β = 0.25 low-skill): flat 0.607, MIDIAN 0.667, MIDIAN-VA 0.813, trusted halving 0.893, LinUCB 0.257,
  oracle 0.897 — written down before the grid so the pre-registration is not fitted to it:
  **T3-17.** (i) MIDIAN-VA ≥ flat_online + 0.10 at every β (ties among 5,000 3/3 scorers make any argmax a lottery; the
  tree's cohort elimination + online root updates are the difference); (ii) VA within 0.03 of itself across β (flat in β);
  (iii) peer halving loses ≥ 0.10 from β = 0 to β = 0.5 low-skill; (iv) LinUCB-honest < random (5,000 × 16 arms, 300
  tasks); (v) MIDIAN-V − MIDIAN ≥ +0.05 at β = 0 (verified root pick over 5,000 candidates), and V collapses ≥ 0.10 at
  β = 0.5 low-skill; (vi) trusted halving within 0.02 of the oracle.

**E. MIDIAN-VA everywhere, scale to 10k / 100k (added 2026-09-04 02:40, before launch).** VA (and A where absent) added
to live_f1_n1000 (+ its replay/bernoulli mirrors), fw_live_n1000 (+ n100 / lowskill mirrors), churn_n1000, budget_sweep,
budget_b10_shapes, live_n10k_v2, midian_r20; frameworks get a VA shortlist arm (`retrieval: midian_va`, r = 10, n = 1000);
grid `scale_100k` = every MIDIAN variant + the self-contained rivals at n = 10,000 and 100,000 on the calibrated bernoulli
backend (3 shapes × 3 β × 2 liar selections × 3 seeds; no LLM calls, frameworks not applicable). Live-LLM 100k is
estimated, not launched: 4.8M fresh probe calls per (population, seed) ≈ 13 h of the whole fleet each.
- **T3-18.** VA's collusion gap over V (+0.06–0.07 at n ≤ 10k) persists at 100k (≥ +0.05 at β = 0.5 low-skill) and VA
  stays flat in β within 0.03 at 100k; peer halving's β = 0.5 low-skill collapse persists at 100k (≥ 0.10).
- **T3-19.** VA-shortlisted frameworks are within ±0.02 of V-shortlisted ones at β ≤ 0.25 and ≥ +0.03 above them at
  β = 0.5 (the cohort is chosen with audited reports).
- **T3-20.** Under churn VA − V ≥ +0.03 at 30% churn (V's verified cache goes stale faster than the audited estimates).
- **T3-21.** In the budget sweep VA ≥ V at every b (audits cost 5% of the budget, returned under β = 0.25 liars).
