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
