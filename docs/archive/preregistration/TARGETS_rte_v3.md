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
