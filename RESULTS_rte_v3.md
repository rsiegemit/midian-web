# RESULTS_rte_v3.md — external comparisons (DRAFT, 2026-09-03 evening)

Pre-registration: `TARGETS_rte_v3.md` (A, B written 18:10; C 18:20–18:45; all before the runs). Motivation: RESULTS_rte_v2.md
§11, last bullet. Three parts: **A** our arm on RouterBench's own protocol against its own routers (done);
**B** RouteLLM's released routers against our arm on their model pair with their metrics (running);
**C** RouterBench's routers as methods inside our benchmark at n = 100 / 1,000 / 10,000 against every MIDIAN variant
and the top self-contained rivals (running; frameworks pair from the fw grids on identical cells).

## A. RouterBench on its own terms (`scripts/routerbench_terms.py`; figure X1)

Data: RouterBench 0-shot table, 36,093 prompts after dropping eval_names with < 60 prompts (67 families remain), 11
models, per-prompt score in [0, 1] and dollar cost. Protocol: 5 stratified 70/30 splits; every router is fit on the
train split and routes each held-out prompt by maximising predicted performance − λ · price (price = the model's mean
train cost, known ahead), λ swept over 0 and 25 log-spaced values in [0.1, 1000] $/unit; a router's summary is **AIQ**,
the area under its non-decreasing quality-vs-cost envelope over the single-model cost range, normalised to that range
(our reading of RouterBench's AIQ; the same function scores every router). Baselines as in the paper: oracle (cheapest
correct model per prompt), zero router (the hull of the single models), KNN (k = 20) and MLP (128 hidden) predictive
routers over prompt embeddings. One deviation for every router alike: embeddings are a local all-MiniLM-L6-v2 (no API
key anywhere in this project). The cascading router needs a judge and is not run.

Our arm, the **probe-family router**: b probe prompts per family drawn from the train split, every model looked up on
them (estimate = per-(model, family) accuracy), and at test time the prompt's family is *predicted* by 5-NN over the
probe prompts' embeddings — the router never sees eval_name. `_oraclefamily` = the same estimate with the true family
(upper bound). `knn_bXX` = RouterBench's KNN fit on the probe prompts only (equal labels). Plain MIDIAN at n = 11 is a
max-tree over the same scores and picks identically (asserted on 2,000 prompts per split, T3-3).

| router | AIQ | sd (splits) | quality at λ = 0 | labelled outcomes used |
|---|---|---|---|---|
| knn (RouterBench, full train split) | **0.7135** | 0.009 | 0.758 | 277,915 |
| probe_family_b50_oraclefamily | 0.7129 | 0.014 | 0.783 | 36,850 |
| probe_family_b50 | **0.7074** | 0.014 | 0.779 | 36,850 |
| probe_family_b20_oraclefamily | 0.6946 | 0.025 | 0.749 | 14,740 |
| knn_b20 (equal labels) | 0.6922 | 0.018 | 0.751 | 14,740 |
| knn_b50 (equal labels) | 0.6891 | 0.014 | 0.753 | 36,850 |
| probe_family_b20 | 0.6876 | 0.019 | 0.744 | 14,740 |
| probe_family_b10_oraclefamily | 0.6747 | 0.018 | 0.720 | 7,370 |
| mlp (RouterBench, full train split) | 0.6673 | 0.012 | 0.712 | 277,915 |
| probe_family_b10 | 0.6614 | 0.012 | 0.706 | 7,370 |
| zero router (hull of single models) | 0.6613 | 0.002 | 0.784 (gpt-4) | 0 |
| knn_b5 | 0.6576 | 0.024 | 0.745 | 3,685 |
| probe_family_b5 | 0.6354 | 0.020 | 0.689 | 3,685 |
| oracle (cheapest correct) | — | — | 0.915 | — |

**Reading.** (1) An eval-then-route table with 50 probes per family (37k labelled outcomes, 13% of the train split)
matches RouterBench's best router trained on all 278k (0.707 vs 0.713, inside the split sd) and beats its MLP router by
0.04. (2) At 20 probes per family it is 0.026 below KNN — **T3-1 MISS** (pre-registered within 0.02; it is above the zero
router as predicted, and b = 5 is ≥ 0.03 below KNN as predicted). (3) With equal labels the two are the same thing:
knn_b20 0.692 vs probe_b20 0.688, knn_b50 0.689 vs probe_b50 0.707 — a learned router given only probe outcomes is a
noisier family table. (4) The family classifier costs 0.007 AIQ at b = 20 and 0.006 at b = 50 (predicted vs oracle
family) — **T3-2 HIT** (≤ 0.02; and the oracle-family table at b = 20 is 0.027 *above* MLP, not within 0.01 below it).
(5) Every router is far from the oracle (0.915 at λ = 0): the oracle knows the per-prompt outcome, the routers know at
best the family. **T3-3** held (MIDIAN's tree = argmax at n = 11; the tree contributes nothing at this n).

What this says about "why nobody does this": on the benchmark the routing community uses, the cheap eval table is
competitive with the best learned router and needs 7× fewer labels, so the practice is not exotic — RouterBench's KNN
*is* a smoothed eval table. What the family-level table cannot do is what RouterBench's oracle does: exploit
within-family, per-prompt variation. That is the regime learned routers are built for and our benchmark does not
contain (family-homogeneous instances by construction).

## B. RouteLLM's released routers vs the probe-family router (`scripts/rivals_routellm.py`; figure X2) — RUNNING

Pair: strong = gpt-4-1106-preview, weak = mixtral-8x7b-chat, both RouterBench models, so RouteLLM's routers are scored
on RouterBench's per-prompt outcomes with no new generations and with RouteLLM's own metrics (CPT 20/50/80 %, APGR).
Run: `bert` (routellm/bert_gpt4_augmented, their BERTRouter inference reproduced verbatim), random, optimal, the probe-
family router (b = 20, predicted family), knn_full (fully supervised on the pair's train labels), knn_b20.
**NOT RUN**: `causal_llm` (gated meta-llama/Meta-Llama-3-8B; no Hugging Face token on this account), `mf` and
`sw_ranking` (OpenAI embeddings). T3-5 (their harness on MMLU / GSM8K / MT-Bench) pending the harness check.

## C. Their routers on our terms (`knn_router`, `knn_router(online)`, `mlp_router`; grids learned_f1 / learned_n100 / learned_n10k) — RUNNING

Launched 18:03–18:05 (240 + 360 + 198 shards). Measured speeds before launch (2 CPU threads): MiniLM embedding 3.5 ms
per text (48k probe prompts = 2.8 min at n = 1,000; 480k = 28 min at n = 10,000); knn_router fetch 0.3 / 5.4 / 45.5 ms
at n = 100 / 1k / 10k plus 3.5 ms to embed the task, O(n·K·b) per task; mlp_router fit 72 s at n = 1,000 (48k × 1,384,
30 epochs), fetch 2.5 ms plus the embedding. MIDIAN's fetch is 6 ms and MIDIAN-V's 2 ms at n = 1,000, O(r · log n)
(§6b of RESULTS_rte_v2.md). T3-10 is therefore already decided at n ≥ 1,000 pending the success numbers.
