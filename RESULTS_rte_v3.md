# RESULTS_rte_v3.md — external comparisons (DRAFT, 2026-09-03 evening)

Pre-registration: `TARGETS_rte_v3.md` (A, B written 18:10; C 18:20–18:45; all before the runs). Motivation: RESULTS_rte_v2.md
§11, last bullet. Three parts: **A** our arm on RouterBench's own protocol against its own routers (done);
**B** RouteLLM's released routers against our arm on their model pair with their metrics (done; their own harness running);
**C** RouterBench's routers as methods inside our benchmark at n = 100 / 1,000 / 10,000 against every MIDIAN variant
and the top self-contained rivals (n = 100 and 1,000 done, n = 10k tail running);
**D** RouterEval (EMNLP 2025; pools of 10 / 100 / 1,000 real LLMs): their protocol with their baselines and the SOTA
routers with released code (Avengers, EmbedLLM), and our whole method suite with liars on their real pools (running).

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

## B. RouteLLM's released routers vs the probe-family router (`scripts/rivals_routellm.py`; figure X2)

Pair: strong = gpt-4-1106-preview, weak = mixtral-8x7b-chat, both RouterBench models, so RouteLLM's routers are scored
on RouterBench's per-prompt outcomes with no new generations, with RouteLLM's own metrics (thresholds at the router-score
quantiles so strong calls span 0–100% in 10% steps; CPT(p) = strong-call % needed to recover p of the weak→strong gap;
APGR = average performance-gap recovered), 5 stratified splits. `bert` = routellm/bert_gpt4_augmented with their
BERTRouter inference reproduced verbatim. **NOT RUN**: `causal_llm` (gated meta-llama/Meta-Llama-3-8B, no token on
this account), `mf` and `sw_ranking` (OpenAI embeddings).

| router | APGR | sd | CPT 20% | CPT 50% | CPT 80% |
|---|---|---|---|---|---|
| optimal (their oracle) | 0.993 | 0.005 | 4.7% | 11.8% | 18.9% |
| knn_full (fully supervised on the pair's train labels) | 0.650 | 0.004 | 9.3% | 27.0% | 63.1% |
| probe_family_b20, oracle family | 0.641 | 0.015 | 12.4% | 29.5% | 60.8% |
| **probe_family_b20** (family predicted) | **0.594** | 0.011 | 13.0% | 33.4% | 70.8% |
| knn_b20 (equal labels) | 0.582 | 0.026 | 12.5% | 37.7% | 72.0% |
| random | 0.507 | 0.004 | 19.1% | 49.0% | 79.7% |
| **bert (RouteLLM, released)** | **0.479** | 0.004 | 19.3% | 55.7% | 81.1% |

**T3-4 HIT** (probe table APGR ≥ bert's, CPT50 ≤ bert's), by a margin nobody pre-registered: RouteLLM's released BERT
router is *below random* on RouterBench's outcomes for its own model pair. That is not a bug in our harness — the
random line reproduces their evaluate.py's random baseline, and the strong/weak accuracies (0.784 / 0.549) are
RouterBench's. It is distribution shift: the router was trained on Chatbot-Arena preference battles and RouterBench's
prompts are benchmark items (MMLU, GSM8K, HellaSwag, …), where "GPT-4 wins" correlates with nothing the classifier
learned. A 20-probes-per-family table built on the target distribution recovers 0.59 of the gap; their own
distribution-matched routers should do better on their own benchmarks, which is what T3-5 (their harness on MMLU /
GSM8K / MT-Bench) measures.

**T3-5 — their harness on their benchmarks.** RouteLLM's `evaluate.py` with their precomputed strong/weak outcomes
(MMLU 14,037 prompts, GSM8K 1,307, MT-Bench 144 turns; contaminated prompts removed by their lists) and their `bert`
router. Their script crashes at its metrics step (`routed_pair` arrives as a str in `generate_results`), so APGR and
CPT are computed from its own per-threshold prints with its own formulas (11 thresholds at 0–100% strong calls):

| benchmark | weak (Mixtral) | strong (GPT-4) | bert APGR | CPT 20 / 50 / 80 % |
|---|---|---|---|---|
| MMLU | 68.1 | 80.6 | 0.536 | 17 / 44 / 77 % |
| GSM8K | 63.7 | 85.8 | 0.531 | 17 / 45 / 79 % |
| MT-Bench (score 0–10) | 8.28 | 9.21 | 0.751 | 10 / 20 / 34 % |

On their own distribution the released BERT router recovers half the gap on MMLU / GSM8K and three quarters on
MT-Bench (the Arena-like distribution it was trained on), against 0.48 on RouterBench's items (above) — the shift, not
the router, is what part B measures. Our probe-family router was **NOT RUN** inside their harness (T3-5's expectation
needs the family table on *their* prompts; adding a router class to their package was not worth a fourth harness run
tonight); on the same three benchmarks it would be the per-subject table on MMLU (57 subjects) and a constant on GSM8K /
MT-Bench, i.e. the pre-registered "degenerates to always-strong or always-weak" case.

## C. Their routers on our terms (`knn_router`, `knn_router(online)`, `mlp_router`; grids learned_f1 / learned_n100 / learned_n10k; figure X3)

RouterBench's KNN and MLP predictive routers as methods in our benchmark, on exactly MIDIAN's probe budget (b = 3 per
agent per family), keeping every probe's prompt text (MiniLM embeddings; their own RoBERTa embeddings on the routereval
backend), scoring every agent on the incoming task's text. Paired on identical (cell, seed) units with every MIDIAN
variant and the top self-contained rivals; frameworks from the fw grids on the same cells. Costs: build = flat's probes
(48,000 at n = 1000 plus 2.8 min of embedding), per task n comparisons and 0 messages (flat's), the embedding arithmetic
outside the ledger.

**n = 1000 (learned_f1: 240 cells × 10 seeds, self-described).** Success by β:

| arm | β=0 | β=0.1 | β=0.25 | β=0.5 | β=0.5 low-skill |
|---|---|---|---|---|---|
| sequential_halving_peer | 0.722 | 0.722 | 0.718 | 0.539 | 0.400 |
| midian_va | 0.684 | 0.671 | 0.667 | 0.680 | 0.679 |
| midian_v | 0.683 | 0.684 | 0.676 | 0.570 | 0.532 |
| midian_a | 0.668 | 0.668 | 0.667 | 0.667 | 0.666 |
| flat_probe_argmax_online | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 |
| midian | 0.668 | 0.664 | 0.648 | 0.598 | 0.569 |
| knn_router_online | 0.654 | 0.654 | 0.654 | 0.654 | 0.654 |
| **mlp_router** | 0.647 | 0.647 | 0.647 | 0.647 | 0.647 |
| flat_probe_argmax_frozen | 0.615 | 0.615 | 0.613 | 0.611 | 0.611 |
| **knn_router** | 0.614 | 0.614 | 0.614 | 0.614 | 0.614 |

Paired (60 pairs per β): knn_router − flat_frozen **−0.002 / −0.001 / +0.000 / +0.003** — the learned KNN router *is*
flat probe argmax (**T3-6 HIT**); knn_router_online − flat_online −0.013 at every β (the online store's nearest-probe
rule under-weights the routed outcomes relative to flat's running mean; T3-6's second half misses by 0.003).
mlp_router − knn_router **+0.034** [+0.020, +0.048] at every β: the agent one-hot lets the MLP pool an agent's
success across families, a learned skill prior that the 4-valued family table lacks (T3-7's "within 0.02" MISS, in the
learned router's favour). Both learned routers are exactly flat in β (no report channel: immune to liars, as
pre-registered). Against MIDIAN-VA: VA − mlp **+0.036 / +0.024 / +0.020 / +0.032** by β (+0.032 [+0.016, +0.047] under
low-skill collusion), VA − knn +0.070 / +0.057 / +0.053 / +0.066; peer halving − mlp +0.075 at β ≤ 0.25 and −0.108 at
β = 0.5 (T3-7's other halves HIT). Per task the learned routers spend 1,000 comparisons and 9 ms (5.4 ms scoring + 3.5 ms
embedding); VA spends 31.6 comparisons and ~2 ms; MIDIAN 60 and 6 ms (**T3-10 HIT** at n ≥ 1000).

**n = 100 (learned_n100: the same 240 cells at n = 100, every arm in one grid).** knn = flat (−0.001 / −0.000 / −0.000);
mlp − knn **+0.060**; VA − mlp +0.014 / +0.001 / +0.011 (+0.010 [−0.002, +0.022] at β = 0.5 low-skill: within floor);
VA − knn +0.074 / +0.062 / +0.071; VA − AutoGen +0.13 (29 pairs); VA − peer halving −0.031 / −0.040 / **+0.165**
(+0.285 low-skill). At n = 100 the MLP router's skill prior closes most of the gap to VA in the honest regime; VA's
edge is the collusion regime and the cost.

**n = 10,000 (learned_n10k: specialist, both liar selections, 3 seeds, Q = 300; knn_router and the β = 0.5 halving
rows still running at write time).** midian_va 0.811 / 0.787–0.803 / 0.792–0.810 at β = 0 / 0.25 / 0.5, midian_v
0.812 / 0.803–0.814 / 0.732–0.757, flat_online 0.774, flat_frozen 0.691, peer halving 0.863 at β ≤ 0.25; VA − V
−0.001 / −0.014 / **+0.057** [+0.032, +0.081]; VA − plain MIDIAN +0.058 at β = 0.5 low-skill. **T3-9** (VA − knn gap
grows with n) pending the knn rows; the VA − V collusion gap is +0.058 / +0.074 / +0.057 at n = 100 / 1k / 10k
(learned_n100, replication, learned_n10k).



## D. RouterEval — the benchmark at MIDIAN's scale (pools of 10 / 100 / 1,000 real LLMs; `scripts/routereval_terms.py`, `rte/backends/routereval.py`)

### D1. On its own terms (figure X4; `results/routereval_terms/summary.md`)

Their hard setting: 12 datasets × three pool types (all_strong / all_weak / strong_to_weak) × m ∈ {10, 100, 1000}
REAL LLMs drawn from 8,500 leaderboard models, binary score of every candidate on every prompt, 8:1:1 split, their
RoBERTa prompt embeddings, their metrics μ (mean test score of the routed model) and V_B = μ / best single model,
averaged over the three pools as their harness does. Their baselines re-implemented from their router/ scripts
(PRKnn k = 5, C-RoBERTa-cluster K = 3, LinearR, MLPR — sklearn batch 32 instead of batch 1), plus the SOTA routers
named by LLMRouterBench (2026) that have open algorithms: **Avengers top-1** (AAAI 2026 oral: KMeans K = 64 on query
embeddings, per-cluster accuracy ranking on all train labels, top model; their voting needs generations) and
**EmbedLLM** (ICLR 2025: matrix-factorised model embeddings × projected query embedding, BCE; dim 64 / 5 epochs on CPU
instead of 232 / 50). Our arm: the probe-family table with unsupervised KMeans families (K = 3, 16) and, on mmlu, the
subject named in the prompt. With truthful labels every MIDIAN variant reduces to this table (max-tree = argmax,
asserted on every pool; nothing to audit; V's re-probing at n ≤ 1000 with r = 10 is one extra probe per promoted cell).

| router, m = 1000 | μ | V_B | labelled outcomes |
|---|---|---|---|
| oracle | 0.986 | 1.63 | |
| LinearR (theirs) | **0.661** | 0.99 | 3.35M |
| EmbedLLM MF (ICLR 2025) | 0.658 | 0.96 | 3.35M |
| MLPR (theirs) | 0.645 | 0.95 | 3.35M |
| C-RoBERTa-cluster (theirs, K = 3) | 0.643 | 0.94 | 3.35M |
| Avengers top-1 K = 16 (AAAI 2026) | 0.638 | 0.94 | 3.35M |
| best single model | 0.629 | 0.90 | 3.35M |
| **probe table, 16 clusters × 30 probes** | **0.615** | 0.90 | 0.43M (13%) |
| probe table, 3 clusters × 30 probes | 0.601 | 0.88 | 0.09M |
| Avengers top-1 K = 64 | 0.597 | 0.88 | 3.35M |
| probe table, 16 clusters × 10 probes | 0.532 | 0.78 | 0.15M |
| PRKnn (theirs, k = 5) | 0.489 | 0.73 | 3.35M |
| random | 0.461 | 0.68 | |

(m = 100: LinearR 0.605, MLPR 0.589, cluster 0.589, Avengers-16 0.584, EmbedLLM 0.578, probe-16×30 0.569, best single
0.559, PRKnn 0.531; m = 10: LinearR 0.586, cluster 0.574, MLPR 0.573, probe-16×10 0.571, best single 0.553, PRKnn 0.552.)

Per dataset at m = 1000 (μ): on **mmlu** LinearR 0.748, Avengers-64 0.745, EmbedLLM 0.740, MLPR 0.734, cluster 0.731,
best single 0.724, probe-subject×30 0.687, probe-16×30 0.683, PRKnn 0.500. On the single-task datasets their
prompt-level routers beat the best single model by a lot — bbh 0.720 vs 0.636, musr 0.522 vs 0.408, gpqa 0.497 vs
0.356, ifeval 0.709 vs 0.667 — and the cluster table by less (bbh 0.659, musr 0.496).

Verdicts. **T3-11 HIT** on its letter and wrong in spirit: probe-16×10 is +0.043 *above* PRKnn (0.532 vs 0.489) and
probe-16×30 +0.125, because their kNN router is below random-plus-a-bit at 1,000 candidates (k = 5 neighbours over
binary labels of 1,000 models is a lottery among ties); against the real leaders the probe table is −0.047 (LinearR)
and −0.043 (EmbedLLM) with 13% of the labels, and −0.014 below the best single model. **T3-12 MISS**: probe-3×30 is
−0.042 below C-RoBERTa-cluster on the same partition — the label budget (90k vs 3.35M) is the whole difference.
**T3-13 MISS on both halves**: on mmlu the routers beat the best single model by only +0.024, and on the single-task
datasets they beat it by up to +0.14 (gpqa), i.e. prompt-level routers exploit within-task variation the family table
cannot see. **T3-14 HIT** (tree = argmax). **T3-16 (partial)**: EmbedLLM beats the probe table by +0.043 (> 0.02:
MISS for "none beats it"); Avengers top-1 at their default K = 64 is *below* the probe table at m = 1000 (0.597: 64
clusters × 1,000 candidates over-partitions the 11k train prompts) and above it at K = 16; GraphRouter / RouterDC (D3)
not yet run.

What this says. At 1,000 real LLMs with all labels, a linear probe of the prompt embedding is the strongest published
router and it is +0.03 over the best single model; the strongest cheap thing is a 16-cluster table at 13% of the
labels, 0.015 under the best single model. Every method is 0.3 below the per-prompt oracle. The "model-level scaling"
headroom RouterEval advertises is per-prompt, and nothing family-level or label-cheap reaches it. MIDIAN's own
mechanisms do not act here (no liars, no cost accounting); they act in D2.

