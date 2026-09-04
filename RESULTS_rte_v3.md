# RESULTS_rte_v3.md — external comparisons (FINAL, 2026-09-04)

Pre-registration: `TARGETS_rte_v3.md` (A, B written 18:10; C 18:20–18:45; all before the runs). Motivation: RESULTS_rte_v2.md
§11, last bullet. Three parts: **A** our arm on RouterBench's own protocol against its own routers (done);
**B** RouteLLM's released routers against our arm on their model pair with their metrics (done, incl. their own harness);
**C** RouterBench's routers as methods inside our benchmark at n = 100 / 1,000 / 10,000 against every MIDIAN variant
and the top self-contained rivals (n = 100, 1,000 and 10,000 done);
**D** RouterEval (EMNLP 2025; pools of 10 / 100 / 1,000 real LLMs): their protocol with their baselines and the SOTA
routers with released code (Avengers, EmbedLLM), and our whole method suite with liars on their real pools (done);
**E** scale to 10k / 100k agents; **F** LLMRouterBench on its terms and its 20-model pool with liars; **G** AgentsNet /
RouterArena (not applicable / not run). Every part is complete.

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

**n = 10,000 (learned_n10k: specialist, both liar selections, 3 seeds, Q = 300; complete 22:50).** knn_router
0.686 at every β = flat frozen 0.691 (−0.006 [−0.014, +0.003]); knn_router_online 0.731 vs flat_online 0.774 (−0.043:
the nearest-probe store dilutes the routed outcomes more as n grows); midian_va 0.811 / 0.787–0.803 / 0.792–0.810 at
β = 0 / 0.25 / 0.5; midian_v 0.812 / 0.803–0.814 / 0.732–0.757; peer halving 0.863 at β ≤ 0.25 and **0.730** at β = 0.5
low-skill (0.855 random); warm-start 0.74–0.80. Paired: VA − knn **+0.126 / +0.109 / +0.116** [+0.084, +0.144] by β
(+0.124 at β = 0.5 low-skill); VA − knn_online +0.080 / +0.064 / +0.070; VA − V −0.001 / −0.014 / **+0.057**. Costs at
10k: knn 480k probes and 10,000 comparisons per task; VA 496k probes and 43 comparisons / 6 messages. Their MLP router
does not fit at this n (one-hot over 480k rows).

**T3-9**: the VA − knn gap grows with n — +0.074 / +0.070 / +0.126 at β = 0 and +0.070 / +0.066 / +0.124 at β = 0.5
low-skill for n = 100 / 1k / 10k — so the growth clause HITS; the clause "β = 0 gap within ±0.02" MISSES, because a
learned KNN router on the probe budget *is* flat frozen, and flat frozen is 0.07–0.13 below the tree at every n once
ties among b = 3 probes make the argmax a lottery. **T3-10 HIT**: per-task latency 9 / 54 / 49 ms for the learned
routers at n = 100 / 1k / 10k (measured, TARGETS) against MIDIAN-V's 2 ms and MIDIAN's 6 ms.

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

**D1, validation-tuned (rerun 2026-09-04 02:00; every router's hyperparameters chosen on RouterEval's own validation
split — PRKnn k ∈ {5, 20, 50}, cluster K ∈ {3, 16, 64, 128}, ridge α ∈ {0.1, 1, 10}, MLPR width ∈ {256, 1024}, EmbedLLM
(dim, epochs) ∈ {(64, 5), (232, 20)}, probe table b ∈ {3, 10, 30} × K ∈ {3, 16}).** m = 1,000, mean over 12 datasets × 3
pools: LinearR 0.667 (α = 10 in 19/36 runs), EmbedLLM 0.661, MLPR 0.654, cluster table 0.647 (K = 3 in 18, K = 16 in 12),
**probe table 0.630** (16 × 30 in 18, 3 × 30 in 12; 0.26M labels), best single model 0.629, PRKnn 0.627 (k = 50 in 27 of 36),
random 0.463. Their kNN's default k = 5 was the failure mode (0.489); at k = 50 it is a best-single-model router. Verdicts
under tuning: T3-11 HIT (probe table +0.003 vs PRKnn), T3-12 MISS by 0.017 (was 0.042), T3-13 unchanged (MISS: bbh +0.09,
gpqa +0.07, musr +0.11 over the best single model for prompt-level routers), T3-16 MISS (EmbedLLM +0.031 over the probe
table). Per-dataset tuned table: `results/routereval_terms/summary.md`.

### D2. Every arm with liars on RouterEval's real LLM pools (`rte/backends/routereval.py`; grids routereval_mmlu, routereval_mmlu5k; figure X5)

The same score tables inside our World: agents = the pool's real LLMs, families = the 16 largest MMLU subjects (named in
the prompt), probes = index-seeded train prompts (b = 3 per agent per family), tasks = test prompts, reporters and liars
as in v1/v2 (β of the pool lies about observed outcomes; `low_skill_first` picks the weakest as liars; declarations are
the honest noisy control since no self-descriptions exist here). Pools of 10 / 100 / 1,000 (their three pool types × 5
seeds, Q = 1000) and, from their leaderboard table, **all 5,000 MMLU-scored LLMs** as one pool (3 seeds, Q = 300;
pre-registered T3-17 from one smoke run). Their KNN / MLP routers run on their own RoBERTa embeddings (knn_router,
mlp_router); at n = 5,000 knn_router embeds the 5-shot prompts itself (= flat frozen 0.609 at every β; VA − knn +0.097 / +0.101 at β = 0 /
0.5 low-skill) and mlp_router does not fit.

**Success (mean over pools × seeds) by pool size, β and liar selection.** n = 1000, 100, 10 use their pools; n = 5,000
the leaderboard pool:

| arm | n=1000 β=0 | β=.25 rnd | β=.5 low-skill | n=100 β=0 | β=.5 low-skill | n=10 β=0 | β=.5 low-skill | **n=5000 β=0** | β=.25 low-skill | **β=.5 low-skill** |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 0.747 | 0.748 | 0.748 | 0.669 | 0.669 | 0.666 | 0.666 | 0.902 | 0.902 | 0.902 |
| trusted halving | 0.608 | 0.608 | 0.608 | 0.572 | 0.572 | 0.567 | 0.567 | 0.882 | 0.882 | 0.882 |
| peer halving | 0.608 | 0.602 | **0.483** | 0.572 | 0.521 | 0.567 | 0.545 | 0.882 | 0.856 | **0.564** |
| declared_argmax (honest noisy) | 0.710 | 0.539 | 0.470 | 0.639 | 0.536 | 0.661 | 0.562 | 0.864 | 0.608 | 0.614 |
| warm_start_bandit | 0.646 | 0.569 | 0.516 | 0.608 | 0.563 | 0.643 | 0.619 | 0.822 | 0.694 | 0.634 |
| **MIDIAN-VA** | 0.617 | 0.592 | **0.607** | 0.595 | 0.590 | 0.629 | 0.629 | 0.706 | 0.711 | **0.710** |
| MIDIAN-V | 0.617 | 0.613 | 0.525 | 0.596 | 0.562 | 0.631 | 0.582 | 0.706 | 0.691 | 0.542 |
| MIDIAN-A | 0.586 | 0.585 | 0.585 | 0.598 | 0.597 | 0.638 | 0.638 | 0.643 | 0.643 | 0.643 |
| MIDIAN | 0.587 | 0.576 | 0.531 | 0.597 | 0.563 | 0.638 | 0.576 | 0.643 | 0.617 | 0.599 |
| linucb_honest | 0.618 | 0.616 | 0.616 | 0.598 | 0.598 | 0.650 | 0.650 | 0.609 | 0.609 | 0.609 |
| mlp_router (theirs) | 0.620 | 0.620 | 0.620 | 0.627 | 0.627 | 0.649 | 0.649 | — | — | — |
| flat_probe_argmax_online | 0.503 | 0.503 | 0.503 | 0.578 | 0.578 | 0.649 | 0.649 | 0.621 | 0.621 | 0.621 |
| knn_router (theirs) = flat frozen | 0.380 | 0.379 | 0.380 | 0.475 | 0.475 | 0.556 | 0.556 | 0.609 | 0.609 | 0.609 |
| random | 0.505 | 0.505 | 0.505 | 0.538 | 0.538 | 0.541 | 0.541 | 0.550 | 0.550 | 0.550 |

Paired (identical pool × seed units). n = 1000: VA − knn +0.22 at every β; VA − flat_online +0.09 … +0.11; VA − MLP
+0.008 [−0.016, +0.039] at β = 0 (11 pairs); VA − V +0.000 / −0.021 / +0.082 at β = 0 / 0.25 / 0.5 low-skill; VA − peer
halving +0.009 / −0.010 / **+0.124**; VA − MLP −0.003 / −0.028 / −0.007 (β = 0 / 0.25 / 0.5 low-skill). n = 5,000: VA − flat_online **+0.084 / +0.090 / +0.089** (β = 0 / 0.25 low-skill /
0.5 low-skill); VA − V +0.000 / +0.020 / **+0.168**; VA − MIDIAN +0.062 / +0.094 / +0.111; V − MIDIAN +0.062 at β = 0;
VA − peer halving −0.177 / −0.144 / **+0.146**; trusted halving − oracle −0.020 [−0.060, +0.003]. Costs at 5,000: build
240k probes for every probe arm (VA 248k), 2.16M reports for the MIDIAN family and 2.07M for peer halving, per task 42
comparisons / 6 messages (VA), 80 / 12 (MIDIAN), 5,000 / 0 (flat, LinUCB), 1 / 0 (halving).

Reading, in order of what a reviewer would check. (1) **With real LLMs and three probes per family, a flat argmax is a
lottery**: at 1,000 candidates hundreds tie at 3/3 and flat frozen (= their KNN router, 0.38) is below random; at 5,000
flat_online is 0.62. The tree turns the same probes into 0.59–0.62 at 1,000 and 0.64–0.71 at 5,000: cohort elimination
plus online updates on a root cohort of ten. (2) **Under collusion only two things stay up**: MIDIAN-VA (0.61 at 1,000,
0.71 at 5,000, flat across β) and MIDIAN-A; the declaration reader, the warm-start bandit, MIDIAN-V and peer halving all
lose 0.15–0.32 from β = 0 to β = 0.5 low-skill. (3) **In the honest regime MIDIAN is not the best thing at scale**: at
5,000 the trusted halving sits on the oracle (0.88 vs 0.90) and peer halving keeps that until liars collude; the
declaration reader (0.86) and the warm-start bandit (0.82) beat VA (0.71) by 0.11–0.16 when nobody lies. Adaptive
allocation over 5,000 candidates is worth 0.17 over a static three-probe tree, and MIDIAN-VA's claim is robustness at
fixed cost, not honest-regime accuracy. (4) At 10 and 100 real LLMs their MLP router is the best method (0.649 / 0.627):
a learned skill prior beats a 4-valued table when ties are few; at 1,000 it matches VA (0.620 vs 0.617: VA − MLP −0.003
/ −0.028 / −0.019 / −0.007 at β = 0 / 0.25 / 0.5 random / 0.5 low-skill, 15 pairs each) and is immune to liars by
construction; at 5,000 and 10,000 it does not fit (agent one-hot over 240k / 480k probes) and it pays n comparisons per
task where VA pays 42. (5) LinUCB-honest matches VA at 1,000 (0.618) and is immune by construction, at 5,000 comparisons per task; at
5,000 candidates it is flat frozen (0.609).

Verdicts. **T3-15 HIT**: the v2 ordering reproduces on real outcomes — VA and A flat in β, V best at β = 0 (tied with
VA), peer halving best at β ≤ 0.25 and collapsing under low-skill collusion (−0.125 at 1,000, −0.318 at 5,000), the
learned routers = flat and immune. **T3-17**: (i) MISS by 0.01–0.016 (VA − flat_online +0.084 … +0.090 against a
pre-registered +0.10); (ii) HIT (0.706–0.711 across β); (iii) HIT (peer halving −0.318); (iv) MISS — LinUCB is +0.059
*above* random (the smoke run's 0.257 was one seed; over 3 seeds LinUCB reduces to flat frozen, 0.609); (v) HIT (V −
MIDIAN +0.062 at β = 0; V −0.164 at β = 0.5 low-skill); (vi) HIT on the boundary (trusted halving −0.020 from the
oracle).

### D3. GraphRouter through the released LLMRouter library (`scripts/rivals_llmrouter.py`)

GraphRouter (Feng et al., ICLR 2025) run two ways through the authors' library. **Their defaults**: routing JSONL,
query-embedding tensor, LLM json, `GraphRouter(yaml)` + `GraphTrainer.train()` (hidden 64, AdamW 1e-3, 100 epochs,
4 masked samples per step, mask rate 0.3, 20% validation), LLM node features random-initialised (seeded), label =
their one-hot argmax of performance per query. **Tuned, so that no failure mode is reported as the method**: label =
the edge's own performance (the paper's objective; the release's one-hot argmax breaks 0/1 ties by index, which on
RouterBench makes WizardLM-13B the "best model" on 11,271 of 25,265 train prompts), MiniLM embeddings of model
descriptions as LLM node features, and hyperparameters chosen on their validation split from hidden 64/256 × lr
1e-3/3e-4 × mask 0.3/0.6 at 200 epochs. Scoring in both cases is the batched form of their `route_single` (test
queries appended to the training graph with zero edges, `GNNPredictor.predict`) with the same embedder as the graph.
RouterDC fine-tunes a DeBERTa encoder (GPU): NOT RUN.

| setting | GraphRouter, defaults | GraphRouter, tuned | best single model | other routers on the same split |
|---|---|---|---|---|
| RouterBench split 0 / 1 / 2, quality at λ = 0 | 0.7847 / 0.7831 / 0.7854 (2 / 1 / 3 distinct models, $0.00326) | 0.7849 / 0.7828 / 0.7855 (1 / 3 / 1) | gpt-4: 0.7845 ($0.00329) | KNN 0.760, probe b=50 0.781, MLP 0.718, oracle 0.915 |
| RouterEval mmlu m = 100 strong_to_weak | 0.7867 (1 model) | 0.7867 (1 model) | 0.7867 | LinearR 0.770, EmbedLLM 0.777, MLPR 0.778, cluster 0.783, probe-16×30 0.718 |
| RouterEval mmlu m = 1000 strong_to_weak | 0.8685 (1 model) | 0.8685 (1 model) | 0.8685 | (this pool; the D1 mmlu row averages three pool types) |

The sweep is the evidence: on RouterBench split 0 every one of the eight configurations reaches exactly the constant
router's validation value (0.7851 = best single model on val) and picks 1–5 distinct models on test; on the RouterEval
pools every configuration is the constant router (val 0.8325 at m = 100, one distinct pick). GraphRouter, tuned or
not, **is the best-single-model router** on these data. On RouterBench that is a strong baseline (0.784 vs the probe
table's 0.781 at b = 50 and KNN's 0.760), bought by paying GPT-4's price on every prompt; on the RouterEval pools the
best single model is above every learned router at m = 100 (Avengers-16 0.785, cluster 0.783, MLPR 0.778, EmbedLLM
0.777, LinearR 0.770) and far above the probe tables (0.718 / 0.685). This is the LLMRouterBench (2026) finding —
recent routers "fail to reliably outperform a simple baseline" — reproduced with the authors' code and a fair sweep.
**T3-16**: GraphRouter is not within 0.02 of PRKnn (PRKnn is bad here: 0.56 at m = 100); "none beats the probe table by
> 0.02 μ at m = 1000" was already a MISS via EmbedLLM (D1); GraphRouter beats the probe table wherever a single model
dominates, by being that model.


## E. Scale: every variant and rival at n = 10,000 and 100,000 (grid scale_100k; figure X6)

Calibrated bernoulli backend (S drawn to match the measured live populations; no LLM calls, so the frameworks have no
arm here), 3 shapes × β ∈ {0, 0.25, 0.5} × both liar selections × 3 seeds, b = 3, Q = 1,000. Pre-registered T3-18.

| arm | n=10k β=0 | β=.25 low-skill | β=.5 low-skill | **n=100k β=0** | β=.25 low-skill | **β=.5 low-skill** |
|---|---|---|---|---|---|---|
| oracle | 0.847 | 0.847 | 0.847 | 0.849 | 0.849 | 0.849 |
| trusted halving | 0.846 | 0.846 | 0.846 | 0.846 | 0.846 | 0.846 |
| peer halving | 0.847 | 0.846 | **0.654** | 0.846 | 0.845 | **0.682** |
| declared argmax (honest, noisy) | 0.839 | 0.707 | 0.720 | 0.845 | 0.710 | 0.733 |
| warm-start bandit | 0.837 | 0.787 | 0.770 | 0.845 | 0.791 | 0.779 |
| **MIDIAN-VA** | 0.798 | 0.782 | **0.792** | 0.805 | 0.782 | **0.793** |
| MIDIAN-V | 0.798 | 0.799 | 0.684 | 0.805 | 0.790 | 0.651 |
| MIDIAN-SH+A | 0.778 | 0.779 | 0.751 | 0.767 | 0.768 | 0.737 |
| MIDIAN-A | 0.772 | 0.771 | 0.766 | 0.765 | 0.764 | 0.757 |
| MIDIAN | 0.772 | 0.766 | 0.727 | 0.765 | 0.757 | 0.710 |
| MIDIAN-SH | 0.778 | 0.745 | 0.573 | 0.767 | 0.728 | 0.578 |
| flat probe argmax (online) | 0.757 | 0.757 | 0.757 | 0.767 | 0.767 | 0.767 |
| flat probe argmax (frozen) | 0.704 | 0.704 | 0.704 | 0.708 | 0.708 | 0.708 |
| LinUCB-honest | 0.620 | 0.620 | 0.620 | 0.426 | 0.426 | 0.426 |
| random | 0.408 | 0.408 | 0.408 | 0.419 | 0.419 | 0.419 |

Paired (9 pairs per cell): at n = 100k, VA − V +0.000 at β = 0 and **+0.143** [+0.107, +0.179] at β = 0.5 low-skill;
VA − MIDIAN +0.040 / +0.084; VA − flat online +0.038 / +0.027; VA − peer halving −0.041 / **+0.112**. At n = 10k: VA − V
+0.000 / +0.108, VA − peer halving −0.049 / +0.138. Costs at 100k per task: MIDIAN 100 comparisons / 15 messages, V and
VA 51–53 / 7, flat and LinUCB 100,000 / 0, halving 1 / 0; build 4.8M probes for every probe arm (VA 4.96M) and 43.2M
reports for the MIDIAN family (41.8M for peer halving). LinUCB-honest over 1.6M arms with 1,000 tasks is below random.

**T3-18 HIT** on all three clauses: VA − V under low-skill collusion is +0.108 at 10k and +0.143 at 100k (≥ +0.05);
VA moves by 0.025 across β at 100k (≤ 0.03); peer halving loses 0.164 from β = 0 to β = 0.5 low-skill at 100k (≥ 0.10).
What changes with scale: the verification step is worth more (VA / V − MIDIAN +0.026 at 10k → +0.040 at 100k at β = 0,
because among 100k candidates the unverified cohort winners are more often 3/3 lottery tickets), the collusion
collapse of V and of peer halving does not shrink, and MIDIAN-VA is the only arm that is both within 0.05 of the
adaptive-halving ceiling in the honest regime and flat under collusion — at 53 comparisons per task against 100,000.
A live-LLM run at 100k is not launched: 4.8M fresh probe calls per seed (≈ 13 h of the whole fleet each).


## F. LLMRouterBench (Li et al., Findings@ACL'26) on its own terms (`scripts/llmrouterbench_terms.py`)

Their performance-oriented setting from their bench-release: 15 datasets (AIME, MATH500, MATHBench, HumanEval, MBPP,
LiveCodeBench, BBH, KORBench, Knights & Knaves, MMLU-Pro, GPQA, FinQA, MedQA, EmoryNLP, MELD; 11,481 prompts) × the same
20 lightweight (7–9B) models, per-instance scores; their protocol (70/30 split × their five seeds 42 / 999 / 2024 / 2025 /
3407); their metrics (AvgAcc = mean over datasets of the routed accuracy; Gain@R, Gain@B = relative gain over the random
and best-single-model routers; Gap@O = relative gap to the per-instance oracle). Deviation: MiniLM embeddings for every
embedding router (they use gte-qwen2-7B-instruct). Every router's hyperparameters are chosen on a 20% slice of the
train split (no failure-mode defaults). Their own routers are not re-run through their code; the open algorithms are
ours as in D1, and their Table 11 is quoted for the rest.

| router | AvgAcc (ours, MiniLM) | AvgAcc (their Table 11) | Gain@B | Gap@O | labels used |
|---|---|---|---|---|---|
| oracle (per instance) | 0.916 | 0.916 | +0.392 | 0 | |
| dataset table, full labels ("Dataset Oracle") | 0.718 | 0.731 | +0.051 | 0.222 | 161k |
| Avengers top-1 (K = 128 / 64, tuned) | 0.709 | 0.719 (with voting) | +0.039 | 0.232 | 161k |
| Model-SAT | — | 0.719 | | | |
| **probe table, family given, 30 probes / dataset** | **0.704** | — | +0.030 | 0.238 | 9,000 (5.6%) |
| EmbedLLM MF (dim 232, 20 ep, tuned) | 0.702 | 0.712 | +0.026 | 0.240 | 161k |
| GraphRouter | — | 0.703 | | | |
| probe table, family predicted (5-NN), 30 probes | 0.694 | — | +0.015 | 0.249 | 9,000 |
| best single model (Qwen3-8B) | 0.688 | 0.680 | 0 | 0.252 | |
| linear router (ridge, tuned) | 0.687 | — | +0.007 | 0.255 | 161k |
| KNN router (k = 5, tuned) | 0.670 | — | −0.019 | 0.273 | 161k |
| MLP router (1024, tuned) | 0.637 | — | −0.068 | 0.308 | 161k |
| RouterDC | — | 0.613 | | | |
| random | 0.476 | 0.488 | −0.295 | 0.483 | |

The reproduction is faithful where it can be checked: oracle 0.916 vs 0.916, best single model 0.688 vs 0.680, random
0.476 vs 0.488, dataset table 0.718 vs 0.731, Avengers 0.709 vs 0.719, EmbedLLM 0.702 vs 0.712 (their embeddings are a
7B encoder; MiniLM costs 0.01). On this benchmark the dataset id is a strong family signal, so a table with 30 probes
per dataset per model (5.6% of the labels) sits within 0.005 of Avengers / EmbedLLM and +0.016 over the best single
model; nothing gets within 0.2 of the per-instance oracle. **T3-22**: within 0.02 of the best learned router (HIT,
−0.005 vs Avengers) but +0.016 over the best single model against a pre-registered +0.02 (MISS by 0.004). **T3-23
MISS**: Gap@O is 0.22–0.25 for the top routers, just under the pre-registered 0.25 (the per-instance headroom is real but
smaller than on RouterEval's 1,000-model pools). With truthful labels every MIDIAN variant is the probe table (max-tree =
argmax, nothing to audit); the 20-model pool with liars (grid `llmrouterbench_pool`, T3-24) is in F2 below.


**F2 — LLMRouterBench's 20 models with liars (grid `llmrouterbench_pool`; family = dataset, K = 15, b = 3, Q = 1000, 5 seeds).**

| arm | β=0 | β=.25 random | β=.25 low-skill | β=.5 random | β=.5 low-skill |
|---|---|---|---|---|---|
| oracle | 0.725 | 0.725 | 0.725 | 0.725 | 0.725 |
| declared argmax (honest, noisy) | 0.718 | 0.635 | 0.553 | 0.681 | 0.568 |
| their MLP router / flat online | 0.687 | 0.687 | 0.687 | 0.687 | 0.687 |
| LinUCB-honest | 0.679 | 0.679 | 0.679 | 0.679 | 0.679 |
| warm-start bandit | 0.679 | 0.658 | 0.651 | 0.663 | 0.627 |
| **MIDIAN-A** | 0.677 | 0.676 | 0.676 | 0.676 | **0.677** |
| MIDIAN-VA | 0.669 | 0.664 | 0.666 | 0.649 | 0.666 |
| MIDIAN-V | 0.669 | 0.671 | 0.666 | 0.657 | 0.597 |
| MIDIAN | 0.674 | 0.661 | 0.663 | 0.664 | 0.605 |
| peer halving | 0.664 | 0.663 | 0.660 | 0.638 | **0.464** |
| trusted halving | 0.662 | 0.662 | 0.662 | 0.662 | 0.662 |
| their KNN router (online / frozen) | 0.663 / 0.637 | | | | |
| flat frozen | 0.620 | 0.620 | 0.620 | 0.620 | 0.620 |
| random | 0.489 | 0.489 | 0.489 | 0.489 | 0.489 |

At 20 candidates and 15 families with 3 probes each, the online flat scan and their MLP router are the best arms (0.687,
tied), MIDIAN-A the best MIDIAN variant (0.677, flat in β), and VA 0.02 below them (VA − MLP −0.018 [−0.036, −0.003] at
β = 0, −0.021 at β = 0.5 low-skill); V loses 0.07 and peer halving 0.20 under low-skill collusion, the declaration
reader 0.15. The tree has nothing to compress at n = 20 (one cohort of 10 per level) and the audits' cost shows.
**T3-24 HIT** on all three clauses: VA moves 0.020 across β (≤ 0.03); MLP ≥ VA at β = 0 (+0.018); peer halving −0.200
at β = 0.5 low-skill (≥ 0.05). Costs per task: VA 21 comparisons / 4 messages, MIDIAN 40 / 6, flat and MLP 20 / 0.


## G. Benchmarks considered and not run, with reasons

- **AgentsNet** (Grötschla et al. 2025): five distributed-computing tasks — (Δ+1)-colouring, minimal vertex cover, maximal
  matching, leader election, consensus — solved by 4–100 LLM agents exchanging messages on small-world / scale-free /
  Delaunay topologies; binary task success. It contains no notion of selecting or routing to the best agent for a job
  and no skill estimation, so none of our arms or rivals has a counterpart in it. The overlap is thematic only
  (decentralised protocols over many agents); reported as not applicable rather than run.
- **RouterArena** (ICLR 2026): 8,400 queries over 9 domains × 3 difficulty levels; a router submission outputs one model
  name per query from a candidate list the submitter configures (e.g. gpt-4o-mini, claude-3-haiku, gemini-2.0-flash),
  and the platform then runs those models through paid APIs to score the choices. Per-model responses are not
  precomputed, so an offline evaluation is impossible without provider keys; NOT RUN (the probe table would need the
  same paid calls to build). Leaderboard top-3 at fetch time: Paix2 77.63, KT-ModelRouter 76.28, Sqwish 76.21
  (Acc–Cost Arena score).
- **LLMRouterBench** (Findings@ACL'26): run — part F.
- **TwinRouterBench** (2026, agentic routing): not examined tonight.
