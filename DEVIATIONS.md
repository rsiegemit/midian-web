# DEVIATIONS.md — where the implementation departs from SPEC.md, and why
- 2026-09-02: cluster is NVIDIA/CUDA (FAS RC), not ROCm; `VLLM_ROCM_USE_AITER=0` is irrelevant here.
- 2026-09-02: declared-channel lie uses `clip(D_honest + 0.4)` (so it "inflates on top" of the self-described channel too);
  for `programmatic` D_honest = S + N(0,0.05) this is the spec's `clip(S + 0.4)` up to the declaration noise.
- 2026-09-02: `skill_excess_ratio` (per-agent-mean variance / binomial floor) is structurally ~0 for `specialist`
  populations (everyone has 3 good families → flat per-agent mean). We report it AND `skill_excess_ratio_family`
  (median per-family ratio); the ≥1.5 gate is applied to the family version, stated beside every result.
- 2026-09-02: route-to-many majority is majority-of-binary-outcomes (optimistic proxy for majority-of-answers).
- 2026-09-02: `replay` backend's RouterBench normalization (`scripts/02_download_routerbench.py`) follows the old
  repo's `_normalize_routerbench` verbatim (category = raw `eval_name`, which already splits every MMLU subject
  into its own eval_name; keep categories with >= 60 rows; binarize a model's score at >= 0.5). At that threshold
  67 categories survive on the actual `withmartian/routerbench` 0-shot pickle, not the 64 named in SPEC.md/task
  prose — matches the old repo's own config comments ("N=500, K=67"), so 64 appears to be stale prose rather than
  a real target. K=64 is honored by taking the 16 (or up to 67) categories with the most prompts when a caller
  passes a smaller K; the full table carries K=67. 11 models present, matching SPEC's "11 models".
- 2026-09-02: `replay` backend's handicap rule (masked/handicapped category, of the two options the task offered):
  the agent's recorded outcome is replaced by the recorded outcome of the model with the LOWEST mean accuracy
  IN THAT CATEGORY, at the same prompt index — a deterministic table lookup, no extra randomness, always a real
  recorded outcome (never a synthesized 0/1 draw). No stochastic 0-with-prob-0.7 fallback: one simple rule, not
  a blend of the two offered.
- 2026-09-02: `replay` backend's per-distribution profile draw (real RouterBench models + a per-category mask;
  models ranked by overall mean accuracy across all categories into a "strong half" / "weak half" and a single
  "strongest" model):
    specialist:  model_id ~ uniform over all 11 models; 3 categories per agent unmasked (random), rest masked.
    heavy_tail:  10% of agents = the single strongest model, fully unmasked; 90% = uniform over the weak half,
                 masked on all but 1 (random) category.
    bimodal:     20% of agents ~ uniform over the strong half, 80% ~ uniform over the weak half; NO masking
                 (bimodal's "good vs bad agents" is realized entirely by model choice, per SPEC's llm-profile
                 draw column, which names no handicap for this distribution).
    correlated:  model_id ~ uniform over all 11 models; categories split into 4 groups (`f % 4`, same convention
                 as `world.sample_skill`); one Bernoulli(0.5) mask draw per (agent, group), applied to every
                 category in that group.
    iid_uniform: model_id ~ uniform over all 11 models; mask ~ Bernoulli(0.5) independently per (agent, category).
  All draws vectorized (no per-agent Python loop) using the `rng` World already seeds via `stable_seed_32`.
- 2026-09-02: `replay` backend's `declared("self_described")` is identical to `declared("programmatic")`
  (`S + N(0,0.05)` clipped): there is no LLM in the replay backend to self-describe, matching `bernoulli.py`'s
  same documented deviation.
- 2026-09-02: centralized rivals (`rte/methods/flat_probe_argmax.py`, `ucb_per_family.py`,
  `thompson_per_family.py`, `sequential_halving.py`, `verify_on_claim.py`, `warm_start_bandit.py`,
  `trueskill_per_family.py`):
  - `verify_on_claim` charges `compare(n)` only the first time a family's declared-order ranking is computed
    (D is static, so the order never changes); every later fetch of that family reuses the cached order and
    charges `compare(1)`, per the alternative SPEC.md explicitly names ("or compare(1) after the first ranking
    per family is cached; document").
  - `warm_start_bandit` clips `D` to `[1e-3, 1-1e-3]` before forming the Beta prior (`alpha = n0*D`,
    `beta = n0*(1-D)`): an exact 0 or 1 declared value gives a zero shape parameter, which `numpy`'s Beta
    sampler rejects. Effect is negligible (n0=5, so the prior mean moves by <=0.005).
  - `trueskill_per_family` raises `NotImplementedError` at `n >= 100_000`, as the SPEC's own note anticipates
    ("at n≥1e5 raise NotImplementedError... document in DEVIATIONS.md"): `trueskill.rate_1vs1` has no vectorized
    form, so the update loop is O(n*K*b/2) pure-Python calls (~3.8s at n=1000, K=16, b=3 measured locally);
    not run at n=1e5+ scale.
  - `sequential_halving` runs entirely at build (per family: rounds of halving, one `probe_many` call per
    round, budget `n*b` per family); `fetch` is an O(1) cached lookup charging `compare(1)`, matching "no
    online update; charge compare(1) at fetch."
- 2026-09-02 (decentralized rivals, SPEC §6 "verified outcomes, decentralized"): a referral network cannot afford
  `d` independent probes per edge at the shared `n*K*b` budget. Implemented faithfully to the budget instead: each agent
  is probed exactly `b` times per family (total `n*K*b`, the full cap and no more) and each single outcome is observed
  and reported by exactly ONE peer — a random one of the agent's `d` graph neighbours (`referral_network`) or a random
  peer (`gossip_reputation_greedy`). Consequence: per-edge, per-family coverage is `b/d` (30% at b=3, d=10; 10% at b=1),
  so most (node, neighbour, family) beliefs are empty. This is the price of decentralization at a fixed probe budget,
  and it is the dominant driver of `referral_network`'s low success — not an implementation shortcut.
- 2026-09-02 `referral_network`: exact d-regularity via a union of `d/2` random permutations (slot `2k` of `i` points at
  `sigma_k(i)`, slot `2k+1` of `sigma_k(i)` points back at `i`), so the relation is symmetric with partner slot `s ^ 1`.
  Odd `d` is rounded up to the next even number; self-loops and parallel edges occur with probability O(d^2/n) and are
  left in place. Beliefs are stored float16 (`n*d*K*2` bytes: 320 MB at n=1e6, d=10, K=16; 1.3 GB at K=64); values are
  means of <= b binary outcomes so half precision is exact enough. Unobserved (node, neighbour, family) triples read as
  0.0, i.e. "no evidence" is ranked equal to "observed and failed".
- 2026-09-02 `gossip_reputation_greedy`: EigenTrust runs with no pre-trusted seed (as SPEC §6 specifies), uniform start,
  <= 50 power iterations; rows of peers who reported nothing are dangling and redistribute their mass uniformly rather
  than making the matrix dense. With `collude=True` this is captured completely by the liar clique (liars report 1 for
  liars), which is the known failure mode of seedless EigenTrust and shows up as misroute_to_liar = 1.00 at beta >= 0.25.
  The T-Man similarity overlay is the standard gossip approximation of the exact O(n^2) c-nearest-in-est graph: start
  from c random peers, then 3 rounds over a candidate set of own neighbours + one random neighbour's neighbours + 2
  fresh random peers, keeping the c most cosine-similar; duplicate neighbours are possible and left in place.
  Build holds all `n*K*b` reports in memory (int32 reporter + int8 value) because the trust weights used for
  `est[a,f]` only exist after the trust vector is computed, and the probe budget forbids a second probing pass.
- 2026-09-02 `flat_nsw_router`: hnswlib exposes no count of nodes visited during a search, so per query we charge
  `hop(ceil(log2 n))` (textbook expected greedy-search depth in an NSW graph) and `compare(ef)` (candidate-list size).
  Both are approximations of the true search cost, not measured quantities. The index uses hnswlib's inner-product
  space over unnormalized `est` rows with a one-hot query, which is maximum-inner-product search rather than a metric
  NN search; measured against the exact `argmax_a est[a,f]` on the same estimates it returned an agent tied at the
  maximum est on 100% of queries at n=1e3 (b=3) and n=1e5 (b=1), so the ANN approximation is not what costs it quality.
  At small `b` the est matrix has huge ties at the maximum (114 agents at n=1e3/b=3, ~30k at n=1e5/b=1), so which tied
  agent is returned is arbitrary and the success number is correspondingly high-variance across seeds.
- 2026-09-02 CPU deps (hnswlib, scipy) installed with `pip install --user` into `~/miniconda3/bin/python` per CONTRACT.md
  via `scripts/03_install_deps.sh`; move to `$RTE_DATA/env/rte` once that env exists.
