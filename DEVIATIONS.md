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
- 2026-09-02 `disrouter_cascade`: when nobody in the cost-ordered cascade takes the task (only possible near tau=1,
  or if every declarer under-reports for a family), SPEC/task offered two options ("return the last agent (or the
  highest D[a,f]; document"). We return the highest declarer, not the literal last-in-cost-order agent: returning
  the cheapest-declared (lowest-skill) agent whenever the whole population under-declares for a family would be a
  silent quality cliff with no corresponding "somebody actually claimed capability" signal. `hops`/`messages` still
  charge as if the task forwarded through the full order (n-1 forwards) before the fallback triggers.
- 2026-09-02 `cluster_head_router`: naive k-means with k=ceil(n/r) centroids (r=10) is O(n*k) per assignment sweep —
  at n=1e6 that is k=1e5 centroids, ~1e11 scored (point, centroid) pairs per sweep. Measured at n=1e5, k=1e4: ~9s per
  sweep single-bucket, i.e. an extrapolated ~15 min/sweep at n=1e6 (~100x the flops), infeasible for `build`. Instead
  `build` first splits agents into random buckets of `bucket=20_000` (plain `view.rng.permutation`, no distance
  computation, so it doesn't bias which agents can cluster together beyond bucket membership) and runs the same
  vectorized/chunked k-means independently within each bucket, targeting ceil(bucket_size/r) local clusters per
  bucket. This bounds assignment cost to O(n * bucket / r) instead of O(n^2 / r), while every cluster still has size
  ~r. `fetch`'s two-level lookup (compare over all k heads, then compare within the winning cluster's members) is
  unchanged and still operates over the full global set of ~n/r clusters — only the *build*-time clustering search
  space is bucketed. Consequence: cluster membership can only combine agents that landed in the same random bucket,
  slightly less globally-optimal than exact flat k-means, but on i.i.d.-ish declared-skill rows (no adjacency
  structure to exploit) this costs essentially nothing in practice.
- 2026-09-02: report-channel lie "top-20% honest by j's observed outcomes" has two faithful readings and both exist: the scalar
  `report()` decides ONLINE from what j has observed so far (sequential), the vectorized `report_many()` decides on the batch's
  complete per-agent means. They agree on most reports (28/31 in the checker) and differ only where the ranking changes within a
  batch. Every method at scale uses `report_many`; the scalar path exists for the spec's `report_channel` interface and tests.
- 2026-09-02: `_est.probe_successes` accumulates in float64 (counts are exact either way) so exact-estimate correctness mocks work.
- 2026-09-02: report-channel lie strength grows with n at fixed budget: with one observer per outcome and b=1, a peer sees only
  K·b outcomes, so "zero the top 20% honest agents I saw" degenerates to zeroing nearly every honest agent it saw (ceil(0.2·1)=1 of 1).
  Inherent to the rule, not a batching artifact; stated beside large-n results rather than tuned away. (Found by the decentral-rivals agent.)
- 2026-09-02 (midian): SPEC §3 says "pad n to r^L and reshape". We pad *each level* to a multiple of r instead
  (leaf cohorts = ceil(n/r), then ceil(N/r) per level). Identical tree when n = r^L (the 1e7 point), identical
  depth in general (iterating ceil(N/r) from n reaches 1 in exactly ceil(log_r n) steps), and it avoids the up-to-r x
  memory waste of a full r^L level-0 array at, e.g., n=1234, r=10 (1000 cohorts padded vs 124 real).
- 2026-09-02 (midian): when r does not divide n exactly one leaf cohort is short (padding is contiguous after the
  permutation). Probes are exactly n*K*b always; reports are ((n-m)(r-1) + m(m-1))*K*b with m = n mod r, i.e. exactly
  the SPEC §5(iii) n*K*b*(r-1) only when r | n. A short cohort has fewer peers, so it cannot produce r-1 reports per
  outcome without probing someone twice. The trim is likewise floor(delta*(s-1)) for a cohort of size s, clamped so
  at least one report survives; a cohort of size 1 has no peers and falls back to its own probe mean.
- 2026-09-02 (midian): reports for a whole cohort chunk are sent in ONE view.report_many call covering all K families,
  which is what makes the liar's "top-20% honest by j's observed outcomes" rule pool across families - matching the
  scalar view.report_channel path, whose _obs accumulator also pools across families.
- 2026-09-02 (midian): `stratify=True` needs the declared channel, so the instance widens `needs` to include
  "declared" in __init__; the class attribute stays {"probe","reports"} (the default stratify=False).
- 2026-09-02 (midian): observe()'s running mean is seeded with the number of reports behind the build estimate
  ((r-1)*b - 2*trim) so one online outcome cannot overwrite the probe evidence; per-(agent,family) counts are kept
  in a dict (only routed pairs), not an n x K array, which would be a second 2.6 GB at n=1e7, K=64.
- 2026-09-02 (midian): cohort leaders (SPEC §5) are not materialized. A node's summary is carried by the node's own
  arrays, so nothing reads the leader id; it survives only in the message accounting (member->leader, leader->parent).
- 2026-09-02 (midian): message accounting per the 2026-09-02 CONTRACT section. Build charges
  (n - ceil(n/r)) member->leader messages plus one leader->parent message for every node but the root
  (= sum_l N_l - 1, N_l = nodes at level l); fetch charges 2 per level = 2*depth. Verified in tests and by
  scripts/check_methods.py (n=100 -> 100 build messages, n=1000 -> 1010).
- 2026-09-02 (midian): the probe/report/trim stage lives in `rte/methods/_est.py::peer_reported_estimates`, beside the
  `probe_successes` helper the centralized rivals share. It is a separate function because MIDIAN needs the individual
  b outcomes to feed the report channel, not their sum; the CONTRACT's suggested name `probe_estimates` is what
  `probe_successes(view, b) / b` already is, and renaming it would churn seven files owned by another agent.
- 2026-09-02 (midian_llm_descent): the LLM sees only the r children's summary numbers for the task's family and
  answers with an index; a parse failure, an out-of-range index, or an empty padding slot falls back to the arithmetic
  argmax and is counted in `stats`. Ledger charges are identical to plain MIDIAN, so the ablation is quality-only.
- 2026-09-02 (llm backend): the ladder's non-Qwen half is `google/gemma-2-{2b,9b}-it`, not
  Llama-3.2 (SPEC §1 offers either). Both gemma repos are `gated=manual` on the Hub but this
  account's token has access, so all 7 models downloaded; no Qwen-only fallback was needed.
- 2026-09-02 (llm backend): `reasoning-gym` 0.1.19 ships `propositional_logic` and `graph_color`
  with `answer=None` -- their own gold answer scores 0.0, so no agent can ever be right on them.
  Both are excluded; `syllogism` takes propositional_logic's slot in the K=16 list and
  `circuit_logic` and `self_reference` fill the K=64 list. `scripts/probe_families.py`
  re-verifies every family (gold scores 1, junk scores 0, instances deterministic).
- 2026-09-02 (llm backend): SPEC §1's handicap "difficulty capped" is implemented as a cap on the
  agent's *generation budget* (160 vs 512 max_tokens), not on the generator's difficulty
  parameters. Capping generator difficulty would give the handicapped agent a DIFFERENT (easier)
  instance, which breaks both the paired task stream and the shared verifier -- every agent must
  see the identical instance for `(family, instance)` to mean one thing. The other two handicaps
  (exemplar withheld, family tool removed) are implemented literally.
- 2026-09-02 (llm backend): the response memo is keyed on an agent's *prompt signature*
  `(model, handicapped_on_f, tool_on_f, max_tokens)` rather than on the agent id. Agents sharing a
  signature emit a byte-identical prompt and, at temperature 0, the identical answer, so this is
  the same cache SPEC §1 asks for with a wider (and provably safe) sharing rule. It makes the
  unique-generation count for `true_skill()` scale with the number of signatures (<= 42), not n.
- 2026-09-02 (llm backend): the env at `$RTE_DATA/env/rte` is a `venv` off the Miniforge 3.12
  interpreter, not `conda create`. Concurrent env builds on this cluster deadlock conda's shared
  repodata lock (`BlockingIOError: [Errno 11]`); a venv off the same CPython 3.12 needs no solver.
  `scripts/00_build_env.sh` still takes `RTE_USE_CONDA=1` for the conda path.
- 2026-09-02 (llm backend): the `python` tool is containment, not a security boundary -- a
  throwaway `python -I` subprocess with a 5 s timeout, CPU/address-space/file-size rlimits, an
  empty environment and a socket-blocking preamble. It stops runaway loops and accidental network
  use; it is not hardened against a deliberate escape.
- 2026-09-02 (runner): `rows.csv` is *materialised* from one JSON file per row under
  `results/<grid>/rows.d/` (written temp + `os.replace`) rather than appended in place. SPEC says
  "appended atomically"; a real append cannot be made atomic across forked workers on NFS, and a
  per-row file gives the same guarantee plus O(1) resume (the row's file either exists or does not).
  `rte.run` and `rte.analyze` both rebuild `rows.csv` from that directory.
- 2026-09-02 (runner): SPEC's `wall_clock_per_task` is split in two. `wall_clock_per_task` times
  only `fetch` + `observe` -- the routing cost F3 is about; `wall_clock_per_task_total` also
  includes `world.execute`, which on the llm backend is a real generation and would swamp it.
- 2026-09-02 (runner): for a route-to-many `fetch`, the *scored* outcome is the majority (ties -> 0)
  but `observe` is called once per routed agent with that agent's own outcome, not the majority.
- 2026-09-02 (runner): `success_late` uses the last `min(500, max(1, Q//4))` tasks, so it stays
  meaningful on short smoke streams.
- 2026-09-02 (runner): the oracle line is executed once per (cell, seed) and reused as the regret
  baseline for every method in that cell -- on the llm backend those are real generations.
- 2026-09-02 (runner): `bernoulli_scale` uses b=1 at n >= 1e6 (n*K*b at 1e7 x 64 x 3 is 1.9e9
  probe draws). `backend_kwargs.calibrate_from` falls back to sampling `dist` directly with a loud
  warning when the measured-S file is absent; those points must be labelled uncalibrated.
- 2026-09-02 (runner): the sbatch scripts default to `--account=sompolinsky_lab`. On this cluster
  `kempner_sompolinsky_lab` is the Kempner GPU account; the general CPU partitions used here
  (`sapphire`, `shared`, `bigmem`) run under `sompolinsky_lab`.
- 2026-09-02 (tests): `tests/test_each_method.py` discovers one level into method subpackages
  (`rte/methods/frameworks/`), and skips classes flagged `requires_llm=True` or `runner_only=True`,
  plus anything whose optional dependency is missing at import/construction/build.
- 2026-09-02 (tests): the needs-mutation test reports a method that survives having one of its
  declared needs removed as an **xfail naming the method** (an over-declared need), not a failure.
- 2026-09-02 (mirror grids): on the bernoulli backend `declared_source=programmatic` and
  `self_described` are the same channel (there is no LLM to self-describe), so those mirror cells
  are exact duplicates. They are kept as a pipeline consistency check, not as two conditions.
- 2026-09-02 message accounting for the decentralized rivals, under the lead's CONTRACT directive
  ("referral/gossip = 2 per neighbour consulted per hop"). `referral_network` build charges n*d messages: the graph has
  n*d/2 undirected edges and wiring each costs 2 (offer + accept). `gossip_reputation_greedy` build charges
  nnz(R) * iterations_run + 2*n*rounds: EigenTrust is genuinely distributed, so each power iteration pushes t[j] once
  along every report edge, and each T-Man round is one view exchange (2 messages) per node. That EigenTrust term
  dominates everything else in the benchmark (~1.0e6 build messages at n=1e3, K=16, b=3, i.e. ~20x the probe count) and
  is a real cost of seedless global reputation, not an accounting artifact. `flat_nsw_router` charges 0 messages: it is
  a centrally held index, the same trusted-observer advantage `flat_probe_argmax` has, and we state it rather than hide it.
- 2026-09-02 both graph walks take exactly `depth` steps and pay for a self-loop rather than breaking early, so
  per-fetch hops / comparisons / messages are exact constants (referral: 4 / 40 / 80; gossip: 6 / 60 / 120) and
  `scripts/check_methods.py` can assert them.
- 2026-09-02 `gossip_reputation_greedy` names its estimate table `_est`, not `est`, so that
  `scripts/check_methods.py`'s exact-estimate argmax check skips it: it is a greedy-walk method, not an argmax method,
  and is not expected to return the global argmax even with exact estimates. `flat_nsw_router` keeps `est` and does
  pass that check at 1.00.
