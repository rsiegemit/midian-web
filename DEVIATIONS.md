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
- 2026-09-02 `scripts/check_methods.py`'s exact-estimate argmax check does not apply to any reports-based method
  (gossip here, and MIDIAN). It mocks `probe_many` to return S as fractions, but the reports path then runs them through
  `World.report_many`, whose first line is `outcomes.astype(np.int8)`: verified, reports of [0.83, 0.6, 1.0, 0.2] come
  back as [0, 0, 1, 0], so every estimate below 1.0 truncates to 0 and the check measures truncation, not routing. It is
  reported as 0.09 for gossip, which is an artifact. Gossip is a greedy walk, not an argmax, so the check would not apply
  to it even with a working mock. `flat_nsw_router` takes no reports and does pass at 1.00.
- 2026-09-02 `flat_nsw_router` built its index with M and ef_construction transposed after the simplification pass
  (`init_index(n, efc, M, ...)`; hnswlib's positional order is `(max_elements, M, ef_construction, random_seed)`), so it
  ran at M=200 instead of the SPEC's M=16, with hnswlib silently clamping ef_construction up to M. Fixed by passing
  keyword arguments. Measured effect: n=1e5 build 141 s at M=200 versus 55 s at M=16, and roughly 12x the per-element
  memory, which is the likely origin of SPEC's "tens of GB at 1e7" estimate.
- 2026-09-02 (llm backend env): vLLM 0.22.1 requires `llguidance>=1.7.0,<1.8.0`, whose published
  Linux wheels are `manylinux_2_31` while these nodes run glibc 2.28 (Rocky 8.10). pip therefore
  falls back to the sdist and compiles it from Rust, bootstrapping a ~1.5 GB toolchain into
  `$HOME/.cache` -- on a home quota that is nearly full. `scripts/00_build_env.sh` now vendors the
  llguidance 1.7.6 abi3 build from the validated reference env (compiled on this same cluster, same
  CPython 3.12, same glibc) and redirects CARGO_HOME/RUSTUP_HOME/TMPDIR/XDG_CACHE_HOME to
  $RTE_DATA. The from-source path is still in the script behind `RTE_LLGUIDANCE_SRC`.
- 2026-09-02: MIDIAN's optional `stratify` flag (cohorts stratified by declared mean, SPEC §5 comment) is NOT implemented:
  it was the only path by which MIDIAN could read the declared channel, no grid used it, and its padding layout broke the
  estimator's one-short-cohort assumption. Plain MIDIAN never reads D, full stop.
- 2026-09-02: `check_methods.py`'s exact-estimate argmax check applies to argmax-type routers only; `gossip_reputation_greedy`
  (greedy walk of depth 6 on a 10-neighbour overlay from a random start) is not one and scores ~0.1 on it by design.
  (An earlier note here blamed `flat_nsw_router`'s 0.70 at n=1000 on approximate search; it was a swapped hnswlib argument
  order in the lead's rewrite, caught by the decentral agent. Fixed: 0.85, exact-estimate hit rate 1.00.)
- 2026-09-02 `flat_nsw_router` single-seed numbers are dominated by tie-breaking, MEASURED not argued. At n=1e3, K=16,
  b=3, specialist, ~116 agents tie at the maximum estimate in every family. Holding the estimates fixed and varying only
  hnswlib's index random_seed over 6 values, success ranges 0.698-0.888 and misroute_to_liar ranges 0.046-0.398
  (exact argmax on the same estimates: 0.810 / 0.286). So which tied agent is returned is arbitrary, the spread swamps
  any real effect, and this method must be read only through the runner's 10-seed average. The same nuisance applies to
  every argmax-over-b-probe-estimates method including `flat_probe_argmax`, which resolves the identical ties by lowest
  agent index -- a different arbitrary rule, not a better one. Raising b shrinks the tie set; at b=1 it is far worse
  (~30k tied at n=1e5).
- 2026-09-02 simplicity pass (lead directive): the shared "probe every agent b times per family and have exactly one peer
  observe and report each outcome" step now lives once in `rte/methods/_est.py` as `one_observer_reports(view, b, pick)`,
  and the shared greedy-walk-on-a-graph route as `greedy_walk(view, start, depth, step)`; `flat_nsw_router` reuses
  `probe_successes` from the same file. Both graph methods lost their own copies. `one_observer_reports` now reports a
  whole family in ONE `report_many` batch rather than one per agent-chunk, which is also the semantically right grouping
  for the liar's "top 20% of what I have seen" rule.
- 2026-09-02: framework rivals rows 1-4 (`fw_langgraph`, `fw_crewai`, `fw_autogen`, `fw_magentic_one`).
  All four SPEC §6A pins exist on PyPI and are installed as written (`langgraph 1.2.11` +
  `langgraph-supervisor 0.0.31`, `crewai 1.15.18`, `autogen-agentchat 0.7.5` + `autogen-ext[openai] 0.7.5`);
  `langchain-openai` is unpinned in the SPEC and resolved to 1.6.0. Recipe changes:
  - CrewAI (recipe 2): the SPEC says subscribe to `ToolUsageStartedEvent` and read `tool_args["coworker"]`.
    In 1.15.18 `crewai_event_bus.emit` dispatches handlers on a `ThreadPoolExecutor`, so a handler can
    neither return a value synchronously nor stop the run. We read the identical `coworker` argument one
    frame later at `BaseAgentTool._execute` and raise there. The exception subclasses `BaseException`
    because CrewAI wraps the delegated call in `except Exception` and would otherwise convert the abort
    into an error string fed back to the manager. The self-description goes in `role` as the SPEC requires,
    prefixed with the `agent_XXXXXX` id so the manager's pick is invertible.
  - AutoGen (recipe 3): `MaxMessageTermination(1)` cannot be used — the condition is evaluated on the task
    message itself, so the team terminates before selecting a speaker; we use `MaxMessageTermination(2)`
    plus an explicit break at the first `SelectSpeakerEvent`. Both AutoGen teams also publish the work
    request to the chosen speaker *before* emitting that event and keep running in a background task, so
    breaking at the event is a race: measured against the mock, a plain `AssistantAgent` participant spent
    one model call answering the task. Participants are therefore an `Idle(AssistantAgent)` subclass whose
    `on_messages` returns an empty `Response` without touching the model. Cost is then exactly one model
    call per `fetch`.
  - Magentic-One (recipe 3, second half): parsing `next_speaker` from the first ledger is not enough to
    stop the run (measured 5 model calls per selection even with `max_turns=1`). We patch
    `MagenticOneOrchestrator._log_message`, which announces `"Next Speaker: <name>"` immediately before the
    dispatch, and raise there; the `SelectSpeakerEvent` remains as a fallback. Cost is 3 model calls per
    `fetch` (fact sheet, plan, ledger) — inherent to the framework's outer loop. We also declare
    `structured_output=True` in `model_info` so the ledger is requested as an OpenAI JSON-schema
    `response_format`; this must be validated against the real vLLM endpoint.
  - Not a recipe change but load-bearing: conda envs put `~/.local/lib/python3.12/site-packages` on
    `sys.path`, which let `pip` skip dependencies that exist only in the user site and leave the venv
    broken off-login-node. `scripts/fw_envs/{langgraph,crewai,autogen}.sh` write a `zzz_no_user_site.pth`
    into the env before installing anything. Doing so surfaced that `crewai 1.15.18` imports `numpy`
    without declaring it; `numpy` is pinned in `requirements-frameworks/crewai.txt`.
  - `scripts/mock_openai_server.py` extended twice, generically: (a) the agent-name scan falls back from
    `messages` to the whole request, because CrewAI puts its coworker roster in a tool description rather
    than in the prompt; (b) `_fill` now resolves `$ref`/`allOf` and recurses into nested objects,
    inheriting "agent-ish"-ness from the enclosing field, because Magentic-One's `LedgerEntry` schema
    nests every answer as `{reason, answer}`. Flat-schema behaviour is unchanged.
- 2026-09-02: framework rivals rows 1-4, simplification pass and two follow-ups.
  - Shared worker boilerplate moved to `rte/methods/frameworks/workers/_wk.py` (`openai_kwargs(req)`,
    `run_async(coro)`, `sanitize(names)`); it imports no framework and has no side effects.
  - `scripts/mock_openai_server.py`: when a request carries tools, the mock now prefers the tool that names
    the first agent mentioned in the request, falling back to the previous "first tool that names any agent"
    rule. Without this the mock's documented policy ("the first agent named in the prompt/tools") did not
    hold for LangGraph, whose `create_supervisor` builds handoff tools from a `set` and so presents them in
    Python hash order; the pick was a valid candidate but an arbitrary one, and no test could assert which.
    With it, all four rivals return candidate 0 of the retrieved top-k and `tests/test_fw_a.py` asserts that
    identity rather than mere membership.
- 2026-09-02 (runner/analysis, ownership): `rte/run.py`, `rte/analyze.py`, `configs/grid.yaml` and
  `scripts/run_grid.sbatch` were written concurrently by two agents. The converged on-disk versions
  are the other agent's, with the runner-agent additions folded in (total-communication columns,
  build-probe budget warning, atomic column-ordered `rows.csv`, `imap_unordered` so parallel runs
  log progress as units finish, and `method_specs` dropping `requires_llm` methods off non-llm
  backends so a bernoulli mirror of a framework grid skips them instead of failing all ten).
- 2026-09-02 (sbatch): the two scripts collapsed into one `scripts/run_grid.sbatch`; the 1e6-1e7
  bigmem points are a documented override in its header
  (`sbatch -p bigmem -c 48 --mem 700G --export=ALL,RTE_WORKERS=4 scripts/run_grid.sbatch bernoulli_scale`)
  rather than a second near-identical file (zero-duplication directive).
- 2026-09-02 (bernoulli_scale): the runner brief asked for a warn-and-fall-back when
  `backend_kwargs.calibrate_from` is missing. The converged `cells()` instead ASSERTS the file
  exists, so an uncalibrated 1e6-1e7 point can never be produced by accident. That is the stricter
  reading and it is what is on disk; flagged to the lead because it departs from the brief.
- 2026-09-02 (grids): the frameworks ride `live_core_n100` and the dedicated `fw_live_n100` /
  `fw_live_n1000` / `fw_k_sensitivity` grids (self_described only, 3 seeds, Q=300) rather than
  `live_f1_n1000`. Each framework fetch is a real supervisor generation, so putting ten of them in
  the full F1 grid would be ~2.4M supervisor calls. Flagged to the lead as a scope decision.
- 2026-09-02 (CSV): rows now carry `total_comm_per_task` = (probes+reports+messages+tasks)/Q and
  `build_total_comm`, per the message-accounting directive; F3 and the new F3b plot them against n
  with fitted log-log exponents.
- 2026-09-02 (tests): the exact-estimate mock returns `rint(100*S)` as int8, not a fraction. The
  report channel casts outcomes to int8, so a fractional mock would truncate to zero and MIDIAN's
  peer-reported estimates could not carry it. 100*S is deterministic, monotone in S and int8-safe.
- 2026-09-02 (tests): `fw_echo` is the one `requires_llm=True` class the generic test still runs;
  it is the bridge protocol check and never reaches the endpoint, so it is built with a dummy
  `base_url`. It is excluded from every grid.
- 2026-09-02 (cluster, affects every agent): `fcntl.flock` BLOCKS FOREVER on the /n/netscratch
  mount that holds `$RTE_DATA`. Verified: an flock on a freshly created file under $RTE_DATA never
  returns, while the identical call on $HOME or /tmp returns instantly; sqlite is unaffected
  because it uses POSIX fcntl record locks, which do work there. Consequences: (a) any
  `huggingface_hub` download into `$HF_HOME` wedges, since it takes an flock per file --
  `scripts/complete_snapshots.py` fetches missing files over plain HTTP into the cache layout
  instead; (b) `scripts/_register_endpoint.py` is lock-free (one file per served model under
  `$RTE_DATA/endpoints.d/`, atomic `os.replace`, `endpoints.json` regenerated as a merged view).
  Nothing in this project may use flock/filelock on $RTE_DATA.
- 2026-09-02 (llm backend): model snapshots are downloaded COMPLETE (only duplicate weight formats
  are ignored), not filtered by `allow_patterns`. vLLM calls `snapshot_download(repo,
  local_files_only=True)` with no patterns and refuses to boot on a partial snapshot -- for these
  seven repos the missing files were .gitattributes / LICENSE / README.md, under 9 MiB total.
- 2026-09-02 (framework rivals B, `fw_maf`): SPEC §6A pins `agent-framework 1.16.0`, but that
  meta-package is unresolvable on linux-x86_64 -- its `[all]` extra pulls `agent-framework-hyperlight`,
  which requires `hyperlight-sandbox-backend-wasm`, and that has no distribution for this platform
  (pip: `ResolutionImpossible`). Installed instead the sub-packages the selection primitive needs:
  `agent-framework-core==1.16.0`, `agent-framework-openai==1.14.1`, `agent-framework-orchestrations==1.1.1`.
  The latter two are on separate version lines and 1.16.0 does not exist for either (newest on PyPI are
  1.14.1 and 1.1.1). `GroupChatBuilder`, `HandoffBuilder` and `OpenAIChatCompletionClient` all come from
  these, so the recipe is unchanged.
- 2026-09-02 (framework rivals, affects every fw_* venv): `~/.local/lib/python3.12/site-packages` is on
  `sys.path` of every conda prefix under `$RTE_DATA/env`, and it holds numpy 1.26.4, boto3, pytorch-lightning
  and a `google` namespace package that shadows `google.adk` / `google.genai`. Worse, pip treated those as
  already-satisfied dependencies: `fw_llamaindex` ended up with NO numpy of its own. Fix in two places --
  `scripts/fw_envs/*.sh` export `PYTHONNOUSERSITE=1` before pip, and each worker strips `/.local/lib/` from
  `sys.path` before importing its framework (`_bridge.Bridge._start` does not set PYTHONNOUSERSITE and is
  not modified here). `numpy` is now an explicit pin in `requirements-frameworks/llamaindex.txt`.
- 2026-09-02 (framework rivals B, `scripts/mock_openai_server.py`): three generic extensions, needed to test
  MAF / LlamaIndex without a GPU. (a) `"choice"` in the no-tools JSON answer is now `1`, not `0`:
  LlamaIndex's `SelectionOutputParser` reads it 1-based (`index = choice - 1`), so 0 selected index -1. The
  0-based `"index": 0` key is untouched. (b) A structured-output branch: when the request carries
  `response_format` with a JSON schema, the reply is a minimal instance of that schema and nothing else --
  MAF's `AgentOrchestrationOutput` sets `extra: "forbid"` and rejects the kitchen-sink JSON. (c) An SSE
  branch for `"stream": true` requests, because MAF's handoff path streams its agent runs; without it the
  client parsed an empty stream and no tool call ever arrived. Also `request_queue_size = 512` on the server:
  runs we abort at the pick leave connections queued, and the default backlog of 5 fills up and makes every
  later call fail with "Connection error". Unrelated: line 51 referenced a removed local `text` and raised
  NameError on every no-tools POST; fixed to read the messages.
- 2026-09-02 (framework rivals B): `mode` reaches the worker in the bridge request's `params` field, which
  `_bridge.py` / `_common.py` now carry. (Superseded: an earlier version of this passed it in an `FW_MODE`
  environment variable, which made two framework methods with different modes unsafe in one process.)
- 2026-09-02 (framework rivals B, `fw_maf` handoff): `tests/test_fw_b.py` asserts the pick is the FIRST
  retrieved candidate for every recipe EXCEPT MAF's handoff mode, where it only asserts membership in the
  top-k. `HandoffBuilder` stores a source's targets in a `set` of `HandoffConfiguration` hashed by target id,
  so the `handoff_to_<name>` tool order is Python's per-process randomized string hash order, not candidate
  order. Measured directly: 5 runs over the same 10 candidates picked agents 2, 3, 7, 0 and 5. The mock's
  fixed "first agent named" policy therefore cannot land on candidate 0 there, and the strict check would be
  flaky. Nothing to fix in the recipe -- the interception point is still the framework's own HandoffSentEvent.
- 2026-09-02 (serving): the `gpu_test` partition allocates A100 MIG slices, so SLURM sets
  `CUDA_VISIBLE_DEVICES` to a MIG UUID; vLLM 0.22.1 calls `int()` on it and dies with
  `ValueError: invalid literal for int() with base 10: 'MIG-adfbd773-...'`
  (vllm/platforms/cuda.py via registry.py:951, smoke job 43857516). `scripts/serve_smoke.sbatch`
  therefore runs on `kempner_h100` with one whole GPU instead of the spec's `gpu_test`.
- 2026-09-02 (serving): vLLM 0.22.1's FlashInfer top-k/top-p sampler JIT-compiles with ninja on
  first use, inside the memory-profiling dummy run, and that build FAILS on these nodes -- taking
  the engine down after the weights have already loaded (smoke job 43858361,
  CalledProcessError from flashinfer/jit/cpp_ext.py). Both serve scripts set
  `VLLM_USE_FLASHINFER_SAMPLER=0`; we decode greedily at temperature 0, so the FlashInfer sampler
  is not doing anything for us. They also set FLASHINFER_CACHE_DIR and a HOME shim under $RTE_DATA
  so no library writes build trees into the near-full home quota.
- 2026-09-02 (serving): both serve scripts pass `--generation-config vllm`, which ignores each
  repo's own generation_config.json. Qwen2.5 ships `repetition_penalty=1.1` and a temperature
  there; without this flag the models on the ladder would be sampled under different rules, which
  is a confound in a benchmark that exists to compare them. Requests set temperature 0 and seed 0.
- 2026-09-02 (llm client): the response memo opens its SQLite files with `nolock=1`. On this
  cluster's netscratch mount, where `$RTE_DATA/cache` lives, a plain `sqlite3.connect` from the
  rte env HANGS on the first write -- SQLite 3.50.3 (the env's build) selects a locking style that
  depends on the same broken `flock`; base Python's SQLite 3.51.0 happens not to. Skipping SQLite's
  file locking is sound because the memo has one writer process (the runner) whose threads are
  already serialised by a per-database `threading.Lock`; `rte.llm_client._claim` writes an owner
  stamp so a second live writer on the same host fails loudly instead of corrupting the cache.
  `RTE_LLM_CACHE_NOLOCK=0` restores real locking on a filesystem where it works.
- 2026-09-02 (framework rivals C: smolagents, CAMEL, MetaGPT, AgentScope):
  - `fw_smolagents`: SPEC §6A recipe 8 says read `ActionStep.tool_calls[0].name` from a `step_callback`.
    In smolagents 1.26.0 step callbacks fire in `MultiStepAgent._finalize_step`, which runs only AFTER
    `process_tool_calls` has already executed the selected managed agent — i.e. after a full sub-agent run,
    which is exactly what the interception must prevent. We instead consume `agent.run(task, stream=True)`
    and return on the first `smolagents.memory.ToolCall`, which `process_tool_calls` yields immediately
    before `execute_tool_call`. Same value, one step earlier, nothing executed.
  - `fw_camel_workforce`: `Workforce.add_single_agent_worker(description, worker)` has no `node_id`
    parameter and `BaseNode` defaults it to `str(id(self))`, but the coordinator's `ASSIGN_TASK_PROMPT`
    roster is `<node_id>:<description>:<toolkits>` and it answers with `assignee_id`. To put the candidate
    name where the framework's own primitive reads it, the worker assigns `wf._children[-1].node_id` after
    adding each worker (the one private-attribute touch in these four rivals).
  - `camel-ai` 0.2.90 declares `mcp>=1.3.0` but imports `mcp.server.FastMCP`, which mcp 2.x removed;
    `requirements-frameworks/camel.txt` pins `mcp>=1.3.0,<2` (resolved to 1.29.1).
  - `fw_metagpt`: the SPEC pin and the SPEC recipe are incompatible. `metagpt` 0.8.2 on PyPI contains NO
    `TeamLeader` and no `publish_team_message` (`metagpt/roles/di/` holds only `data_interpreter.py`), and
    0.8.2 routes purely by hardwired `Message.send_to` with no LLM and no agent descriptions read. It is
    also not installable as published: it hard-pins `lancedb==0.4.0`, which no longer exists on PyPI. The
    rival is therefore built against the GitHub `main` revision, which self-reports version 1.0.0 and does
    contain `TeamLeader.publish_team_message(content, send_to)`, installed `--no-deps` with a hand-assembled
    dependency set (see `requirements-frameworks/metagpt.txt`). Appendix material; caveats in NOTES_metagpt.md.
  - `fw_agentscope`: AgentScope 2.0.7 has no multi-agent selection primitive at all (no supervisor, no
    handoff, no speaker selector; `agentscope.pipeline` no longer exists). This rival is a DIY
    structured-output router — one `OpenAIChatModel` call with a json_schema `response_format` forcing
    `{"agent": <name>}` — and measures AgentScope's model layer, not a routing primitive. Report as DIY.
  - `scripts/mock_openai_server.py` (shared, GPU-free test double) extended twice for these rivals:
    (a) `ThreadingHTTPServer` instead of `HTTPServer`, because CAMEL's Workforce issues concurrent
    completions and the single-threaded server dropped connections; (b) when the prompt enumerates
    `Task ID: <id>` lines, the JSON answer also carries `assignments: [{task_id, assignee_id, dependencies}]`,
    which is the shape CAMEL's `ASSIGN_TASK_PROMPT` demands. Without (b) CAMEL falls back to inventing a
    brand-new worker with a uuid id, which is a real (and worth reporting) failure mode with a live model.
  - Build note, not a spec deviation: parallel `conda create` runs sharing `CONDA_PKGS_DIRS` corrupt the
    package cache (`InvalidArchiveError`). These four envs were created serially against a private cache
    at `$RTE_DATA/conda_pkgs_fwc`; `scripts/fw_envs/<name>.sh` is unchanged and still correct when run alone.
- 2026-09-02 (llm backend, MEASURED): the per-family handicap is prompt-side only -- no worked
  exemplar, no family description, no family tool. The generation-budget cap that earlier stood in
  for the spec's "difficulty capped" is OFF by default (`handicap_max_tokens=None`) because it was
  measured to be NON-MONOTONE: on Qwen2.5-0.5B the capped ("handicapped") configuration BEAT the
  uncapped one on syllogism, 0.85 vs 0.30 over 20 probes, and was +0.06 better on average across 8
  families -- a short budget forces a terse answer while a long one lets a small model ramble past
  the answer format. A handicap that sometimes helps would invert the `specialist` distribution.
  With the prompt-side handicap the sign is correct on every family measured (+0.031 mean).
  Handicapped agents still get the answer-format instruction: withholding it would measure format
  compliance rather than skill, and all agents are scored by the same verifier.
  `scripts/calibrate_families.py` reproduces the measurement.
- 2026-09-02: two CrewAI defects found by running the worker for 150 consecutive requests rather than 20,
  both fixed inside `workers/crewai_worker.py`:
  - CrewAI prints to `sys.stdout` (a Rich "Tracing Preference Saved" panel on first run, and
    `[CrewAIEventsBus] Warning:` lines), which is the `_bridge` JSON-lines protocol channel; a stray line
    makes the bridge kill the worker and record an error. The crew now runs inside
    `contextlib.redirect_stdout(sys.stderr)`. `_bridge.Bridge.select` would be more robust if it skipped
    lines that do not parse as JSON instead of treating the first line as the response — flagged, not
    changed, since `_bridge.py` is shared.
  - Aborting inside the delegate tool leaves CrewAI's `tool_usage_started` scope unclosed; the scope stack
    is a process-wide `ContextVar` whose `push_event_scope` raises at depth 100, so a persistent worker
    started failing after ~100 requests. The worker resets it per request with
    `crewai.events.event_context.restore_event_scope(())`.
- 2026-09-02 (llm backend, MEASURED calibration, `scripts/calibrate_families.py`, 20 probes/cell,
  first 8 families of the K=16 list, tool=none both arms):
    Qwen2.5-0.5B-Instruct   unhandicapped 0.14   handicapped 0.11
    Qwen2.5-7B-Instruct     unhandicapped 0.46   handicapped 0.33
  The ladder therefore does produce spread (0.14 -> 0.46 across two rungs) and the handicap sign is
  correct on every family measured. But SPEC §3 wants `specialist` specialty families at 0.70-0.95,
  and only gcd (0.85) and syllogism (0.80) reach that for the 7B; `basic_arithmetic` (0.25) and
  `leg_counting` (0.00) sit far below. Inspecting responses confirms this is genuine model error,
  not an extraction or verifier bug (gold answers rescore 1). The 14B/gemma-9b rungs and the tools
  will lift it; whether the population clears `skill_excess_ratio_family >= 1.5` must be settled by
  a real `python -m rte.backends.llm --measure` run before the grid.
- 2026-09-02 (llm backend, MEASURED tools, Qwen2.5-7B, 15 probes/cell): the `python` sandbox is a
  real capability axis -- basic_arithmetic 0.47 -> 0.60, gcd 0.80 -> 0.93, leg_counting 0.00 -> 0.07.
  The `calculator` tool is NOT: 0.40 / 0.73 / 0.07 on the same cells, at or below the no-tool arm,
  because one arithmetic expression cannot carry a multi-step task and emitting it costs a turn.
  Consider drawing tools from {python, none} rather than the spec's three-way set.
  - Hardening, not a spec deviation: `$RTE_DATA/env/fw_*/bin/python` is a conda prefix, so it still puts
    `~/.local/lib/python3.12/site-packages` on `sys.path`. Before this was noticed, fw_smolagents, fw_camel
    and fw_agentscope had no numpy of their own and were resolving `import numpy` to the user-site copy
    (1.26.4). numpy is now pinned into each of those three requirement files and installed with
    `PYTHONNOUSERSITE=1` (without it pip treats the user-site copy as already satisfying the requirement).
    The general fix, for whoever owns `_bridge.py`, is `env.setdefault("PYTHONNOUSERSITE", "1")` in
    `Bridge._start` beside the existing proxy stripping.
