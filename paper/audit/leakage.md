# Leakage audit (information / test-to-train leakage)

Auditor: audit-leakage agent, 2026-09-23. Read-only on code and data. Findings appended as confirmed.

## Findings

### L1. View isolation is by convention; no method breaks it (CLEAN, residual risk LOW)

- `View` stores the world as `self._w` (`rte/world.py:172`); `__getattr__` (`:229-230`) only fires for *missing*
  attributes, so `view._w.S`, `view._w.liars`, `view._w.backend` are reachable by any method. It is Python convention, not a sandbox.
- Grep of `rte/methods/**/*.py` for `_w`, `vars(`, `__dict__`, `true_skill`, `._S`, `.liars`, `oracle`, `World`, `backend`,
  `_snap`: no hit except `current_backend()` in `frameworks/_common.py:181-182, 201-202` (L2). `view.probe_at` / `probe_text`
  (the only `_w`-adjacent helpers) go through `_require("probe")` (`world.py:203-211`).
- Framework workers (`rte/methods/frameworks/workers/*.py`) get only `{name, description}` payloads and the task text over the
  bridge; every framework agent is built with `tools=[]` (e.g. `smolagents_worker.py:18-21`, `crewai_worker.py:53-58`,
  `langgraph_worker.py:25`), so a supervisor has no code-execution or file tool that could read `S.npy`/`profiles.json`.
- `view.text` / `view.embedding` are ungated (`world.py:213-217`); they return prompt text/embeddings only (no outcomes).
- Severity LOW (no current violation; a future method could silently cheat). Affects nothing drawn.

### L2. `current_backend()` hands framework rivals the population directory and the TRUE specialty (MEDIUM, favours RIVALS, not MIDIAN)

- `FrameworkMethod._texts` (`frameworks/_common.py:179-185`) reads `be.descriptions()`, `be.family_descriptions()`,
  `be.task_text` from the live backend; `_popdir` (`:194-203`) returns `be.dir`, the population directory that also holds
  `S.npy`, `profiles.json` and `D_self_described.npy` (`rte/backends/llm.py:168-200, 215`). The code only reads/writes its own
  embedding/shortlist caches there (`descriptions_*.npy`, `families_*.npy`, `shortlist_sota_*.npy`, `_common.py:231-270`);
  it never opens `S.npy` or `profiles.json` (grep: no hit in `rte/methods`). Clean as used.
- But what `descriptions()` returns is itself privileged: `llm.py:241-248` appends `Declared areas: <profiles[a]["specialty"]>`,
  and the describe-self prompt (`prompts.py`, `describe_self`) is told the agent's **base model id, tool and specialty set**,
  i.e. the agent's full prompt signature (`population.py:85-90`), which is exactly what determines S on live. Example
  (`$RTE_DATA/populations/specialist_n100_K16_seed1/descriptions.json` agent 0): "I'm Gemma ... Declared areas:
  simple_equations, time_intervals, calendar_arithmetic." with profile `{'model': 'google/gemma-2-2b-it', 'specialty': [9, 12, 15]}`.
- Consequences: (i) under the cartel, liars' text still states their TRUE specialty (erratum 27) while `view.declared` is
  inflated; (ii) even honest, live figure grids use `declared_source: self_described` (self-rating, corr ~0.36 with S per
  guide §0.3 #18) whereas the text names the ground-truth unhandicapped families. Text shortlists (TF-IDF, BM25, MiniLM, Qwen3
  dense, rerank) and `llm_supervisor` therefore see a cleaner skill signal than any declared-channel arm and than MIDIAN's
  inputs. This is information outside the View's contract.
- Bias direction: in favour of framework / supervisor rivals (figures E, F, G; live parts of A if llm_supervisor appears),
  against MIDIAN / MIDIAN-VA. Does not inflate MIDIAN. Should be stated in captions (guide §0.3 #18 already flags the text clause;
  it does not mention that the prose also carries the base-model id).
- Minor: `current_backend()` is a process-global "last LLMBackend constructed" (`llm.py:270-277`), gated only on `be.n == view.n`
  (`_common.py:183, 203`). A process that ran a live unit and then a non-live unit with equal n (run.py forces workers=1 when any
  unit is live, `run.py:256`) would hand the non-live framework the live population's texts. Wrong-texts bug, not truth leak; LOW.

### L3. Replay (RouterBench): probes and routed tasks are the SAME prompt rows; S is the in-sample pool mean (LOW, no differential bias found)

- `ReplayBackend.execute` / `execute_many` (`rte/backends/replay.py:100-110`) both map an instance seed to
  `row_start[f] + inst % n_prompts[f]`; there is no train/test split. S (`:72-85`) is the effective model's accuracy over
  **all** rows of the category, i.e. over the very rows tasks are drawn from. Outcome depends only on (row, effective model),
  where effective model = own model, or the category's weakest model when masked (`:103`).
- Measured on `$RTE_DATA/data/routerbench_cells.npz` (K = 64 largest categories: 100 / 210 / 10,042 prompts min / median /
  max; M = 11 models). Probes landing on each prompt row per family = n·b / n_prompts:
  n = 10^3, b = 3: median 14 per row; n = 10^5: median 1,430; **n = 10^6, b = 1: median 4,766 (min 100); b = 3: median
  14,297 (min 299)**. Under a uniform-model approximation, P(a given (row, model) pair was probed at least once) is 1.000 for
  every category at 10^6 (b = 1 and 3), >= 0.93 at 10^5. So at the replay 10^6 cell of figure B, **every routed task's exact
  (prompt, model) outcome has been observed during the build**, many times over.
- Why this does not (currently) favour any arm: no method can map a task to its row. `World.text` on replay is synthetic,
  "A task of family X (instance <seed>)" (`world.py:325-329`), and two seeds hitting the same row have unrelated numbers, so
  `knn_router` / `mlp_router` embeddings cannot identify the row; all other arms are per-(agent, family). Every estimator's
  target is S itself, and declared = clip(S + N(0, 0.05)) (`backends/__init__.py:15-19`) is equally in-sample.
- What it does mean: replay numbers measure routing to the best model **on a fixed pool the build already sampled**, not
  generalisation to unseen prompts; that is fine for a per-family routing benchmark but must be stated (B, replay 10^6
  bars; C's cost curve is unaffected). Any future prompt-aware arm on replay would be scoring in-sample. Guide §1.4.3
  states "Probes and tasks draw from the same prompt pool" correctly.

### L4. Live: probe, measurement and task SEED families are disjoint, but the QUESTION TEXTS collide in small-space families (LOW)

- Seeds: tasks `stable_seed_32(seed, "inst", i, f)` (`world.py:292`); probes `probe_seed(stable_seed_32(seed, "probes"), a, f, k)`
  (`world.py:100-105, 262, 319`, a different mixer and salt); S-measurement `stable_seed_32("measure", fam, r)`, r < 200
  (`backends/llm.py:178`); prompt furniture `stable_seed_32("aux", kind, family)` (`families.py:117-120`). Three separate
  namespaces; a 31-bit seed collision is ~1e-6 per pair. Seed-level separation: CLEAN.
- But Reasoning-Gym maps many seeds to the same problem. Measured (scratch script, seed 1, first 3,000 tasks vs the k = 0
  probe of 3,000 agents, i.e. the n = 10^3, b = 1 slice; `$RTE_DATA/env/rte/bin/python`):

  | family | distinct / 3,000 tasks | share of task questions also a probe question | share also an S-measurement question |
  |---|---|---|---|
  | prime_factorization | 0.316 | **0.949** | **0.192** |
  | lcm | 0.803 | 0.383 | 0.031 |
  | chain_sum | 0.782 | 0.296 | 0.056 |
  | basic_arithmetic | 0.933 | 0.097 | 0.015 |
  | calendar_arithmetic | 0.954 | 0.086 | 0.008 |
  | binary_alternation | 0.988 | 0.030 | 0.001 |
  | other 10 families | >= 0.997 | <= 0.005 | 0 |

  At n >= 10^4 (>= 30,000 probes per family) the probe coverage of these families is higher still.
- Because the memo keys on the full request (`llm_client.py:127-139`) and the prompt depends only on the signature and the question
  (`backends/llm.py:108-119`), a probe of any same-signature agent on a colliding question returns **exactly** the routed task's
  outcome. Impact on arms: per-(agent, family) estimators (MIDIAN*, flat, bandits, declared) cannot exploit this. Only
  `knn_router` sees question text, and it keeps only the routed agent's *own* b probes per family (`knn_router.py:28-36`), so
  the chance that agent a itself probed task q is about b / (#distinct problems), ~0.3 % even for prime_factorization. Negligible.
- The oracle's S is partly measured in-sample for prime_factorization (19 % of task questions are in the 200-question
  measurement set) and marginally for chain_sum / lcm. The oracle is the reference line in A, E, F, H and the **normaliser of
  every B bar**; the effect on it is tiny (one or three of 16 families, a partial overlap) but the "S is measured on instances
  disjoint from tasks" wording (guide §1.5, "three different seed families") is true of seeds, not of problems.
- Direction: none systematic. Severity LOW; worth one sentence in the methods.

### L5. RouterEval / LLMRouterBench: routed TEST prompts repeat, and `knn_router(online=True)` memorises their outcomes (MEDIUM, inflates a RIVAL, against MIDIAN)

- Probe vs task separation itself is CLEAN on these backends: probes map to TRAIN rows, tasks to TEST rows
  (`rte/backends/routereval.py:201-222`: `text/embedding(probe=True)` and `execute_many` use `_tr`; `execute` uses `_te`).
  The learned routers train on the train split: `_learned.probe_set` embeds probe prompts with `probe=True`
  (`rte/methods/_learned.py:39-46`), `task_vec` uses the test prompt (`:55-56`). No learner features or training data come from
  test prompts at build time. S (oracle) = mean TRAIN score (`routereval.py:162`).
- But the stream re-draws a finite test pool: task row = `_te[f][instance % len(_te[f])]` (`:213`), instance =
  `stable_seed_32(seed, "inst", i, f)` (`world.py:292`), Q = 1000 over K = 16 (mmlu) / 15 (LLMRouterBench). Measured repeat rate
  (share of tasks whose (family, test row) already occurred earlier in the same stream; seeds 1-5, scratch script
  replicating `run.run_method` with per-task rows):

  | pool | test prompts per family min / median / max | repeat share of the 1,000 tasks |
  |---|---|---|
  | RouterEval mmlu m = 10 / 100 / 1,000 (Q = 1,000) | 27 / 34 / 154 | **0.483** |
  | LLMRouterBench n = 20 (Q = 1,000) | 26 / 302 / 376 | **0.203** |

- `knn_router(online=True)` appends every routed (task embedding, agent, outcome) to that agent's store (`knn_router.py:36-40`)
  and predicts with the mean of the k = b nearest stored items (`:25-29`). A repeated test prompt has cosine 1 to its stored copy,
  so the router recalls the *test label* it saw. Exact counterfactual (same run; at every repeat task, re-score with the stored
  exact copies of that prompt masked, execute the counterfactual pick on the same row):

  | pool | knn-online success on repeat tasks: actual vs without exact copies | memorisation gain in overall success (per-seed) |
  |---|---|---|
  | LLMRouterBench 20 | 0.764 vs 0.644 | **+0.024** (0.023-0.025) |
  | RouterEval mmlu m = 10 | 0.782 vs 0.642 | **+0.067** (0.057-0.082) |
  | RouterEval mmlu m = 100 | 0.693 vs 0.453 | **+0.116** (0.098-0.144) |
  | RouterEval mmlu m = 1,000 | 0.522 vs 0.280 | **+0.117** (0.103-0.133) |

  Offline `knn_router` (same probes, no store growth) shows no repeat advantage at mmlu (first 0.392 vs repeat 0.396 at m = 100).
- Affected arms/figures: `knn_router_online` is a candidate of B's "best learned router" and is the cross-fitted pick at
  LLMRouterBench cartel b = 1 (2 of 5 seeds) and b = 5 (2 of 5) (`figures/condensed_sample/B_families_allb.csv`); it is in the RouterEval
  5k pool (not yet landed there: "INCOMPLETE POOL, missing knn_router_online"); `routereval_mmlu` (`configs/grid.yaml:316-357`) and any
  RESULTS table built on it. On RouterEval 5k (leaderboard, 80/20 split per subject, Q = 300) the test pools are larger and repeats rarer,
  so the gain there will be smaller; on live, tasks are fresh generations (L4), so memorisation needs a text collision (rare).
- Direction: **inflates a rival**, i.e. biases AGAINST MIDIAN / MIDIAN-VA. Also makes the per-seed "best learned router"
  pick partly a memorisation contest on LLMRouterBench cartel cells.

### L6. Live "Declared areas: <true specialty>" clause (erratum 27): who benefits (MEDIUM, favours text-shortlist RIVALS under the cartel; not MIDIAN)

- Mechanism: L2. The clause and the prose come from `profiles[a]["specialty"]` + base model (`backends/llm.py:241-248`), so they are
  identical in every liar regime; only `view.declared` moves with lying.
- Who reads it: (a) every text retriever (TF-IDF, BM25, MiniLM, Qwen3 dense, hybrid, rerank; `_common.py:294-331`);
  (b) **every** framework supervisor, whatever the shortlist, because the payload is `self.desc[a]` for each candidate
  (`_common.py:355, 366`), including the `declared` top-k and the `va_cohort` (MIDIAN-VA cohort) shortlists in E/F/H;
  (c) `llm_supervisor` (`llm_supervisor.py:74-75`). Nobody else: MIDIAN*, flat, bandits, learned routers, declared argmax never
  read descriptions (grep: `desc` only in `frameworks/_common.py` and `llm_supervisor.py`).
- Measured with the paired `lie_text` condition at live n = 10^5 (`$RTE_DATA/results/fw_live_n100k_lietext`, 8 frameworks x 3 seeds per
  cell, dedup on). `lie_text=True` rewrites the clause from `view.declared`:

  | cell | TF-IDF, true clause | TF-IDF, lie_text | BM25, true clause | BM25, lie_text |
  |---|---|---|---|---|
  | beta = 0 (low_skill_first) | 0.316 | 0.375 | 0.391 | 0.404 |
  | beta = 0.5 cartel | 0.304 | 0.352 | 0.388 | 0.403 |
  | cartel minus honest | -0.013 | -0.024 | -0.003 | -0.002 |

  With `claim_threshold: 0.7` (`fw_live_n100k_lietext_th`) TF-IDF falls 0.414 -> 0.358 (-0.056) under the cartel, BM25 0.410 -> 0.420.
  So the true clause shields TF-IDF frameworks from the cartel by roughly 0.01-0.04 at 10^5, and BM25 hardly at all. Surprisingly,
  at beta = 0 the true clause scores LOWER than the self-rated clause (0.316 vs 0.375 TF-IDF), most likely because the relabelled
  texts change the dedup pool; so the clause is not a clean "oracle hint" for frameworks at 10^5.
- Direction: under the cartel it makes text-shortlist framework bars look more liar-robust than they are (E, F, G hatched bars; erratum 27
  already says so). It inflates rivals, not MIDIAN. Caveat for the `va_cohort` bars: their supervisor also reads true-specialty text,
  so any "MIDIAN-VA cohort + framework" gain is partly a text-channel gain the adversary never attacks.

### L7. LLM memo cache: no cross-method information channel (CLEAN; one reproducibility caveat, LOW)

- Key = blake2b over `(model, messages, max_tokens)` (`rte/llm_client.py:127-139`); the agent prompt is built from the signature and
  the question only (`backends/llm.py:108-119`), so the cached value is a function of (signature, question). A cache hit returns what
  a fresh temperature-0 call would (modulo vLLM nondeterminism), whichever method or the oracle line caused it. Methods never read the
  memo: the only `llm_client` callers in `rte/methods` are `llm_supervisor.py:82` and `midian_llm_descent.py:32`, for their own prompts.
  What a method observes before routing (probes, reports, declarations) is therefore independent of which methods ran first or of the
  oracle line (`run.py:162`, which executes the oracle's picks on the task stream first and so fills the memo with the tasks' answers for
  the oracle's signatures). CLEAN.
- Caveat: the memo is first-writer-wins per process and merges other processes' shards every 30 s (`llm_client.py:96-120`, `INSERT OR
  REPLACE`, `_mem.update`). When two processes generate the same key concurrently with different (nondeterministic) answers, a unit can
  score the oracle line and a later method on different answers for the same (signature, question). Noise, not directional; plausibly part
  of the unexplained live rerun drift (guide §0.3 #8). LOW.

### L8. observe()/fetch() ordering and prefetch (CLEAN)

- `run_method` (`rte/run.py:140-149`): `m.fetch(task)` -> `execute` -> `m.observe(task, a, outcome)`, strictly per task; the outcome list
  is appended after observe; the next task is not visible to observe. `world.reset` and `m.build` happen before the loop (`:132-133`).
- `FrameworkMethod.prefetch(stream)` (`frameworks/_common.py:340-360`) sees the whole stream up front but only calls
  `retrieve` + the supervisor; it executes nothing and reads no outcomes, and is skipped for the online MIDIAN-cohort modes. Clean.
- `MidianA.observe` (`midian_a.py:39-51`) audits reporters against the routed outcome it has just been given; that outcome is legitimately
  known to every arm via observe. The build-time audit `view.probe_at` (`world.py:385-389`) re-executes an already-drawn probe instance,
  charged as a probe; it returns the same deterministic outcome on every backend (llm memo; bernoulli `execute_many` hashes (inst, seed)
  only, `bernoulli.py:274-276`; replay/routereval table lookups). This is a trusted re-observation of 5 % of probes, a protocol choice
  (charged, <= 1.05x budget, guide §0.3 #6), not leakage.

### L9. Tuned warm-start bandit n0 = 0.5 (CLEAN on disjointness; LOW under-tuning note)

- `tune_wsb_n1000` (`configs/grid.yaml:1031-1041`): live, specialist, n = 1,000, beta = 0, b = 3, seeds 11-15, Q = 1,000. Rows
  (`$RTE_DATA/results/tune_wsb_n1000`): n0 = 0.5 -> 0.7624, 1.0 -> 0.7450, 2.0 -> 0.7500, 5.0 -> 0.7448 (oracle 0.8618), seeds 11-15 only.
- Disjointness: every reported live 10^3 cell uses seeds 1-10 (`fw_live_n1000` seeds 1-10, `pool_seeds_n1000` 6-10, `va_b_n1000`/
  `rivals_b_n1000` mirror those); resolving every live n = 1,000 block with `rte.run.blocks` finds seeds 11-15 only in `tune_wsb_n1000` and
  `midian_v_replication` (seeds 11-20, a MIDIAN-V replication, not a WSB cell). Live populations are per-seed directories, task instances
  are seeded by the world seed, so seeds 11-15 share no agents, tasks or probe instances with reported cells. The bernoulli grids calibrate
  S from `specialist_n1000_K16_seed1` (a reported seed, not a tuning seed). Tuning did not touch n = 100 / 10^4 / 10^5, bernoulli, replay,
  RouterEval or LLMRouterBench, where the tuned value is applied out of sample. CLEAN.
- Note: 0.5 is the smallest value tried and the curve is still rising toward it, so the rival may be under-tuned (n0 < 0.5 untested). Slight
  bias in favour of MIDIAN in every "best bandit" bar that picks `warm_start_bandit[n0=0.5]` (A: live 10^5; B: bernoulli, replay). LOW.

### L10. Repeated test prompts also leak into per-(agent, family) ONLINE learners, incl. MIDIAN and MIDIAN-VA (MEDIUM on RouterEval mmlu m <= 1,000: H's MIDIAN-VA line inflated vs stateless frameworks; none detected on the 5k pool or LLMRouterBench)

- Repeat rates per backend (share of the Q tasks whose exact prompt already occurred in the same stream):
  live and bernoulli: task instance seeds all distinct (1,000 / 1,000 at Q = 1,000), prompts collide only as in L4;
  replay K = 64, Q = 1,000: 0.043 / 0.038 / 0.036 (seeds 1-3); RouterEval mmlu m <= 1,000: 0.483; LLMRouterBench: 0.203 (L5).
- Per-family learners cannot recall a prompt, but a repeated prompt is deterministic, so an agent credited for solving p is more likely
  to be picked again for families where p recurs, and then solves p again. Test: same seeds and family sequence, test rows re-drawn
  WITHOUT replacement per family (cycled only when a pool is exhausted; this lowers, not removes, repeats: mmlu 0.482 -> 0.347,
  LLMRouterBench 0.205 -> 0.073). Success(natural) - success(fewer repeats), 10 seeds, mean (se):

  | arm | mmlu m = 10 | mmlu m = 100 | LLMRouterBench 20 |
  |---|---|---|---|
  | midian_va | +0.011 (0.006) | +0.005 (0.005) | -0.001 (0.006) |
  | midian | +0.004 (0.004) | **+0.021 (0.004)** | -0.003 (0.007) |
  | flat_probe_argmax online | -0.002 (0.004) | +0.014 (0.006) | +0.002 (0.009) |
  | warm_start_bandit | -0.003 (0.006) | +0.016 (0.005) | -0.002 (0.007) |
  | thompson_per_family | +0.002 (0.007) | +0.012 (0.006) | +0.004 (0.008) |
  | linucb_honest | -0.005 (0.005) | +0.015 (0.009) | -0.015 (0.008) |
  | knn_router online | +0.027 (0.008) | **+0.045 (0.005)** | +0.010 (0.006) |
  | offline controls (flat, knn, declared argmax) | -0.002 to +0.002 | -0.005 to 0.000 | -0.007 to -0.002 |
  | oracle | -0.003 | +0.003 | -0.002 |

  Only 13.5 pp of mmlu's 48 pp repeat share was removed, so the full repeat effect is roughly 2.5-3.5x these numbers (for knn the
  exact counterfactual of L5 gives +0.116 vs +0.045 here, ratio 2.6): about +0.05 for plain MIDIAN and the online bandits, ~+0.015 for
  MIDIAN-VA, at mmlu m = 100.
- mmlu m = 1,000 (5 seeds; repeats 0.483 -> 0.348): midian_va **+0.023 (se 0.011)**, midian +0.020 (0.014), linucb +0.026 (0.010),
  thompson +0.017 (0.009), flat online +0.007 (0.006), warm_start_bandit -0.004 (0.005), knn online +0.036 (0.013); offline flat /
  knn / declared +0.003 / 0.000 / -0.002; oracle -0.004. Extrapolated to zero repeats (x2.5-3.5): **MIDIAN-VA ~+0.06-0.08 at
  m = 1,000** from test-prompt repeats, versus ~0 for the offline arms and for the stateless frameworks.
- RouterEval 5,000-LLM leaderboard (the B "RouterEval n = 5,000" group; Q = 300; test prompts per family 53-307): repeat share
  **0.104** (seeds 1-3), fully removed in the re-drawn stream. Differences (3 seeds, noisy at Q = 300, offline flat moved -0.043 by
  prompt mix alone): midian_va -0.018, midian -0.004, flat online -0.014, warm_start_bandit -0.001, declared argmax -0.001. No detectable
  repeat gain there; knn_router online not run (no shipped embeddings; MiniLM on 5,000 agents too heavy for a login node), by analogy
  with L5 its memorisation gain would be about 0.10 x 0.24 ~ +0.02.
- Affected: every online arm on `routereval_mmlu` (and anything built on it, e.g. cohort_routereval, the H reference line
  "MIDIAN-VA on the whole population" at m = 10 / 100 / 1,000). Framework arms are stateless (no online learning, `_common.py:340-344`),
  so in **H the MIDIAN-VA line at m = 1,000 gains roughly +0.02 (measured, partial removal) to +0.06-0.08 (extrapolated) from repeats the
  framework bars cannot use**: a bias in favour of MIDIAN-VA (smaller at m = 10 / 100, none detectable on the 5k leaderboard).
  Exception: the `va_cohort` framework bars run MIDIAN-VA online inside the adapter (`_common.py:337-338`), so they share the gain.
  In A/B, MIDIAN-VA's main comparators (flat online, bandits, knn online) are also online and gain as much or more; the offline arms
  (declared argmax, mlp_router, cluster_head, disrouter) do not. LLMRouterBench: no measurable per-family effect.
- Fix options: draw test rows without replacement where the pool allows (and set Q <= pool size per family), or report success on
  first occurrences only.

### L11. Report channel is honour-system: MIDIAN holds the true probe outcomes before passing them through `report_many` (CLEAN as coded; LOW)

- `World.report_many(reporters, agents, outcomes)` (`world.py:358-383`) takes the outcomes FROM THE CALLER; the method got them
  from `view.probe_many` a line earlier (`_est.py:73-75`, `midian_sh.py:69`, `_est.py:104-120`). So MIDIAN's "the router only
  sees peer reports" is a coding discipline, not an enforced one: the true outcomes sit in local variables.
- Checked every MIDIAN path used by the drawn arms: plain MIDIAN level 0 (`_est.peer_reported_estimates`) and MIDIAN-VA level 0
  (`MidianSH._level0` with `halving=False`) keep only the reports (`_, per = peer_estimate(...)`, `midian_sh.py:69`); verification
  (`midian.py:79-97`) folds only `peer_estimate`'s trimmed report mean; audits use `probe_at`, charged. The only direct uses of true
  probe outcomes in MIDIAN code are the labelled cohort variants `stratify` / `block` / `specialty` (`midian.py:38-46`, grouping key,
  not drawn in A-H) and `churn()` for a cohort with no peers (`midian.py:181-182`, no churn in A-H). CLEAN for MIDIAN / MIDIAN-VA.
- `Bus` (`world.py:145-159`) delivers nothing but the sender's own payload; it only charges. No information path.

### L12. Note for the protocol audit (not leakage in the code): MIDIAN-VA was composed after seeing V and A on the reported cells

- `midian_va` = MIDIAN-V + MIDIAN-A, "added 2026-09-03 15:00 before any run" (`TARGETS_rte_v2.md:23-26`), after V and A had been run on
  the same live cells and seeds; audit rate 0.05 and STRIKES = 2 are fixed defaults (`midian_a.py:13, 19`), not tuned per cell (guide §2.3.8).
  No held-out seed set exists for the choice of VA as the headline among >= 6 MIDIAN variants, whereas the one tuned rival (L9) was tuned on
  held-out seeds. Selection among variants on the reporting seeds is a mild winner's-curse bias in favour of MIDIAN-VA on live; the other
  backends (bernoulli, replay, RouterEval, LLMRouterBench) postdate the choice and are out of sample. Flagged for audit-protocol2.

---

## What was checked and found clean

1. View `needs` enforcement: every probe/report/declared/bus accessor calls `_require` (`world.py:180-227`); `__getattr__` blocks unknown
   attributes. No method reads `_w`, `S`, liars, oracle, backend or population files (L1). Frameworks reach `current_backend()` only for
   descriptions, family descriptions, task text and the cache directory (L2).
2. Framework workers: `tools=[]` everywhere, no code-execution or file tools; the bridge carries names, descriptions and task text only (L1).
3. Seed families: tasks / probes / S-measurement / prompt furniture use disjoint hash namespaces on live; bernoulli task and probe outcomes use
   different hash formulas (`bernoulli.py:270-276`); RouterEval / LLMRouterBench / leaderboard probes are train rows and tasks test rows
   (L4, L5).
4. Every learner and bandit builds only from `view.probe*` (probe instances; train split on RouterEval) or `view.declared`:
   knn/mlp (`_learned.probe_set`, `probe=True`), flat, flat-NSW, UCB, Thompson, warm-start, LinUCB, TrueSkill, sequential halving (trusted
   and peer), verify-on-claim (probes at fetch time are fresh probe instances, charged), cluster-head, DisRouter, CNP, declared softmax,
   route-to-k (executes k agents on the task, charged as tasks), gossip, referral. No learner's build features or labels come from
   test-split prompts (L5).
5. `run_method` ordering fetch -> execute -> observe; prefetch reads no outcomes (L8).
6. LLM memo: no method-visible channel; cached answers are content-addressed (L7).
7. MIDIAN / MIDIAN-VA use only reports for estimates; audits are charged re-probes (L8, L11).
8. Tuned warm-start bandit: tuning seeds 11-15 disjoint from every reported cell (L9).
9. Guide claims verified against code: §1.1.6 (View, `_w`, `current_backend`), §1.4.3 (replay shares the pool), §1.4.4 (train/test),
   §1.5 ("three different seed families": true for seeds, not for problem texts, L4).

## Summary

| # | Finding | Severity | Direction | Figures |
|---|---|---|---|---|
| L5 | knn_router online memorises repeated TEST prompts: +0.024 (LLMRouterBench), +0.067 / +0.116 / +0.117 (RouterEval mmlu m = 10 / 100 / 1,000) absolute success | MEDIUM | inflates a rival (against MIDIAN) | B (LLMRouterBench cartel best-learned picks), routereval_mmlu tables |
| L10 | Repeats also lift per-family online learners, incl. MIDIAN-VA (~+0.02 measured with partial removal, ~+0.06-0.08 extrapolated at mmlu m = 1,000); offline arms and stateless frameworks get 0 | MEDIUM | favours MIDIAN-VA over frameworks in H m <= 1,000 and over offline arms; neutral vs other online arms | H, routereval_mmlu tables; not B's 5k / LLMRouterBench |
| L2 / L6 | Live descriptions carry the true specialty and base model (outside the View); shields text shortlists from the cartel by ~0.01-0.04 (TF-IDF) at 10^5 | MEDIUM | favours framework rivals (against MIDIAN) | E, F, G |
| L3 | Replay: probes and tasks share rows; every (row, model) of the 10^6 cell is probed during the build | LOW | none found (no arm can map task to row) | B replay |
| L4 | Live problem texts collide across seed families (prime_factorization 95 % of tasks appear among probes; 19 % among S-measurement questions) | LOW | none systematic; oracle slightly in-sample | A, B normaliser |
| L7 | Memo first-writer-wins across processes | LOW | noise | A (live drift) |
| L9 | n0 = 0.5 at the edge of the tuning grid | LOW | slightly favours MIDIAN | A, B best bandit |
| L12 | VA chosen among variants on reporting seeds (protocol) | LOW | favours MIDIAN-VA on live | A |

No finding lets any method read S, the liar set, the oracle or a task's outcome before routing it. The one mechanism that inflates
MIDIAN-VA is test-prompt repetition on RouterEval mmlu m <= 1,000 (L10). The fix is to draw test rows without replacement (Q per family
<= pool) or to score first occurrences only; the same fix removes L5.

Scratch scripts (not part of the repo): `/tmp/claude-68287/-n-home02-rsiegelmann/bdcd0628-942c-4cea-8c96-748671ba0e73/scratchpad/`
`live_overlap.py`, `re_repeat.py`, `re_fresh.py`, `agg.py` and their `rep_*.jsonl` / `fresh_*.jsonl` outputs.
