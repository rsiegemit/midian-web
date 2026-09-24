# Audit: rival fairness (audit-rivals2, 2026-09-23)

Scope: rivals in figures A/B + agent frameworks. Findings appended as confirmed.
Severity: CRITICAL / HIGH / MEDIUM / LOW. Bias direction relative to MIDIAN.
Names follow the 2026-09-24 rename (CHANGES_AND_ERRATA §8g): MIDIAN = both defenses (formerly VA); MIDIAN w/o defenses = the
plain tree; MIDIAN w/o audits / w/o verification = the single-defense ablations. Code references are to the code audited.

## Findings

Reproduction harness (scratch, not in repo): `World(..., backend="bernoulli", calibrate_from=$RTE_DATA/populations/specialist_n1000_K16_seed1/S.npy)`
(= the bernoulli_scale_v5 world), K = 16, specialist, `rte.run.run_method`, seeds 1-3 unless stated. Q = 300 at n >= 1e4 (as in the
live grids), Q = 1000 at n = 1e3.

### F1. linucb_honest is broken by its own exploration bonus: it picks the WEAKEST of the tied top agents (HIGH)

- Evidence (live): `$RTE_DATA/results/rivals_b_n100k/rows.d`, n = 1e5, b = 1, honest: 0.377 / 0.377 / 0.423 vs
  flat_probe_argmax online 0.583 / 0.673 / 0.730 and warm_start 0.623-0.713. At b = 3 (`results/live_n100k`) it is
  0.390 / 0.367 / 0.400, i.e. AT random (live mean skill 0.419), while flat online is 0.703 / 0.737 / 0.807.
- Reproduced on calibrated bernoulli (so not a live/LLM artefact):
  | n, b | flat online | linucb (alpha=1) | linucb alpha=0 | random |
  |---|---|---|---|---|
  | 1e3, b=3 | 0.751 | 0.704 | 0.714 | 0.430 |
  | 1e4, b=1 | 0.667 | 0.463 | - | ~0.42 |
  | 1e4, b=3 | 0.723 | 0.479 | 0.679 | 0.416 |
  | 1e5, b=1 | 0.676 | 0.381 | 0.578 | 0.418 |
  | 1e5, b=3 | 0.774 | **0.340** (< random) | 0.658 | 0.418 |
  The collapse grows with n, exactly as on live.
- Mechanism (diagnosed, n = 1e4, b = 3, seed 1): with only b probes per arm, thousands of agents tie at est = 1.0.
  The shared ridge model learns theta ~ [0, 1, 0, 0] (the target IS the `mean` feature), so the predicted means of the
  tied agents are equal and the pick is decided by the bonus alpha*sqrt(x^T A^-1 x). That bonus is largest for the most
  ATYPICAL context, and among agents with est_f = 1 the atypical ones are those with a LOW `mean over families` feature
  (`linucb_honest.py:23`): agents that got lucky on family f and fail everywhere else, i.e. weak agents. Measured: 62 % of
  linucb's picks lie in the bottom 5 % of true mean skill among the agents tied at/above its estimate (mean percentile
  0.18); with alpha = 0 it is 12 % (percentile 0.68). Removing the sqrt(count) feature (which is constant = sqrt(b) in
  warm-up, hence collinear with the intercept; eigenvalue of A in that direction = 1 = ridge only) does NOT fix it
  (0.477 at 1e4 b = 3): the cause is the bonus x the irrelevant `mean over families` feature, not the collinearity.
- Also not faithful to LinUCB (Li et al. 2010): arms are agents with no side information, so a textbook contextual
  bandit here is either per-arm (disjoint) UCB, which is `ucb_per_family`, or needs real context. The code cites no paper
  (`linucb_honest.py:1-5`; FIGURE_DATA_GUIDE 2.7.4). alpha = 1.0 is arbitrary (not tuned, not pre-registered in SPEC §6).
- Figure impact: bounded, because best-bandit is cross-fitted and linucb rarely wins; but it IS the pick in some bars
  (live 1e3 b = 3 seeds 6-10 `linucb_honest x5`, RouterEval 5k cartel x1, LLMRouterBench), where it drags the best-bandit bar
  down. Any table quoting linucb_honest as a standalone rival (e.g. "contextual bandits fail at scale") is quoting a bug.
- Bias: FAVOURS MIDIAN. Fix: report it as broken / drop from pool, or alpha = 0 / random tie-break; it would still
  sit near flat online, below MIDIAN at 1e4 b = 3 (0.679 vs MIDIAN 0.796), so the headline ranking does not flip.

### F2. The one budget-matched rival that beats MIDIAN when honest is hidden from every figure (HIGH)

- `scripts/extra_figs.py:120` `HIDE_HALVING = True` ("TEMPORARY (2026-09-22, user request)") removes EVERY
  sequential-halving arm, including `sequential_halving(peer_reported=True)`, which DEVIATIONS.md:613-614 introduces as
  "the apples-to-apples control for MIDIAN: same budget, same peer-report channel ... no trusted observer" and which
  METHODS.md §3 lists as "reported (peer halving)". It is also not in A/B's `ARMS` or either pool
  (`scripts/condensed_figs.py:29-34`).
- Measured (calibrated bernoulli, b = 3, seeds 1-3):
  | cell | MIDIAN | peer halving | trusted halving | oracle |
  |---|---|---|---|---|
  | n = 1e3 honest | 0.784 | **0.834** | 0.834 | 0.848 |
  | n = 1e3 beta = 0.5 lsf cartel | **0.787** | 0.626 | 0.834 | 0.848 |
  | n = 1e4 honest | 0.796 | - | **0.862** | 0.845 |
  DEVIATIONS.md:625-627 already concedes "the residual gap to peer-reported halving at beta <= 0.25 is structural".
- So "MIDIAN beats every rival on the same probes" holds under the cartel but NOT honest, and the arm that shows this is
  off the figures. Bias: FAVOURS MIDIAN in the honest panels of A/B. Recommendation: restore peer halving to A/B (or
  state in the caption that it beats MIDIAN when honest and loses under the cartel). This is a presentation choice made on
  user request; flagged, not overridden.

### F3. "Trusted observer" is applied inconsistently: it excludes halving but every drawn probing rival is one (MEDIUM)

- Erratum 26 (CHANGES_AND_ERRATA.md:260-264) withdraws plain `sequential_halving` because "its tournament is scored by a
  trusted observer the setting never provides". But `flat_probe_argmax` (FIGURE_DATA_GUIDE 2.5: "observed directly by
  the router (a trusted observer)"), all five bandits, knn/mlp and flat_nsw read probe outcomes directly
  (`_est.probe_successes`, `_learned.probe_set`), and the runner hands every online method the TRUE executed outcome
  (`rte/run.py:145-147`). Only MIDIAN (every variant) and peer halving pay the report channel.
- Consequence: either the setting grants trusted probe outcomes (then trusted halving, 0.834-0.862, is a legitimate
  budget-matched rival and beats MIDIAN honest by +0.05-0.07), or it does not (then flat argmax and the bandit/learned pools are
  given an information advantage MIDIAN lacks, which biases AGAINST MIDIAN and makes MIDIAN's wins over them conservative).
  The paper should pick one reading and say it. As drawn: the flat/bandit comparison is conservative for MIDIAN (fair to
  rivals), the omission of trusted halving is favourable to MIDIAN.

### F4. Where MIDIAN beats flat probe argmax on the same probes, and why that is legitimate (FAIR; one LOW caveat)

- Measured (calibrated bernoulli, seeds 1-3):
  | cell | MIDIAN | MIDIAN w/o audits | MIDIAN w/o defenses | flat frozen | flat online |
  |---|---|---|---|---|---|
  | 1e3 b=3 honest | 0.784 | - | 0.763 | - | 0.751 |
  | 1e4 b=3 honest | 0.796 | 0.796 | 0.758 | 0.723 | 0.723 |
  | 1e4 b=3 cartel | 0.797 | 0.742 | 0.721 | 0.723 | 0.723 |
  | 1e4 b=1 honest | 0.692 | - | - | - | 0.667 |
  Live 1e5 b=3 honest (results/live_n100k): MIDIAN 0.81 / 0.84 / 0.86 vs flat online 0.70 / 0.74 / 0.81.
- Honest, MIDIAN == MIDIAN w/o audits exactly, so the whole honest gain over flat is verification at promotion: MIDIAN spends b-1 probes
  on everyone and re-spends the freed n*K probes on the few candidates forwarded up the tree, instead of spreading all
  b uniformly. Flat argmax with b probes has ~116 (1e3) to ~30,000 (1e5, b=1) agents tied at est = 1.0
  (DEVIATIONS.md:234-241, 606) and picks one blindly (winner's curse). Adaptive re-allocation of the same budget is a
  real algorithmic advantage -- the same one that makes halving win (F2) -- not a bug. At b = 1 there is nothing to
  verify (erratum 22) and the gap shrinks to ~0.025, consistent with the mechanism.
- Under the cartel the extra gain over MIDIAN w/o audits (0.797 vs 0.742) is the audits removing lying reporters; flat is immune to
  liars (never reads reports), so MIDIAN vs flat under the cartel is again the verification gain.
- Checked no leakage path: MIDIAN never touches S; its verification probes are fresh index-seeded instances; its audits'
  `probe_at` re-runs are used only for striking reporters, not for estimates (then `midian_a.py:32-37`; now `Midian._audit`).
- LOW caveat (already GUIDE 0.3 #6): MIDIAN spends 1.050 / 1.029 / 1.037 x n*K*b at b = 1 / 3 / 5 (measured, n = 1e3,
  cartel, seed 1) plus ~0.41 audit reports per task; rivals spend exactly 1.000. The extra probes only drive reporter
  exclusion, so honest they cannot help; bias negligible but should be in the caption.
- Also note: at Q = 300 on 16 families (~19 tasks/family) "online" gives flat nothing (1e4: frozen 0.723 = online 0.723),
  so the online/frozen distinction is not what separates MIDIAN from flat on live 1e4-1e5.

### F5. "Best learned router" is partly a declared-channel arm, and on bernoulli the declared channel is nearly the answer key (HIGH, labelling)

- `cluster_head_router` (needs {declared}, `cluster_head_router.py:42,50`) and `disrouter_cascade` (needs {declared, bus},
  `disrouter_cascade.py:15,23`) spend zero probes and learn nothing from outcomes (no `observe`). They are k-means /
  threshold heuristics over self-reports, not learned routers. Yet they are in `LEARNED` (`scripts/condensed_figs.py:29`)
  and win the cross-fit in bernoulli 1e7 (x100), replay 1e6 honest, LLMRouterBench honest b = 1/5 (GUIDE 0.3 #1).
- Why they win on bernoulli/replay honest: the honest declared channel there is `noisy_declared` = clip(S + N(0, 0.05))
  (`rte/backends/__init__.py:20-23`; `bernoulli.py:19,40-41`), i.e. near-oracle. Measured, n = 1e4, b = 3, honest:
  declared_argmax 0.858, cluster_head 0.831 vs oracle 0.845 and MIDIAN 0.796. On live the channel is the model's own
  self-rating (corr 0.36 with S; GUIDE 0.3 #18), which is why they do not win there.
- Under the cartel, disrouter's "cheap first" order (ascending MEAN declared skill, `disrouter_cascade.py:24`) puts every
  `inflate` liar (+0.4 in all families) at the back of the queue by construction: misroute-to-liar 0.227 vs
  declared_argmax 0.753 (n = 1e4, b = 3, beta = 0.5 lsf, seed 1). That is an artefact of the uniform-inflate lie model,
  not robustness (a `squat` or single-family lie would not be pushed back). It also depends entirely on tau = 0.7 vs the
  channel's calibration: with declared noise 0.3 it scores 0.210 honest at 1e3 (random is 0.43).
- Direction: mixed. Where these arms win, they RAISE the "best learned/declared router" bar using information MIDIAN does not read
  (conservative for MIDIAN). But the label misleads the reader about what learned routers achieve, and in bernoulli honest
  cells the bar is an answer-key artefact of the synthetic backend. Recommendation: move both to their own
  "declared-channel router" slot (or merge into declared argmax) and keep LEARNED = {knn, knn_online, mlp, flat_nsw};
  caption B that bernoulli/replay honest declarations are S + N(0, 0.05).

### F6. Hyperparameter sensitivity: tuning does not flip the ranking vs MIDIAN on live-like channels, with one exception (MEDIUM)

Calibrated bernoulli, n = 1e3, K = 16, b = 3, Q = 1000, seeds 1-3; MIDIAN = 0.784 honest / 0.787 cartel (beta = 0.5 lsf).
- `ucb_per_family` c (default sqrt(2), `ucb_per_family.py:11`; untuned): c = sqrt(2) 0.694, 0.3 0.723, 0.1 0.722 (same both
  regimes). +0.03 from tuning, still -0.06 below MIDIAN. Structural: t[f] starts at n*b, so the bonus c*sqrt(ln t / cnt)
  always prefers a fresh tied agent over a re-used one; at b = 1 any c > 0 makes UCB pure exploration within Q = 300
  (1e4 b = 1: c = 0.1 and 0.5 give identical 0.571 vs flat 0.667). Faithful to UCB1, just hopeless with n >> Q arms.
- `thompson_per_family`: 0.711 (no knob besides the Beta(1,1) prior). Fair.
- `linucb_honest` alpha: see F1 (alpha = 0: 0.714 at 1e3; 0.679 at 1e4 where alpha = 1 gives 0.479).
- `warm_start_bandit` n0 (the pool winner; pre-registered 5, tuned 0.5 on unreported live seeds 11-15, `grid.yaml:1029-1041`):
  - declared noise 0.05 (the bernoulli default): honest n0 = 0.5 / 2 / 5 / 20 -> 0.809 / 0.823 / 0.821 / 0.832, all ABOVE
    MIDIAN (the prior is near-oracle, F5); cartel 0.756 / 0.769 / 0.771 / 0.765, all below MIDIAN 0.787.
  - declared noise 0.3 (closer to live's self-ratings; `DN=0.3` backend_kwargs): honest n0 = 0.1 / 0.5 / 2 / 5 ->
    0.757 / 0.787 / **0.804** / 0.793 -- n0 = 2 beats MIDIAN (0.784) by +0.02; cartel 0.741 / 0.763 / 0.772 / 0.761 vs MIDIAN 0.787.
  - So the exception: HONEST, a declared+probe Thompson bandit with a moderate prior can match or beat MIDIAN whenever the
    declared channel carries real signal; MIDIAN's margin is a cartel-robustness result. Live data agree in direction but MIDIAN
    still wins there (live 1e5 b = 3 honest: warm_start 0.68 / 0.75 / 0.78 vs MIDIAN 0.81 / 0.84 / 0.86), because live
    self-ratings are poor. n0 = 0.5 was a reasonable, properly held-out tuning; the n0 = 5 drop is harmless.
- Bias: untuned c / alpha favour MIDIAN slightly but do not change the ranking. The n0 finding means the honest-panel claim
  "MIDIAN beats the best bandit" is channel-dependent and should be phrased as such.

### F7. Frameworks: faithful interception; the handicaps come from the pre-registered adapter, and are disclosed (MEDIUM)

Read: `rte/methods/frameworks/_common.py`, `workers/{langgraph,google_adk,openai_agents,crewai}_worker.py`, `NOTES_autogen.md`.
- Faithful: each worker builds the framework's own selection primitive (LangGraph `create_supervisor` handoff tools,
  ADK `sub_agents` + `transfer_to_agent`, OpenAI Agents handoffs + `on_handoff`, CrewAI hierarchical `Delegate work to
  coworker`, AutoGen `SelectorGroupChat`), aborts at the pick, temperature 0, fresh team per request. Name recovery is
  lenient (CrewAI regex `agent_\d{6}` on the coworker arg, `crewai_worker.py`), so the 7B is not punished for copying the
  long role string imperfectly. Prompts ask for exactly one delegate and forbid self-answering. No handicapping bug found.
- Erratum 29 (`_common.py:26-29, 371-378`): an ADK/OpenAI-Agents "Tool 'agent_x' not found" or MAF missing next_speaker
  is the supervisor's own invalid action -> non-pick -> declared argmax inside the shortlist, 0 under `success_strict`.
  That is the right classification (the same as an unparseable reply), and the lenient metric still scores the
  fallback agent, so it cannot push a framework below declared-argmax-in-shortlist. Fair.
- Adapter-driven handicaps (all disclosed, all favour MIDIAN):
  1. Pre-registered hashed TF-IDF, stable sort, no dedup: at n = 1e5 the top-10 is ten clones of one agent, so every
     framework scores 0.378889 in every seed and regime (GUIDE §0.1 G; erratum 25). That bar measures the retriever, not
     the framework. The dedup / MiniLM / SOTA / MIDIAN-cohort shortlists are drawn beside it, so the reader can see this.
  2. Supervisor = Qwen2.5-7B for all ten (`_common.py:20`); only Magentic-One has a 14B arm, and it is dropped from the
     shortlist figures (`scripts/shortlist_figs.py:47`). ADK's invalid tool calls are a 7B capability failure. A frontier
     supervisor would likely raise the 15-35 % "recovery" (METHODS §5b), but its ceiling is still the shortlist's
     best-in-list, so the ranking vs MIDIAN is bounded by retrieval, not by the supervisor.
  3. k = 10 shortlist; k sensitivity (5/10/20) exists only for LangGraph/AutoGen (`fw_k_sensitivity*`), not drawn.
- RouterEval task text "A task of family X (instance i)" (`_common.py:192`): the supervisor never sees the MMLU prompt
  (GUIDE 3.9 #3). Relative to MIDIAN this is FAIR -- MIDIAN, flat and the bandits also route per family only -- but it is weaker
  than `knn_router`, which on RouterEval embeds the real prompt (`_learned.vec` via `view.embedding`). State it.
- Advantages the adapter gives frameworks (conservative for MIDIAN): on live the numeric lie never reaches the prompt and the
  "Declared areas" clause states the TRUE specialty (erratum 27, GUIDE 3.9 #15); AutoGen's silent fallback to
  `participants[0]` after 3 failed selections is counted as a real pick (NOTES_autogen.md), and position 1 is the
  retriever's best.
- `llm_supervisor` (not a framework, not in A/B) is not "the frameworks' shared adapter" as METHODS §4 says: default
  k = 20 (`llm_supervisor.py:22`) vs 10, and its listing adds "Declared competence on this family: 0.xx" (`:33-34`), which
  no framework receives. LOW, documentation.

### F8. knn_router / mlp_router / flat_nsw_router: faithful; k = b and max_iter = 30 are not handicaps (LOW)

Bernoulli with SYNTHETIC prompt embeddings (family centroid + instance noise, d = 64, monkeypatched `_learned.vec` in
scratch only; MiniLM would not load on the login node within 2 min, so the live text geometry was not checked).
- knn, n = 1e3, b = 3, honest, seeds 1-3: knn (k = b) 0.692 == flat frozen 0.692 exactly (the k = b nearest probes are the
  agent's own same-family probes, so it reduces to the probe mean); k = 9 0.704; knn_online 0.733 vs flat online 0.751.
  k = b is the natural choice given b probes per family and is not a strawman. Ties -> lowest id, as flat.
- mlp, n = 300, b = 3, seeds 1-2: max_iter 30 -> 0.710; max_iter 150 -> 0.693; flat frozen 0.738. So the
  ConvergenceWarning at 30 is harmless: more training does not help (the one-hot per-agent input is a noisy restatement of
  the probe means). `random_state = 0` fixed across world seeds (GUIDE 2.8.2) is a minor variance understatement.
- flat_nsw_router: an approximate argmax of the flat probe means (inner product with one-hot(f), ef = 50). At n = 1e3 b = 3 it
  scores 0.657 vs flat frozen 0.692: the HNSW graph over estimates on the {0, 1/3, 2/3, 1} grid returns a tied agent chosen
  by graph geometry, sometimes not even a top-tied one. Faithful to "an ANN index router"; inherits flat's winner's curse.
- Budget: all three spend exactly n*K*b build probes, no reports, trusted outcomes (see F3). Same information as flat.

## What I checked and found FAIR

- Budget parity: every probing rival spends exactly n*K*b (`_est.probe_successes`, `_learned.probe_set`); trueskill
  0.989-0.999x (pairs a1 = a2 dropped, GUIDE 2.7.5); MIDIAN 1.029-1.050x (F4). Probes are index-seeded, so all methods see the
  identical k-th probe outcome of every (agent, family) (`world.py:311-320`, reset per method in `run.py:133`).
- Same online information: the runner hands every method's `observe` the true executed outcome (`run.py:145-147`);
  flat online, bandits and knn_online all use it; no rival ignores outcomes except the ones that have no online
  update by design (frozen flat, mlp, nsw, trueskill, the declared-only arms).
- Argmax direction: correct in every file (np.argmax on success estimates / UCB scores / Beta samples / mu).
- Ties: every argmax breaks ties to the lowest id, the same rule for flat, knn, ucb, declared argmax. Agent ids are
  independent of skill on bernoulli and live (`population.draw_profiles` draws i.i.d. per id; `low_skill_first` only
  labels liars, it does not reorder ids), so lowest-id is equivalent to a random tie-break. Not a handicap.
- Seeding: `view.rng` depends on (seed, needs) (`world.py:177`); Thompson/warm_start sample from it per fetch; no rival
  reuses one fixed seed across worlds except mlp's `random_state = 0` (F8).
- Warm-start prior uses the declared channel with the right orientation (alpha0 = n0*D, beta0 = n0*(1-D),
  `warm_start_bandit.py:14-17`); the n0 = 0.5 tuning was held out (seeds 11-15, never reported) -- proper.
- Declared argmax, random: trivially faithful.
- Cross-fitting of the pools (`seed_tables.crossfit`) avoids the max-of-noisy-means bias in the rivals' favour and against
  them alike.
- Flat argmax vs MIDIAN: MIDIAN's win on the same probes is adaptive re-allocation + audits, not leakage (F4).

## Summary (severity, direction)

| # | finding | sev. | bias |
|---|---|---|---|
| F1 | linucb_honest's UCB bonus selects lucky WEAK agents among ties; at/below random at n >= 1e4 (live 1e5: 0.37-0.42) | HIGH | favours MIDIAN (bounded by cross-fit) |
| F2 | peer-reported halving (the stated apples-to-apples control) beats MIDIAN honest (0.834 vs 0.784 at 1e3) but is hidden by HIDE_HALVING | HIGH | favours MIDIAN (honest panels) |
| F3 | "trusted observer" excludes halving but every drawn probing rival is a trusted observer | MEDIUM | mixed; must be stated |
| F4 | MIDIAN > flat on the same probes is legitimate (verification + audits); +3-5 % probes | FAIR / LOW | negligible |
| F5 | declared-only cluster_head / disrouter in "best learned/declared router"; bernoulli declared = S + N(0, 0.05) | HIGH (label) | raises the rival bar; mislabels |
| F6 | tuning c / alpha / n0 does not flip the cartel ranking; honest, warm_start n0 = 2 beats MIDIAN when declarations carry signal | MEDIUM | honest claim is channel-dependent |
| F7 | frameworks faithful; TF-IDF clones, 7B supervisor, k = 10 are adapter handicaps (disclosed) | MEDIUM | favours MIDIAN |
| F8 | knn k = b, mlp max_iter = 30 are not handicaps | LOW | none |
