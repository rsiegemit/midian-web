# RTE / MIDIAN condensed figures A, B, C, D, E, F, G, H: data guide

A reference for anyone who has to write about figures A, B, C, D, E, F, G and H in `figures/condensed_sample/`. For every bar, line
and marker it says what the number is, which grids and cells and seeds it comes from, how it is averaged, and how each
method and backend behind it is implemented. It was written from the code, not from the project's markdown docs. Where the
two disagree, the code is taken as correct and the disagreement is listed. Citations are `path:line` in `~/rte` at the
working tree of 2026-09-23 ~02:30 EDT; Part 6 (C and D) and every `scripts/condensed_figs.py` citation are against the
tree of ~12:00 EDT the same day. Figure values are those of the CSVs now in `figures/condensed_sample/` (A and B written
02:05–02:06, C and D 12:16, E–H 01:52). Result and seed counts are a snapshot from the same time: runs are still landing.

**How it was built.** Six independent passes, one per part, each re-reading the code and re-deriving plotted numbers from
the raw rows (`$RTE_DATA/results/<grid>/rows.csv` + `rows.d/`). Each pass ends with its own list of discrepancies. Part 0
(this page) gathers the ones that matter for the paper.

| Part | Covers |
|---|---|
| 1 Foundations | world, true skill S vs declared D, tasks, probes, the ledger, skill distributions, liars / β / cartel / lie modes, the four backends (live, bernoulli, replay, RouterEval + LLMRouterBench), the probe budget b, grid mechanics, seeds, pairing, aggregation and CIs, errata and asterisks, glossary |
| 2 Methods | every non-framework arm in A/B: MIDIAN, MIDIAN-V, MIDIAN-VA (deepest), oracle, flat probe argmax, declared argmax, random, the bandit pool, the learned-router pool, how "best learned" / "best bandit" is picked |
| 3 Frameworks and shortlists | the framework adapters, supervisor LLM, fallback, prefetch, every shortlist variant (TF-IDF, BM25, MiniLM, Qwen3 dense ± instructions, fusion + reranker, declared top-k, VA cohort), caches, which combinations exist |
| 4 Figures A and B | every bar, shade, star, error bar in A (live, n = 10^2…10^5) and B (all families, / oracle), with value tables |
| 5 Figures E, F, G, H | the two-stage shortlist pipeline, each figure's bars, dots, lines, stars, the G lift, H on RouterEval, with value tables |
| 6 Figures C and D | C (routing work per query vs n) and D (energy per query vs queries served): the ledger cache, MIDIAN-VA's build probes, the energy model of `scripts/energy.py`, the framework band, slopes and break-even tables |
| Appendix | per-row A/B table with seed counts and source grid |

---

## 0.1 The figures in one paragraph each

- **A — live RTE, specialist population, n = 10², 10³, 10⁴, 10⁵.** Mean task success per arm, honest (β = 0) and
  β = 0.5 low-skill cartel side by side. There are two versions. In `_allb` each budget b = 1 / 3 / 5 gets its own bar,
  shaded light, mid and dark. In `_stacked` the bars for the three budgets of one arm are overlapped: full-height bars
  drawn tallest first, not added segments. The arms are MIDIAN-VA, MIDIAN, flat probe argmax (online), best learned
  router, best bandit, declared argmax and random; the oracle is a dotted line. Error bars are 95 % percentile bootstraps
  over seed means, and there are none on the stacked version. For the five single arms, b = 3 values come from the
  per-cell bar CSVs and b = 1 / 5 values from the `va_b_*` and `rivals_b_*` grids. "Best learned router" and "best
  bandit" are cross-fitted from per-seed tables at every b: each seed's arm is chosen on the other seeds (§2.1.3). A `*`
  above such a bar means a pool candidate has not landed at that b. See §4.1.
- **B — every family at its largest pool, success / oracle.** The same arms and b layout, for live 10⁵,
  bernoulli 10⁷, replay 10⁶ (RouterBench models replayed on RouterBench categories), RouterEval (the 5,000-LLM leaderboard pool) and the
  LLMRouterBench pool. Every bar and its interval is divided by one number, the honest b = 3 oracle of that group. See §4.2.
- **C — routing work per query vs n.** Messages plus comparisons that the ledger charges per routed query, for
  n = 10² … 10⁷ on calibrated bernoulli (`bernoulli_scale_v5`, b = 3, honest), one line per arm, log-log. The legend
  classifies each arm's fitted log-log slope over n ≥ 10³ as "constant", "∝ log n" or "∝ n^s". MIDIAN-VA goes 25 → 80 and
  MIDIAN 46 → 161, a fixed step per tree level, so linear in depth ≈ log n (∝ log n). Flat probe argmax, declared argmax
  and the two bandits are exactly n (∝ n^1.00). The flat-NSW index router is a constant 50.
  It is a count, not an estimate, and there are no error bars (the counts are identical across seeds). See §6.2.
- **D — energy per query vs queries served T (an estimate, marked `*`).** MIDIAN-VA's probing build divided by T plus its
  per-query messages and comparisons, at n = 10³ / 10⁵ / 10⁷ (lighter to darker) and b = 1 / 3 / 5 (dotted, solid,
  dashed), against a grey band: the per-query supervisor energy of the ten agent frameworks measured live at n = 1,000
  (21–220 J, i.e. 1 to 10.7 supervisor-call equivalents, a latency ratio to AutoGen). Each × is a break-even point. Energies come from `scripts/energy.py` (a probe 4.04 J, a 7B supervisor call
  20.6 J, a message 1e-3 J, a comparison 1e-8 J, 700 W). See §6.3.
- **E — frameworks by shortlist across n (live, specialist).** For each n, one bar per shortlist: the unweighted mean
  over frameworks of each framework's seed mean. Solid is honest, hatched is cartel. The oracle is a dotted line and
  MIDIAN-VA on the whole population a solid line; both use honest values only. A `*` at the baseline marks an empty slot
  (not run). A `*` above a bar marks a framework rerun from erratum 28 still listed as outstanding. See §5.2.
- **F — n = 10⁵, every shortlist, best to worst.** Same bars; a black dot marks the best single framework. See §5.3.
- **G — gain over the pre-registered TF-IDF shortlist at n = 10⁵.** Per framework, shortlist minus TF-IDF, averaged over
  the frameworks both have. The whisker is 1.96·sd/√k across frameworks, a normal interval, not a t-interval. At 10⁵
  TF-IDF is a floor: 0.378889 for every framework, seed and regime (erratum 25). See §5.4.
- **H — RouterEval shortlists.** E's layout and E's shortlist slots on RouterEval, pool m = 10 / 100 / 1,000
  (strong-to-weak) plus the 5,000-LLM leaderboard. A slot not yet run is an empty `*`. See §5.5.

## 0.2 Definitions a caption needs (details in §1)

- **Success.** The fraction of the unit's Q tasks whose routed agent actually solved the task. On live this is a real LLM
  answer, graded. On bernoulli it is a coin flip with the agent's skill. On replay and RouterEval it is the recorded
  correctness of that model on that prompt.
- **Honest (β = 0).** No liars.
- **β = 0.5 cartel.** Half the agents lie. The liars are the round(βn) agents with the lowest mean true skill
  (`liar_select = low_skill_first`). With `lie_mode = inflate`, each liar declares its true skill + 0.4 in every family,
  clipped to 1, but still executes at its true skill. `collude = true` is set in every figure grid, so liars also corrupt
  reports: they report 1 for fellow liars and 0 for the top 20 % of honest agents they observe. See §1.3.
- **Probe budget b.** A method may spend at most n·K·b probes when it is built. A probe runs one agent on one instance of one
  family and returns 0/1 (a training-split prompt on RouterEval / LLMRouterBench; a fresh generated or synthetic instance on
  live / bernoulli / replay). Exceeding the budget only logs a warning (§1.5). MIDIAN-VA can spend a few % over;
  see 0.3.
- **Seed.** A new population, liar set and task stream, all drawn together. Every method in a (cell, seed) unit sees the
  same world, the same tasks and the same probe instances. That makes comparisons paired within a unit (§1.1, §1.6), with
  the RouterEval exceptions in §1.6.4 (pool fixed across seeds) and §1.10 #1 (family order of older rows).
- **Methods never see true skill S or the liar set.** Only the oracle and offline diagnostics read S.

## 0.3 Issues that can change what a reader concludes (with where they are explained)

Severity: **H** = can change a conclusion, **M** = comparability or labelling caveat that belongs in a caption.

| # | Issue | Figures | Sev. | Where |
|---|---|---|---|---|
| 1 | **The "best learned router" pool includes arms that read only the declared self-assessments and spend no probes** (`cluster_head_router`, `disrouter_cascade`). They win the cross-fitted pick in several B cells: bernoulli 10⁷ (all seeds, both regimes, b = 1 / 3), replay 10⁶ (honest at every b, cartel at b = 1) and LLMRouterBench honest at b = 1 / 5. | A, B | H | §2.8, §4.5 #2 |
| 2 | **Pool candidates are still landing.** The pool is the same at every b in a cell (`POOLS` minus `NOT_RUNNABLE`), but many candidates have no rows yet at some b; a `*` above the bar marks it. The `pool_fill_*` / `pool_seeds_n1000` grids that supply them are queued. The `*` flags only a candidate with no rows at all: at live 10³ b = 3 seven candidates have seeds 1–5 only, so seeds 6–10 choose among the rest (`warm_start_bandit` ×5, `linucb_honest` ×5). | A, B | H | §2.1.3, §4.5 #3 |
| 3 | **Frameworks whose rows were all quarantined (CrewAI, ADK) drop out of E–H means with no star.** Most non-TF-IDF bars average 8 frameworks (7 on RouterEval) against 9–10 for TF-IDF. RouterEval declared top-k has all 9. | E–H | H | §5.6 D1, D2 |
| 4 | **Different framework sets per bar.** At 10⁴ the honest bars include Magentic-One (the best framework there) but the cartel bars do not; without it, honest MiniLM equals cartel (0.4173). Magentic-One at the 10⁴ cartel is queued (`fw_live_n10k_cartel_magentic`). | E | H | §5.6 D2, §3.9 #5 |
| 5 | **ADK has no honest rows at live 10² / 10³.** Its Google ADK supervisor calls a non-existent tool named after the agent ("Tool 'agent_000030' not found"). Those units were refused as infrastructure errors past the erratum-28 threshold, so ADK is absent from the 10² / 10³ honest bars. The current adapter counts this error as a supervisor non-pick instead (erratum 29; `INVALID_ACTION`, `_common.py:29, 371-378`). The 33 units that failed this way (31 ADK, plus `re_sl_declared_1k` MAF seed 3 and `re_sl_declared_5k` OpenAI Agents seed 1) are queued in the focus pack plan. Whether ADK belongs in the means is an author decision. | E | M | §5.1.4, §5.6 D1 |
| 6 | **MIDIAN-VA can overspend the build budget** (5 % audit re-probes of level 0). On LLMRouterBench the mean is 1.050× / 1.033× / 1.041× n·K·b at b = 1 / 3 / 5; on bernoulli / replay / RouterEval-mmlu at b = 3 the mean is 0.92–0.96× (verification funding depends on the cell's candidate count) with per-cell maxima ≈ 1.04×; at b = 1 ≈ 1.05× everywhere. Rivals spend ≤ 1×. | A, B | M | §2.3.4, §1.5 |
| 7 | **At b = 1 MIDIAN-VA verifies nothing** (no probes are left for verification), so its b = 1 bar is MIDIAN-A with a cached root. It equals plain MIDIAN at b = 1 in bernoulli, replay, LLMRouterBench and live 10⁴. | A, B | M | §2.3, §4.5 #11 |
| 8 | **Live b = 3 bars average rows from several grids per seed, and those rows disagree** by up to 0.076 at n = 100 (e.g. `live_core_n100` vs `fw_live_n100`). Live n = 1,000 rows are not reproducible across reruns either; the cause is not determined. | A | M | §4.5 #5, §1.10 #2 |
| 9 | **Unequal seed counts within a group.** B's b = 5 bars use 51 / 36 seeds (bernoulli MIDIAN-VA) and 12–38 seeds (replay) against 100 at b = 1 / 3, and nothing marks this. | A, B | M | §4.5 #6 |
| 10 | **Only TF-IDF is not deduplicated.** Every other shortlist carries `dedup: true`. At 10⁵ TF-IDF is ten clones of one agent, so G's "gain over TF-IDF" includes the dedup gain. | E–G | M | §3.4.1, §5.6 D10–D11 |
| 11 | **Dedup is not a no-op on RouterEval under lying.** Liars whose top 5 all clip to 1.00 render identical descriptions: 4–34 % of the pool at β = 0.5. | H | M | §3.9 #1 |
| 12 | **RouterEval TF-IDF is not "degenerate".** It shortlists agents that claim the family, and under the cartel liars fill it (one family's top 10 was 100 % liars at 1k). The RouterEval supervisor sees only "A task of family X", never the prompt. | H | M | §3.9 #2–3 |
| 13 | **The reference lines in E–H are honest-only.** Hatched cartel bars are drawn against the honest MIDIAN-VA line (10⁵: 0.8356 honest vs 0.8278 cartel). | E–H | M | §5.6 D7 |
| 14 | **G's whisker is a normal interval**, about 17 % narrower than t at k = 8, and it measures spread across frameworks, not seed uncertainty. | G | M | §5.6 D4 |
| 15 | **Erratum-28 stars persist after the rerun lands** (they are keyed on the quarantine list until `logs/DONE_stage2` exists). One quarantined seed stars a whole bar. | E–H | M | §5.6 D9, §1.10 #10 |
| 16 | **Live rivals at b = 1 / 5 and the tuned bandit have no rows yet**, except `rivals_b_n10k` honest b = 1. `rivals_b_n100`, `rivals_b_n1000` and `tuned_wsb_*` are queued; `rivals_b_n100k` is pending. A's rival bars are therefore b = 3 only elsewhere. | A | M | §2.12 #5, §4.5 #10 |
| 17 | **"Cartel" in the figures means `liar_select = low_skill_first`.** `collude = true` is on in every grid, including the random-liar ones. | all | M | §1.10 #13 |
| 18 | **Live declarations are the model's own self-rating** (mean 0.686 vs true 0.419, correlation 0.36). On live, the description text includes an honest "Declared areas" clause that lying does not change (erratum 27), which helps every text shortlist. | E, F | M | §1.3, §3.9 #15 |
| 19 | **Supervisor picks are not bit-reproducible on the real fleet:** random replica per call, no seed sent, batched vLLM. Pick-level run-to-run variance is unmeasured. | E–H | M | §3.9 #8 |
| 20 | **RouterEval m ≤ 1,000 rows written before the family-order tie-break may come from the other of two subject orders**, so cross-grid pairing there is noisier than it looks. | H | M | §1.10 #1 |
| 21 | **D's advantage is over the frameworks, not over flat probing.** Flat probe argmax pays an n·K·b build (MIDIAN-VA 3–5 % more) and 1e-5 to 0.1 J per query (n comparisons), so its curve would sit on MIDIAN-VA's. The advantage over flat methods is C's routing work, and the flat-NSW index router is constant in C because it is centralised. | C, D | H | §6.4 #3–4 |
| 22 | **D's framework band is a live n = 1,000 measurement applied at every n**, pooled over all β and all three population shapes. Its upper edge is Magentic-One at 10.7 supervisor call-equivalents, a latency ratio to AutoGen rather than a count of calls. The mean latency is used, not the median the `energy.py` comments name, and there is no erratum-28 quarantine. The frameworks' retrieval scan over n descriptions and their n-message registration are not charged. | D | M | §6.4 #5–7 |
| 23 | **D mixes sources.** Build probes at n = 10⁷ are bernoulli ledger counts, each priced at the live specialist probe energy (4.04 J), while 10³ and 10⁵ use live ledgers. Every row's `build_source` says "ledger" either way. | D | M | §6.4 #1–2 |

Lower-severity documentation drift (stale docstrings, stale grid comments, METHODS.md / README.md vs code) is listed in each
part's final section.

---

## 1. Foundations: the world model, backends, experimental axes and aggregation

Everything below was checked against the code in `~/rte` (paths are relative to that repo root unless they start with
`$RTE_DATA`, which is `/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte`). Citations are `path:line`. Where a `.md`
document says something different from the code, the code is taken as correct and the mismatch is listed in §1.10.
Numbers quoted "from data" were computed read-only from the files named next to them on 2026-09-23.

---

### 1.1 The world (`rte/world.py`)

#### 1.1.1 Objects

| Object | What it is | Code |
|---|---|---|
| **agents** | integer ids `0..n-1`. What an agent *is* depends on the backend (§1.4). | `rte/world.py:239-252` |
| **families** | `K` task categories, indexed `0..K-1`, with a name list `world.families`. A task always belongs to exactly one family. | `rte/world.py:250` |
| **true skill `S[n,K]`** | float32 in [0,1]: the probability (bernoulli), measured accuracy (live), recorded accuracy (replay, RouterEval) of agent *a* on family *f*. Supplied by the backend's `true_skill()`. **Runner-only**: used for the oracle, liar selection (`low_skill_first`) and offline diagnostics; no method can read it (§1.1.6). | `rte/world.py:251-252` |
| **declared `D[n,K]`** | what agents *claim* about themselves. `D_honest = backend.declared(declared_source)`, then liars' rows are overwritten by `apply_lying`. Read-only copy `D_view` is what methods see. | `rte/world.py:257-259` |
| **liars** | bool mask `[n]`, chosen once per world by `select_liars` (§1.3). Runner-only. | `rte/world.py:255-256` |
| **demand** | probability vector over families used to draw the task stream: `uniform` (1/K) or `skewed` (Zipf(1)). Every grid behind figures A–H uses `uniform`. | `rte/world.py:279-285` |
| **ledger** | six counters `probes, reports, messages, hops, comparisons, tasks`; each has exactly one increment method. | `rte/ledger.py:5-24` |

#### 1.1.2 Construction and seeding

`World(n, K, dist, beta, liar_select, collude, seed, backend, lie_mode, declared_source, demand, backend_kwargs)`
(`rte/world.py:235-264`). All randomness goes through `stable_seed_32(*parts)` (blake2b of the `repr` of the parts,
masked to 32 bits; `rte/stable_hash.py:47-97`), so it is identical across processes. The streams:

| RNG | seeded by | used for |
|---|---|---|
| world rng | `(seed, "world", n, K, dist, backend)` `rte/world.py:244` | handed to the backend: bernoulli skill draws / calibrated row resampling, replay profile draws. The llm and routereval backends ignore it. |
| liar rng | `(seed, "liars")` `rte/world.py:256` | `liar_select="random"` only |
| honest-declaration noise | `(seed, "declared")` `rte/backends/__init__.py:20-23`, `rte/backends/llm.py:211` | `D = clip(S + N(0, 0.05))` |
| task stream | `(seed, "stream", K, demand)` `rte/world.py:289-290` | which family each task is from |
| task instance | `(seed, "inst", i, f)` `rte/world.py:292` | the concrete instance of task *i* |
| probe salt | `(seed, "probes")` `rte/world.py:262` | index-seeded probe instances (§1.5) |
| method view rng | `(seed, "view", sorted(needs))` `rte/world.py:177` | a method's own randomness (two methods with the same `needs` get the same stream) |
| llm population | `(seed, "profiles", n, K, dist)` `rte/backends/population.py:54` | live agent profiles |

Consequences worth knowing:
- The **task stream does not depend on n, dist, beta, liar_select or declared_source** — only on (seed, K, demand, Q).
  For a given seed and K, the live n = 100, 1,000, 10^4 and 10^5 cells route the *same* sequence of families and the same
  instance seeds. `rng.choice(size=Q)` is prefix-consistent (verified: the first 300 draws of a Q = 1000 stream equal a
  Q = 300 stream), so Q = 300 cells see the first 300 tasks of the Q = 1000 stream.
- The **liar set, lies, probes and tasks do not depend on the order in which methods run** (§1.1.5).
- Because `select_liars` returns an empty mask for `round(beta*n) = 0` (`rte/world.py:110-113`) and no other RNG uses
  `liar_select`, **β = 0 cells with `random` and `low_skill_first` are the same world**. The grid comment at
  `configs/grid.yaml:226-227` states they were verified bit-identical over 690 cells.

#### 1.1.3 Tasks (`rte/world.py:35-40, 287-292`)

`Task(id, family, instance)`. `World.tasks(Q)` draws `Q` family indices from `demand` and gives task *i* the instance
seed `stable_seed_32(seed, "inst", i, f)`. The backend regenerates the concrete problem from `(family, instance)`
(live: a Reasoning-Gym instance; replay / RouterEval: a recorded prompt row, see §1.4). **Methods receive the whole
`Task` in `fetch(task)`**, so they always know the task's family index (and its instance seed, and can read its text,
§1.1.6). Routing is therefore "pick an agent for a task of known family f".

#### 1.1.4 `execute` and the oracle (`rte/world.py:295-305`)

- `execute(a, task)` charges `ledger.tasks += 1` and returns the backend's 0/1 outcome of agent *a* on that task
  (plus the churn epoch rule, irrelevant for A–H because no grid behind A–H has churn — all resolved `churn=None`).
  **Liars execute at their true skill**; lying only touches declarations and reports (`rte/world.py:125-126`).
- `oracle(task) = argmax_a S[a, task.family]` (numpy `argmax`: ties go to the lowest agent id). The oracle row is a
  *realised* success on the same stream (`rte/run.py:122-128`), not `max S`. It is computed once per unit, before any
  method. Because S is an expectation (bernoulli), a measurement on a different instance set (live) or a train-split
  mean (RouterEval), a method can occasionally beat the oracle on a finite stream (negative regret).

#### 1.1.5 Pairing: one world, one stream, every method (`rte/run.py:131-175`, `rte/world.py:391-399`)

`run_unit(cell, seed, specs)` builds **one** `World` and **one** stream (`world.tasks(Q)`), runs the oracle line, then
for each method: `world.reset()` (zero the ledger, forget reporters' memories, set every probe index back to 0, undo
churn), `m.build(view, Budget(b))`, then for each task `m.fetch(task)` → `execute` → `m.observe(task, a, outcome)`.
So within a (cell, seed) unit every method faces the same agents, the same liars, the same declarations, the same task
sequence, and — because probe index k of (a, f) always maps to the same instance (§1.5) — the same probe outcomes.
Differences between methods in a unit are therefore paired. Caveats on pairing *across* units/grids are in §1.10
(RouterEval family-order and live drift items).

#### 1.1.6 What a method can see (`rte/world.py:163-230`, `rte/methods/base.py:1-40`)

A method gets a `View` constructed from its declared `needs ⊆ {"declared", "probe", "reports", "bus"}`
(`rte/world.py:29`). Any access outside `needs` raises `AccessError`; any attribute not defined on `View` raises
("S and liars are never exposed", `rte/world.py:229-230`).

| Always available | `n`, `K`, `families` (names), `ledger`, `rng`, `text(f, inst, probe)` (prompt text of an instance), `embedding(f, inst, probe)` (backend-supplied embedding or None) |
|---|---|
| `needs "declared"` | `view.declared` → read-only `D` (after lying) |
| `needs "probe"` | `probe`, `probe_many`, `probe_text`, `probe_at` (§1.5) |
| `needs "reports"` | `report_channel(j, a, outcome)`, `report_many(reporters, agents, outcomes)` (§1.3.4) |
| `needs "bus"` | `bus.send / send_many / broadcast` — only charge `messages` |
| via the runner | `fetch(task)` gets the `Task` (family + instance); `observe(task, agent, outcome)` gets the **realised 0/1 outcome** of every task it routed — the online-learning channel. |

Never visible to a method: `S`, the liar mask, `beta`, `liar_select`. Two honest caveats: (a) this is enforced by
Python attribute access, not a sandbox (`View._w` is the world object); a grep of `rte/methods/` finds no method
touching `_w`, `true_skill`, `.liars` or `._S`. (b) The framework adapters reach the llm backend through
`rte.backends.llm.current_backend()` (`rte/methods/frameworks/_common.py:179-204`) to read the agents'
self-descriptions, family descriptions and task text — the only out-of-View access in `rte/methods/`.

#### 1.1.7 Ledger and what "communication" means (`rte/ledger.py`, `rte/run.py:95`)

`probes` and `tasks` are charged by the world on every probe/execution, `reports` by the report channel, and
`messages / hops / comparisons` by methods themselves. "Total communication" is defined as
`probes + reports + messages + tasks` (`rte/run.py:95`); `hops` and `comparisons` are recorded but not summed into it.

#### 1.1.8 Churn (brief; not used by A–H)

`World.churn(frac)` replaces `round(frac*n)` agents in place, re-draws their liar status at rate β, recomputes D, and
the first task routed to a replaced, never-re-observed agent scores 0 (`rte/world.py:10-16, 267-276, 295-299`).
`reset()` restores the original population for the next method. Only `churn_n1000*` grids use it.

---

### 1.2 Skill distributions (`dist`)

The `dist` axis names the *population shape*. Its meaning is backend-specific:

#### 1.2.1 bernoulli, uncalibrated (`rte/world.py:47-65`)

| dist | S draw |
|---|---|
| `specialist` | each agent has exactly 3 random "good" families: U(0.70, 0.95) there, U(0.05, 0.30) elsewhere (`_profiles.pick_k_per_agent`, `rte/backends/_profiles.py:12-17`) |
| `heavy_tail` | `0.05 + 0.90 * Beta(0.5, 3)` i.i.d. per cell |
| `bimodal` | 20 % of agents U(0.75, 0.95) everywhere, 80 % U(0.20, 0.40) everywhere |
| `correlated` | 4 family groups (`f % 4`): one U(0.15, 0.90) per (agent, group) + N(0, 0.05), clipped |
| `iid_uniform` | U(0.20, 0.90) i.i.d. |

**Every bernoulli grid behind the figures passes `calibrate_from`, which overrides `dist` entirely**: each agent's
S-row is a uniformly resampled (with replacement) row of the measured live matrix
`$RTE_DATA/populations/specialist_n1000_K16_seed1/S.npy` (`rte/backends/bernoulli.py:24-30`). The `dist` column then
still reads `specialist` but is only a label. That source matrix (1000 × 16) has 887 distinct rows, mean 0.419 and a
mean-over-families of the per-family max of 0.845 (its `summary.json`), which is why the bernoulli oracle sits at
≈0.845 for n ≥ 10^3 (`bernoulli_scale_v5/matrix_success.csv`: 0.845–0.846).

#### 1.2.2 live / llm (`rte/backends/population.py:51-82`, `configs/models.yaml`)

The shape is realised by how **profiles** `{model, specialty, tool}` are drawn; S is then *measured* (§1.4.1). The
ladder (`configs/models.yaml:13-19`, ascending size): Qwen2.5-0.5B, Qwen2.5-1.5B, Gemma-2-2B, Qwen2.5-3B, Qwen2.5-7B,
Gemma-2-9B, Qwen2.5-14B. Bands (`configs/models.yaml:25-28`, `population.py:42-48`): small = ≤ 1.5B {0.5B, 1.5B};
big = ≥ 7B {7B, 9B, 14B}; "large" (measured with 60 instead of 200 probes) = ≥ 9B {Gemma-9B, Qwen-14B}; the `python`
tool only for models ≥ 3B (`population.py:25-31`).

| dist | profile draw |
|---|---|
| `specialist` | model uniform over all 7; 3 random specialty families |
| `heavy_tail` | 10 %: a random big model, specialty = all families; 90 %: a random small model, specialty = none |
| `bimodal` | 20 %: the smallest big model (Qwen2.5-7B), specialty = all; 80 %: the smallest small model (Qwen2.5-0.5B), specialty = none |
| `correlated` | model uniform; each of 4 family groups is a specialty with prob 0.5 |
| `iid_uniform` | model uniform; each family a specialty with prob 0.5 |

On a non-specialty ("handicapped") family the agent's prompt has no family description, no worked example and no tool
(`rte/backends/prompts.py:1-7, 21-32`, `population.py:85-90`). An agent's behaviour on family f is fully determined by
its **signature** `(model, handicapped?, tool, max_tokens)`: at most 7 × 2 = 14 distinct behaviours per family
whatever n is. Measured on `seed1`, n = 1000 populations: specialist 887 distinct S-rows (mean S 0.419),
heavy_tail 5 distinct rows (mean 0.261), bimodal 2 distinct rows (mean 0.237).

#### 1.2.3 replay (`rte/backends/replay.py:26-54`; DEVIATIONS.md 2026-09-02)

An agent = one of the 11 RouterBench models + a per-category mask; a masked category executes as the category's
*weakest* model. Models are ranked by mean accuracy over the K used categories; "strong half" = top 6, "weak half" =
bottom 5 (`half = (M+1)//2`, `replay.py:29-30`).

| dist | draw |
|---|---|
| `specialist` | model uniform over 11; 3 random unmasked categories, the other K−3 masked |
| `heavy_tail` | 10 %: the single best model (gpt-4-1106-preview), unmasked; 90 %: uniform weak-half model, 1 unmasked category |
| `bimodal` | 20 % uniform strong-half, 80 % uniform weak-half; no masks |
| `correlated` / `iid_uniform` | group-level / per-cell Bernoulli(0.5) masks |

With K = 64 the specialist mask leaves 61 of 64 categories at the weakest model, so mean S is tiny (0.076 at n = 1000,
seed-independent draw checked) while the per-family max is 0.790 (gpt-4). Replay results in B/C are "all shapes
pooled" = the per-seed mean over specialist, heavy_tail and bimodal.

#### 1.2.4 RouterEval / LLMRouterBench (`rte/backends/routereval.py`)

Here `dist` is not a shape but the **name of the pool**: `pool = pool or dist` (`routereval.py:22`). For
`dataset: mmlu` it selects RouterEval's hard-setting pool config `strong_to_weak | all_strong | all_weak`
(`routereval.py:36-44`); for `leaderboard_mmlu` and `llmrouterbench` the value `all` is only a label (the pool is fixed).
The population is the real pool itself and is **the same for every seed** (§1.6.4).

---

### 1.3 Liars

#### 1.3.1 How many and who (`rte/world.py:108-120`)

`m = round(beta * n)` liars.
- `liar_select = random`: `m` agents uniformly without replacement (rng `(seed,"liars")`).
- `liar_select = low_skill_first`: the `m` agents with the **lowest mean true skill across families**
  (`argsort(S.mean(1), kind="stable")[:m]`; ties → lowest ids). Deterministic given S. At β = 0.5 on the live
  specialist n = 1000 seed-1 population the liar half has mean S 0.274 vs 0.565 for the honest half.

#### 1.3.2 The declared-channel lie (`rte/world.py:123-141`)

`apply_lying(D_honest, liars, lie_mode)`; honest rows are untouched.
- `inflate` (**every grid behind A–H**): `D[liar] = clip(D_honest[liar] + 0.4, 0, 1)` (`DELTA_INFLATE = 0.4`,
  `rte/world.py:31`). Preserves the liar's ranking of its own families and adds 0.4.
- `max`: `D[liar] = 1` everywhere (only the `lie_max_*` grids, 2026-09-23).
- `squat`: `D[liar, top-3-demand families] = 1` (unused by the figure grids).

#### 1.3.3 What honest declarations look like (`declared_source`)

- `programmatic`: `clip(S + N(0, 0.05))` — an honest, slightly noisy copy of true skill
  (`rte/backends/__init__.py:20-23`; llm version `rte/backends/llm.py:209-212`). bernoulli, replay and RouterEval
  *only* have this (their `declared()` ignores `source`: `bernoulli.py:42-43`, `replay.py:97-98`, `routereval.py:86`).
- `self_described` (llm backend only; all live cells in A, B, E–H): each signature's own model is asked to rate its
  competence on the family 0–1 given the family description and an example problem (`prompts.py:66-73`), parsed by
  `parse_rating` (`prompts.py:56-63`; unparseable → 0.5). Cached per population in `D_self_described.npy`.
  It is coarse and over-confident: on specialist n = 1000 seed 1 it takes 11 distinct values
  {0, .2, .5, .6, .7, .75, .8, .85, .9, .95, 1}, mean 0.686 vs mean S 0.419, corr(S, D) = 0.36
  (heavy_tail: mean 0.526 vs S 0.261; bimodal 0.581 vs 0.237). After `inflate` a specialist liar's mean D is ≈ 0.882.
- Self-description *text* (llm): one ≤ 70-word paragraph per agent written by its own model, plus the appended clause
  "Declared areas: <its TRUE specialty list>" (`rte/backends/llm.py:232-251`). It is generated once per population
  (`descriptions.json`, path has no β) and **is not touched by the lie** — erratum 27 (§1.8).

#### 1.3.4 The report channel and collusion (`rte/world.py:336-383`)

Decentralised methods learn through reports: a method passes `(reporter j, agent a, outcome)` and gets back the value
j reports. If `collude` is **False**, or j is honest, the true outcome comes back. If `collude` is True and j is a liar:
1. liar j reports **1** for any liar a (vouching);
2. liar j reports **0** for the top `ceil(20 %)` of *honest* agents ranked by j's observed mean (in `report_many`, the
   mean within that batch per reporter; in the scalar path, j's cumulative observations);
3. otherwise the true outcome.

Every grid behind A–H has `collude = true` (the default, `configs/grid.yaml:10`) — **random liars collude too**. The
only thing that distinguishes the regimes called "cartel" is `liar_select`.

#### 1.3.5 Definition: "β = 0.5 cartel"

In code and in every figure script, a *cartel* is `liar_select = low_skill_first` (`fw_variant_numbers.regime`,
`scripts/fw_variant_numbers.py:55-59`). Concretely, "β = 0.5 cartel" =
`beta = 0.5, liar_select = low_skill_first, collude = true, lie_mode = inflate`: the half of the population with the
lowest mean true skill (i) declares `D + 0.4` (clipped) and (ii) as reporters vouches 1 for each other and reports 0 on
the best-observed 20 % of honest agents; (iii) executes tasks honestly at its true skill. Regime labels used downstream:
`beta0` (no liars, one cell), `beta{β}_random`, `beta{β}_cartel`, and `cartel` ≡ β = 0.5 low_skill_first
(`fw_variant_numbers.py:55-59`; `bar_figs.py:24-26`).

#### 1.3.6 `lie_text` (framework parameter, erratum 27)

Not a world axis: a framework-adapter flag (`rte/methods/frameworks/_common.py:138-145, 166-177`). When on, the
trailing "Declared areas:" clause of *every* agent's description is re-derived from `view.declared` (top 3 families, or
all above `claim_threshold`), so liars' text claims match their inflated D. The LLM prose is unchanged (partial text
lie). It appears only in `fw_live_n100k_lietext*` grids, which `shortlist_figs.py:38` explicitly excludes.

---

### 1.4 Backends

A backend supplies `n, K, families, true_skill(), declared(source), execute(a, task), execute_many(agents, fams,
inst), stats()` (+ churn hooks) (`rte/backends/__init__.py:1-13`); `make()` dispatches on
`bernoulli | replay | llm | routereval` (`rte/backends/__init__.py:26-39`). The grid axis value for the live backend
is `llm`; figure scripts call that family "live".

#### 1.4.1 live (`backend: llm`) — `rte/backends/llm.py`, `prompts.py`, `families.py`, `tools.py`, `rte/llm_client.py`

- **Agent**: a profile (model from the 7-model ladder, specialty set, tool) — §1.2.2. Real vLLM-served models,
  temperature 0, `seed=0` (`rte/llm_client.py:180-182`). n = 10^5 means 10^5 agent ids mapped onto ≤ 14 behaviours per
  family and (specialist) at most 7 × C(16,3) = 3,920 distinct self-description prompts (erratum 25's count).
- **Families / tasks**: K = 16 Reasoning-Gym generators (`families.py:28-31`: basic_arithmetic, chain_sum,
  letter_counting, syllogism, family_relationships, gcd, lcm, prime_factorization, number_sorting, simple_equations,
  count_bits, number_format, time_intervals, word_sequence_reversal, binary_alternation, calendar_arithmetic);
  basic_arithmetic and chain_sum use eased settings (`families.py:51-54`). A task is one generated problem, the
  instance seed being the dataset seed.
- **Success**: the agent's answer (last `<answer>…</answer>`, else last line; one optional python-tool round for tool
  holders) scored by the Reasoning-Gym verifier, `correct = score >= 0.99` (`families.py:112-114`, `llm.py:97-152`).
- **S is measured, per signature**, on a fixed project-wide instance set `stable_seed_32("measure", fam, r)`,
  r < 200 (60 for Gemma-9B / Qwen-14B), not on task instances (`llm.py:163-202`). Cached as
  `$RTE_DATA/populations/<dist>_n<n>_K<K>_seed<seed>/S.npy`. Available populations (dir count): specialist n = 100 and
  1,000 seeds 1–20, 10^4 seeds 1–5, 10^5 seeds 1–3; heavy_tail / bimodal n = 100, 1,000 seeds 1–20 and 10^4 seeds 1–3;
  correlated / iid_uniform n = 1,000 seeds 1–5.
- **LLM cache / memo** (`rte/llm_client.py:1-12, 127-136, 217-250`): every request is keyed by blake2b of
  `(model, messages, max_tokens)`; answers are stored in per-process SQLite shards under `$RTE_DATA/cache/` and every
  process reads all shards. Agents sharing a signature send byte-identical prompts, so they get identical answers on
  the same instance: **same-signature agents are exact clones**. This is what makes n = 10^5 affordable (the probe
  build for a (n, dist, seed) costs `n·K·b` lookups, not new generations, once warmed) and is also the root of
  erratum 25.
- **Supervisor**: framework arms (not agents) call a supervisor LLM, `SUPERVISOR = "Qwen/Qwen2.5-7B-Instruct"`
  (`rte/methods/frameworks/_common.py:20`), over a top-k shortlist; covered in the frameworks part.

#### 1.4.2 bernoulli — `rte/backends/bernoulli.py`

Synthetic: outcome ~ Bernoulli(S[a, f]) from a hash of (seed, agent, family, instance) (`bernoulli.py:45-51`). Agents
are rows of S; with `calibrate_from` (all figure grids) each row is resampled from the live specialist n = 1000 seed-1
S (§1.2.1), so a bernoulli agent is "a clone of a random live specialist agent's skill profile with i.i.d. Bernoulli
outcomes". Declared = `clip(S + N(0, 0.05))` for both sources. n goes up to 10^7 (`bernoulli_scale_v5`, K = 16).
Note the module docstring's own policy: "Never a headline number" (`bernoulli.py:1-2`).

#### 1.4.3 replay (RouterBench) — `rte/backends/replay.py`

Data: `$RTE_DATA/data/routerbench_cells.npz`, built by `scripts/02_download_routerbench.py` from RouterBench's 0-shot
pickle: category = `eval_name`, categories with ≥ 60 prompts kept (67 survive), score binarised at ≥ 0.5; 11 models,
36,093 prompts. With K = 64 the 64 largest categories are used (`replay.py:67-69`). An agent = (model, mask) (§1.2.3);
`execute` = the recorded outcome of the agent's model (or the category's weakest model if masked) on prompt
`instance % n_prompts[f]` (`replay.py:100-110`). Probes and tasks draw from the **same** prompt pool. S = the
(model or weakest-model) accuracy over the whole category. n up to 10^6 in the figures (`replay_scale_v5`).
Declared = `clip(S + N(0, 0.05))`.

#### 1.4.4 RouterEval (`dataset: mmlu`, `leaderboard_mmlu`) and LLMRouterBench (`dataset: llmrouterbench`) — `rte/backends/routereval.py`

All three are recorded per-prompt 0/1 scores of **real, distinct LLMs**; n must equal the pool size (asserted); no
resampling of agents.

| dataset | pool (n) | families (K) | train / test split | S |
|---|---|---|---|---|
| `mmlu` | RouterEval "hard" pools m ∈ {10, 100, 1000} × config {strong_to_weak, all_strong, all_weak} (Open-LLM-Leaderboard models) | the K = 16 MMLU subjects with the most train prompts, parsed from the prompt text (`routereval.py:68-78`) | RouterEval's own (11,215 train / 1,430 test prompts; 6,253 / 790 fall in the 16 subjects) | mean train score |
| `leaderboard_mmlu` | all 5,000 `leaderboard_old` LLMs, synthetic names `llm0..llm4999` | 16 largest `harness_hendrycksTest_*` subjects | fixed 80/20 per subject, rng 0 (`routereval.py:52-65`) | mean train score |
| `llmrouterbench` | 20 models (7–9B open models, e.g. Qwen3-8B, DeepSeek-R1-0528-Qwen3-8B, gemma-2-9b-it, …) | 15 datasets (aime, math500, mathbench, humaneval, mbpp, livecodebench, bbh, korbench, kandk, mmlupro, gpqa, finqa, medqa, emorynlp, meld) | global 70/30, rng 0; score binarised ≥ 0.5 (`routereval.py:24-31`) | mean train score |

**Probes read TRAIN prompts, tasks read TEST prompts** of the family (`routereval.py:88-110`), so the oracle (argmax of
train accuracy) is not guaranteed to be the test-best agent. Declared = `clip(S + N(0, 0.05))`; no self-descriptions
exist (frameworks get the declaration vector rendered as text, `_common.py:188-191`). Churn is a no-op.
From the backend (seed 1): mmlu strong_to_weak mean S 0.551 / 0.545 / 0.545 at m = 10 / 100 / 1000, per-family max
mean 0.876 / 0.799 / 0.890; leaderboard 5,000: mean 0.542, per-family max mean 0.917; LLMRouterBench: mean 0.481,
per-family max mean 0.725.

---

### 1.5 The probe budget `b`

- **Formula**: `Budget(b).total_probes(n, K) = n · K · b` (`rte/budget.py:14-15`) — "b probes per (agent, family)" on
  average. Every probing method receives the same `Budget(b)`; `b` is the grid's `b` axis (default 3,
  `configs/grid.yaml:14`). Examples: live 10^5 at b = 3 → 4.8 M probes; bernoulli 10^7 at b = 1 → 1.6 × 10^8.
- **What a probe is**: executing agent *a* on a *fresh* instance of family *f* and returning 0/1, charged
  `ledger.probes += 1` (`rte/world.py:311-320`). The k-th probe of (a, f) uses instance seed
  `probe_seed(salt, a, f, k)` (`rte/world.py:100-105`); `reset()` puts every k back to 0 per method
  (`rte/world.py:395`), so **every method sees the same outcome for the k-th probe of (a, f)** (and on live they share
  memoised generations). `probe_text` also returns the instance seeds (so a router can read the probe prompt via
  `view.text(f, inst, probe=True)`); `probe_at(a, f, k)` re-runs a past instance for audits, charged as a probe, index
  untouched (`rte/world.py:385-389`). Live probe instances, measurement instances and task instances are three
  different seed families; RouterEval probes are train prompts.
- **How it is checked**: only in `run_method`, after `build`: if `build_probes > n·K·b` a `[WARNING]` is **logged**;
  nothing is enforced and the row is written (`rte/run.py:135-136`). Run-time probes (during `fetch/observe`) are not
  checked at all; they appear in `probes_per_task`. From data (build_probes / nKb): flat probe argmax, KNN/MLP routers,
  LinUCB, warm-start bandit, plain MIDIAN and MIDIAN-V spend exactly 1.000; peer/trusted sequential halving 0.833 on
  LLMRouterBench; **MIDIAN-A / MIDIAN-SHA ≈ 1.04–1.06; MIDIAN-VA 1.050 / 1.033 / 1.041 at b = 1 / 3 / 5 on LLMRouterBench, but a mean of 0.92–0.96 at b = 3 on bernoulli / replay / RouterEval-mmlu (max ≈ 1.04)** (the 5 % audit re-probes, documented in
  `rte/methods/midian_a.py:8` as "≤ 1.05×"; observed up to 1.0625 on `routereval_mmlu`); `verify_on_claim` spends 0 at
  build and ≈ 2.4 probes per task at run time on live 10^5 (range 1.4–4.1 across cells); declared argmax and random spend 0.
- **What changing b does**: more probes per cell → sharper estimates for probe users; declaration-only arms are
  unaffected (the condensed figures draw declared argmax / random once, at b = 3, `scripts/condensed_figs.py:37`).
  For the verified MIDIAN variants the build splits b into `b0 = b − 1` level-0 probes and verification probes
  `e = (b − b0)·n / C` (`rte/methods/midian.py:104-112`); at **b = 1, b0 = 1 and e = 0, so verification is unfunded:
  MIDIAN-V ≡ plain MIDIAN and MIDIAN-VA ≡ MIDIAN-A** (also stated at `configs/grid.yaml:232-233`). The condensed A/B
  b = 1 bars inherit this.
- Budget-less arms still carry a `b` value in their row and a distinct row id per b.

---

### 1.6 The experimental unit and grid mechanics (`rte/run.py`, `configs/grid.yaml`)

#### 1.6.1 Cell, unit, row

- **CELL axes** (`rte/run.py:16`): `backend, n, K, dist, beta, liar_select, collude, declared_source, lie_mode,
  demand, b, Q`. A **cell** is one point of their Cartesian product (`rte/run.py:62-71`); `backend_kwargs` (e.g.
  `calibrate_from`, `dataset`) and `churn` ride along.
- A **unit** = (cell, seed): one `World`, one stream, the oracle line, then every requested method (§1.1.5)
  (`rte/run.py:155-175`). A method that raises is logged and skipped; the unit continues.
- **row_id** = blake2b-128 of the CELL values + backend_kwargs + method + params + seed (+ churn) (`rte/run.py:74-77`).
  It does not include the grid name; results are separated by directory `$RTE_DATA/results/<grid>/`. Each row is
  written atomically as `rows.d/<rid>.json` (`rte/run.py:171-174`); a rerun of the same unit overwrites the same files.
- **Resume**: a (cell, seed, method) is skipped if its rid is in `rows.d` or `rows.csv` (`rte/run.py:240-255`).
- **Consolidation** (`rte/run.py:178-225`): `rows.d/*.json` are merged into `rows.csv` (additive, dedup on `rid`,
  keep last), optional prune; skipped if `.merge_owner` exists unless forced. Note `rte.analyze.load` calls
  `consolidate` (`rte/analyze.py:52-58`), so figure scripts that use it rewrite `rows.csv`;
  `scripts/fw_variant_numbers.load` instead reads `rows.d` + `rows.csv` directly and dedups on `rid` and then on
  `(n, b, dist, beta, liar_select, seed, method, params)` (`fw_variant_numbers.py:24-32`).

#### 1.6.2 Grid resolution

`blocks(cfg, grid)` (`rte/run.py:49-59`): `defaults` < `mirror_of` source (chains allowed) < the grid's own keys <
each `blocks:` entry. Defaults (`configs/grid.yaml:6-19`): backend bernoulli, K 16, liar_select random, collude true,
declared_source programmatic, lie_mode inflate, demand uniform, b 3, Q 1000, seeds 1–5, methods `all` + six paired
variants. `methods: all` = every method file except LLM-only ones off the llm backend (`rte/run.py:31-46`);
`allow_llm_methods: true` lets framework arms run on routereval. Seeds spec `"1-10"` → `[1..10]` (`rte/run.py:22-28`).

#### 1.6.3 Resolved grids behind figures A–H (exact values from `rte.run.blocks`)

All rows: `K = 16` unless noted, `collude = true`, `lie_mode = inflate`, `demand = uniform`, no churn.

| grid (yaml line) | backend | n | dist / pool | β | liar_select | decl. source | b | Q | seeds |
|---|---|---|---|---|---|---|---|---|---|
| `live_core_n100` (105) | llm | 100 | spec, heavy, bimodal | 0, .1, .25, .5 | random, lsf | prog, self | 3 | 1000 | 1–5 |
| `live_f1_n1000` (106) | llm | 1000 | spec, heavy, bimodal | 0, .1, .25, .5 | random, lsf | prog, self | 3 | 1000 | 1–5 |
| `fw_live_n100` / `fw_live_n1000` (112/109) | llm | 100 / 1000 | spec, heavy, bimodal | 0, .1, .25, .5 | random | self | 3 | 1000 | 1–10 |
| `fw_live_n100_lowskill` / `_n1000_lowskill` (146/145) | llm | 100 / 1000 | spec, heavy, bimodal | .5 | lsf | self | 3 | 1000 | 1–10 |
| `learned_n100` (272), `variants_f1` (152), `learned_f1` (265) | llm | 100 / 1000 / 1000 | spec, heavy, bimodal | 0, .1, .25, .5 | random, lsf | self | 3 | 1000 | 1–10 |
| `learned_n10k` (291) | llm | 10^4 | spec | 0, .25, .5 | random, lsf | self | 3 | 300 | 1–3 |
| `live_n10k_v2` (130) | llm | 10^4 | spec | 0, .25 | random | self | 3 | 300 | 1–3 |
| `fw_live_n10k_cartel` (515), `live_n10k_cartel_random` (1027) | llm | 10^4 | spec | .5 | lsf | self | 3 | 300 | 1–3 |
| `learned_n10k_beta01` (986) | llm | 10^4 | spec | .1 | random, lsf | self | 3 | 300 | 1–3 |
| `live_n100k` (460), `live_n100k_fill` (979) | llm | 10^5 | spec | 0, .25, .5 | random, lsf | self | 3 | 300 | 1–3 |
| `live_n100k_beta01` (989) | llm | 10^5 | spec | .1 | random, lsf | self | 3 | 300 | 1–3 |
| `va_b_n100` / `va_b_n1000` (1047-8) | llm | 100 / 1000 | spec | 0, .5 | lsf | self | **1, 5** | 1000 | 1–10 |
| `va_b_n10k` / `va_b_n100k` (1049-50) | llm | 10^4 / 10^5 | spec | 0, .5 | lsf | self | **1, 5** | 300 | 1–3 |
| `rivals_b_n100/n1000/n10k/n100k` (1068-71) | as their `va_b_*` twin | | | | | | 1, 5 | | |
| `routereval_mmlu` (316) | routereval, `mmlu` | 10, 100, 1000 | s2w, all_strong, all_weak | 0, .25, .5 | random, lsf | prog | 3 | 1000 | 1–5 |
| `routereval_mmlu5k` (359) | routereval, `leaderboard_mmlu` | 5000 | all | 0, .25, .5 | random, lsf | prog | 3 | 300 | 1–3 |
| `va_b_routereval5k`, `rivals_b_routereval5k` (1051/1072) | routereval, `leaderboard_mmlu` | 5000 | all | 0, .5 | lsf | prog | 1, 5 | 300 | 1–3 |
| `llmrouterbench_pool` (584) | routereval, `llmrouterbench` | 20 | all | 0, .25, .5 | random, lsf | prog | 3 | 1000 | 1–5 (K = 15) |
| `va_b_llmrouterbench`, `rivals_b_llmrouterbench` (1052/1073) | same | 20 | all | 0, .5 | lsf | prog | 1, 5 | 1000 | 1–5 (K = 15) |
| `bernoulli_scale_v5` (217) | bernoulli, calibrated | 10, 100, 1000, 10^4 / 10^5 / 10^6 / 10^7 | spec (label) | 0; .25, .5 | random; random, lsf | prog | 3 / 3 / 1, 3 / 1, 3 | 1000 | 1–1000 / 1–500 / 1–200 / 1–100 |
| `va_b_bernoulli_1e7`, `rivals_b_bernoulli_1e7` (1053/1074) | bernoulli, calibrated | 10^7 | spec | 0, .5 | lsf | prog | **5** | 1000 | 1–100 |
| `replay_scale_v5` (240) | replay | 10–10^4 / 10^5 / 10^6 | spec, heavy, bimodal | 0; .25, .5 | random; random, lsf | prog | 3 / 3 / 1, 3 | 1000 | 1–1000 / 1–200 / 1–100 (K = 64) |
| `va_b_replay_1e6`, `rivals_b_replay_1e6` (1055/1075) | replay | 10^6 | spec, heavy, bimodal | 0, .5 | lsf | prog | **5** | 1000 | 1–100 (K = 64) |
| `fw_routereval_small` / `_1k` / `_5k` (970/503/528) | routereval | 10, 100 / 1000 / 5000 | 3 configs / 3 configs / all | 0, .5 | random, lsf | prog | 3 | 1000 / 1000 / 300 | 1–5 / 1–5 / 1–3 |
| `tuned_wsb_n100/n1000/n10k/n100k` (1060-3) | as their `va_b_*` twin | | | 0, .5 | lsf | self | 3 | | |
| `pool_fill_n100` / `_n10k` / `_n100k` (1105-8) | llm | 100 / 10^4 / 10^5 | spec | 0, .5 | lsf | self | 1, 3, 5 | 1000 / 300 / 300 | 1–10 / 1–3 / 1–3 |
| `pool_fill_routereval5k`, `pool_fill_llmrouterbench` (1109/1112) | as their `va_b_*` twin | 5000 / 20 | all | 0, .5 | lsf | prog | 1, 3, 5 | 300 / 1000 | 1–3 / 1–5 |
| `pool_fill_bernoulli_1e7`, `pool_fill_replay_1e6` (1115/1119) | bernoulli / replay | 10^7 / 10^6 | spec / 3 shapes | 0, .5 | lsf | prog | 1, 3, 5 | 1000 | 1–100 |
| `pool_seeds_n1000` (1125) | llm | 1000 | spec | 0, .5 | lsf | self | 3 | 1000 | **6–10** |

(lsf = low_skill_first; spec = specialist; prog = programmatic; self = self_described.) The live framework-shortlist
grids (`fw_live_n*_{dd,em,sota,verified_va,…}`) mirror their source grid cell for cell (same n, β, liar_select,
seeds, Q); they are resolved in the frameworks part. For bernoulli and replay, β = 0 is run with `random` only and the
b = 1 rows at 10^6 / 10^7 come from the scale grids, the b = 5 rows from `va_b_*`/`rivals_b_*`
(`scripts/condensed_figs.py:151-183`).

The `pool_fill_*` grids hold only the candidates of "best learned router" / "best bandit" that no other grid ran in
that cell at that b, so that every b of a cell has the same pool (comment at `configs/grid.yaml:1099-1104`). Their
method lists differ per grid and per b block (`:1105-1122`). `pool_seeds_n1000` gives the seven live 10^3 b = 3
candidates that `live_f1_n1000` ran on seeds 1–5 their seeds 6–10 (`:1123-1125`). `tuned_wsb_*` runs
`warm_start_bandit[n0=0.5]` at b = 3. `scripts/seed_tables.py:65-72` reads all three families of grids for the
cross-fitted pools (§2.1.3); `bar_figs.py` does not.

Row-count snapshot of the b-axis grids at ~02:30 on 2026-09-23 (rows.d files): `va_b_n100` 80, `va_b_n1000` 80,
`va_b_n10k` 24, `va_b_n100k` 6, `va_b_routereval5k` 24, `va_b_llmrouterbench` 40, `va_b_bernoulli_1e7` 202,
`va_b_replay_1e6` 452, `rivals_b_llmrouterbench` 280, `rivals_b_replay_1e6` 1,420, `rivals_b_bernoulli_1e7` 80,
`rivals_b_n10k` 24 (b = 1, honest). `rivals_b_routereval5k` has an empty directory; `rivals_b_n100`, `rivals_b_n1000`,
`rivals_b_n100k`, every `tuned_wsb_*`, every `pool_fill_*` and `pool_seeds_n1000` have no results directory yet.
`pool_fill_{routereval5k,llmrouterbench,replay_1e6}` are pending in SLURM. `pool_fill_bernoulli_1e7` and
`rivals_b_bernoulli_1e7` are pending as 20 seed-block jobs each (10 seeds × 2 betas; `$RTE_DATA/logs/launch_pool_fill.txt`,
`launch_rivals_b.txt`). `pool_fill_n100`, `pool_fill_n10k`, `pool_seeds_n1000`, `rivals_b_n100`, `rivals_b_n1000`,
`tuned_wsb_*` and `pool_fill_n100k` at b = 1 and 3 are queued in the focus pack plan
(`$RTE_DATA/logs/focus_pack_plan.tsv`). `pool_fill_n100k` at b = 5 is three SLURM jobs (47904099–47904101) that wait
on the MIDIAN-VA 10^5 pack (`--dependency=afterany:47887198`, logged in `launch_pool_fill.txt`). In the condensed A/B figures an absent budget is drawn as `*` (§1.8.4).

#### 1.6.4 What a seed is

A seed re-draws everything that is random in the world: the population (live profiles; bernoulli row resampling;
replay profiles), the honest-declaration noise, the random liar set, the task stream and task instances, the probe
instances, and each method's view rng (§1.1.2). Exceptions: on **RouterEval / LLMRouterBench the pool, S, and (for
`low_skill_first`) the liar set are identical in every seed** — a seed there only re-draws declaration noise, random
liars, tasks and probes. On live, the self-rating and the measured S of a *signature* are the same in every
population (fixed measurement set), so seeds change which signatures are present and in what proportion. Seeds with the
same number but different n are different populations (the profile seed includes n).

#### 1.6.5 The metrics row (`rte/run.py:98-108, 150-152, 157-174`)

Cell columns + `seed, grid, method, params, backend_kwargs, n_agents, n_liars`, world summary columns prefixed
`skill_` (`skill_mean`, `skill_per_agent_std`, `skill_p90_p10`, `skill_family_max_mean`, `skill_excess_ratio`,
`skill_excess_ratio_family`, plus backend stats), and:

| column | definition |
|---|---|
| `success` | mean 0/1 outcome over the Q routed tasks. Route-to-many (a list from `fetch`): majority of the executed agents' outcomes, ties → 0 (`rte/run.py:111-115`) |
| `success_late`, `n_late` | mean over the last `min(500, max(1, Q//4))` tasks (250 at Q = 1000, 75 at Q = 300) |
| `success_by_block` | per-100-task means |
| `oracle_success` | the unit's oracle-line success; `regret = oracle_success − success` |
| `misroute_to_liar` | fraction of tasks whose (first) routed agent is a liar |
| `build_{probes,reports,messages,hops,comparisons}`, `build_total_comm` | ledger after `build` (`total_comm` = probes + reports + messages) |
| `{probes,reports,messages,hops,comparisons,tasks}_per_task`, `total_comm_per_task` | run-phase ledger / Q |
| `wall_clock_build`, `wall_clock_per_task`, `wall_clock_per_task_total` | seconds; memo-dependent, not used as costs (`rte/analyze.py:27`) |
| `method_stats` | JSON of the method's `stats` (e.g. framework picks / fallbacks) |

The oracle row has zero build/run communication except `tasks_per_task = 1`.

---

### 1.7 Aggregation conventions (general machinery)

- **Labels** (`rte/analyze.py:20-24, 60-86`): `label = method` if params are `{}`, else `method[k=v,…]` (sorted, floats
  `%.3g`), then `ALIAS`: `flat_probe_argmax → flat_probe_argmax_frozen`, `flat_probe_argmax[online=True] →
  flat_probe_argmax_online`, `knn_router[online=True] → knn_router_online`, `midian[cached=True,verify=True] →
  midian_v`, `midian[cached=True,r=5,verify=True] → midian_v_r5`, `sequential_halving[peer_reported=True] →
  sequential_halving_peer`, `midian[stratify=True] → midian_stratified`, and the two churn-mode halving labels.
  `scripts/seed_tables.py:26-29` builds the same label for the b = 1/5 rows and the per-seed tables.
- **Unit of replication = the seed.** The figure CSVs (`scripts/bar_figs.py:78-87`, `condensed_figs.py:169-175`, `seed_tables.py:48-62`,
  `scale_matrix.py:41-52`) first average each label's rows **per seed** (over shapes where a family pools them, and
  over duplicate rows of the same cell from different grids), then report the mean of the per-seed means.
- **95 % CIs are percentile bootstraps over seeds, B = 2000**, not t-intervals: `extra_figs.ci`
  (`scripts/extra_figs.py:89-97`, rng 0; resamples seeds when the series is seed-indexed), `fw_variant_numbers.ci`
  (`fw_variant_numbers.py:62-65`, rng 0), `rte.analyze.boot` (`rte/analyze.py:88-94`, rng 12345). With 3 seeds (live
  10^4/10^5, leaderboard 5k) the bootstrap has only 10 distinct resamples, so the interval is essentially the seed
  range; with 1 seed it collapses to the point. The one non-bootstrap interval in the condensed set is figure G's
  whisker: `1.96 · sd / √k` across frameworks (`scripts/shortlist_condensed.py:111`).
- **Regimes** (`fw_variant_numbers.regime`, `bar_figs.REGIMES`): β = 0 is one regime (`beta0`); for bar CSVs the β = 0
  row set is filtered to `liar_select = random` when present (`bar_figs.py:80-81`), because the two liar selections are
  the same world at β = 0.
- **Paired comparisons**: `rte.analyze.paired` (`rte/analyze.py:108-129`) — per cell, pivot seed × label, delta
  `ref − rival` per seed, bootstrap CI + sign test, `WITHIN_FLOOR` if |mean delta| ≤ MIDIAN's seed envelope.
  `scripts/paired_gaps.py:27-70` (b = 3 only, not a condensed figure): per condition, `midian_va − rival` per unit (unit =
  seed, or seed|shape where shapes are not pooled), 95 % bootstrap CI of that difference, "win/loss/tie" by CI sign;
  requires ≥ 2 shared units.
- **Figure filters**: `extra_figs.excluded` (`scripts/extra_figs.py:111-135`) drops MIDIAN variants with r ≠ 10 or
  δ ≠ 1/3, MIDIAN-SH / -SHA, trusted-observer `sequential_halving`, `route_to_k_majority`, the LLM-descent ablation,
  online-off and cohort/churn variants, and — while `HIDE_HALVING = True` (line 120, "TEMPORARY 2026-09-22") — every
  label containing "halving", including peer-reported halving.

---

### 1.8 Errata and asterisks (`CHANGES_AND_ERRATA.md`)

#### 1.8.1 Erratum 25 — clone-filled framework shortlist (`CHANGES_AND_ERRATA.md:241-258`)

Same-signature agents have identical, memoised self-descriptions and answers (§1.4.1), so the pre-registered TF-IDF
top-10 (stable sort, lowest ids) fills with copies of the best-matching text: bimodal has 2 and heavy_tail 5 distinct
descriptions at any n; specialist has 3,920, giving ~4.4 distinct signatures in the top-10 at 10^3, ~2.6 at 10^4 and
exactly one at 10^5. Those cells measured the retriever, not the framework. Fix: labeled variant `dedup: true` (one
agent per distinct text), rerun into separate `*_dd` grids; RouterEval descriptions are all unique and unaffected.
The pre-registered numbers stay as quoted. The document's statement that the 10^5 TF-IDF row is a single clone agrees
with the code path (`descriptions()` is per agent but identical prompts produce one memoised text).

#### 1.8.2 Erratum 26 — trusted-observer halving withdrawn (`CHANGES_AND_ERRATA.md:260-264`)

`sequential_halving` (no params) is scored by a trusted observer the setting does not provide; only
`sequential_halving[peer_reported=True]` is a legitimate rival. Rows remain in the grids; `scale_matrix.py:42` and the
`DO_NOT_ADD` set drop the label.

#### 1.8.3 Erratum 27 — the lie does not touch description text (`CHANGES_AND_ERRATA.md:352-365`)

`apply_lying` edits only the D matrix; `descriptions.json` is per population with no β in its path, and its
"Declared areas:" clause lists the agent's true specialty (`rte/backends/llm.py:241-247`). Therefore every
text-retrieval shortlist (TF-IDF, dedup, BM25, MiniLM, dense, hybrid, sota) is **identical across all liar regimes**;
its only β-dependence is the declared-argmax fallback when a framework fails to pick. Cross-regime robustness of text
arms is an artefact of the threat model; only declared-channel arms are attacked by the lie. `lie_text` (§1.3.6) is
the partial remedy.

#### 1.8.4 Erratum 28 — framework rows that measured declared argmax, and the asterisks (`CHANGES_AND_ERRATA.md:368-386`)

Broken library symlinks in 7 framework conda envs made workers crash on some nodes; the adapter then fell back to
declared argmax (`_common.py:389, 398`) and wrote normal-looking rows. CrewAI and ADK were hit hardest (~2/3 of units)
and were biased *upward* (declared argmax scores 0.619 honest). Fix: envs restored, 4,269 rows moved to
`results/<grid>/quarantine/`, their units rerun, and `fetch` now fails the unit past 2 % infra errors.

**Asterisks**. Two different `*` markers exist:
1. *Shortlist figures E–H* (`scripts/shortlist_figs.py:81-99`, `shortlist_condensed.py`): a framework × shortlist
   bar is starred if `(grid, framework, dist, regime)` appears in `pending_reruns()` (`fw_variant_numbers.py:35-44`).
   That set is every row of `$RTE_DATA/results/quarantine_units.tsv` (grid, method, dist, beta, liar_select, seed)
   mapped to regimes, **unless** `$RTE_DATA/logs/DONE_stage2` exists (then empty); if only `DONE_stage1` exists it is
   restricted to the `fw_live_n{100,1000}[_lowskill]_sota` grids. On 2026-09-23 neither marker exists and the tsv has
   3,572 unit rows, so every combination listed there is starred **regardless of whether its rerun has already
   landed**, and a single quarantined seed stars the whole (grid, framework, dist, regime) bar.
2. *Condensed A/B* (`scripts/condensed_figs.py:10, 105-107, 112`): a `*` at the baseline = that budget has no rows yet
   for an arm that exists in the cell; a `*` above a "best learned router" / "best bandit" bar = the pool still misses a
   candidate at that b (`INCOMPLETE POOL` in the CSV `chosen` column). Both are missing-data markers, unrelated to
   erratum 28.

#### 1.8.5 Rival note 8d (`CHANGES_AND_ERRATA.md:388-398`)

The warm-start bandit (prior Beta(n0·D, n0·(1−D)), n0 = 5, on self-described claims) scores higher under the cartel
than honest at live 10^4/10^5 (0.743 → 0.783, 0.738 → 0.778) because liars' D ≈ 1 removes their prior failures; the
tuned n0 = 0.5 arm (`tune_wsb_n1000`, seeds 11–15) is reported beside it.

---

### 1.9 Glossary

| term | meaning in code |
|---|---|
| **n** | number of agents in the world (`World.n`). live: agent profiles; bernoulli: resampled skill rows; replay: (model, mask) agents; RouterEval/LLMRouterBench: the real pool size m |
| **K** | number of task families (16 live, bernoulli, RouterEval; 64 replay; 15 LLMRouterBench) |
| **family** | a task category: a Reasoning-Gym generator (live), a RouterBench `eval_name` (replay), an MMLU subject (RouterEval), a dataset (LLMRouterBench), or `famNN` (bernoulli) |
| **task / Q** | one routed request of known family; Q = tasks per unit (1000, or 300 for live 10^4/10^5 and leaderboard 5k) |
| **stream** | the Q tasks of a unit; same for every method in the unit, and for every n/dist/β at a given seed |
| **true skill S** | per-(agent, family) success probability / measured accuracy; hidden from methods |
| **declared D** | per-(agent, family) claim, honest = S + N(0, .05) (programmatic) or the model's self-rating (self_described), liars +0.4 |
| **declared_source** | `programmatic` or `self_described` (live only) |
| **self-description** | live agent's LLM-written paragraph + "Declared areas: <true specialty>", read by frameworks |
| **probe** | one charged execution of agent a on a fresh index-seeded instance of family f; returns 0/1 |
| **b** | probe budget per (agent, family): build budget n·K·b |
| **report** | a peer's relayed probe outcome; corrupted if the reporter is a colluding liar |
| **β (beta)** | liar fraction, `round(β n)` liars |
| **liar** | agent whose D is inflated and who, if `collude`, corrupts its reports; executes honestly |
| **honest** | non-liar; "honest regime" = β = 0 |
| **liar_select** | `random` or `low_skill_first` (lowest mean S first) |
| **collude** | liars corrupt reports (vouch 1 for liars, 0 for top-20 % honest); true in every A–H grid |
| **cartel** | liar_select = low_skill_first (with collude); "β = 0.5 cartel" = half the population, the least skilled |
| **lie_mode** | `inflate` (+0.4, used throughout A–H), `max`, `squat` |
| **oracle** | routes every task to argmax_a S[a, family]; realised success on the same stream |
| **success** | fraction of the Q tasks whose routed agent's outcome is 1 |
| **regret** | oracle_success − success |
| **misroute_to_liar** | fraction of tasks routed to a liar |
| **signature** | (model, handicapped, tool, max_tokens) — determines a live agent's answers; same signature = clone |
| **shape / dist** | population shape (§1.2); on RouterEval the pool config; on bernoulli-calibrated only a label |
| **pool m** | RouterEval pool size (10, 100, 1000 hard-setting pools; 5,000 leaderboard) |
| **shortlist** | the top-k candidate agents a framework's supervisor chooses among (frameworks part) |
| **cell** | one combination of the 12 CELL axes |
| **unit** | (cell, seed): one world, one stream, all methods |
| **seed** | replicate index; re-draws population, noise, liars, stream, probes (§1.6.4) |
| **grid** | a named experiment in `configs/grid.yaml`; results in `$RTE_DATA/results/<grid>/` |
| **row / rid** | one (cell, method, params, seed) result; rid = blake2b of those |
| **regime** | beta0, beta{β}_random, beta{β}_cartel, cartel (= β .5 lsf) |
| **b = 1 / 3 / 5 bars** | same arm at different probe budgets; never pooled |
| **`*`** | E–H: above a bar, erratum-28 rerun listed as outstanding; at the baseline, slot not yet run. A/B: at the baseline, budget not yet run; above a pooled bar, pool candidate not yet run |

---

### 1.10 Discrepancies / open questions

1. **Older RouterEval MMLU rows come from one of two family orders (affects `dataset: mmlu` only).**
   `RouterEvalBackend._families` ranks the subjects by train-prompt count and breaks ties by subject name
   (`rte/backends/routereval.py:72`; test `tests/test_routereval_family_order.py`). "High school biology" and
   "philosophy" tie at 248 train prompts, at family indices 9 / 10. Rows written before the tie-break took their order
   from Python set iteration, which varies with `PYTHONHASHSEED` (under hash seeds 0–11 the order flips for 6, 8, 9 and
   11), so about half of them come from the other order. The subject set is the same either way; only the two indices
   swap. Task instances, probes and declaration noise are keyed by family index, so rows of the same (cell, seed) from
   different processes can come from two slightly different worlds. Observed: in `routereval_mmlu/rows.csv` 240 of 270
   units have rows whose `oracle_success` disagrees (spread up to 0.013), and `fw_routereval_1k` 28 of 30 units.
   Averages are unbiased. Within one unit (one process) pairing holds; pairing *across* grids or reruns on the mmlu
   pools (e.g. `fw_routereval_*` vs `routereval_mmlu`, `re_sl_*`) is noisier than it looks for rows written before the
   tie-break. `leaderboard_mmlu`, `llmrouterbench`, live, bernoulli and replay are deterministic (`routereval_mmlu5k`,
   `llmrouterbench_pool`: 0 units with disagreement).
2. **Live n = 1000 rows are not bit-reproducible across reruns.** `fw_live_n1000`: 54/120 units, `variants_f1`:
   64/240 units carry more than one `oracle_success` value (spread ≤ 0.006); duplicate cells across grids differ
   (plain `midian` in `fw_live_n1000` vs `variants_f1`: 39 % of seed-pairs differ, max 0.011; vs `live_f1_n1000` up
   to 0.075 over all shared methods). n = 100, 10^4, 10^5 and the va_b grids show no oracle drift. Cause not determined
   (candidates: memo entries regenerated non-deterministically, code changes between runs); figure CSVs average such
   duplicates per seed.
3. **Build-budget contract vs code.** `CONTRACT.md` says probing methods spend "AT MOST" n·K·b in build; MIDIAN-A /
   -SHA spend up to 1.0625× and MIDIAN-VA up to ≈ 1.04× per cell (audit re-probes; mean 1.033× at b = 3 on LLMRouterBench), and `run.py:135-136` only logs a warning. The
   `midian_a.py:8` docstring's "≤ 1.05×" is also exceeded on small pools (1.0625 on `routereval_mmlu`).
4. **README / docstrings say three backends** (`README.md:62`, `rte/world.py:3`); there are four (`routereval`).
5. **README: "bernoulli: synthetic S with five population shapes, calibrated to the measured live S"** — with
   `calibrate_from` the shape is ignored; every figure grid resamples rows of one live specialist population.
6. **README: "S is measured per signature (200 probes)"** — 60 for the ≥ 9B models (`llm.py:186-187`,
   `models.yaml:27`).
7. **`View` docstring says `dist` is always available** (`rte/world.py:165`); `View.__init__` never sets it and
   `view.dist` would raise `AccessError`.
8. **`_common.py:142` comment** says honest agents' declared "is true skill plus 0.05 noise"; on the live
   `self_described` channel (where `lie_text` is used) honest D is the model's coarse self-rating (mean 0.686 vs S
   0.419), not S + noise.
9. **Figure G whisker**: `shortlist_condensed.py` docstring says "95 % t-interval across frameworks", the code uses
   `1.96 · sd / √k` (normal), narrower than t for k ≈ 9–10.
10. **Erratum 28 counts vs the quarantine file**: the erratum says 2,904 units were rerun; `quarantine_units.tsv` has
    3,572 unit rows (it also lists `*_sota`, `*_em`, `*_verified*` and RouterEval-framework grids). Whether the tsv
    grew after the erratum was written is not determined. Because `pending_reruns()` keys only on the DONE markers, the
    E–H asterisks do not reflect which reruns have landed.
11. **Stale grid comment**: `fw_live_n1000` comment "(v2 0.4: 5 seeds, Q=1000 (was 3/300))" (`configs/grid.yaml:111`)
    vs actual `seeds: 1-10`.
12. **`bernoulli.py:1-2` says bernoulli is "never a headline number"**, while the condensed figure B includes a
    bernoulli 10^7 bar and C/D include bernoulli cells. A policy question for the author, not a code error.
13. **"Cartel" naming**: random liars also collude (`collude = true` everywhere), so "cartel" in figures means
    *low-skill-first selection*, not "collusion on vs off". Readers may assume the latter.
14. **Open**: live oracle is argmax of S measured on a fixed 200/60-instance set; how often the oracle agent is not
    the best on the actual task instances was not quantified here.

---

## 2. The non-framework methods in figures A and B, plus the oracle and MIDIAN-VA reference lines in E, F and H

All paths are relative to `/n/home02/rsiegelmann/rte`. Every claim below was checked against the code on 2026-09-23. Where
METHODS.md, DEVIATIONS.md or CHANGES_AND_ERRATA.md disagree with the code, the code is described here and the disagreement
is listed in §2.12.

---

## 2.1 Which methods appear in A and B, and how they get there

### 2.1.1 The seven arms that are drawn

`scripts/condensed_figs.py` draws exactly the arms in `ARMS` (`scripts/condensed_figs.py:31-33`):

| key in code | legend label | colour | what the key resolves to |
|---|---|---|---|
| `midian_va` | MIDIAN-VA | green | method `midian_va`, params `{}` (§2.3) |
| `midian` | MIDIAN | red | method `midian`, params `{}` (plain MIDIAN, §2.2) |
| `flat_probe_argmax_online` | flat probe argmax (online) | blue | method `flat_probe_argmax`, params `{online: true}`. The alias comes from `rte/analyze.py:19-20` (`FLAT_ON`, `ALIAS`) |
| `best_learned` | best learned router | orange | the cross-fitted best of the `LEARNED` pool in this cell, regime and b (§2.1.3) |
| `best_bandit` | best bandit | purple | the cross-fitted best of the `BANDIT` pool plus `warm_start_bandit[n0=0.5]`, in this cell, regime and b (§2.1.3) |
| `declared_argmax` | declared argmax | grey | method `declared_argmax`, params `{}` (not cached) |
| `random` | random | light grey | method `random` |

The **oracle** is not an arm. It is the dotted horizontal line (`scripts/condensed_figs.py:116`). In B every bar is divided by
the oracle's mean (`norm=True`).

Rows whose label starts with `fw_` (frameworks) are removed at load time (`scripts/condensed_figs.py:46`; `seed_tables.py:39` for the pooled arms). Every label also
goes through the do-not-add filter `extra_figs.excluded` (`scripts/condensed_figs.py:56,168,182`). Because `ARMS` and the two
pools list every drawable label, anything outside them is never drawn in A or B, whether or not it is excluded. That covers
MIDIAN-V, MIDIAN-A, verify_on_claim, cnp_self_bid, declared_softmax, gossip, referral, sequential halving and the rest.
No condensed figure draws them; they are in the per-family bar figures (`figures/bars/`, `scripts/bar_figs.py`).

### 2.1.2 Where each budget's numbers come from

- **The five single arms at b = 3** (MIDIAN-VA, MIDIAN, flat probe argmax, declared argmax, random). From
  `figures/bars/<family>.csv` (`scripts/condensed_figs.py:44-57`). `scripts/bar_figs.py` writes those files from the
  grids listed in `LIVE_GRIDS` (`scripts/bar_figs.py:29-31`) and the equivalents for the other families.
- **The same arms at b = 1 and b = 5.** From the `va_b_*` grids (MIDIAN-VA only) and the `rivals_b_*` grids (the
  budget-matched rivals) (`scripts/condensed_figs.py:151-176`; grids at `configs/grid.yaml:1047-1056` and `1068-1075`).
  For bernoulli and replay, b = 1 comes from the `bernoulli_scale_v5` and `replay_scale_v5` matrices
  (`scripts/condensed_figs.py:177-183`).
- **"Best learned router" and "best bandit" at every b.** From per-seed tables built straight from the raw rows by
  `seed_tables.tables()` (`scripts/seed_tables.py:65-81`, attached to each cell at `condensed_figs.py:188-189`). Per cell
  it reads: live, `LIVE_GRIDS[n]` + `va_b_*` + `rivals_b_*` + `pool_fill_*` + `tuned_wsb_*` + `pool_seeds_*`;
  RouterEval 5,000, `routereval_mmlu5k` + the b-grids; LLMRouterBench, `llmrouterbench_pool` + the b-grids; bernoulli 10^7
  and replay 10^6, their scale grids + the b-grids. The `pool_fill_*`, `pool_seeds_n1000` and `tuned_wsb_*` grids are
  queued and have no rows yet (§1.6.3).
- **Budgetless arms.** `declared_argmax` and `random` are drawn only from the b = 3 data, as one bar each (`BUDGETLESS`,
  `scripts/condensed_figs.py:37,74`).

### 2.1.3 How "best learned router" and "best bandit" are chosen (`arms_at`, `crossfit`)

Both are **cross-fitted**: no bar is the maximum of noisy means over the seeds it reports.

- **Pools** (`POOLS`, `scripts/condensed_figs.py:34`):
  - `LEARNED = [knn_router, knn_router_online, mlp_router, flat_nsw_router, cluster_head_router, disrouter_cascade]`
    (`:29`);
  - `BANDIT = [ucb_per_family, thompson_per_family, warm_start_bandit, linucb_honest, trueskill_per_family]`, plus
    `warm_start_bandit[n0=0.5]` (`:30, 34`).
- **The pool is the same at every b of a cell.** `arms_at` (`:60-75`) takes the pool minus `NOT_RUNNABLE(family, n)`
  (`:35-36`): `trueskill_per_family` at n ≥ 10^5, `mlp_router` at n ≥ 5,000, and `knn_router`, `knn_router_online`,
  `mlp_router` on bernoulli and replay (no prompt text).
- **Per-seed tables.** `seed_tables.tables()` gives, for each (cell, regime, b), a seed × arm table of mean success
  (`seed_tables.py:48-62`). Rows of one (seed, arm) from several grids are averaged, as `bar_figs` does. β = 0 counts as
  honest whatever its `liar_select` tag; the cartel is β = 0.5 with `low_skill_first` (`:44-45`). Live keeps
  `dist = specialist` and the self-described channel (`:76-78`). Replay pools the three shapes per seed and keeps only
  seeds where the arm ran on all three (`:55-58`).
- **Cross-fitting** (`crossfit`, `seed_tables.py:84-94`). For each seed s, among the arms that ran on s, the arm with
  the highest mean over the **other** seeds is picked, and its success **on s** is s's score. The bar is the mean of
  those per-seed scores. The CI is the 95 % percentile bootstrap over the same per-seed scores (`extra_figs.ci`,
  `condensed_figs.py:69`). A cell with fewer than 2 scored seeds gets no bar (`:68`).
- **The CSV `chosen` column holds the pick counts**, e.g. `warm_start_bandit x5; linucb_honest x5` (`:70`). The picked
  arm can differ between seeds, regimes and b values.
- **Incomplete pools.** If a candidate has no column in the table at that b, `chosen` ends with
  `| INCOMPLETE POOL, missing <arms>` and a `*` is drawn above the bar (`:69-70`, `:112`). The flag covers only
  candidates with no rows at all. A candidate that ran on some seeds only competes on those seeds: at live 10^3 b = 3,
  `warm_start_bandit`, `cluster_head_router`, `disrouter_cascade`, `flat_nsw_router`, `ucb`, `thompson` and `trueskill`
  have seeds 1–5 only (`live_f1_n1000`), so seeds 6–10 choose among the rest. `pool_seeds_n1000` supplies seeds 6–10.
- **The "learned router" pool includes two declaration-only methods.** `cluster_head_router` and `disrouter_cascade`
  (§2.8) spend no probes and read only the declared channel. `flat_nsw_router` is an ANN index over flat probe means.
  Declaration-only methods win the "best learned router" pick in several B cells (bernoulli 10^7, replay 10^6,
  LLMRouterBench honest b = 1 and 5).

What was picked in the current `figures/condensed_sample/*.csv` (written 02:05–02:06). `*` = incomplete pool.

| figure / cell | best learned router | best bandit |
|---|---|---|
| A, live n = 10^2, b = 3 | `knn_router_online` ×10 * | `warm_start_bandit` ×10 * |
| A, live n = 10^3, b = 3 | `knn_router_online` ×10 | `warm_start_bandit` ×5, `linucb_honest` ×5 * |
| A, live n = 10^4, b = 1 (honest) and b = 3 | `knn_router_online` ×3 * | `warm_start_bandit` ×3 * |
| A / B, live n = 10^5, b = 3 | `knn_router_online` ×3 * | `warm_start_bandit` ×3 * |
| B, bernoulli 10^7 | `cluster_head_router` ×100 (b = 1 *, honest b = 3); `disrouter_cascade` ×100 (cartel b = 3) | `warm_start_bandit` ×100 * |
| B, replay 10^6 | honest: `cluster_head_router` (b = 1 *, 3, 5); cartel: `cluster_head_router` (b = 1 *), `flat_nsw_router` (b = 3, 5) | `warm_start_bandit` * |
| B, RouterEval 5,000, b = 3 | `knn_router` ×3 * | honest `warm_start_bandit` ×3 *; cartel `warm_start_bandit` ×2, `linucb_honest` ×1 * |
| B, LLMRouterBench 20 | honest: `cluster_head_router` ×5 (b = 1, 5), `mlp_router` ×5 (b = 3 *); cartel: mixed picks at b = 1 and 5, `mlp_router` ×5 (b = 3 *) | `warm_start_bandit` / `linucb_honest`, every bar * |

### 2.1.4 Shared machinery every method uses

- **Interface** (`rte/methods/base.py:1-41`):
  - `needs` is a subset of {declared, probe, reports, bus}. The `View` raises `AccessError` on any other access
    (`rte/world.py:180-182,229-230`), and it never exposes S or the liar set.
  - `build(view, budget)` is the pre-emptive phase.
  - `fetch(task)` returns an agent id.
  - `observe(task, agent, outcome)` is the online update. The runner passes the **true executed outcome** directly
    (`rte/run.py:145-147`), so every online method learns from trusted outcomes, not from reports.
- **Budget** (`rte/budget.py:5-15`): `total_probes(n, K) = n·K·b`, with b = `probes_per_agent_family`. The runner does
  **not** enforce it. It only logs a warning when build probes exceed it (`rte/run.py:135-136`).
- **Probes are index-seeded** (`rte/world.py:311-320`). The k-th probe of (agent, family) is the same instance for every
  method. `World.reset` zeroes the probe index before each method (`rte/world.py:391-395`), so two methods probing the
  same cell see identical outcomes.
- **Method randomness** comes from `view.rng = default_rng(stable_seed_32(seed, "view", sorted(needs)))`
  (`rte/world.py:177`). It depends only on the world seed and the method's `needs`. Methods with the same `needs`
  therefore start from the same RNG stream. Checked: MIDIAN, MIDIAN-V and MIDIAN-VA build **identical leaf cohorts** at
  a given seed (n = 100 and 1,000, b = 1, 3, 5).
- **Declared-channel helpers** (`rte/methods/_decl.py`):
  - `declared(view)` charges n messages once, for collecting the registry.
  - `scan` charges n comparisons per read of one family's column.
- **What liars do.** The world layer is described in Part 1; summarised here so each method's liar behaviour reads
  on its own.
  - Declared lie (`rte/world.py:123-141`): `inflate` sets D = clip(D_honest + 0.4). `max` sets D = 1 everywhere.
    `squat` sets D = 1 on the top-3 demand families.
  - Report lie, when `collude` is on (`rte/world.py:358-383`): a liar reporter says 1 about any liar, and 0 about
    the top 20% (by its observed mean in this batch) of the honest agents it reports on. Otherwise it reports the
    truth.
  - Liars **execute at their true skill**.

---

## 2.2 Plain MIDIAN (`midian`, label "MIDIAN")

**File:** `rte/methods/midian.py`, class `Midian`. **Needs:** `{probe, reports}` (`:17`). **Params used in A/B:** defaults,
`r=10, delta=1/3, online=True, verify=False, cached=False, top=1, cohort="random"` (`:19-36`). The grid lists it as a plain
string or `{}`. METHODS.md calls it pre-registered (SPEC §5) and never changed since the first run. The parameters r = 10
and δ = 1/3 are defaults, not tuned. `extra_figs.excluded` drops any MIDIAN with r ≠ 10 or δ ≠ 1/3
(`scripts/extra_figs.py:123-135`).

### 2.2.1 The idea

Put agents in small random groups ("cohorts") of r. Each agent is probed, and its cohort peers, not a trusted central
observer, report what they saw. A trimmed mean over those reports absorbs a few lying peers. The cohorts are then stacked
into an r-ary tree. Each tree node remembers, per task family, the best estimate anywhere below it and which child holds
it. Routing a task walks down the tree from the root, about log_r n steps, instead of scanning all n agents.

### 2.2.2 Tree construction (`_cohorts`, `_structure`, `:48-77`)

- **Leaf cohorts.** `c = rng.permutation(n)`, padded with −1 to a multiple of r and reshaped to `(ceil(n/r), r)`
  (`:51-54`). A "leaf cohort" is one row of `self.leaves`: r agent ids (only the last row can be short). `self.leaf_of[a]`
  maps each agent to its row (`:109-110`).
- **Upper levels.** At each level the m nodes are randomly permuted and grouped r at a time into parents (`:72-76`),
  until one node is left. `self.children[l]` holds, for each node at level l, its r children (agent ids at level 0, node
  ids above). `self.parent[l]` maps a node to its parent.
- **Depth.** `depth = len(children)`. Checked: depth = 2 at n = 100 (10 leaf cohorts, 1 root) and 3 at n = 1,000.

### 2.2.3 Level-0 estimation (`peer_reported_estimates`, `rte/methods/_est.py:86-122`)

- **Probes.** Every agent is probed b times on every family. That is n·K·b probes, exactly `Budget.total_probes`.
- **Reports.** Every probe outcome is reported by **every other member of the agent's cohort** (s − 1 reporters). That is
  Σ_c s_c(s_c − 1)·K·b reports. At r = 10 this is 9 × the probes. Checked: 43,200 reports for 4,800 probes at n = 100,
  b = 3.
- **Aggregation for plain MIDIAN** (`by_reporter=False`, because `verify=False`; `midian.py:101`):
  - The (s − 1)·b individual reports about (member, family) are pooled, and a trimmed mean drops
    `trim_k(delta, s, b) = max(0, min(floor(delta·(s−1)), ((s−1)·b − 1)//2))` **single reports** from each end
    (`_est.py:43-45,121`).
  - At r = 10, δ = 1/3 that is **3 reports per side** at b = 1, 3 and 5, out of 9, 27 and 45 reports.
  - A colluding peer contributes all b of its reports. So at b = 3 the trim removes about one liar peer's worth of
    reports per side, and at b = 5 about 0.6. The code's own comment on `trimmed_by_reporter` (`_est.py:55-56`) says
    per-report trimming "under-trims" for this reason. The by-peer trim is what MIDIAN-V and MIDIAN-VA use.
- **Singleton cohort.** A cohort of size 1 has no reporters, so its estimate is its own probe mean (`_est.py:105-107`).
- **Build messages** (`midian.py:137`): (n − #leaves) member-to-leader messages plus (#nodes − 1) node-to-parent messages.
  Checked: 100 at n = 100 and 1,010 at n = 1,000. No comparisons are charged at build.

### 2.2.4 Summaries (`build`, `:115-136`)

- At every node and for every family, the children's candidate values are sorted with a stable sort.
- `best[l][node, f]` is the child slot holding the maximum.
- `summary[l][node, f]` is that maximum value.
- `cand[l][node, f]` is the agent holding it.
- `topc` holds the top `top` (= 1) forwarded agents.
- **Ties** go to the lowest slot index. Slots are randomly permuted, so this amounts to a random tie-break fixed at
  build time.

### 2.2.5 Routing (`fetch`, `:149-156`)

- Start at the root and descend `depth` levels. At each level take `child = best[l][node, f]`, and return the level-0
  member reached.
- **Cost per task:** 1 hop, r comparisons and 2 messages per level, so depth hops, r·depth comparisons and 2·depth
  messages.

### 2.2.6 Online update (`observe`, `:168-172`, with `online=True`)

- The routed agent's estimate gets a running-mean update: `est += (outcome − est)/k`. The count k starts at
  `w0 = max(1, (r−1)·b − 2·trim_k)` (`:114`): 3 at b = 1, 21 at b = 3, 39 at b = 5. So the build estimate carries
  the weight of that many observations, even though only b independent probe outcomes lie behind it.
- The family's summaries are then recomputed up the agent's path (`_recompute`, `:158-166`). This charges r comparisons
  and 1 message per level (`:160-161`).
- **Only the routed agent is ever updated.** The update is greedy exploitation: a routed agent that keeps failing sinks
  until another subtree's summary is higher.

### 2.2.7 Churn and budget

- **Churn** (`:174-184`) re-probes arrivals b times per family through their cohort peers. It is not relevant to A/B,
  whose cells have no churn.
- **Probe spend** is exactly n·K·b, and nothing at run time.
- **Declarations are never read** (with `cohort=random`).

### 2.2.8 Behaviour with liars

- Liars can corrupt the level-0 estimates of every member of any cohort they sit in, through the report channel.
- Per-report trimming of 3 reports per side absorbs only a few colluding reports.
- Online updates use true outcomes, so a liar that was over-estimated and gets routed is corrected after its failures.

---

## 2.3 MIDIAN-V and MIDIAN-VA ("MIDIAN-VA" is drawn in A and B; it is the reference line in E, F and H)

### 2.3.1 Class chain

- `MidianVA(MidianA)` sets `verify=True, cached=True` (`rte/methods/midian_va.py:10-14`).
- `MidianA(MidianSH)` sets `halving=False, audit=0.05` (`rte/methods/midian_a.py:16-21`).
- `MidianSH(Midian)` supplies the level-0 engine (`rte/methods/midian_sh.py:29-86`).
- So MIDIAN-VA = plain MIDIAN's tree + MIDIAN-SH's per-peer level-0 engine with halving **off** + MIDIAN-A's audits and
  reporter exclusion + MIDIAN-V's verification at promotion + a cached root pick.
- The "VA" name stands for **V**erified promotion + **A**udited reports (`midian_va.py:1-6`).
- It is pre-registered as V2-11 in TARGETS_rte_v2.md (`midian_va.py:6`) and is a labelled variant added 2026-09-03.
- **Needs:** `{probe, reports}`, inherited. It **never reads declarations** in A/B (`cohort="random"`).

### 2.3.2 MIDIAN-V for reference (`rte/methods/midian_v.py:21-26`)

- It is `Midian(verify=True, cached=True)`. The ALIAS `midian[cached=True,verify=True]` maps to `midian_v`
  (`rte/analyze.py:21`).
- It is not drawn in A or B (it is not in `ARMS`); it is in the per-family bar figures (`figures/bars/`).
- It differs from VA only in level 0: plain MIDIAN's `peer_reported_estimates` with `by_reporter=True` (per-peer trim),
  and no audits.

### 2.3.3 Build, step by step (`Midian.build`, `midian.py:103-137`, with VA's overrides)

1. **Split the budget.** `b0 = max(1, min(b, self.b0 or b − 1))` (`:105`). With `b0=None` this gives b0 = b − 1 for
   b ≥ 2 and **b0 = 1 at b = 1**. Level 0 spends b0 probes per (agent, family). The rest, n·K·(b − b0), is kept for
   verification.
2. **Tree.** Identical to plain MIDIAN (`_structure`). It uses the same `view.rng`, so at the same seed it builds the
   same leaf cohorts as plain MIDIAN (§2.1.4).
3. **Verification probes per candidate** (`:111-112`).
   - `C = top · Σ_{l≥1} (#valid child slots at level l)` is the number of forwarded candidates across all upper levels.
   - `e = floor((b − b0)·n / C)` fresh probes are spent per (candidate, family).
   - For r = 10, C ≈ n/9, so **e ≈ 9·(b − b0) = 9 at both b = 3 and b = 5**. Checked: e = 10 at n = 100 and e = 9 at
     n = 1,000.
   - **At b = 1, b − b0 = 0, so e = 0 and there is no verification at all.** VA at b = 1 is MIDIAN-A with a cached
     root (DEVIATIONS "Erratum 22", `DEVIATIONS.md:838-839`).
4. **Level 0, audited** (`MidianSH._level0`, `midian_sh.py:40-78`, with `halving=False`, so `_schedule` returns a single
   round `[(s, b0)]`, `:14-26`).
   - Each member is probed b0 times per family. Each of the s − 1 cohort peers reports every outcome
     (`_est.peer_estimate`, `_est.py:69-75`).
   - Per-peer report sums and counts are stored in `rsum` / `rcnt` (`midian_sh.py:45`, accumulated at `:72`), and each member's reporter ids in `peer_of`
     (`:59`).
   - The estimate is `_estimates` (`:80-86`):
     - First average each peer's reports about the member (the per-peer mean).
     - Mask the excluded peers (and peers with no reports), unless every peer is excluded.
     - Then apply a **trimmed mean over peers**, dropping `t = min(floor(δ·(s−1)), (v−2)//2)` peers from each end,
       where v is the number of peers left (`_est.py:54-66`). With 9 peers and none excluded, t = 3: the middle 3 of
       9 per-peer means are averaged.
5. **Audit during the build** (`MidianA._audit`, `midian_a.py:32-37`).
   - For every level-0 probe instance, the auditor draws uniformly with rate 0.05.
   - Each drawn instance is **re-run on the same index-seeded instance** with `view.probe_at` (`rte/world.py:385-389`,
     charged as a probe, probe index untouched). The true outcome is compared with **every** peer's report about that
     instance.
   - Each mismatch is a strike (`_strike`, `:23-30`). A reporter with `STRIKES = 2` mismatches (`:13`) is excluded from
     every later aggregation. Its later reports are still charged but never used.
   - An honest reporter reports the observed outcome verbatim, so it can only be struck if re-running the same instance
     gives a different outcome. That depends on backend determinism (see Part 1). A liar is struck only when its lie
     actually flips an outcome.
6. **Verification at promotion** (`Midian._verify`, `midian.py:79-97`, called at every level l ≥ 1, `:123-126`).
   - Every candidate forwarded by a child (its per-family best) is re-probed `e` times per family.
   - **Who reports.** The reporters are `self.rep[l−1]` of the **other** r − 1 children of the same parent node. `rep` is
     **one uniformly random member of each child's subtree** (`:135-136`; comment `:123`: "reporters: RANDOM members of
     the sibling"). The docstring's "representatives lead[M, r]" is a parameter name. The `self.lead` array computed at
     `:134` is not used for reporting.
   - With the default `observers = r − 1` all 9 report. Each report is trimmed by reporter exactly as at level 0. For VA,
     excluded reporters are masked (`:93-95`).
   - The new mean is folded into the candidate's running estimate, weighted by probe count:
     `est = (est·k + m_new·e)/(k + e)`, where k starts at b0 (`:97`, `:114`). At b = 3, e = 9 outweighs the b0 = 2
     level-0 probes. A candidate forwarded again at the next level is verified again.
   - The child's summary value is rewritten with the verified estimate (`:125-126`) before the parent compares its
     children (`:127-136`).
7. **Build messages** are the same as plain MIDIAN (`:137`).

### 2.3.4 Probe spend (checked by instantiating the classes on a bernoulli world, K = 16, β = 0.5 cartel, seed 1)

| n | b | MIDIAN probes / n·K·b | MIDIAN-V | MIDIAN-VA |
|---|---|---|---|---|
| 100 | 1 | 1.000 | 1.000 | **1.045** |
| 100 | 3 | 1.000 | 1.000 | **1.031** |
| 100 | 5 | 1.000 | 1.000 | **1.037** |
| 1,000 | 1 | 1.000 | 1.000 | **1.050** |
| 1,000 | 3 | 1.000 | 0.997 | **1.029** (49,383 vs 48,000) |
| 1,000 | 5 | 1.000 | 0.998 | **1.037** |

- MIDIAN-VA overspends by the audit re-runs: about 0.05·b0/b of the budget, which is ≈5% at b = 1, 3.3% at b = 3 and
  4% at b = 5.
- The floor in `e` makes the verification slightly under-spend.
- This matches DEVIATIONS: "Audited builds exceed the n*K*b cap by the audit rate by design" (`DEVIATIONS.md:690-693`),
  and the "build spent 49380 probes > budget 48000" warning (`DEVIATIONS.md:904-905`, recorded, not corrected).
- The runner only warns (`rte/run.py:135-136`), so these rows are kept.

### 2.3.5 Routing (`fetch` with `cached=True`, `midian.py:150-151`)

- The route returns `cand[-1][0, f]`, the agent the root holds for family f.
- It is charged 1 comparison, 2 messages and 0 hops.
- The choice is the same agent a full descent would reach. Only the charged cost differs.

### 2.3.6 Online updates (`MidianA.observe`, `midian_a.py:39-51`)

- It first runs plain MIDIAN's `observe`: running mean with pseudo-count `w0` computed from the full b (`midian.py:114`),
  then a path recompute. With `cached=True` this also refreshes the cached root pick (`midian.py:164-165`).
- Then, with probability 0.05, it runs an **online audit**:
  - The routed outcome is put to the routed agent's cohort peers through the report channel, charged s − 1 reports.
  - Each peer's claim is compared with the true outcome, adding strikes.
  - If a reporter is newly excluded, every member of that reporter's cohort is re-aggregated from the stored level-0
    per-peer means, and the cohort's path is recomputed for all K families.
- That re-aggregation **overwrites** those members' online running-mean updates and their promotion-verification folds.
  DEVIATIONS notes the first effect ("rare event; noted, not fixed", `DEVIATIONS.md:694-695`). The loss of the
  verification fold is not mentioned there.
- **Per-task run cost:**
  - 1 comparison and 2 messages to route.
  - r·depth comparisons and depth messages for the path recompute.
  - On average 0.05·(r − 1) ≈ 0.45 audit reports.
  - **No probes at run time.**

### 2.3.7 Behaviour with liars

- VA never reads declarations, so declared lies have no effect on it.
- Report lies are fought three ways:
  1. **By-peer trimming** removes 3 of 9 peers per side, which is robust to a colluding peer's b correlated reports.
  2. **Audits** exclude a reporter caught twice.
  3. **Promotion re-probes** use reporters drawn from **other subtrees**, so a cohort-local cartel cannot vouch for its
     own member at the higher levels.
- **At b = 1 defence 3 is absent** (e = 0).

### 2.3.8 Hyperparameters

- r = 10 and δ = 1/3 are plain MIDIAN's defaults.
- The audit rate 0.05 is the default in `MidianA.__init__` (`:19`), and `STRIKES = 2` is marked "work order 1.2" (`:13`).
- `b0 = b − 1` and `observers = r − 1` are defaults.
- None of these was tuned on reported seeds, as far as the code and grid comments show.

### 2.3.9 The "leaf cohort" and the VA-cohort shortlist (used by the frameworks' `va_cohort` source)

- With `retrieval="midian_va"`, `FrameworkMethod.build` builds its own `MidianVA(r=self.r)` inside the framework unit
  (`rte/methods/frameworks/_common.py:290-292`). That instance spends its own probes, including the audit overspend.
- `retrieve` then returns VA's pick first, followed by the other members of **that pick's leaf cohort**,
  `leaves[leaf_of[a]]` (`_common.py:313-317`). That is at most r = 10 agents, all probed and peer-reported together at
  level 0.
- The framework's `observe` forwards the outcome to the embedded VA (`_common.py:337-338`).
- The framework's `needs` gain probe and reports (`_common.py:161`) on top of declared. The embedded VA's `view.rng` is
  therefore seeded with a **different `needs` set** than a standalone `midian_va`, so its cohorts need not match the
  standalone arm's cohorts at the same seed.

### 2.3.10 MIDIAN-VA as the reference line in E, F and H

- `scripts/shortlist_figs.py:104,121` takes rows with `method == "midian_va"` at b = 3 (`:65-70`) from `REF_GRIDS`
  (`:55-59`), matched on (n, dist, regime). The grid with the most seeds wins.
- The condensed E, F and H figures draw the **honest (β = 0)** line only, across both the honest and the cartel bars
  (`scripts/shortlist_condensed.py:46-50`: `ref.loc[(n, "beta0")]`).
- In H the line comes from `routereval_mmlu` (m = 10, 100, 1,000) and `routereval_mmlu5k` (5,000) (`REF_GRIDS`,
  `shortlist_figs.py:59`). It is drawn over every H shortlist, including the `re_sl_*` bars that `shortlist_figs.py:39`
  collects. At m ≤ 1,000 the line and older framework rows may come from different family orders (§1.10 #1).

---

## 2.4 The oracle (dotted line in A, B, E, F and H; the normaliser in B)

- `World.oracle(task) = argmax_a S[a, task.family]` (`rte/world.py:301-302`). It is the **only** routing rule that reads
  true skill S. Ties go to the lowest agent id (`np.argmax`).
- The runner runs it once per (cell, seed) in `oracle_line` (`rte/run.py:122-128`). It executes the chosen agent on the
  same task stream as every method (`world.execute`), so **oracle success is the realised success** of the best-S agent
  on these task instances, not max S itself.
- After a churn event it re-picks using the new S (`rte/run.py:125-126`); A/B cells have no churn.
- It is written as the row `method="oracle"`, with a zero build ledger (`rte/run.py:162-164`), and used as the regret
  baseline (`:173`).
- **Liars do not affect it.** Liars execute at true skill, S and the task stream do not depend on β, and the liar RNG is
  separate (`rte/world.py:244,255-256,287-292`). The world seed does not include b either, so the oracle is also
  independent of b.
- In B, each group divides **both regimes'** bars by the **honest** cell's oracle mean
  (`scripts/condensed_figs.py:97`), and that oracle comes from the b = 3 bar CSV for all b.

---

## 2.5 Flat probe argmax, online (label "flat probe argmax (online)")

**File:** `rte/methods/flat_probe_argmax.py`. **Needs:** `{probe}`. **Params:** `online=True`, `cached=False`
(ALIAS `rte/analyze.py:20`). METHODS.md (SPEC §6) calls it "the key control": MIDIAN's probes without the tree or the
reports.

- **Build** (`:16-20`): `probe_successes(view, b)` (`_est.py:9-16`), i.e. n·K·b probes, observed directly by the router
  (a **trusted observer**). The estimate is the probe mean, and the per-family argmax is stored.
- **Fetch** (`:22-27`): `argmax(est[:, f])`, charged n comparisons (not cached). **Ties go to the lowest agent id.**
  - After b probes the estimates lie on the grid {0, 1/b, ..., 1}, so ties at the maximum are large.
  - DEVIATIONS reports about 116 agents tied at n = 10^3, b = 3, specialist, and about 30,000 at n = 10^5, b = 1
    (`DEVIATIONS.md:234-241`).
  - DEVIATIONS calls lowest-id tie-breaking "a different arbitrary rule, not a better one".
- **Observe** (`:29-34`, `online=True`): running mean with count starting at b, then re-argmax of that family. The
  argmax costs O(n) compute but is not charged. The effect is sequential elimination among the tied top agents: a routed
  agent that fails drops out of the tie.
- **Probe spend:** exactly n·K·b at build, none at run time.
- **Liars:** it never reads declarations or reports, so liars cannot influence it. Checked in A's CSV: honest and cartel
  are identical at 10^2, 10^4 and 10^5, and differ by 0.0002 at 10^3, presumably because the regimes pool different seed
  or grid sets. `random` is likewise identical across regimes.
- **Hyperparameters:** none.

---

## 2.6 Declared argmax (label "declared argmax") and random

### 2.6.1 `declared_argmax`

**File:** `rte/methods/declared_argmax.py`. **Needs:** `{declared}`. **Params:** `cached=False`.

- **Build:** collect D, charged n messages (`_decl.py:4-7`).
- **Fetch:** `argmax_a D[a, f]` (`:22-26`), with an O(n) comparison scan. Ties go to the lowest id.
- **Probes:** none, so b does not apply and it is drawn once, from b = 3.
- **Liars:** it **trusts declarations completely**. Inflated liars move to the top whenever D_honest + 0.4 beats the
  honest maximum. Under `lie_mode=max` every liar claims 1.0 and the lowest-id liar wins every family.
- **Online:** none.

### 2.6.2 `random`

**File:** `rte/methods/random.py`. **Needs:** none.

- **Fetch:** `view.rng.integers(0, n)`. No build, no cost charged.
- It is the floor, and is drawn once.

---

## 2.7 The bandit pool (candidates for "best bandit")

All five spend exactly n·K·b warm-up probes as trusted observations (checked: ratio 1.000, except TrueSkill), learn online,
and charge n comparisons per fetch.

### 2.7.1 `ucb_per_family` (UCB1; `rte/methods/ucb_per_family.py`)

- **Needs:** `{probe}`. **Default:** c = √2 (`:11`).
- **Build:** probe means, `cnt = b`, and per-family pull count `t[f] = n·b` (`:15-19`).
- **Fetch:** `argmax mean[:, f] + c·sqrt(ln t[f] / cnt[:, f])` (`:21-24`). It is deterministic, and ties go to the
  lowest id.
- **Observe:** running mean, with cnt and t incremented.
- **Declarations:** ignored.

### 2.7.2 `thompson_per_family` (`rte/methods/thompson_per_family.py`, via `_est.BetaBandit`, `_est.py:19-37`)

- **Needs:** `{probe}`.
- **Prior:** Beta(1, 1) + b probes, so α = 1 + s and β = 1 + b − s.
- **Fetch:** one Beta sample per agent from `view.rng`, then argmax. It is random per fetch.
- **Observe:** α or β += 1.

### 2.7.3 `warm_start_bandit` (label `warm_start_bandit`, n0 = 5; and `warm_start_bandit[n0=0.5]`)

**File:** `rte/methods/warm_start_bandit.py`. **Needs:** `{declared, probe}`.

- **Prior** (`:14-17`):
  - Charges n messages, for collecting the declarations.
  - D is clipped to [1e-3, 1 − 1e-3].
  - The prior is α0 = n0·D and β0 = n0·(1 − D).
- **Build:** the prior plus b probes (BetaBandit).
- **Fetch and observe:** Thompson sampling, as in 2.7.2.
- **It trusts declarations as a prior** worth n0 pseudo-observations: 5 against b = 3 real probes at the default.
- **Liars:** an inflated liar has D ≈ 1, so its prior failure count is ≈ 0 and its first observed probe failures
  dominate. CHANGES_AND_ERRATA §8d (`CHANGES_AND_ERRATA.md:388-398`) uses this to explain why this arm scores **higher
  under the cartel** than honest (live 10^4: 0.743 → 0.783; 10^5: 0.738 → 0.778). §8d calls the honest arm
  "under-tuned" and states that reported numbers keep the pre-registered n0 = 5.
- **Tuned variant `n0 = 0.5`:**
  - Chosen by `tune_wsb_n1000` (`configs/grid.yaml:1029-1041`): live, n = 1,000, specialist, β = 0, self-described,
    b = 3, **seeds 11-15**, n0 ∈ {0.5, 1, 2, 5}.
  - The grid comment (`configs/grid.yaml:1058-1059`) says seeds 11-15 are never reported (reported live 10^3 cells use
    seeds 1-10) and that the result was 0.762 for n0 = 0.5 against 0.745-0.750 for the others.
  - The tuned arm is run "beside" the pre-registered one in `tuned_wsb_n{100,1000,10k,100k}` at b = 3
    (`:1060-1063`) and in the live `rivals_b_n*` grids at b = 1 and 5 (`:1068-1071`).
  - In the condensed figures it only enters the best-bandit pool (`scripts/condensed_figs.py:34`), at every b. None of
    its grids has rows yet, so every best-bandit bar that could include it carries the incomplete-pool `*` (§2.1.3).
  - §8d as written does **not** mention the tuning run. The "CHANGES 8d" citation in `grid.yaml:1029` points at the
    motivation, not at a record of the tuning.

### 2.7.4 `linucb_honest` (`rte/methods/linucb_honest.py`)

- **Needs:** `{probe}`. **Default:** α = 1.0.
- It is a labelled v2 rival (2026-09-03). The code cites no paper. "Honest" means its context never uses model
  identity, specialty or declarations (`:1-5`).
- **Context of agent a for family f:** `x = [1, mean_af, sqrt(cnt_af), mean over families of a]` (`:20-23`).
- **Model:** a ridge model per family with `A_f = I + Σ x xᵀ` and `b_f = Σ x y`.
- **Warm-up:** each arm's b probes are added as b copies of (x, mean) (`:30-32`). This is equivalent to adding the b
  individual outcomes, because x is constant during the warm-up.
- **Fetch:** `argmax xᵀA⁻¹b + α·sqrt(xᵀA⁻¹x)` (`:34-38`).
- **Observe:** updates A, b, the count and the mean (`:40-45`).
- **Declarations:** ignored.

### 2.7.5 `trueskill_per_family` (`rte/methods/trueskill_per_family.py`)

- **Needs:** `{probe}`. Uses the `trueskill` package's default environment.
- **Build** (`:24-59`), for each family:
  - Draw n·b//2 random pairs, with replacement, and drop pairs where a1 = a2. It therefore spends slightly **less** than
    the budget. Checked: ratio 0.989-0.999.
  - Probe each agent of a pair once and apply a 1-vs-1 update: a win, a loss, or a draw when the outcomes are equal.
  - The docstring says both are probed "on the same instance". **The code probes each agent at its own index-seeded
    instance** (`:47-48`), because instance seeds depend on the agent id (`rte/world.py:100-105,319`).
- **Fetch:** argmax μ.
- **No online update** (`:65-66`).
- It raises `NotImplementedError` at n ≥ 100,000 (`:17,32-36`), so it is absent from live 10^5, bernoulli 10^7 and
  replay 10^6.

---

## 2.8 The "learned router" pool (candidates for "best learned router")

### 2.8.1 `knn_router` and `knn_router_online` (`rte/methods/knn_router.py`)

- **Needs:** `{probe}`. The code docstring calls it "RouterBench's KNN predictive router (Hu et al. 2024) on our terms".
- **Build:** `probe_set` (`_learned.py:39-46`) probes every agent b times per family, n·K·b probes, **keeping the prompt
  of every probe**.
  - Each probe prompt is embedded with the backend's own vectors when it ships them (RouterEval RoBERTa), otherwise with
    all-MiniLM-L6-v2 on the prompt text (`_learned.py:49-52`).
  - Bernoulli has no real text: its "text" is `"A task of family X (instance i)."` (`rte/world.py:325-329`).
  - The embedding arithmetic is not charged in the ledger (`_learned.py:6`).
- **Predicted success of agent a** = the mean outcome of a's **k nearest** probes by cosine to the task's embedding.
  k defaults to b (`:21`), so k = 1 at b = 1.
- **Fetch:** argmax over agents, charged n comparisons. Ties go to the lowest id.
- **Online variant** (`online=True`, ALIAS `knn_router_online`): appends every routed (task embedding, outcome) to the
  agent's store (`:34-38`).
- **Liars:** it reads no declarations and no reports, so liars have no effect. Checked: the honest and cartel values are
  identical in A.

### 2.8.2 `mlp_router` (`rte/methods/mlp_router.py`)

- **Needs:** `{probe}`. The code cites RouterBench's MLP router (Hu et al. 2024).
- **Model:** one sklearn `MLPRegressor(hidden=128, max_iter=30, random_state=0)` on the input
  [agent one-hot (n dims) ⊕ prompt embedding], fit on the n·K·b probes (`:19-24`).
  - `random_state=0` is fixed, so it does not vary with the world seed.
- **Fetch:** predict for all n agents, then argmax (n comparisons).
- **No online update.**
- The one-hot input makes memory O(n²·K·b). The grids exclude it at n = 10,000 and at the 5,000-LLM pool
  (`DEVIATIONS.md:727`; `configs/grid.yaml:291` comment). There is no guard in the code itself.

### 2.8.3 `flat_nsw_router` (`rte/methods/flat_nsw_router.py`)

- **Needs:** `{probe}`. METHODS.md calls it the "E7 flat rival".
- **Build:** the same trusted probe means as `flat_probe_argmax`, inserted into an hnswlib inner-product index
  (M = 16, ef = 50, ef_construction = 200). The index seed comes from `view.rng`.
- **Fetch:** a one-hot(f) query, i.e. an approximate argmax of `est[:, f]`, charged ⌈log2 n⌉ hops and ef comparisons
  (`:30-33`). Which of the tied top agents it returns depends on the index seed (`DEVIATIONS.md:234-241`).
- **No online update.**

### 2.8.4 `cluster_head_router` (`rte/methods/cluster_head_router.py`): declared-only, no probes

- **Needs:** `{declared}`. METHODS.md describes it as "AgentNet++-style prior art".
- **Build** (`:48-66`):
  - Split the agents into random buckets of 20,000 (DEVIATIONS `:93-100`).
  - Run k-means on D inside each bucket, with k = ceil(size/10) and 5 iterations (+1 assignment pass).
  - The **head** of each cluster is the member with the highest **mean over families** of D.
- **Fetch** (`:68-75`):
  1. Pick the cluster whose head has the highest D[head, f]: compare over all heads, 1 hop, 2 messages.
  2. Pick `argmax D[member, f]` within that cluster: compare over its members, 1 hop, 2 messages.
- **No online update.**
- **Liars:** it **trusts declarations completely**.
- b does not affect it. Its b = 1 and 5 "bars" are the same method re-run under the b = 1 and 5 grids.

### 2.8.5 `disrouter_cascade` (`rte/methods/disrouter_cascade.py`): declared-only, no probes

- **Needs:** `{declared, bus}`.
- **Order:** agents sorted by **ascending** mean declared skill ("cheap first").
- **Fetch:** the first agent in that order with D[a, f] ≥ τ = 0.7 takes the task. It is charged pos hops and pos
  messages, where pos is the taker's position in the order and can be O(n).
- **Fallback:** if nobody qualifies, route to the highest declarer (DEVIATIONS `:87-92`).
- It picks the **lowest-mean declarer that clears the threshold**, not the best one.
- **Liars:** it trusts declarations. **No online update.**

---

## 2.9 Methods that are run but excluded from A and B (brief)

| label | file | why it is not in A/B | one-line mechanism |
|---|---|---|---|
| `midian_v` | `midian_v.py` | not in `ARMS`; appears in D | §2.3.2 |
| `midian_a` | `midian_a.py` | not in `ARMS`; appears in D | VA without promotion verification and without the cached pick. At b = 1, VA ≡ A in its picks |
| `midian_sh`, `midian_sha` | `midian_sh.py`, `midian_sha.py` | `DO_NOT_ADD` (`scripts/extra_figs.py:111`) | successive halving inside each cohort, spending exactly s·b probes. `_schedule` raises when b is too small (`midian_sh.py:23-25`) |
| `midian[r=5]`, `midian_v_r5`, r = 20 | `midian.py` | `excluded`: r ≠ 10 | same tree, different r |
| `midian[online=False]` | `midian.py` | `DO_NOT_ADD` | frozen internals ablation |
| `midian[cohort=...]`, `stratify=True` | `midian.py:23-66` | `_VARIANT` regex | budget-neutral cohort modes (`block`, `specialty`, `declared`; `declared` adds needs `declared`) |
| `midian_llm_descent` | `midian_llm_descent.py` | `DO_NOT_ADD` | an LLM chooses among children at each level; argmax fallback |
| `sequential_halving`, `sequential_halving_peer` | `sequential_halving.py` | `HIDE_HALVING = True` (TEMPORARY, 2026-09-22, `scripts/extra_figs.py:120,128`); the plain trusted-observer arm is also in `DO_NOT_ADD` | per-family fixed-budget best-arm identification, n·b probes per family. `peer_reported` scores through r − 1 random peer reports, per-reporter trimmed |
| `route_to_k_majority` | `route_to_k_majority.py` | `DO_NOT_ADD`: executes 3 agents per task | top-3 by D; the runner majority-votes (`rte/run.py:111-115`) |
| `verify_on_claim` | `verify_on_claim.py` | not in any A/B pool; appears in D | rank by D, then at **fetch time** probe the top unverdicted candidate k = 3 times, accept if mean ≥ D − 0.15, up to 5 new verifications per fetch; verdicts cached. It spends **run-time** probes and no build probes |
| `declared_softmax` | `declared_softmax.py` | not in `ARMS`; appears in D | sample ∝ exp(D/0.1) |
| `cnp_self_bid` | `cnp_self_bid.py` | not in `ARMS`; appears in D | Contract Net: broadcast, bids D + N(0, 0.02), argmax; 2n messages |
| `gossip_reputation_greedy` | `gossip_reputation_greedy.py` | not in `ARMS`; appears in D | one random peer reports each probe; seedless EigenTrust; T-Man overlay; greedy walk |
| `referral_network` | `referral_network.py` | not in `ARMS`; appears in D | d = 10 regular graph; neighbour beliefs from single-observer reports; greedy referral walk |
| `llm_supervisor` | `llm_supervisor.py` | live only; not in A/B pools | framework adapter's retrieval + one direct supervisor call. Code default k = 20 (`:22`) |
| `flat_probe_argmax_frozen` | `flat_probe_argmax.py` | only the online version is in `ARMS` | same as §2.5 without `observe` |

---

## 2.10 Summary table

"Build probes" is relative to `Budget.total_probes = n·K·b`. "Trusts declarations" means the routing decision uses D.

| label in A/B | code name + params | reads (`needs`) | build probes | run-time probes | trusts declarations? | tuned? |
|---|---|---|---|---|---|---|
| MIDIAN-VA | `midian_va` {} (r = 10, δ = 1/3, b0 = b − 1, audit 0.05, 2 strikes, cached) | probe, reports | n·K·b·(1 + 0.05·b0/b): **+5% / +3.3% / +4%** at b = 1 / 3 / 5 (the floor in e trims slightly) | 0 (≈0.45 audit **reports** per task) | no | no (pre-registered V2-11; defaults) |
| MIDIAN | `midian` {} (r = 10, δ = 1/3, online) | probe, reports | exactly n·K·b | 0 | no | no (SPEC §5) |
| flat probe argmax (online) | `flat_probe_argmax` {online: true} | probe | exactly n·K·b | 0 | no | no |
| best learned router ∋ | `knn_router` / `knn_router[online=True]` (k = b) | probe | n·K·b | 0 | no | no |
| | `mlp_router` (128 hidden, 30 iter, random_state 0) | probe | n·K·b | 0 | no | no |
| | `flat_nsw_router` (M 16, ef 50, efc 200) | probe | n·K·b | 0 | no | no |
| | `cluster_head_router` (r 10, 5 iter, bucket 20k) | declared | **0** | 0 | **yes** | no |
| | `disrouter_cascade` (τ 0.7) | declared, bus | **0** | 0 | **yes** | no |
| best bandit ∋ | `ucb_per_family` (c = √2) | probe | n·K·b | 0 | no | no |
| | `thompson_per_family` (Beta(1, 1)) | probe | n·K·b | 0 | no | no |
| | `warm_start_bandit` (n0 = 5) | declared, probe | n·K·b | 0 | **yes, as a prior** worth n0 | no (pre-registered) |
| | `warm_start_bandit[n0=0.5]` | declared, probe | n·K·b | 0 | yes, weak prior | **yes**: n0 picked on unreported seeds 11-15, live 10^3 honest |
| | `linucb_honest` (α = 1) | probe | n·K·b | 0 | no | no |
| | `trueskill_per_family` | probe | ≈0.99·n·K·b (pairs with a1 = a2 dropped) | 0 | no | no |
| declared argmax | `declared_argmax` {} | declared | 0 | 0 | **yes** | no |
| random | `random` | nothing | 0 | 0 | no | no |
| oracle (line) | `World.oracle` | **true S** (runner only) | 0 | 0 | no | n/a |

---

## 2.11 Cost per routed task (what the ledger charges; not drawn in A/B, but it defines the arms)

| arm | comparisons | hops | messages | reports |
|---|---|---|---|---|
| MIDIAN | r·depth (fetch) + r·depth (observe) | depth | 2·depth + depth | 0 |
| MIDIAN-VA | 1 + r·depth | 0 | 2 + depth | ≈0.05·(r − 1) |
| flat probe argmax (online), bandits, knn, mlp | n | 0 | 0 | 0 |
| flat_nsw_router | ef = 50 | ⌈log2 n⌉ | 0 | 0 |
| cluster_head_router | #heads + cluster size | 2 | 4 | 0 |
| disrouter_cascade | 0 | pos | pos | 0 |
| declared_argmax | n | 0 | 0 | 0 |
| random | 0 | 0 | 0 | 0 |

---

## 2.12 Discrepancies and open questions

1. **Plain MIDIAN trims reports, not reporters.** METHODS.md §2 says plain MIDIAN takes "a trimmed mean over reporters
   (drop ⌊δ(r−1)⌋ from each end) ... so up to that many liars per cohort are absorbed". The code trims `trim_k` = 3
   **single reports** per side out of (r − 1)·b (`_est.py:43-45,117-121`). DEVIATIONS `:688` states this correctly.
   METHODS.md overstates plain MIDIAN's robustness at b ≥ 2.
2. **MIDIAN-A / VA per-task cost.** METHODS.md §2 says MIDIAN-A costs "1.05× build probes, nothing per task". The code's
   online audit charges s − 1 reports on 5% of routed outcomes (`midian_a.py:39-46`), and the observe path recompute is
   charged too.
3. **MIDIAN-VA overspend** is 5% of the level-0 probes, i.e. 5% / 3.3% / 4% of the budget at b = 1 / 3 / 5. It is not a
   flat "+5%" and not "≤ 5% of build" in the usual sense. The runner only warns (`rte/run.py:135-136`). A/B bars compare
   VA at up to 1.05 n·K·b against rivals at ≤ n·K·b.
4. **At b = 1, MIDIAN-VA has no promotion verification** (e = 0), so its b = 1 bar is "MIDIAN-A + cached root". This
   matches DEVIATIONS Erratum 22 (`:838-839`). Paper text should not describe VA's b = 1 bars as "verified".
5. **The tuned warm-start bandit has no rows yet.** `warm_start_bandit[n0=0.5]` is in the bandit pool at every b
   (`condensed_figs.py:34`). Its live b = 3 rows are to come from `tuned_wsb_*` and its b = 1 / 5 rows from `rivals_b_n*`
   and `pool_fill_*`; none has landed, so every best-bandit bar that could include it carries the incomplete-pool `*`.
6. **CHANGES §8d does not record the tuning.** §8d (`CHANGES_AND_ERRATA.md:388-398`) says "Reported numbers are
   unchanged (the rival is as pre-registered)" and does not mention the n0 = 0.5 tuning or its 0.762 result. The only
   record is the grid comment (`configs/grid.yaml:1029-1030,1058-1059`). I found no results file confirming 0.762.
7. **The "best learned router" pool contains declaration-only and non-learned methods.** It includes
   `cluster_head_router` and `disrouter_cascade` (no probes, declared-only) and `flat_nsw_router` (an ANN over probe
   means). In B they win at bernoulli, replay and LLMRouterBench. The legend label "best learned router" is misleading
   for those cells, since the CSV's `chosen` column shows a declared-channel method.
8. **Candidates that ran on only some seeds.** The pool is the same at every b, but `crossfit` lets a candidate
   compete only on the seeds it has (`seed_tables.py:87-91`). The incomplete-pool `*` does not flag this. At live 10^3
   b = 3 the best bandit is `warm_start_bandit` on seeds 1–5 and `linucb_honest` on seeds 6–10 for this reason.
9. **The CIs of the pooled bars are bootstraps of the cross-fitted per-seed scores.** They include the variation of
   the pick across seeds, but with 3 seeds (live 10^4 / 10^5, RouterEval) each seed's pick rests on 2 others.
10. **TrueSkill docstring vs code.** The docstring says pairs are probed "on the same instance". The code probes each
    agent at its own index-seeded instance (`trueskill_per_family.py:47-48`, `world.py:319`).
11. **`_verify` docstring vs code.** The docstring says reporters are the "representatives lead[M, r]". The call passes
    `self.rep` (a random subtree member, `midian.py:123-124,135-136`). `self.lead` is computed (`:134`) but never used
    for reporting.
12. **Padded upper-level nodes.** In `_verify` (`midian.py:86-87`), empty slots are filled with slot 0's representative.
    If the candidate's own child is slot 0, a reporter from the candidate's own subtree (possibly the candidate itself)
    can be used. Could not determine how often this occurs; it only affects short nodes.
13. **A new exclusion discards the verification fold.** When an online audit excludes a reporter, the cohort's estimates
    are rebuilt from level-0 reports only (`midian_a.py:47-51`), discarding promotion-verification folds as well as
    online updates. DEVIATIONS mentions only the online updates.
14. **Embedded VA vs standalone VA.** The framework adapter's embedded `MidianVA` uses a `view.rng` seeded with the
    framework's `needs` ({declared, probe, reports}), so its cohorts differ from the standalone `midian_va` at the same
    seed. The E/F/H "MIDIAN-VA (whole population)" line and the `va_cohort` shortlist therefore do **not** share a tree.
15. **Reference-line filter.** `shortlist_figs.collect` filters reference rows by `method == "midian_va"` without
    checking params (`scripts/shortlist_figs.py:104`). If a REF_GRID held `midian_va` with non-default params (for
    example `cohort=block`, `r=20`), those rows would be pooled into the line. I did not check the row files.
16. **Honest-only reference lines.** The E/F/H lines use the honest regime's MIDIAN-VA and oracle for the cartel bars
    too (`shortlist_condensed.py:46-48`). B also normalises cartel bars by the honest oracle. The oracle is identical
    across regimes by construction, but MIDIAN-VA is not.
17. **METHODS.md minor points.**
    - `llm_supervisor` is listed with k = 10; the code default is k = 20 (`llm_supervisor.py:22`). The grids may
      override it; not checked.
    - `mlp_router` "does not fit above 10^4": it is in fact excluded **at** 10^4 by the grids (`DEVIATIONS.md:727`),
      and the code has no guard.
18. **MIDIAN's online pseudo-count overstates information.** MIDIAN's online `w0` counts reports (21 at b = 3),
    not independent probes (3), so a build estimate carries 21 pseudo-observations. For VA, `w0` uses the full b while
    level 0 used b0 (`midian.py:114`). This is a design choice, not documented anywhere I found.

---

## 3. Agent frameworks and shortlist (retrieval) variants: how they are implemented

Scope: every framework arm (`fw_*`) and every shortlist that sits in front of the frameworks in condensed Figures E/F/G/H.
Everything here was checked against code on 2026-09-23. Paths are relative to `/n/home02/rsiegelmann/rte` unless they
start with `$RTE_DATA` (= `/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte`). Where the code and the docs disagree,
the code is described here and the conflict is listed in 3.9.

---

## 3.1 What a framework arm is, in one paragraph

A framework arm does not route over the whole population. Per task it does four things:
1. It builds a **shortlist** of about k = 10 agents with a retriever (`FrameworkMethod.retrieve`).
2. It sends the **task text** plus the shortlist's `{name, description}` pairs to a real agent-framework library. That library runs in its own venv, as a JSON-lines subprocess.
3. It lets that library's own delegation primitive (the supervisor, manager, router or triage agent) name one agent. The run stops before the named agent executes.
4. It maps the name back to an agent id (`rte/methods/frameworks/_common.py:362-398`).

The supervisor LLM is `Qwen/Qwen2.5-7B-Instruct` (`_common.py:20`), served by the vLLM fleet. The arm never probes, never
reads reports, and never learns from outcomes, except when the shortlist itself is a MIDIAN cohort (3.4.10). When the
framework names no valid candidate, the adapter falls back to declared argmax within the shortlist (3.2.6).

---

## 3.2 The shared pipeline (`rte/methods/frameworks/_common.py`)

### 3.2.1 Class and constructor
- `FrameworkMethod(Method)` (`_common.py:109`) has `needs = {"declared"}` (`:111`) and `requires_llm = True` (`:112`). The View
  therefore raises `AccessError` on any probe, report or bus access (`rte/world.py:180-195`). The one exception is the MIDIAN
  retrieval modes, which add `{"probe", "reports"}` (`_common.py:160-161`).
- Constructor defaults (`_common.py:116-120`):
  - `k = 10`
  - `supervisor = Qwen/Qwen2.5-7B-Instruct`
  - `retrieval = "tfidf"`
  - `r = 10`
  - `dedup = False`
  - `embed_model = all-MiniLM-L6-v2` (`MINILM`, `rte/methods/_learned.py:12`)
  - `rerank_model = Qwen/Qwen3-Reranker-4B`
  - `rerank_pool = 50`
  - `embed_instruct = ""`
  - `shuffle = False`
  - `lie_text = False`
  - `claim_threshold = 0.0`
- Only `k, supervisor, retrieval, r` and any extra `**params` go into `self.params` (`:121`, `rte/methods/base.py:24-25`).
  Those are what the worker receives in `req["params"]` (`_common.py:356,367`), which is how `mode=` reaches the MAF and
  LlamaIndex workers. The retrieval settings (`dedup`, `embed_model`, ...) are not forwarded to the worker. They are still
  recorded in the row's `params` JSON, because the runner writes the grid spec's params.
- Each subclass sets only `name`, `env` (venv name) and `worker` (script in `workers/`); see the table in 3.3.

### 3.2.2 `build(view, budget)` (`_common.py:284-292`)
1. `self._index(view)` computes the texts, retrieval vectors and shortlist tables (3.2.3).
2. `self.bridge = Bridge(self.env, self.worker)` sets up one worker subprocess, which is started lazily. `self._pre = {}`.
3. `self.base_url = _endpoint(supervisor)` reads `$RTE_DATA/endpoints.json`. It accepts the key `model` and any replica keys `model#<jobid>`,
   and picks one URL uniformly at random with an unseeded `np.random.default_rng()` (`_common.py:97-106`). The pick is
   repeated on every call (`:367`), so replicas that join mid-run get traffic.
4. `view.ledger.message(view.n)`: every agent sends its description to the registry once (`:289`).
5. MIDIAN modes only: it builds `Midian(verify=True, cached=True, r)` (MIDIAN-V) or `MidianVA(r)` on the same view and budget (`:290-292`). **This spends the probe budget** (3.4.10).

### 3.2.3 `_index(view)` (`_common.py:294-310`)
`scripts/embed_routereval.py` calls exactly this function, on a GPU, to pre-warm caches.
- `self.desc, self.fdesc, self._task_text = self._texts(view)` (3.2.4). If `lie_text` is set, `self.desc = self._relabel(...)` (3.4.12).
- Agent names are `agent_{a:06d}` (`:299`). `_name2id` inverts them.
- Vectors:
  - Dense modes (`embed`, `hybrid`, `sota`; `DENSE`, `:23`): `self._Xa, self._Xf = self._embeddings(view)` (3.6).
  - Every other mode, including `declared` and `bm25` whose rankings never use it: one hashed TF-IDF matrix over `desc + fdesc` (`:304-305`).
- BM25 matrix `self._B = _bm25(desc, fdesc)` (K × n) is built only for `bm25`, `hybrid` and `sota` (`LEXICAL`, `:25,306`).
- Dedup pool: `self._pool` holds the lowest agent id per distinct description string, in sorted order (`:307-309`). It is always
  computed and used only when `dedup=True`, and always for `sota`, whose cached table was built with dedup (3.6).
- `self._sota = self._sota_table(view)` for `sota` only (`:310`).

### 3.2.4 What text the supervisor and the retriever see (`_texts`, `_common.py:179-192`)

**Live backend (`backend: llm`).** `_texts` returns the backend's own texts when `rte.backends.llm.current_backend()` is set
and has the same n (`:180-184`):
- **Agent descriptions** come from `LLMBackend.descriptions()` (`rte/backends/llm.py:232-251`) and are cached in
  `$RTE_DATA/populations/<dist>_n<n>_K16_seed<s>/descriptions.json`.
  - Each is written by the agent's own base model. The system prompt asks for "one short paragraph (<=70 words) describing yourself as a service in an agent marketplace". The user prompt is `Your base model is {model}. Your tool is: {tool}. You are equipped for these task families: {specialty}...` (`rte/backends/prompts.py:76-84`).
  - The adapter then appends ` Declared areas: {true specialty list}.` (`llm.py:247`).
  - The text is generated per agent from its profile (true model, tool, specialty). **It does not depend on beta or on who lies.**
  - Descriptions can name the base model, for example "As an agent in the Qwen/Qwen2.5-0.5B-Instruct model ...", as seen in `specialist_n1000_K16_seed1/descriptions.json`. The text therefore carries a model-size signal.
- **Family descriptions** (the retrieval queries) are `families.describe(f)`, which returns `"<family name> problems. Example question: <one example question, ≤240 chars>"` (`rte/backends/families.py:124-128`).
- **Task text** (what the supervisor is asked to route) is the real task question, `families.question(family, instance)` (`llm.py:256-260`).

**Every other backend, including RouterEval.** `_texts` falls back to text rendered from the declaration matrix (`_common.py:187-192`):
- Agent: `"Self-rated competence: <fam> 0.xx, <fam> 0.xx, ..."`, listing the agent's **top-5 families by `view.declared`**, with two decimals.
- Family query: `"Tasks of family <fam>"`.
- Task text: `"A task of family <fam> (instance <i>)."`

**Consequences on RouterEval:**
- The text **is** the declared channel. `view.declared` already includes the lie (`world.py:258/273`, `apply_lying`, `world.py:123-141`: `inflate` adds `DELTA_INFLATE = 0.4`, clipped to 1, `world.py:31`). A liar's inflated numbers therefore appear in its description, and in which families make its top 5.
- **The supervisor never sees the actual MMLU prompt, only the subject name.**
- Example I printed at n = 1,000, strong_to_weak, seed 1: desc `"Self-rated competence: high school biology 0.70, elementary mathematics 0.70, philosophy 0.69, ..."`, query `"Tasks of family professional law"`, task `"A task of family professional psychology (instance 7)."`

### 3.2.5 `retrieve(task)` (`_common.py:312-331`)
- **MIDIAN modes:** MIDIAN's pick `a = self.mid.fetch(task)` goes first, followed by the other members of `a`'s leaf cohort, with padding removed. With `shuffle`, that list is permuted by `stable_seed_32(a, "fw_shuffle", family)`. `k` is ignored here: the list length is the cohort size, at most `r`.
- **`sota`:** row `f` of the cached (K, k) table, with -1 padding dropped.
- **Everything else:**
  - `pool = _pool if dedup else arange(n)`.
  - Score on the pool:
    - `declared`: `view.declared[pool, f]`.
    - `bm25`: `_B[f][pool]`.
    - otherwise: the cosine `(_Xa @ _Xf[f])[pool]`, where rows are L2-normalised. For `hybrid` it is `_rrf(bm25, cosine)`.
  - The result is `pool[argsort(-score, stable)][:k]`: best first, **ties broken by lowest agent id**.
  - The shortlist depends only on the task's **family**, never on the instance text. Every task of family f gets the same list, except in the MIDIAN modes, whose pick can change online.
  - Any unrecognised `retrieval` string falls silently into the hashed-TF-IDF branch (`:328-329` and `:303-305`).

### 3.2.6 `fetch(task)`: calling the framework, parsing, fallback (`_common.py:362-398`)
1. `cand = retrieve(task)`. Ledger: `compare(len(cand))`, `hop(1)`, `message(len(cand)+2)` for k descriptions read plus request and reply (`:364-365`).
2. Payload: `[{"name": "agent_000123", "description": desc[a]} for a in cand]`, in shortlist order (`:366`).
3. `resp` is the prefetched response if one exists (3.2.8), otherwise `bridge.select(...)`. If `resp["error"]` is set and is not a supervisor invalid action, the call is retried once (`:368-369`).
4. **Supervisor invalid action** (`INVALID_ACTION`, `:26-29`; handled at `:371-378`; erratum 29, `CHANGES_AND_ERRATA.md` §8e). An error matching `Tool '…' not found` (ADK, OpenAI Agents: a tool named after the agent instead of the routing tool), `ModelBehaviorError` or `next_speaker must be provided` (MAF) is the framework failing to route, not infrastructure:
   - it increments `stats["invalid_action"]`, sets `_picked = False` and routes by declared argmax over `cand`;
   - it is **not retried** (no other non-pick gets a second sample) and never counts as an infrastructure error;
   - it enters `fallback_rate` and scores 0 under `success_strict`.
   Rows written before this rule are unchanged; the 33 units it had failed (31 ADK, plus `re_sl_declared_1k` MAF seed 3 and `re_sl_declared_5k` OpenAI Agents seed 1) are queued in the focus pack plan.
5. **Infrastructure error, still failing after the retry** (`:379-389`):
   - It increments `stats["infra_errors"]`, sets `_picked = False` and routes by declared argmax over `cand`.
   - Once `infra_errors > max(3, 0.02 × calls)`, it raises `RuntimeError("... refusing to write a fallback-contaminated row")`, so the unit writes **no row**.
   - This guard was added after erratum 28 (`CHANGES_AND_ERRATA.md` §8c; `OPS_RULES.md` D1).
6. **Valid pick:** `choice` is a known name whose id is in `cand`. Then `stats["picks"] += 1` and the arm routes to it (`:390-394`).
7. **Otherwise it is a genuine framework non-pick.** It is classified as follows (`:395-396`):
   - `failures` if the worker's `raw` begins with `"FAILURE"`, meaning the supervisor answered the task itself instead of delegating. The CrewAI, ADK and Magentic-One workers emit this.
   - `fallbacks` if `choice is None`.
   - `bad_name` if the model named a non-candidate or a non-agent, for example smolagents calling its `final_answer` tool.
   - In all three cases the adapter routes to **declared argmax within the shortlist** (`:397-398`).
8. **Scoring:**
   - The row's `success` is **lenient**: the fallback agent is executed and scored.
   - `success_strict` counts an outcome only when `_picked` is true (`observe`, `:333-336`).
   - `fallback_rate = 1 - picks/(picks+fallbacks+failures+bad_name+invalid_action)` (`:336`). `infra_errors` is excluded from the denominator.
   - All of these go in the row's `method_stats` JSON (`rte/run.py:152`).
9. **What "fallback rows" or "contamination" means (erratum 28).**
   - Before 2026-09-22, broken venvs made workers crash. `choice=None` then fell through to declared argmax, and the row looked normal while actually measuring declared argmax under the framework's name.
   - `scripts/ops/quarantine_fallback_rows.py:3-6,16-24` moved such rows to `results/<grid>/quarantine/`. The criterion was fallback > 0.05 for CrewAI and ADK, and ≥ 0.9 for every other framework. The units were then rerun.
   - Genuine non-picks are still routed by declared argmax. So even clean rows contain a declared-argmax component equal to `fallback_rate`, and that component is **the only way the inflate lie reaches a text-retrieval arm on live** (erratum 27).

### 3.2.7 What the framework does NOT get
- **No probes, no reports, and a build probe budget of 0.** The View denies access (`needs={"declared"}`), except in the MIDIAN modes.
- **No outcomes.** `observe` only updates `stats` (`:333-338`), except in the MIDIAN modes, where it forwards the outcome to MIDIAN.
- **No declared numbers in its prompt on live.** Only name and description are sent. Declared numbers are used by the adapter for `retrieval="declared"` and for the fallback. On RouterEval, though, the description *is* the rendered declared vector.
- **No true skill, no liar labels, and no family label on live.** On live the supervisor sees the real question text. On RouterEval it sees the family name and nothing else.
- **A fresh team per request, with no memory.** Every worker constructs a new team inside `select()`.

### 3.2.8 Parallel prefetch (`RTE_FW_PARALLEL`, `_common.py:340-360`)
- If `RTE_FW_PARALLEL > 1` and the retrieval is not a MIDIAN mode:
  - `prefetch(stream)` opens N extra Bridges (one worker process each).
  - It sends every task's request concurrently through a `ThreadPoolExecutor`.
  - It stores the responses by `id(task)`.
  - `fetch` then consumes them in order and falls back to a live call on an error.
- `rte/run.py:138-139` calls it before the task loop, except on churn grids. Its wall time is charged to routing.
- It is sound because each request is stateless and uses temperature 0.
- `tests/test_fw_sota_retrieval.py:255-268` shows identical picks, stats and ledger at 1 and 8 for tfidf, bm25 and declared. That test uses a mock endpoint.
- Against real vLLM, results are only as reproducible as greedy decoding:
  - Replicas are chosen at random.
  - Batch composition varies.
  - No `seed` is sent by any worker.
- So parallelism does not change results *systematically*, but bit-identity with a sequential run is not guaranteed on the real fleet, and it is not guaranteed between two sequential runs either.
- `launch_units.sh:35` passes `RTE_FW_PARALLEL` (default 1). `OPS_RULES.md` T1 prescribes 8, with `--mem=40G`.

### 3.2.9 The supervisor LLM and serving
- **Model.** `Qwen/Qwen2.5-7B-Instruct` (`_common.py:20`) for every framework. The exception is one labeled arm, `fw_magentic_one` with `supervisor: Qwen/Qwen2.5-14B-Instruct`, which runs in `fw_live_n1000`, `fw_live_n100` and their `_lowskill`, `_dd` and `_em` mirrors. That arm is dropped from the shortlist figures (`scripts/shortlist_figs.py:47`).
- **Endpoints.** `$RTE_DATA/endpoints.json`: keys `model#<slurm job>` are replicas.
- **Supervisor-only replicas.** `configs/fleet_supervisor.yaml` places just the 7B at `gpu_share 0.90` on one GPU. `serve_fleet.sbatch:64-72` selects the models through `RTE_FLEET_PLACEMENT` and notes that "the 7B supervisor took 487,222 requests in 16 h".
- **vLLM flags** (`scripts/_serve_env.sh:54-66`):
  - `--generation-config vllm` ignores Qwen's own repetition_penalty and temperature defaults.
  - `--enable-auto-tool-choice --tool-call-parser hermes` is needed by the tool-calling frameworks.
  - `--enable-prefix-caching`, max-model-len 8192, max-num-seqs 64.
  - `VLLM_USE_FLASHINFER_SAMPLER=0` (`:24`).
- **Temperature 0 is set client-side in every worker**: LangGraph `ChatOpenAI(temperature=0)`, CrewAI `LLM(temperature=0)`, AutoGen `OpenAIChatCompletionClient(temperature=0)`, MAF `default_options.temperature=0.0`, OpenAI Agents `ModelSettings(temperature=0.0)`, ADK `LiteLlm(temperature=0.0)`, LlamaIndex `OpenAILike(temperature=0.0)`, smolagents `OpenAIServerModel(temperature=0.0)`, CAMEL `model_config_dict={"temperature": 0.0}`, AgentScope `Parameters(temperature=0.0)`.

### 3.2.10 The Bridge subprocess protocol (`rte/methods/frameworks/_bridge.py`)
- **Protocol** (`:3-8`), one JSON object per line.
  - Request: `{"id", "task", "candidates": [{"name","description"}], "model", "base_url", "api_key", "params"}`.
  - Response: `{"id", "choice": str|null, "error": str|null, "raw"}`.
- **Interpreter.** `$RTE_DATA/env/<env>/bin/python` (`:22-26`).
- **Worker environment** (`:37-45`): `PYTHONPATH=workers/`, proxies stripped, `OPENAI_API_KEY=EMPTY` by default, `PYTHONNOUSERSITE=1`.
- **Reads.** Per-request timeout of 120 s (`:30`). Non-JSON stdout lines are skipped (`:78-91`).
- **Failures.** On a timeout, dead worker or bad JSON, the worker is killed, `{"error": ...}` is returned, and the worker restarts on the next call (`:61-66`).
- **Worker side.** `serve_worker(select_fn)` never dies on one bad request: an exception becomes `resp["error"]` (`:94-110`).
  - **An exception raised inside a worker is therefore reported as an infrastructure error, not as a framework non-pick.** Examples:
    - LlamaIndex's selector failing to parse the model output.
    - AgentScope `json.loads` failing.
    - `back[...]` KeyErrors in MAF, OpenAI Agents or LlamaIndex handoff.
  - Such errors count toward the 2 % unit-failure guard.

---

## 3.3 The frameworks

The code lists 13 `fw_*` classes (`rte/methods/frameworks/fw_*.py`).
- The **ten grid frameworks** are the YAML set `frameworks` (`configs/grid.yaml:38-39`): LangGraph, CrewAI, AutoGen, Magentic-One, MAF, OpenAI Agents SDK, Google ADK, LlamaIndex, smolagents and CAMEL Workforce.
- **AgentScope** appears only in the appendix grids `fw_appendix` and `fw_appendix_dd` (n = 100, specialist, beta ∈ {0, 0.25}, seeds 1-3, Q = 300; `grid.yaml:122-123,655`).
- **MetaGPT** is `NotImplementedError` (`fw_metagpt.py:1-11`).
- **Echo** is a protocol check that picks the first candidate (`fw_echo.py`, `workers/echo_worker.py`) and is never in a grid.

Versions are the pins in `requirements-frameworks/<fw>.txt`. I confirmed each against the installed `*.dist-info` in `$RTE_DATA/env/<env>/lib/python3.12/site-packages`, and they match.

| arm (`name`) | venv (`env`) | library, pinned = installed | worker | selection primitive driven, and how the pick is captured | what the supervisor reads |
|---|---|---|---|---|---|
| `fw_langgraph` | fw_langgraph | langgraph 1.2.11, langgraph-supervisor 0.0.31, langchain-openai 1.6.0 | `langgraph_worker.py` | `create_supervisor(agents, prompt=...)`, with one `create_react_agent(model, tools=[], name=...)` per candidate. The pick is the first tool call named `transfer_to_<name>`; the stream is then closed, aborting the graph (`:23-35`) | Its prompt holds the roster `- name: description`, "Assign the user's task to exactly one agent by calling its transfer tool. Do not do any work yourself." (`:17-19`), plus the task as the user message |
| `fw_crewai` | fw_crewai | crewai 1.15.18 | `crewai_worker.py` | `Process.hierarchical` crew with an explicit `manager_agent` "Dispatcher" (`:55-58`). The coworkers are `Agent(role="<name>: <description>", backstory=description)`. `BaseAgentTool._execute` is monkey-patched to raise `Picked(coworker)` (`:37-41`), and the name is taken with the regex `agent_\d{6}` (`:64`). A kickoff with no delegate call returns `"FAILURE: manager answered itself"` (`:66`) | The manager's delegate-tool description lists the coworker roles; the task is "Delegate this to exactly one coworker: {task}" |
| `fw_autogen` | fw_autogen | autogen-agentchat 0.7.5, autogen-ext[openai] 0.7.5 | `autogen_worker.py` | `SelectorGroupChat(Idle participants, allow_repeated_speaker=True, MaxMessageTermination(2))`. The pick is the first `SelectSpeakerEvent` (`:44-60`). Participants are `Idle` agents, so none executes (`:26-35`). **Library-internal fallback:** after 3 failed selector attempts, AutoGen itself returns the previous speaker, or else `participants[0]`, which is shortlist position 1 (`autogen_agentchat/.../_selector_group_chat.py:301-307` in the venv). The adapter counts that as a valid pick | The selector prompt's roster `"<name>: <description>"` |
| `fw_magentic_one` | fw_autogen (shared) | same as AutoGen | `magentic_one_worker.py` | `MagenticOneGroupChat(max_turns=1)`, with the ledger requested as structured output. `_log_message` is patched to catch `"Next Speaker: <name>"` (`:43-53`). With `robust=True`, the default and never overridden in any grid, a rejected ledger is read by mention: the first `agent_\d{6}` after "next_speaker". A satisfied ledger, or one naming nobody, is a `FAILURE` (`:37-40,66-69`) | The orchestrator's `{team}` roster built from the descriptions |
| `fw_maf` | fw_maf | agent-framework-core 1.16.0, -openai 1.14.1, -orchestrations 1.1.1 | `maf_worker.py` | Default `mode="groupchat"` (`fw_maf.py:9`; no grid sets `handoff`): `GroupChatBuilder(participants, orchestrator_agent=router, max_rounds=1)`, and the pick is the first `GroupChatRequestSentEvent.participant_name` (`:37-42`). `handoff` mode, never run, would use `HandoffBuilder` and `HandoffSentEvent.target`. Names are sanitised to identifiers and mapped back | Router instruction "You are a router. Pick the single participant best suited to solve the task. Do not solve it yourself." plus the participants' `name` and `description` |
| `fw_openai_agents` | fw_openai_agents | openai-agents 0.22.0 | `openai_agents_worker.py` | Triage `Agent(handoffs=specialists)`. Each specialist has `handoff_description=description`, which becomes a `transfer_to_<name>` tool. `RunHooks.on_handoff` raises `Picked(to_agent.name)` (`:26-28`). `max_turns=1`, `use_responses=False`. If the run ends without a handoff, the result is `None` → counted as `fallbacks` | "You are a triage agent. Hand the task off to the single specialist best suited to solve it." |
| `fw_google_adk` | fw_google_adk | google-adk 2.8.0, litellm 1.99.0 | `google_adk_worker.py` | Router `LlmAgent(sub_agents=...)` with auto-delegation via `transfer_to_agent`. The pick is the first `event.actions.transfer_to_agent` (`:37-40`). No transfer returns `"FAILURE: router answered itself"` (`:41`) | ADK's roster of `Agent name:` / `Agent description:`, plus "You are a router. Transfer the task to the single sub-agent best suited to solve it." |
| `fw_llamaindex` | fw_llamaindex | llama-index-core 0.14.24, llama-index-llms-openai-like 0.8.0 | `llamaindex_worker.py` | Default `mode="selector"` (`fw_llamaindex.py:9`; no grid sets `handoff`): one `LLMSingleSelector.aselect([ToolMetadata(name, description)], task)` call, taking `selections[0].index`. An out-of-range index gives `None` → `fallbacks`. A parser exception becomes a worker error (`:40-50`) | The selector prompt listing the numbered descriptions |
| `fw_smolagents` | fw_smolagents | smolagents 1.26.0 | `smolagents_worker.py` | Manager `ToolCallingAgent(managed_agents=...)`, `max_steps=1`, `run(stream=True)`. The pick is the first `ToolCall` event's name, taken before execution (`:23-27`). A `final_answer` call → `bad_name` | Managed agents as tools, `name` plus `description` |
| `fw_camel_workforce` | fw_camel | camel-ai 0.2.90 | `camel_worker.py` | `Workforce` with one `add_single_agent_worker(description, ...)` per candidate. `node_id` is overwritten with the candidate name (`:37-39`). A `WorkforceCallback.log_task_assigned` raises `Picked(worker_id)` (`:16-22,40-43`) | The coordinator's `<node_id>:<description>:<toolkits>` list |
| `fw_agentscope` (appendix) | fw_agentscope | agentscope 2.0.7 | `agentscope_worker.py` | No selection primitive, so this is a DIY router: one `OpenAIChatModel` call json_schema-constrained to `{"agent": name}` (`:11-30`) | Roster plus task in a hand-written prompt |

Notes that apply to all of them:
- Every worker stops **before the chosen agent runs**, so the only LLM traffic is the supervisor's. The routed task is then executed by the benchmark world, not by the framework.
- The agents inside the framework are placeholders; only the manager or router makes model calls.
- Names such as `agent_000123` are already identifier-safe, so `sanitize` (`workers/_wk.py:19-22`) is a no-op on them.
- **Magentic-One exclusions.** It is left out of every RouterEval framework grid and of `fw_live_n10k_cartel` and its `_dd`, `_em`, `_sota`, `_shapes` and `_verified_va` mirrors, "for time (18 s/task)" (`grid.yaml:515-526`, `528-540`; `DEVIATIONS.md` §"Frameworks on RouterEval's 5,000-LLM pool"). It **is** included in:
  - `fw_live_n10k_cartel_backfill` (it inherits `*fw10_backfill`, all ten frameworks);
  - `fw_live_n10k_cartel_magentic` (`grid.yaml:1129-1132`): Magentic-One alone at the 10^4 cartel, under TF-IDF, MiniLM,
    fusion + reranker and the VA cohort, with the honest grids' params (queued, no rows yet);
  - `lie_max_fw_n10k` (`*fw10_declared`);
  - every other live grid.
- The CAMEL rows of `fw_routereval_1k` are partial: 8/30 units, excluded from the published ranges according to DEVIATIONS.

---

## 3.4 Every shortlist variant

The figure label comes from `source(params)` in `scripts/shortlist_figs.py:42-54`. It works as follows:
- Rows with `supervisor` or `lie_text` in their params → not drawn.
- `retrieval` absent: `dedup` → not drawn; otherwise `tfidf`.
- `midian_va` with `r == 10` and no `shuffle` → `va_cohort`.
- `embed` → `dense[_icomp|_idemo]` if the `embed_model` contains "Qwen", else `embed` (MiniLM).
- `sota` → `sota[_icomp|_idemo]`. The tag comes from the substring "competent" or "demonstrated" in `embed_instruct`.
- `bm25` and `declared` keep their names.
- Everything else (`hybrid`, `midian`, `tfidf` + dedup) → not drawn.

Display names come from `SOURCES` (`shortlist_figs.py:30-36`); short legend names come from `SHORT` (`scripts/shortlist_condensed.py:25-27`).
- Fig E draws `MAIN`, which is every source except `bm25` and `dense` (`:28`).
- Fig H draws `MAIN` too (`:134-135`), on the RouterEval rows.

### 3.4.1 `tfidf`: pre-registered hashed TF-IDF (label "hashed TF-IDF (pre-registered)" / "TF-IDF (pre-reg.)")
- **Params:** none, i.e. the defaults `retrieval="tfidf"`, `dedup=False`, `k=10`.
- **Grids:**
  - `fw_live_n1000` and `fw_live_n100` (+ `_lowskill`)
  - `live_n10k_v2` and `live_n100k` (their `*frameworks` arms)
  - `fw_live_n10k_cartel`, `fw_live_n10k_beta01`, `fw_live_n10k_shapes`, `fw_live_n10k_cartel_shapes`, `live_n100k_beta01`
  - `fw_routereval_small`, `_1k`, `_5k`
- **Algorithm** (`_hash_tfidf`, `_common.py:33-43`):
  1. Tokens are `[a-z0-9]+` on lowercased text, so `basic_arithmetic` splits into `basic` and `arithmetic`.
  2. Each token is hashed with `blake2b(digest_size=4) mod 4096`, giving counts in 4,096 buckets.
  3. IDF is `log((N+1)/(df+1))` with N = n + K. It is fitted on the agent descriptions **and** the 16 family queries together.
  4. Rows are L2-normalised, and similarity is cosine against the family description.
- **Erratum 25: clone shortlists** (`CHANGES_AND_ERRATA.md` item 25).
  - Agents that share a prompt signature have byte-identical memoized descriptions and identical answers.
  - The stable sort therefore fills the top 10 with clones of the best-matching text.
  - bimodal has 2 distinct texts and heavy_tail 5, at any n. specialist has ~3,920 distinct texts, giving ~4.4 signatures per top 10 at 10^3, ~2.6 at 10^4, and **exactly 1 at 10^5**.
  - At 10^5 every framework therefore gets ten copies of one agent, and its choice cannot matter. `shortlist_condensed.py:9-10` states that TF-IDF is "0.379 for every framework" at 10^5.
  - **Fig G's baseline is this clone floor.** Every other drawn text arm uses `dedup: true`, so Fig G's "gain over TF-IDF" includes the dedup gain.
- **Liars:**
  - Live: they cannot move the ranking, because the text is independent of lying (erratum 27). They reach the row only through the fallback.
  - RouterEval: they **can**. The query `"Tasks of family X"` matches every agent whose declared top 5 contains X. Inflation (+0.4, clipped to 1) pushes liars into many top-5s and, with ties broken by id, to the front. In my check (routereval mmlu n = 1,000, strong_to_weak, beta 0.5 low_skill_first, seed 1), family 0's TF-IDF top 10 was **100 % liars** (mean true skill 0.276 against a population mean of 0.428). See 3.9 on the "degenerate" wording.

### 3.4.2 `dedup` (TF-IDF + `dedup: true`): computed but not drawn
- **Params:** `{dedup: true}` (`&fw10_dd`, `grid.yaml:634-657`).
- **Grids:** `fw_live_n1000_dd`, `fw_live_n100_dd`, the `_lowskill_dd` pair, `fw_live_n10k_dd`, `fw_live_n100k_dd`, `fw_live_n10k_cartel_dd`, `fw_k_sensitivity_dd`, `fw_appendix_dd`, `budget_b10_fw_dd` and `churn_n1000_fw_dd`.
- **Mechanism:** the same ranking, restricted to `_pool`, the lowest id per distinct description string (`_common.py:307-309,324`).
- **Figures:** excluded by `source()`.
- **Liars:** the same as TF-IDF. The one difference is which member of a clone group is offered: the lowest id, whatever its liar status.

### 3.4.3 `bm25` (label "BM25")
- **Params:** `{retrieval: bm25, dedup: true}` (`&fw10_bm25`, `grid.yaml:692-701`).
- **Grids:**
  - `fw_live_n1000_sota` only, at n = 10^3, all 3 shapes × 4 random-liar betas, seeds 1-10. This is the "four-arm ablation".
  - Also `fw_live_n100k_lietext` (honest and lied text). The `GRIDS` filter excludes `lietext` grids, so these never reach the figures (`shortlist_figs.py:38`).
- **Algorithm:** Okapi BM25 with k1 = 1.5 and b = 0.75 over an inverted index; IDF is `log(1 + (N − df + 0.5)/(df + 0.5))`. Documents are the n agent descriptions and queries are the K family descriptions, including the example question's tokens, numbers too (`_common.py:46-64`).
- **Figures:** Fig E leaves it out (`MAIN`).
- **Liars:** text only, so the same as TF-IDF.

### 3.4.4 `embed`: MiniLM (label "MiniLM")
- **Params:** `{retrieval: embed, dedup: true}`, with `embed_model` at its default `all-MiniLM-L6-v2` (`&fw10_em`, `grid.yaml:664-677`; `&fw9_em`, `grid.yaml:995-1003`).
- **Grids:**
  - `fw_live_n1000_em`, `fw_live_n100_em`, the `_lowskill_em` pair, `fw_live_n10k_em`, `fw_live_n100k_em`, `fw_live_n10k_cartel_em`
  - `fw_routereval_1k_em`, `_5k_em`, `_small_em`
- **Algorithm:** cosine between unit-norm MiniLM embeddings of the descriptions and the family descriptions. There is no query prompt (`_common.py:253`: `qp=None` for MiniLM), and it runs on CPU (`_learned.py:22-36`).
- **Cache:**
  - Live: `descriptions_minilm.npy` in the population dir.
  - RouterEval: no cache, unless `RTE_EMBED_CACHE_DIR` is set. It is recomputed on CPU per unit, which is cheap.
- **Liars:** live text only. RouterEval rendered numbers, so manipulable as in 3.4.1.

### 3.4.5 `dense`: Qwen3-Embedding-8B with its stock instruction (label "Qwen3-8B dense" / "dense")
- **Params:** `{retrieval: embed, dedup: true, embed_model: Qwen/Qwen3-Embedding-8B}` (`&fw10_dense`, `grid.yaml:702-711`).
- **Grids:** `fw_live_n1000_sota` only (the 10^3 ablation). Not in Fig E's `MAIN`.
- **Query side:**
  - With no `embed_instruct`, the family texts are encoded with `prompt_name="query"`. That is the model's shipped prompt `"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"` (`$RTE_DATA/models/Qwen3-Embedding-8B/config_sentence_transformers.json`).
  - Documents get no prompt (`_common.py:253-254,270`).
- **Hardware:** bf16 on GPU (`_learned.py:29-31`). A routing unit with no cached block and no GPU raises instead of computing (`_common.py:259-263`).

### 3.4.6 `dense_icomp` / `dense_idemo`: Qwen3 dense with a task instruction (labels "Qwen3-8B dense, I-competent" / "I-demonstrated"; "dense, I-comp" / "dense, I-demo")
- **Exact instruction strings** (`grid.yaml`; each appears on 30 / 20 lines, plus 18× on each of the three flow-style `re_sl_embed_*` lines):
  - **I-comp:** `"Given a task family, retrieve agents that are competent at solving tasks of that family"` (cache tag `_i0258be9d`)
  - **I-demo:** `"Given a task family, retrieve agents whose demonstrated skill at that family is highest"` (cache tag `_i06c7d226`)
- **Query prompt:** `"Instruct: {embed_instruct}\nQuery:"` (`_common.py:254`). Only the K = 16 family vectors change (`families_<slug>_i<hash>.npy`); the n document vectors are reused.
- **Params:** `{retrieval: embed, dedup: true, embed_model: Qwen/Qwen3-Embedding-8B, embed_instruct: <string>}` (`&fw10_dense_ic`, `&fw10_dense_id`, from `grid.yaml:802`).
- **Grids:**
  - `fw_live_n100k_dense_instruct`
  - the backfill grids `fw_live_n{100,1000}[_lowskill]_backfill`, `fw_live_n10k_backfill`, `fw_live_n10k_cartel_backfill` (`grid.yaml:1018-1023`)
  - `re_sl_embed_{small,1k,5k}` (`grid.yaml:1095-1097`)
- **Liars:** live text only. On RouterEval, rendered numbers (manipulable).

### 3.4.7 `hybrid`: RRF(BM25, Qwen3 dense), not drawn
- **Params:** `{retrieval: hybrid, dedup: true, embed_model: Qwen/Qwen3-Embedding-8B}`, plus an I-comp variant in `fw_live_n100k_sota_instruct`.
- **Algorithm:** `_rrf` gives each list the score `1/(60 + 1 + rank)` and sums the two lists, ranking over the pool only (`_common.py:67-73,330`).
- **Grids:** every `*_sota` grid and `fw_live_n100k_sota_instruct`.
- **Figures:** excluded by `source()` ("fusion without the reranker").

### 3.4.8 `sota`, `sota_icomp`, `sota_idemo`: fusion + cross-encoder (labels "fusion + reranker[, I-competent / I-demonstrated]"; "rerank[, I-comp / I-demo]")
- **Algorithm** (`sota_shortlist`, `_common.py:76-86`), per family f:
  1. `fused = RRF(BM25[f][pool], (Xa @ Xf[f])[pool])`.
  2. Take the top `rerank_pool = 50` of the pool by `fused`.
  3. Rerank them with `CrossEncoder(Qwen/Qwen3-Reranker-4B).predict([(family description, agent description)])`. The model's own template is used (`_rerank`, `:272-282`).
  4. Keep the top k = 10, padding with -1 if fewer.
- **Setup:** `pool` is `_pool` because every grid sets `dedup: true`. The table is (K, k) and depends only on the population.
- **Params:**
  - Stock: `{retrieval: sota, dedup: true, embed_model: Qwen/Qwen3-Embedding-8B, rerank_model: Qwen/Qwen3-Reranker-4B}` (`&fw10_sota`, `grid.yaml:722-731`).
  - Instruct variants add `embed_instruct`. The instruction affects only the dense half's **query** vectors, and therefore the fused pool. The reranker never sees it.
- **Grids:**
  - Stock: `fw_live_n1000_sota`, `fw_live_n100_sota`, `fw_live_n1000_lowskill_sota`, `fw_live_n100_lowskill_sota`, `fw_live_n10k_sota`, `fw_live_n100k_sota`, `fw_live_n10k_cartel_sota` (9 frameworks), and `re_sl_embed_*`.
  - I-comp and I-demo: `fw_live_n100k_sota_instruct` (sota I-comp, sota I-demo, hybrid I-comp), the backfill grids, and `re_sl_embed_*`.
- **Liars:** live text only. On RouterEval, rendered numbers (manipulable).

### 3.4.9 `declared`: declared top-k (label "declared-claim top-k" / "declared top-k")
- **Params:** `{retrieval: declared, dedup: true}` (`&fw10_declared`, `grid.yaml:832-843`).
- **Grids:**
  - `fw_live_n100k_declared`
  - the backfill grids
  - `lie_max_fw_n1000`, `_n10k`, `_n100k` (`grid.yaml:1084-1086`, with `lie_mode: max`)
  - `re_sl_declared_{small,1k,5k}` (`grid.yaml:1092-1094`)
- **Algorithm:** `view.declared[pool, f]`, stable-sorted descending, top 10 (`_common.py:325-326`). There is no text retrieval. The framework still receives the **text** descriptions of these 10.
- **Setup:**
  - `_index` still builds a hashed-TF-IDF matrix, which is not used.
  - `dedup` restricts the ranking to one representative per distinct text, so liars who are not the lowest id of their text group are invisible to it.
- **Interpretation.** This idealises agent-card or registry discovery: an A2A-style registry where each agent publishes a structured skill claim and the orchestrator shortlists by that claim (`grid.yaml:1077-1079` calls it "an idealized A2A Agent-Card registry").
- **Liars: fully manipulable.**
  - `inflate` (+0.4) moves liars up. The grid comment at `grid.yaml:831` reports 74 % liars in the top 10 at beta = 0.5 random, against a 50 % population rate.
  - `max` sets `D[liars] = 1` (`world.py:132-133`). Ties are broken by lowest id, so the shortlist is typically ten liars.
  - My RouterEval check (beta 0.5 low_skill_first, `inflate`) gave 100 % liars in the declared top 10 for families 0, 3 and 7.
  - Honest agents' declared values are themselves noisy: on live, `D_self_described` is the model's self-rating (`llm.py:212-230`); on RouterEval, `noisy_declared(S)`.
- **Hardware:** needs no GPU or cache.

### 3.4.10 `va_cohort` (retrieval `midian_va`) and `midian` (MIDIAN-V cohort, not drawn)
- **Params:**
  - `{retrieval: midian_va, r: 10}`. Grids: `fw_live_n1000_verified_va`, `fw_live_n1000_verified_va_lowskill`, `fw_live_n100_verified_va[_lowskill]`, `fw_live_n10k_verified_va`, `fw_live_n10k_cartel_verified_va` (9 frameworks), `fw_live_n100k_verified_va`, and `fw_routereval_{small,1k,5k}_va` (9).
  - `midian`: `{retrieval: midian, r: 10 | 5}` in `fw_live_n1000_verified` and `fw_live_n100_verified`. The label is None (the MIDIAN-V cohort is not drawn).
- **Mechanism:**
  - `build` constructs `MidianVA(r=10)`, i.e. `MidianA(verify=True, cached=True)` (`rte/methods/midian_va.py:10-14`), and calls its full `build(view, budget)` (`_common.py:290-292`).
  - `retrieve` returns `[VA's cached root pick for the family] + the other members of that pick's leaf cohort`, up to 10 agents, pick first (`_common.py:313-319`). The cohort is random (plain `cohort="random"`).
  - `fetch` also calls `self.mid.fetch`, which charges 1 comparison and 2 messages (`midian.py:149-151`), and then charges the framework ledger on top.
  - `observe` forwards the **framework's chosen agent's** outcome to MIDIAN-VA (`_common.py:337-338`). MIDIAN-VA keeps learning online, including its 5 % report audits, from the framework's routing. That is why prefetch is disabled (`:346`).
- **Yes, it spends probes: the same budget as a MIDIAN-VA arm.**
  - At the grid's `b = 3`, the level-0 estimate uses `b0 = b − 1 = 2` probes per (agent, family). The remaining `(b − b0)·n` per family is spent on verification re-probes at promotion (`midian.py:105,111-113`).
  - The total is n·K·b = 48n probes at K = 16: 48,000 at n = 10^3 and 4.8 M at 10^5.
  - MIDIAN-A's build audits add up to 5 % (`midian_a.py:1-8`).
  - Probes and peer reports are charged to the row's `build_*` ledger columns.
  - The fallback, still declared argmax within the cohort, needs `declared`, which is kept in `needs`.
- **Liars:** they can distort the cohort only through the peer reports, which the audits and exclusion check, and through the fallback. Descriptions play no part in which agents are shortlisted.

### 3.4.11 `shuffle`: position control, not a shortlist source
- **Params:** `{retrieval: midian_va, r: 10, shuffle: true}`. Grid: `fw_live_n100k_verified_va_shuffled` (`grid.yaml:852-871`).
- **Mechanism:** the same members in a deterministic permutation seeded by (pick, "fw_shuffle", family) (`_common.py:317-318`). It separates "better material" from "the pick is first".
- **Figures:** excluded by `source()`.

### 3.4.12 `lie_text` (erratum 27): not drawn
- **Params:** `{retrieval: tfidf|bm25, dedup: true, lie_text: true}` in `fw_live_n100k_lietext`, which also holds the honest-text twins `{retrieval: tfidf|bm25, dedup: true}` (`grid.yaml:872-914`). `fw_live_n100k_lietext_th` adds `claim_threshold: 0.7` (`grid.yaml:920-942`).
- **Mechanism** (`_relabel`, `_common.py:166-177`), applied to **every** agent's description:
  1. Strip the trailing `Declared areas: ...` clause.
  2. Append `Declared areas: <families>.`, built from `view.declared`: the top 3 families, or with `claim_threshold > 0` every family with D > threshold (at least the argmax).
  3. The LLM prose is untouched, so the prose still describes the true specialty. It is a partial text lie.
  4. Cache names get `_lt<hash of the texts>` (`_common.py:214-220`).
- **Liars:** this is the only variant in which the live lie reaches the text. With top 3, it barely changes anything, because `inflate` preserves rank: "only 14 % of cartel liars' top-3 changes" (`_common.py:146-149`). With threshold 0.7, a cartel liar claims about 15 of 16 families.
- **Figures:** excluded both by `source()` and by the `GRIDS` name filter.

### 3.4.13 k sensitivity
`fw_k_sensitivity` and `fw_k_sensitivity_dd` run LangGraph and AutoGen at k ∈ {5, 10, 20}. The grid name does not start with `fw_live_n`, so they are not in the shortlist figures.

### 3.4.14 Summary table

| shortlist label (`source`) | grid params | signal used | manipulable by liars? | GPU / cache needed |
|---|---|---|---|---|
| `tfidf`, "hashed TF-IDF (pre-registered)" | none (`retrieval` absent, no dedup) | hashed TF-IDF cosine, desc vs family description; **clone-filled** (erratum 25) | live: no, the text ignores the lie (fallback only). RouterEval: **yes**, the text is the rendered declared vector | no |
| (not drawn) TF-IDF + dedup | `dedup: true` | same, one agent per distinct text | as `tfidf` | no |
| `bm25`, "BM25" | `retrieval: bm25, dedup: true` | Okapi BM25 (k1 = 1.5, b = 0.75) | as `tfidf` | no |
| `embed`, "MiniLM" | `retrieval: embed, dedup: true` | all-MiniLM-L6-v2 cosine | as `tfidf` | CPU; live cache `descriptions_minilm.npy` |
| `dense`, "Qwen3-8B dense" | `retrieval: embed, dedup: true, embed_model: Qwen/Qwen3-Embedding-8B` | Qwen3-Embedding-8B cosine, stock web-search query prompt | as `tfidf` | GPU pre-warm; `descriptions_qwen_qwen3_embedding_8b.npy`, `families_..._8b.npy` |
| `dense_icomp` / `dense_idemo` | the above + `embed_instruct: "...competent at solving..."` / `"...whose demonstrated skill..."` | same, task-specific query instruction | as `tfidf` | GPU pre-warm; `families_..._i0258be9d.npy` / `_i06c7d226.npy` |
| (not drawn) `hybrid` | `retrieval: hybrid, dedup: true, embed_model: Qwen3-8B` | RRF(k = 60) of BM25 and dense | as `tfidf` | GPU pre-warm (dense block) |
| `sota`, "fusion + reranker" | `retrieval: sota, dedup: true, embed_model: Qwen3-8B, rerank_model: Qwen/Qwen3-Reranker-4B` | RRF(BM25, dense) → top 50 → cross-encoder → top 10 | as `tfidf` | GPU pre-warm; `shortlist_sota_..._k10p50_dd.npy` |
| `sota_icomp` / `sota_idemo` | the above + `embed_instruct` | same; the instruction changes the dense query side only | as `tfidf` | GPU pre-warm; `..._k10p50_dd_i<hash>.npy` |
| `declared`, "declared-claim top-k" | `retrieval: declared, dedup: true` | self-declared D[:, f], top 10 | **yes, directly** (inflate, max) | no |
| `va_cohort`, "MIDIAN-VA leaf cohort" | `retrieval: midian_va, r: 10` | MIDIAN-VA's probed and audited pick + its random leaf cohort | only via peer reports (audited) and the fallback | no GPU; **spends 48n probes (+≤5 % audits)** + reports |
| (not drawn) MIDIAN-V cohort | `retrieval: midian, r: 10/5` | MIDIAN-V pick + cohort | reports + fallback | spends probes |
| (not drawn) shuffle | `retrieval: midian_va, r: 10, shuffle: true` | VA cohort, permuted | as `va_cohort` | as `va_cohort` |
| (not drawn) lie_text | `lie_text: true` [+ `claim_threshold: 0.7`], tfidf/bm25 | text with the Declared-areas clause taken from D | **yes (partially)**, by construction | no |

---

## 3.5 Which (framework × shortlist × n × regime) combinations exist

I resolved these with `rte.run.blocks` (mirror chains followed) on `configs/grid.yaml`. The field values are:
- live `dist` ⊂ {specialist, heavy_tail, bimodal}
- `declared_source = self_described` on every live framework grid
- `b = 3` on all of them except `budget_b10_*`
- RouterEval `declared_source = programmatic`
- "10" = the ten frameworks and "9" = without Magentic-One
- "cartel" = beta 0.5 with `low_skill_first` liars

| grid(s) | backend, n | dist | beta × liar_select | seeds, Q | frameworks × shortlist | figure label |
|---|---|---|---|---|---|---|
| `fw_live_n1000`, `fw_live_n100` | llm 10^3 / 10^2 | 3 shapes | {0, .1, .25, .5} × random | 1-10, 1000 | 10 × tfidf, + Magentic-One 14B | tfidf |
| `fw_live_n1000_lowskill`, `fw_live_n100_lowskill` | llm 10^3 / 10^2 | 3 | .5 × low_skill_first | 1-10, 1000 | 10 × tfidf, + M1-14B | tfidf |
| `live_n10k_v2` (fw arms) | llm 10^4 | specialist | {0, .25} × random | 1-3, 300 | 10 × tfidf | tfidf |
| `fw_live_n10k_cartel` | llm 10^4 | specialist | cartel | 1-3, 300 | 9 × tfidf | tfidf |
| `fw_live_n10k_cartel_magentic` | llm 10^4 | specialist | cartel | 1-3, 300 | Magentic-One × {tfidf, MiniLM, sota, VA cohort} | tfidf, embed, sota, va_cohort |
| `fw_live_n10k_beta01` / `fw_live_n10k_shapes` / `fw_live_n10k_cartel_shapes` | llm 10^4 | spec / {ht, bim} / {ht, bim} | .1 random / {0, .25} random / cartel | 1-3, 300 | 10 / 10 / 9 × tfidf | tfidf |
| `live_n100k` (fw arms), `live_n100k_beta01` | llm 10^5 | specialist | {0, .25, .5} × {random, low_skill_first}; {.1} | 1-3, 300 | 10 × tfidf | tfidf |
| `*_dd` (11 grids) | as their sources | | | | 10 (9 cartel-10^4) × tfidf + dedup | not drawn |
| `fw_live_n{1000,100}[_lowskill]_em`, `fw_live_n10k_em`, `fw_live_n100k_em`, `fw_live_n10k_cartel_em` | as sources | | | | 10 (+M1-14B at 10^2 and 10^3) / 9 × MiniLM | embed |
| `fw_live_n1000_sota` | llm 10^3 | 3 | {0, .1, .25, .5} × random | 1-10, 1000 | 10 × {bm25, dense, hybrid, sota} | bm25, dense, –, sota |
| `fw_live_n100_sota`, `fw_live_n{1000,100}_lowskill_sota`, `fw_live_n10k_sota`, `fw_live_n100k_sota` | as sources | | | | 10 × {hybrid, sota} | –, sota |
| `fw_live_n10k_cartel_sota` | llm 10^4 cartel | spec | cartel | 1-3, 300 | 9 × {hybrid, sota} | –, sota |
| `fw_live_n100k_sota_instruct` | llm 10^5 | spec | 6 regimes as live_n100k | 1-3, 300 | 10 × {hybrid-Icomp, sota-Icomp, sota-Idemo} | –, sota_icomp, sota_idemo |
| `fw_live_n100k_dense_instruct` | llm 10^5 | spec | 6 regimes | 1-3, 300 | 10 × {dense-Icomp, dense-Idemo} | dense_icomp, dense_idemo |
| `fw_live_n100k_declared` | llm 10^5 | spec | 6 regimes | 1-3, 300 | 10 × declared | declared |
| `fw_live_n{100,1000}_backfill` | llm 10^2 / 10^3 | spec | 0 | 1-3, 1000 | 10 × {dense-Ic, dense-Id, sota-Ic, sota-Id, declared} | same labels |
| `fw_live_n{100,1000}_lowskill_backfill` | llm 10^2 / 10^3 | spec | cartel | 1-3, 1000 | same 10 × 5 | same |
| `fw_live_n10k_backfill` / `fw_live_n10k_cartel_backfill` | llm 10^4 | spec | 0 random / cartel | 1-3, 300 | **10** × 5 (includes Magentic-One even in the cartel grid) | same |
| `fw_live_n1000_verified_va` / `fw_live_n100_verified_va` | llm 10^3 / 10^2 | 3 | {0, .1, .25, .5} × random | 1-10, 1000 | 10 × VA cohort | va_cohort |
| `fw_live_n{1000,100}_verified_va_lowskill` | llm | 3 | cartel | 1-10, 1000 | 10 × VA | va_cohort |
| `fw_live_n10k_verified_va` / `fw_live_n10k_cartel_verified_va` / `fw_live_n100k_verified_va` | llm 10^4 / 10^4 / 10^5 | spec | {0, .25} random / cartel / 6 regimes | 1-3, 300 | 10 / 9 / 10 × VA | va_cohort |
| `fw_live_n100k_verified_va_shuffled` | llm 10^5 | spec | 6 | 1-3, 300 | 10 × VA shuffled | not drawn |
| `fw_live_n{1000,100}_verified` | llm | 3 | 4 × random | 1-10, 1000 | 10 × MIDIAN-V cohort, r ∈ {10, 5} | not drawn |
| `fw_live_n100k_lietext` / `_lietext_th` | llm 10^5 | spec | 6 | 1-3, 300 | 10 × {tfidf, tfidf-lie, bm25, bm25-lie} / {tfidf-lie-th, bm25-lie-th} | not drawn |
| `lie_max_fw_n1000` / `_n10k` / `_n100k` | llm 10^3 / 10^4 / 10^5, `lie_mode: max` | spec | cartel | 1-3; Q 1000 / 300 / 300 | 10 × declared | not drawn (grid-name filter; see 3.9) |
| `fw_routereval_small` (n ∈ {10, 100}) / `_1k` | routereval mmlu | 3 pools | {0, .5} × {random, low_skill_first} | 1-5, 1000 | 9 × tfidf | tfidf (Fig H) |
| `fw_routereval_5k` | routereval leaderboard_mmlu 5,000 | all | same | 1-3, 300 | 9 × tfidf | tfidf (H) |
| `fw_routereval_{small,1k,5k}_em` / `_va` | as above | | | | 9 × MiniLM / VA cohort | embed / va_cohort (H) |
| `re_sl_declared_{small,1k}` / `_5k` | routereval | strong_to_weak / all | {0, .5} × low_skill_first | 1-5, 1000 / 1-3, 300 | 9 × declared | declared (H) |
| `re_sl_embed_{small,1k}` / `_5k` | same | same | same | same | 9 × {dense-Ic, dense-Id, sota, sota-Ic, sota-Id} | dense_icomp, dense_idemo, sota, sota_icomp, sota_idemo (H) |
| `fw_k_sensitivity[_dd]` | llm 10^3 | spec | {0, .25} random | 1-3, 300 | LangGraph and AutoGen × k ∈ {5, 10, 20} | not drawn |
| `fw_appendix[_dd]` | llm 10^2 | spec | {0, .25} random | 1-3, 300 | AgentScope | not drawn |
| `budget_b10_shapes` / `_fw_dd`, `churn_n1000` / `_fw_dd` | llm 10^3 | | | | LangGraph, AutoGen | not drawn |

**Results on disk at ~02:30 on 2026-09-23.**
- `$RTE_DATA/results/` has a directory for every grid above **except**:
  - the six `*_backfill` grids and `fw_live_n10k_cartel_magentic`, which are queued in the focus pack plan
    (`$RTE_DATA/logs/focus_pack_plan.tsv`, 30 units per backfill grid, 3 for Magentic-One);
  - `lie_max_fw_n100k`;
  - `re_sl_embed_5k`.
- `lie_max_fw_n1000` has 23 rows and `lie_max_fw_n10k` none. `re_sl_declared_small` has 194 `rows.d` entries,
  `re_sl_declared_1k` 86 and `re_sl_declared_5k` 58. `re_sl_embed_small` and `re_sl_embed_1k` have empty directories; their
  units are queued behind the cache pre-warm (`scripts/embed_routereval.py`).
- Fig E's empty `*` slots at n ≤ 10^4 are the backfill shortlists (dense I-comp / I-demo, rerank I-comp / I-demo,
  declared). H's empty slots are the `re_sl_*` shortlists that have not landed.
- The RouterEval embedding cache root `$RTE_DATA/cache/embed_routereval/` holds 66 content-keyed directories.

---

## 3.6 Embedding and shortlist caches

- **`_popdir(view)`** (`_common.py:194-212`) decides where the cache files for a population live.
  - **Live:** the backend's population dir, `$RTE_DATA/populations/<dist>_n<n>_K16_seed<s>/`.
  - **Other backends:** a directory under `$RTE_EMBED_CACHE_DIR`, named by `blake2b(digest 12)` of the exact agent texts, a `\x00` separator, and the family texts. It exists **only if that variable is set**; unset means no cache.
  - Keying the directory on the texts means that any change to the declared vector, and therefore to the rendered text, lands in a different directory. For RouterEval, every (n, beta, liar_select, seed) world gets its own directory. The root in use is `$RTE_DATA/cache/embed_routereval/` (66 dirs).
- **`_embeddings(view)`** (`_common.py:244-270`) holds two blocks.
  - Agent block: `descriptions_<slug><ltag>.npy`. The slug is `minilm` or `qwen_qwen3_embedding_8b`.
  - Family block: `families_<slug><ltag><itag>.npy`, where `itag = _i<blake2b-4 of the instruction>`.
  - A file is reused only if its row count matches. Otherwise it is recomputed. A non-MiniLM model without a GPU **raises** with a pointer to `scripts/embed_populations.py`.
  - Writes are atomic: tmp file, then `os.replace`.
- **`_sota_table(view)`** (`_common.py:228-242`) is the (K, k) table `shortlist_sota_<embed>_<rerank>_k<k>p<pool>[_dd][_i<hash>].npy`, named by `sota_cache_name` (`:89-94`, where `instruct` is `embed_instruct + ltag`). It is reused if its row count equals K. Otherwise it is computed, and the reranker then runs on whatever device is present (on CPU it is slow, but it does not raise).
- **`scripts/embed_populations.py` (live).**
  - Walks every population dir that has `descriptions.json` (`:36-51`) and writes `descriptions_<slug>.npy`.
  - Writes `families_<slug>[_i<hash>].npy`, with the query prompt `"Instruct: ...\nQuery:"` or `prompt_name="query"` (`:62`).
  - With `--sota`, it also writes the reranked table through the **same** `sota_shortlist` and dedup pool (`:83-97`) and `sota_cache_name(..., dedup=True, instruct)`.
  - The family texts are `families.describe(f)` for `FAMILIES_16`. I checked that `families.names(16) == FAMILIES_16`, so they match the backend's texts.
  - An example 10^5 population dir holds `descriptions_minilm.npy`, the Qwen description block, three family blocks (stock, `_i0258be9d`, `_i06c7d226`) and three sota tables.
- **`scripts/embed_routereval.py` (RouterEval).**
  - Asserts that `RTE_EMBED_CACHE_DIR` is set.
  - For each grid block, cell and seed, it builds the `World` and calls `FrameworkMethod._index` once per distinct dense setting, i.e. the exact code a routing unit runs (`:24-37`).
  - `--check` replaces `_rerank` with a function that raises, so it passes only if every file hits. The routing units must be launched with the same `RTE_EMBED_CACHE_DIR`. No launcher script sets it (grep finds it only in `_common.py`, the test and `embed_routereval.py`).
- **What never needs a cache:** `tfidf`, `bm25`, `declared`, the MIDIAN modes and `lie_text` tfidf/bm25 are computed inside the job.

---

## 3.7 Ledger accounting of a framework arm
- **Build:** `message(n)`, 0 probes and 0 reports. MIDIAN modes add their own build costs.
- **Per task:** `compare(len(cand))`, `hop(1)`, `message(len(cand) + 2)` (`_common.py:364-365`). MIDIAN modes add MIDIAN's `compare(1)`, `message(2)` and its observe-time costs.

---

## 3.8 Facts a paper author should state beside the framework numbers
1. **Live text arms ignore the lie.**
   - On live, every text shortlist is bit-identical across liar regimes (erratum 27).
   - Honest-vs-cartel differences for text arms come only from the declared-argmax fallback.
   - The exception is `lie_text`, which is not drawn.
2. **On RouterEval the text arms read the lied declared vector**, and the supervisor sees only the subject name, never the question.
3. **The pre-registered TF-IDF bar is non-dedup**, so it is clone-filled at scale. Every other drawn text bar is dedup. Fig G's lift therefore mixes the retriever gain and the dedup gain.
4. **Declared top-k and the VA cohort are the only drawn shortlists that are not text retrieval.** VA spends a full MIDIAN-VA probe budget. Declared spends none and is directly attackable.
5. **Every shortlist is ordered best-first.** The shuffle control exists because frameworks largely consume position 1 (`METHODS.md` §5b).
6. **AutoGen's own selector falls back to the first participant** after 3 failed attempts, and the adapter cannot distinguish that from a real pick.

---

## 3.9 Discrepancies / open questions

1. **Dedup is not a no-op on RouterEval under liars.**
   - `grid.yaml:985` says "dedup is a no-op there: every rendered description is unique". Erratum 25 says 1,000/1,000 and 5,000/5,000 unique.
   - That holds only at beta = 0. Inflated liars whose top-5 all clip to `1.00` render identically. My count of distinct texts from `_texts` at seed 1:

     | pool | beta 0.5 low_skill_first | beta 0.5 random |
     |---|---|---|
     | n = 1,000 | 844/1,000 | 694/1,000 |
     | n = 100 | 96/100 | 78/100 |
     | n = 5,000 | 4,052/5,000 | 3,310/5,000 |

   - So every `dedup: true` RouterEval arm at beta 0.5 (`fw_routereval_*_em`, `re_sl_*`) removes 4-34 % of agents, mostly liar clones, from the retrievable pool. The pre-registered `fw_routereval_*` TF-IDF rows (no dedup) are unaffected.
2. **"TF-IDF is degenerate on rendered numbers" (`grid.yaml` fw_routereval_5k comment; `DEVIATIONS.md` §Frameworks on RouterEval (a)) is imprecise.**
   - The query `"Tasks of family X"` matches the family-name tokens in each agent's declared top 5.
   - The retriever therefore shortlists agents that *claim* X among their top 5, and liars flood it: 100 % liars for one family in my 1k check.
   - It carries declared-channel signal, and adversarial signal. It is not a no-signal ranking.
3. **The RouterEval supervisor never sees the prompt.** The task text is `"A task of family <subject> (instance i)."` (`_common.py:192`). DEVIATIONS does not state this.
4. **Which grids the shortlist figures read is decided by name.**
   - `shortlist_figs.py:38-39` collects live grids starting with `fw_live_n` (without `lietext`), plus `live_n10k_v2`
     and `live_n100k`, and RouterEval grids starting with `fw_routereval_` or `re_sl_`. H draws `MAIN` on the RouterEval
     rows (`shortlist_condensed.py:134-135`).
   - `lie_max_fw_*` is excluded by name (no `fw_live_n` prefix). That exclusion is necessary: `regime()` keys on beta
     and liar_select only, so `lie_mode: max` rows would otherwise be pooled with `inflate` "cartel" rows.
5. **Magentic-One coverage differs across 10^4 cartel grids.** It is excluded "for time" from `fw_live_n10k_cartel`
   and its `_dd`, `_em`, `_sota`, `_shapes` and `_verified_va` mirrors (9 frameworks), but included in
   `fw_live_n10k_cartel_backfill` and `lie_max_fw_n10k` (10), and in every live 10^5 grid. `fw_live_n10k_cartel_magentic`
   adds it for the four shortlists E draws at 10^4 (TF-IDF, MiniLM, rerank, VA cohort); it is queued and has no rows
   yet. Until it lands, a "mean over frameworks" at 10^4 cartel averages 7–8 frameworks against 8–9 honest.
6. **`METHODS.md` §5 (line ~119) says `retrieval="declared"` is "grid written, NOT yet run".** `results/fw_live_n100k_declared` has 162 csv rows. The doc is stale.
7. **`METHODS.md` §5 table:** "`fw_langgraph` | `create_react_agent` supervisor". In the code the supervisor is `langgraph_supervisor.create_supervisor`; `create_react_agent` builds the candidate agents (`langgraph_worker.py:25-27`).
8. **Reproducibility or determinism.**
   - `OPS_RULES.md` T1 says prefetch gives "identical picks; tested". The test uses a mock server.
   - On the real fleet:
     - The replica is chosen per call with an unseeded RNG (`_common.py:106`).
     - No worker sends a `seed`.
     - vLLM greedy decoding under varying batches is not guaranteed bit-stable.
   - `DEVIATIONS.md:385` ("Requests set temperature 0 and seed 0") describes the agents' `llm_client` calls, not the framework workers.
   - I did not measure the pick-level variance of a framework row between two sequential runs.
9. **Some worker exceptions still count as infrastructure errors.** `INVALID_ACTION` (`_common.py:29`) covers only the three error classes that have failed a unit. LlamaIndex selector parse failures, AgentScope JSON errors and handoff name KeyErrors are also raised inside the worker; they become `error`, count as `infra_errors` and can fail a unit past 2 %. Arguably they are framework behaviour (non-picks). This matters only if some framework's parse-failure rate is near 2 %.
10. **`success_strict` after an error is scored as a non-pick.** Both error paths set `_picked = False` before routing by declared argmax (`_common.py:376, 388`), so `observe` scores such a task 0 under the strict metric.
11. **Erratum 28 wording and counts differ.** CHANGES_AND_ERRATA says "7 framework **conda** envs ... 240 dangling library symlinks each". `OPS_RULES.md` D2 says "98 dangling library symlinks each". The code-side comment (`_common.py:382`) says ~4,500 rows; the erratum says 4,269 rows were quarantined. The envs are venvs (`_bridge.py:22-26`).
12. **Stale grid comment:** `fw_live_n1000`'s inline comment says "v2 0.4: 5 seeds, Q=1000", but the grid runs `seeds: 1-10` (`grid.yaml:110-111`).
13. **The unused TF-IDF matrix** is built for `declared` and `bm25` (`_common.py:303-305`), which is harmless. Unknown `retrieval` strings silently run hashed TF-IDF (`:328-329`). No current grid uses a misspelled value: every value I resolved is one of tfidf, bm25, embed, hybrid, sota, declared, midian or midian_va.
14. **MiniLM family vectors are not on disk at 10^5.** `specialist_n100000_K16_seed1/` has `descriptions_minilm.npy` but no `families_minilm.npy`, even though `_embeddings` would write it (`:270`). The `_em` rows there were probably produced before the family block was cached, or with a different family-cache name. It is harmless, since 16 MiniLM vectors cost nothing, but it is unexplained.
15. **Open: the "Declared areas" clause on live gives the true specialty list.** It is structured metadata that no real agent card would guarantee honest. Every live text arm (and the supervisor) benefits from it; `lie_text` is the only condition that corrupts it.

---

## 4. Figures A and B: where every bar, line, colour and number comes from

Scope: `figures/condensed_sample/A_live_allb.{png,pdf,csv}`, `A_live_stacked.*`, `B_families_allb.*` and `B_families_stacked.*`. A was written at **2026-09-23 02:05** and B at **02:06** by `python scripts/condensed_figs.py`. Figures C and D (efficiency) are written by `scripts/efficiency_figs.py` and covered in Part 6.
Code, repo-relative to `~/rte`: `scripts/condensed_figs.py` (190 lines) and `scripts/seed_tables.py` (94 lines). They import `scripts/extra_figs.py` (legend wrapper, `excluded`, `ci`), `scripts/fw_variant_numbers.py` (`load`, `regime`) and `rte/analyze.py` (`ALIAS`). `condensed_figs.py` also reads the CSVs that `scripts/bar_figs.py` wrote.
Data root: `RTE_DATA=/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte`, with results under `$RTE_DATA/results/<grid>/`.

**How this section was checked (read-only).**
1. I rebuilt the script's internal dict `C` by calling its own `load() → cells() → add_budgets()`, attaching `seed_tables.tables()` and calling `arms_at()` without saving anything (`scratchpad/repro_ab.py`), then compared the result row by row with the CSVs (value to 1e-9 and the `chosen` string).
   - **A: all 73 rows are identical.**
   - **B: 121 of 127 rows are identical.** The other 6 are replay b = 5 rival bars whose `rivals_b_replay_1e6` seeds grew after 02:06. The current rows would also add 8 bernoulli b = 5 rival bars (`rivals_b_bernoulli_1e7`, 4 seeds) that the figure draws as `*`. The tables below give the figure's values; the seed counts of the 6 replay bars were counted from the rows written before 02:06.
2. Independent checks from raw `rows.d/*.json` or `rows.csv` with plain pandas, all matching:
   - A, n=100, honest, MIDIAN-VA b=1 = **0.7105**, 95% CI **[0.68979, 0.731605]**: the mean over 10 seeds of `va_b_n100` rows with method `midian_va`, b=1, β=0, bootstrapped with `default_rng(0)` and `choice`.
   - A, n=1000, honest, MIDIAN-VA b=5 = **0.8377**, the mean of 10 per-seed values from `va_b_n1000`.
   - A b=3 values recomputed from the `LIVE_GRIDS` `rows.csv` (self_described, specialist), via `rte.analyze.prepare` labels and a per-seed mean:
     - `midian` n=100: 0.7747 honest / 0.7359 cartel
     - n=1000: 0.7890 / 0.7379
     - n=10k: 0.7850 / 0.7522
     - n=1e5: 0.7522 / 0.7056
     - `oracle` n=100 / n=1000 / n=10k / n=1e5: 0.8449 / 0.8612 / 0.8589 / 0.8622

     All of these match `figures/bars/live.csv` and the A CSV.
   - B, LLMRouterBench, honest, best learned router b=1: `cluster_head_router` is picked on all 5 seeds; its mean over `rivals_b_llmrouterbench` is 0.7174, and divided by the b=3 oracle 0.7250 this gives **0.98952**, matching the CSV value 0.9895.

---

### 4.0 Machinery shared by A and B (read this first)

#### 4.0.1 The pipeline (`condensed_figs.py:186-190`)
```
d = load(); C = cells(d); add_budgets(C)
for key, bb in tables().items():                 # per-seed tables for the cross-fitted pooled arms
    if key in C: C[key]["seeds"] = bb
fig_A(C); fig_B(C)                    # C / D: scripts/efficiency_figs.py
```
A per-seed table whose cell has no b = 3 bar row is dropped (`if key in C`). This script draws only A and B.
1. **`load()`** (`:44-46`) concatenates the five per-family bar CSVs `figures/bars/{live,bernoulli,replay,routereval,llmrouterbench}.csv` and **drops every label starting with `fw_`**, so framework arms are never drawn.
   - These CSVs are *not* regenerated by condensed_figs. They were written by `scripts/bar_figs.py` at **2026-09-22 22:39**.
   - They hold **b = 3 only** (see 4.0.2).
2. **`cells(d)`** (`:49-57`) groups the rows by `(family, group, n, regime)`.
   - A cell is kept only if it has an `oracle` row.
   - Each cell stores `oracle = (mean, ci_lo, ci_hi)` and `raw[3] = {label: (mean, lo, hi)}` for every non-oracle label that is not on the do-not-add list (`excluded(l)`).
   - **So every b=3 number of the five single arms in A/B is a bar_figs number, unchanged.**
3. **`add_budgets(C)`** (`:151-183`) adds `raw[1]` and `raw[5]`. See 4.0.3.
4. **`seed_tables.tables()`** (`seed_tables.py:65-81`) builds seed × arm tables per (cell, regime, b) from the raw rows; they feed only the two pooled arms. See 4.0.4.
5. **`arms_at(cell)`** (`:60-75`) turns `raw[b]` and the tables into the seven plotted "arms". See 4.0.4.
6. **`budget_bars(...)`** (`:87-121`) draws the figure and writes the CSV. See 4.0.5.

#### 4.0.2 Where the b = 3 numbers come from (`scripts/bar_figs.py`)
- **live** (`family_live`, `bar_figs.py:127-142`):
  - Rows are the concatenation of `LIVE_GRIDS[n]` (`bar_figs.py:29-31`), loaded through `rte.analyze.load`, which first runs `consolidate` to merge `rows.d` into `rows.csv` (`analyze.py:52-61`, `run.py:178`).
  - Filtered to `declared_source == "self_described"` (`:134`) and `dist == dist` (`:135`).
  - Per-regime statistics come from `stats_from_rows` (`:78-87`):
    - Filter `beta` ≈ β and `liar_select == liar`.
    - **Honest (β=0) uses `liar_select == "random"`**. It falls back to all β=0 rows only if that set is empty (`:81`).
    - Then per label: `per = groupby("seed").success.mean()`; plotted mean = `per.mean()`; CI = `extra_figs.ci(per)` (`:84-86`).
  - Grids per n: 100 → `fw_live_n100, learned_n100, live_core_n100, fw_live_n100_lowskill`; 1000 → `fw_live_n1000, live_f1_n1000, variants_f1, learned_f1, fw_live_n1000_lowskill`; 10000 → `learned_n10k, live_n10k_v2, fw_live_n10k_cartel, live_n10k_cartel_random`; 100000 → `live_n100k`.
  - Verified cell axes of those rows after the filter: all `b = 3`, `K = 16`, `lie_mode = inflate`, `demand = uniform`, `collude = True`; `Q = 1000` at n ≤ 10^3 and `Q = 300` at n ≥ 10^4.
  - **There is no deduplication across grids.** If a (label, seed) appears in several grids, all of its rows are averaged into that seed's value (details in 4.5).
- **bernoulli / replay** (`family_matrix`, `bar_figs.py:145-156`):
  - Read `results/<grid>/matrix_success.csv` for `bernoulli_scale_v5` (group label `"specialist"`) and `replay_scale_v5` (group `"all shapes pooled"`), filtered to `b == 3`.
  - The matrix was written by `scripts/scale_matrix.py` (`:39-50`) on 2026-09-17:
    - mean = mean over all (dist, seed) units;
    - CI = `extra_figs.ci` on the (dist, seed)-indexed series, which is a seed bootstrap;
    - honest = β=0 with `liar_select = random` (`scale_matrix.py:17-22`).
  - Replay at n=1e6 has 300 units = 3 shapes × 100 seeds, so it is balanced.
- **routereval** (`bar_figs.py:159-177`):
  - n=5000 comes from `big = routereval_mmlu5k + fw_routereval_5k` (`:161`). `fw_routereval_5k` contains only `oracle` rows once frameworks are dropped.
  - Cell axes: `dist = all`, K=16, Q=300, b=3, `declared_source = programmatic`.
  - **`big` is not filtered by pool** (`:172`), so the n=5000 bars are identical in the `strong_to_weak`, `all_strong` and `all_weak` groups. B picks the `strong_to_weak` copy (`PRIMARY`, `condensed_figs.py:39`), but the data is the 5000-LLM leaderboard pool, `dist = all`.
- **llmrouterbench** (`bar_figs.py:180-187`): grid `llmrouterbench_pool`, n=20, K=15, Q=1000, b=3, programmatic, 5 seeds.

#### 4.0.3 Where the b = 1 and b = 5 numbers of the single arms come from (`add_budgets`, `condensed_figs.py:151-183`)
- **Grid list** (`:158-160`, `:162`): for each tag, both `va_b_<tag>` (MIDIAN-VA only) and `rivals_b_<tag>` (budget-matched rivals) are read. The tags are:
  - `n100, n1000, n10k, n100k` → key `("live","specialist",n)`
  - `routereval5k` → `("routereval","strong_to_weak",5000)`
  - `llmrouterbench` → `("llmrouterbench","20 models",20)`
  - `bernoulli_1e7` → `("bernoulli","specialist",1e7)`
  - `replay_1e6` → `("replay","all shapes pooled",1e6)`
- **Row loading.** Rows come from `fw_variant_numbers.load` (`fw_variant_numbers.py:24-32`):
  - it reads `rows.d/*.json` plus `rows.csv` directly, with **no consolidate** and no write;
  - `drop_duplicates("rid")`;
  - then `drop_duplicates` on `(n, b, dist, beta, liar_select, seed, method, params)`.
- **Labels.** `label(method, params)` (`seed_tables.py:26-29`, imported at `condensed_figs.py:23`) rebuilds the same label string as `rte.analyze.prepare` (`analyze.py:71`), then applies `ALIAS` (`analyze.py:20-24`). For example, `flat_probe_argmax{"online":true}` becomes `flat_probe_argmax_online`, and `knn_router{"online":true}` becomes `knn_router_online`.
- **Grouping** (`:166`): `groupby(["n","b","beta","liar_select","label"])`, mapped to a regime by `fw_variant_numbers.regime(beta, ls)` (`fw_variant_numbers.py:55-59`):
  - β = 0 → `beta0`, whatever `liar_select` is;
  - β = 0.5 with `low_skill_first` → `cartel`.
- **Rows kept** (`:168`): only if the key already exists in `C` (so the b=3 bar CSV must have that cell), b ∈ {1, 5}, the label is not `oracle`, and `excluded(l)` is false.
- **Aggregation**:
  - **Non-replay** (`:174-175`): `per = q.groupby("seed").success.mean()`; value = `per.mean()`; CI = `extra_figs.ci(per.values)`, which is the numpy-array path of `ci` (`extra_figs.py:89-97`: 2000 resamples with `rng.choice`, `default_rng(0)`, 2.5 and 97.5 percentiles).
  - **Replay, "all shapes pooled"** (`:169-173`):
    - `by = groupby(["seed","dist"]).success.mean().unstack()`;
    - if all 3 shapes (specialist, heavy_tail, bimodal) exist, keep only **seeds that have every shape** (`by.dropna()`); otherwise the cell is empty;
    - per-seed value = mean over the 3 shapes; value = mean over those seeds; CI = seed bootstrap.
    - **For the single arms the intersection rule only exists at b = 5.** Their b=1 and b=3 replay numbers come from the matrix, which averages all (shape, seed) units. That matrix is balanced, 100 × 3, so the two rules agree there.
- **Matrices for b=1** (`:177-183`): bernoulli and replay b=1 come from `bernoulli_scale_v5` / `replay_scale_v5` `matrix_success.csv`, filtered to `b == 1`, the `success` metric, and regimes `"beta=0 (no liars)"` / `"beta=0.5 CARTEL (low-skill-first)"`. Mean and CI are copied from the matrix. The `va_b_*` / `rivals_b_*` bernoulli and replay grids contain only b=5, so nothing is overwritten.
- **Nothing is ever pooled across b.** Each bar is one budget.

#### 4.0.4 The seven arms and how "best learned" / "best bandit" are chosen (`arms_at`, `:60-75`)
`ARMS` (`:31-33`), in plotting order, with key, legend label and colour:

| order | key | legend label | colour |
|---|---|---|---|
| 1 | `midian_va` | MIDIAN-VA | `#2ecc71` |
| 2 | `midian` | MIDIAN | `#c0392b` |
| 3 | `flat_probe_argmax_online` | flat probe argmax (online) | `#3498db` |
| 4 | `best_learned` | best learned router | `#ff7f0e` |
| 5 | `best_bandit` | best bandit | `#9467bd` |
| 6 | `declared_argmax` | declared argmax | `#5d6d7e` |
| 7 | `random` | random | `#bbbbbb` |

- **Single arms** (1–3, 6, 7): `arms_at` copies `raw[b][key]` for every b that has it (`:72-74`).
- **Pooled arms** (4, 5): for each b with a per-seed table in the cell (`:64-71`):
  - `want` = the pool (`POOLS`, `:34`) minus `NOT_RUNNABLE(family, n)` (`:35-36`). It is the same at every b of a cell.
  - `crossfit(T, want)` (`seed_tables.py:84-94`) scores each seed with the arm that has the best mean on the other seeds, among arms that ran on that seed. The bar is the mean of the per-seed scores, and the CI the seed bootstrap of those scores (`:69`).
  - Fewer than 2 scored seeds: no bar (`:68`).
  - `chosen` = the pick counts, plus `| INCOMPLETE POOL, missing …` when a `want` arm has no column at that b (`:70`).
  - The rules and caveats are in §2.1.3.
- **What the pools really contain**, per METHODS.md:
  - `cluster_head_router` and `disrouter_cascade` are **"Declared channel only"** arms (METHODS.md:35-44). They read declared skill D and never probe. They are also in `bar_figs.DECL` (`bar_figs.py:90`).
  - `flat_nsw_router` is a "verified outcomes, centralized" arm (METHODS.md:60).
  - Only `knn_router(_online)` and `mlp_router` are the published RouterBench routers (METHODS.md:73-74).
  - So in the honest bernoulli/replay columns the "best learned router" is `cluster_head_router`, whose value is b-invariant and close to declared argmax. In the bernoulli matrix, `cluster_head_router` is 0.8384 at both b=1 and b=3.
  - `warm_start_bandit` reads declared skill plus probes (METHODS.md:54).
- **Which pool candidates are still missing** (from the `chosen` column of the current CSVs):

| cell | b | learned candidates missing | bandit candidates missing |
|---|---|---|---|
| live 10^2 | 3 | `flat_nsw_router`, `cluster_head_router`, `disrouter_cascade` | `ucb`, `thompson`, `linucb_honest`, `trueskill`, `warm_start_bandit[n0=0.5]` |
| live 10^3 | 3 | none | `warm_start_bandit[n0=0.5]` |
| live 10^4 | 1 (honest), 3 | `flat_nsw_router`, `cluster_head_router`, `disrouter_cascade` | `ucb`, `thompson`, `trueskill` (+ `warm_start_bandit[n0=0.5]` at b = 3; + `linucb_honest` in the cartel) |
| live 10^5 | 3 | `flat_nsw_router`, `cluster_head_router`, `disrouter_cascade` | `ucb`, `thompson`, `warm_start_bandit[n0=0.5]` |
| bernoulli 10^7, replay 10^6 | 1 | `disrouter_cascade` | `linucb_honest`, `warm_start_bandit[n0=0.5]` |
| bernoulli 10^7, replay 10^6 | 3, 5 | none | `warm_start_bandit[n0=0.5]` |
| RouterEval 5k | 3 | `knn_router_online`, `flat_nsw_router`, `cluster_head_router`, `disrouter_cascade` | `ucb`, `thompson`, `trueskill`, `warm_start_bandit[n0=0.5]` |
| LLMRouterBench | 1, 5 | none | `warm_start_bandit[n0=0.5]` |
| LLMRouterBench | 3 | `flat_nsw_router`, `cluster_head_router`, `disrouter_cascade` | `ucb`, `thompson`, `trueskill`, `warm_start_bandit[n0=0.5]` |

  The `pool_fill_*`, `pool_seeds_n1000` and `tuned_wsb_*` grids supply these (§1.6.3). Pooled bars at live b = 1 / 5 (except 10^4 honest b = 1), RouterEval b = 1 / 5 and bernoulli b = 5 are absent from the figure: no rows yet.
- **Budget-less arms** (`BUDGETLESS = {declared_argmax, random}`, `:37`, `:74`): only their b=3 entry is used, drawn as **one bar per regime in its base colour** (no shading), and written to the CSV with `b = "-"`. They spend no probes, so b does not apply. Their b=3 value is the bar_figs number.
- **Every other arm** has a bar for each b ∈ {1, 3, 5} for which data exists.

#### 4.0.5 Drawing (`budget_bars`, `:87-121`)
- **Groups**: one group per x tick (A: n; B: family). **Regimes**: `REG = {"beta0": "honest", "cartel": "β=0.5 cartel"}` (`:41`).
- **Slots within a group** (`:91-94`):
  - `arms` = the ARMS present in any cell.
  - **_allb** has one slot per (arm, regime, b), ordered arm → regime (honest, then cartel) → b (1, 3, 5). Budget-less arms get one slot per regime.
  - With 5 budgeted and 2 budget-less arms that is 5·2·3 + 2·2 = **34 slots**, bar width `w = 0.86/34`.
  - **_stacked** has one slot per (arm, regime): **14 slots**, `w = 0.86/14`.
  - Slot positions are fixed, so a missing bar leaves a gap (or a star).
- **Colour and shade** (`shade`, `:82-84`):
  - b=1 = the base colour mixed 50% toward white (light);
  - b=3 = the base colour;
  - b=5 = base × 0.6 (dark).
- **Hatch**: the cartel regime has `hatch="////"` and `alpha=0.8`; honest bars are solid (`:110-111`). Every bar has a black edge, lw 0.3. There are no hollow bars in A/B.
- **Stars.** Two kinds:
  - **Missing budget** (`:105-107`): if an arm has at least one b in that (cell, regime) but not this b, a `*` is placed at the slot's x, at y = 0.205, just above the 0.2 axis floor. If an arm has **no** b at all in a (cell, regime), nothing is drawn: no bar and no star. Currently every arm has b=3 wherever the cell exists, so every gap carries a star.
  - **Incomplete pool** (`:112`): a pooled bar whose `chosen` contains `INCOMPLETE` gets a `*` 0.004 above its CI top (above the bar top in _stacked).
- **Oracle line** (`:116`): a dotted grey (`#7f8c8d`, lw 1.2) horizontal line across each group (±0.46).
  - A: y = the **honest (β=0) b=3 oracle mean** from the bars CSV.
  - B: y = 1.0.
  - The cartel oracle is not drawn separately. It is numerically identical: oracle success does not depend on liars, and I verified this in every grid here, e.g. live oracle 0.8449 / 0.8612 / 0.8589 / 0.8622 in both regimes and in the va_b grids.
- **Y axis** (`:118`): A shows `success` on ylim (0.2, 0.95); B shows `success / oracle` on ylim (0.2, 1.05). Bars start at 0, so anything below 0.2 is hidden. None currently is: the lowest bar is replay random at 0.242.
- **Normalisation (B only)** (`:97`, `:108`): `z = C[(family, group, n, "beta0")]["oracle"][0]`, the **β=0, b=3 oracle mean from the bar CSV**. Mean, `ci_lo` and `ci_hi` of **every** bar in the group (both regimes, all b) are divided by that one scalar.
  - This is a ratio of means, not a per-seed ratio.
  - The CI ignores the oracle's own uncertainty.
  - For live, RouterEval and LLMRouterBench, the b=1/5 grids' own oracles equal the b=3 oracle exactly: 0.8622, 0.9022 and 0.7250. They mirror the same task streams.
  - For bernoulli and replay, the b=5 bars cover a seed subset whose own oracle differs slightly: MIDIAN-VA's 51 honest bernoulli seeds have 0.8456 against 0.8462, and its 38 honest replay seeds 0.7885 against 0.7900. So the b=5 ratios are 0.1–0.2 % off a same-seed normalisation.
- **Error bars** (`:113-114`): drawn **on every bar in _allb** as `yerr = [m − lo, hi − m]` (dark grey, lw 0.4, cap 0.6). They are **95% percentile-bootstrap CIs over seeds**, using per-seed means (for replay b=5, the per-seed mean of the 3 shape means). The implementation is `extra_figs.ci` (`:89-97`) with B = 2000 and `default_rng(0)`:
  - seed-indexed Series (b=3 live/RouterEval/LLMRouterBench via bar_figs; b=1/3 matrices): resample seeds with `rng.integers`;
  - plain arrays (`add_budgets`, b=1/5; the cross-fitted scores of the pooled arms): resample with `rng.choice`.
  
  With 3 seeds (live n ≥ 10^4, RouterEval) the bootstrap has at most 10 distinct resample means, so those CIs are coarse.
- **_stacked has no error bars.** The condition `if not nested or (b == 3 and not stacked)` is false whenever `stacked=True` (`nested` is forced true at `:93`). The comment gives the reason: "an interval inside the stack misreads; see _allb" (`:113`).
- **Legend** (`:109`, `:120`):
  - Each arm gets its handle from **group 0, honest, b=3** (`first = i == 0 and not h and b == 3`); the oracle gets its handle from group 0.
  - `ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.07))` goes through the global wrapper `extra_figs._ranked` (`extra_figs.py:38-54`, installed at `:58`). That wrapper sorts entries by `_plotted(handle)`, descending:
    - a bar handle's value = the height of that single bar, i.e. group 0's honest b=3 value;
    - the oracle's `LineCollection` value = its y.
  - `_rowmajor` (`:32-35`) then lays the entries out row-major: best top-left, then left to right.
  - **A** ranks by n=100 honest b=3: oracle, MIDIAN-VA .7816, flat .7805, MIDIAN .7747 / learned .7376, bandit .7300, declared .5992, random .4258.
  - **B** ranks by live-1e5 honest b=3 ratios: oracle 1.0, VA .969, MIDIAN .872, flat .869 / learned .860, bandit .856, declared .722, random .513.
  - Both orders match the PNGs.
  - An arm that lacked an honest b=3 bar in group 0 would get **no legend entry**. That does not happen currently.
- **CSV** (`:115`, `:121`): one row per drawn bar with columns `group, regime, arm, b ('-' for budget-less), chosen, value, ci_lo, ci_hi`. The values are **absolute bar heights**; in B they are already divided by the oracle. **_stacked CSVs contain the same rows as _allb** in a different order (`sort | diff` is empty). Neither CSV lists the missing-budget stars or seed counts; the incomplete-pool flag is in `chosen`, and the pooled arms' pick counts add up to their seed count.

#### 4.0.6 What "honest" and "β=0.5 cartel" mean here
- **honest** = β = 0, no liars.
  - At b=3: `liar_select = random` rows (bar_figs `:81`; the matrices' "beta=0 (no liars)" cell).
  - At b=1/5 (live/RouterEval/LLMRouterBench `va_b_*` / `rivals_b_*`): the rows are tagged `liar_select = low_skill_first, collude = True`, with `n_liars = 0`. The tag is inert at β = 0, as the `configs/grid.yaml:1043-1046` comment notes.
- **β=0.5 cartel** = β = 0.5, `liar_select = low_skill_first`, `collude = True`, `lie_mode = inflate`. Half the agents lie, chosen low-skill first, and they collude.
  - live: `declared_source = self_described`.
  - bernoulli/replay/RouterEval/LLMRouterBench: `programmatic`.
- **What changes with b** (the probe budget: probes per agent per family, per METHODS.md:46-52):
  - Every probe-using arm changes with b: MIDIAN-VA, MIDIAN, flat probe argmax, the bandits' warm-up, and the learned routers' training probes (knn: k = b).
  - Declared-channel arms (`declared_argmax`, `cluster_head_router`, `disrouter_cascade`) and `random` do not. In the bernoulli/replay matrices `cluster_head_router`, `declared_argmax` and `random` are identical at b=1 and b=3.
- **Structural identity at b = 1.** Several MIDIAN and MIDIAN-VA b=1 values are equal:
  - bernoulli honest: 0.8015 = 0.8015 (matrix: `midian`, `midian_v`, `midian_a` and `midian_va` are all 0.6782 raw);
  - replay honest: 0.8272 = 0.8272;
  - LLMRouterBench honest: 0.8880 = 0.8880;
  - live 10^4 honest: 0.6678 = 0.6678 (`va_b_n10k` vs `rivals_b_n10k`).
  
  This is consistent with README.md:344-345: "never run b = 1 beside b = 3 in one table (verification is unfunded at b = 1)". The condensed figures put b = 1 beside b = 3 deliberately (see 4.5).
- **Identical honest and cartel b=5 at live n=100.** MIDIAN-VA b=5 is 0.8121 in both regimes. I checked per seed:
  - all 10 seeds have identical success at β=0 and β=0.5 (for example seed 1: 0.801 / 0.801);
  - `n_liars` is 0 against 50, and `misroute_to_liar` is > 0 in the cartel rows (seed 2: 0.091).
  
  So the routing is identical, and "misroutes" land on liar-flagged agents whose executions are real. At b=5 VA's selection does not depend on declarations. This is a genuine result, not duplicated rows.

---

### 4.1 Figure A: live RTE, specialist population, n = 10^2 … 10^5

**Question answered.** On the live LLM backend with the specialist population, how does MIDIAN-VA compare with plain MIDIAN, the flat probe-argmax control, the best learned router, the best bandit, declared argmax and random, at every scale n, honest versus a β = 0.5 low-skill colluding cartel, and at each probe budget b = 1, 3, 5? It also shows how much each budget step buys.

**Groups** (`fig_A`, `:129-133`): n = the n values with a `("live","specialist",n,"beta0")` cell in `C`, i.e. 100, 1000, 10000 and 100000. The x labels are `n = 100`, `n = 1,000`, etc.

**_allb** (`A_live_allb`): in each group, for each arm left to right (MIDIAN-VA, MIDIAN, flat, best learned, best bandit, declared, random):
- solid bars for honest b=1, b=3 and b=5 (light, mid, dark), then hatched bars for cartel b=1, b=3 and b=5;
- one solid and one hatched bar for declared and random;
- a `*` wherever a budget is missing;
- dotted line = oracle success, i.e. the honest b=3 oracle: 0.8449, 0.8612, 0.8589, 0.8622;
- y = success, not normalised.

**_stacked** (`A_live_stacked`, `stacked=True`, `:133`): one full-width slot per (arm, regime).
- The b values present are sorted **tallest first** (`:102`) and each is drawn **from 0 to its own full height**, with the full slot width. Drawing order is z-order 2, 3, 4 (tallest behind, shorter in front).
- So what you see is the shortest bar in its shade at the bottom, and above it the part of each taller bar that sticks out. When success rises with b, that reads as "b=1 base, gain to b=3, gain to b=5", which is what the title says.
- **Negative gains are never drawn as negative segments.** If b=3 < b=1, the bar just shows the lighter b=1 shade *above* the mid shade: out-of-order shades, as the docstring (`:9`) warns.
- In A, the arms with more than one b are MIDIAN-VA (every n) and, at n = 10^4 honest, MIDIAN, flat probe argmax and the two pooled arms (b = 1 and 3). All are monotone in b, so A's stacks are all in order.
- Missing budgets: a `*` at the slot's x; two missing budgets put two overlapping stars at the same x.
- No error bars (see 4.0.5). The CSV holds the absolute values, not gains.

**Provenance.** The single arms' b=3 source grids and seed counts are listed in 4.0.2 and in the table below; the pooled arms' sources are in §2.1.2.

| n | b=3 source of the single arms (bars CSV `live.csv`, filtered self_described + specialist) | b=1 / b=5 source |
|---|---|---|
| 100 | `midian`, `midian_va`, `oracle`, `declared_argmax`: pooled from `fw_live_n100` (honest) or `fw_live_n100_lowskill` (cartel) + `learned_n100` + `live_core_n100` (5 seeds, honest and cartel). `flat_probe_argmax_online`: `learned_n100` only. `random`: fw grids + `live_core_n100`. **10 seeds.** | `va_b_n100` (MIDIAN-VA only; 10 seeds; K16 Q1000; mirror of `fw_live_n100`) |
| 1000 | `midian`/`midian_va`: `fw_live_n1000`/`_lowskill` + `variants_f1` + `live_f1_n1000` (5 seeds). `flat_online`: `variants_f1` (10 seeds) + `live_f1_n1000` (5). **10 seeds.** | `va_b_n1000` (10 seeds) |
| 10000 | `learned_n10k` (+ `live_n10k_v2` honest for `midian`, `midian_va`, `flat_online`, `random`, `oracle`; `live_n10k_cartel_random` / `fw_live_n10k_cartel` cartel `oracle` / `random`). **3 seeds.** | `va_b_n10k` (3 seeds; K16 Q300); `rivals_b_n10k` (b = 1 honest, 3 seeds) |
| 100000 | `live_n100k`. **3 seeds.** | `va_b_n100k`: **b=1 honest only, 3 seeds**. No cartel b=1 and no b=5 in any regime. |

- `rivals_b_n100` and `rivals_b_n1000` are queued and have no rows; `rivals_b_n100k` is pending; `rivals_b_n10k` has only b = 1 honest (4.3).
- **So MIDIAN, flat, learned and bandit are b=3 only in A, except at n = 10^4 honest, which has b = 1.** Stars mark the other b=1/b=5 slots.

**A value table (current CSV, 02:05).** Format: value, seeds (source); for the pooled arms, value, seeds, pick counts. `*` after a pooled entry = incomplete pool (a candidate has no rows yet); `*` alone = budget not in yet; n/a = budget-less. CIs and missing candidates are in the Appendix.

| group | regime | arm | b=1 | b=3 | b=5 |
|---|---|---|---|---|---|
| n = 100 | honest | MIDIAN-VA | 0.7105 10s (va_b_n100) | 0.7816 10s (bars) | 0.8121 10s (va_b_n100) |
| n = 100 | cartel | MIDIAN-VA | 0.6972 10s (va_b_n100) | 0.7680 10s (bars) | 0.8121 10s (va_b_n100) |
| n = 100 | honest | MIDIAN | * | 0.7747 10s (bars) | * |
| n = 100 | cartel | MIDIAN | * | 0.7359 10s (bars) | * |
| n = 100 | honest | flat probe argmax (online) | * | 0.7805 10s (bars) | * |
| n = 100 | cartel | flat probe argmax (online) | * | 0.7805 10s (bars) | * |
| n = 100 | honest | best learned router | * | 0.7376 10s, `knn_router_online` ×10 * | * |
| n = 100 | cartel | best learned router | * | 0.7376 10s, `knn_router_online` ×10 * | * |
| n = 100 | honest | best bandit | * | 0.7300 10s, `warm_start_bandit` ×10 * | * |
| n = 100 | cartel | best bandit | * | 0.7183 10s, `warm_start_bandit` ×10 * | * |
| n = 100 | honest | declared argmax | n/a | 0.5992 10s (bars) | n/a |
| n = 100 | cartel | declared argmax | n/a | 0.5181 10s (bars) | n/a |
| n = 100 | honest | random | n/a | 0.4258 10s (bars) | n/a |
| n = 100 | cartel | random | n/a | 0.4258 10s (bars) | n/a |
| n = 1,000 | honest | MIDIAN-VA | 0.6913 10s (va_b_n1000) | 0.8134 10s (bars) | 0.8377 10s (va_b_n1000) |
| n = 1,000 | cartel | MIDIAN-VA | 0.6914 10s (va_b_n1000) | 0.8074 10s (bars) | 0.8375 10s (va_b_n1000) |
| n = 1,000 | honest | MIDIAN | * | 0.7890 10s (bars) | * |
| n = 1,000 | cartel | MIDIAN | * | 0.7379 10s (bars) | * |
| n = 1,000 | honest | flat probe argmax (online) | * | 0.7871 10s (bars) | * |
| n = 1,000 | cartel | flat probe argmax (online) | * | 0.7869 10s (bars) | * |
| n = 1,000 | honest | best learned router | * | 0.7656 10s, `knn_router_online` ×10 | * |
| n = 1,000 | cartel | best learned router | * | 0.7656 10s, `knn_router_online` ×10 | * |
| n = 1,000 | honest | best bandit | * | 0.7501 10s, `warm_start_bandit` ×5; `linucb_honest` ×5 * | * |
| n = 1,000 | cartel | best bandit | * | 0.7518 10s, `warm_start_bandit` ×5; `linucb_honest` ×5 * | * |
| n = 1,000 | honest | declared argmax | n/a | 0.6154 10s (bars) | n/a |
| n = 1,000 | cartel | declared argmax | n/a | 0.5202 10s (bars) | n/a |
| n = 1,000 | honest | random | n/a | 0.4321 10s (bars) | n/a |
| n = 1,000 | cartel | random | n/a | 0.4321 10s (bars) | n/a |
| n = 10,000 | honest | MIDIAN-VA | 0.6678 3s (va_b_n10k) | 0.8111 3s (bars) | 0.8356 3s (va_b_n10k) |
| n = 10,000 | cartel | MIDIAN-VA | 0.6700 3s (va_b_n10k) | 0.8100 3s (bars) | 0.8311 3s (va_b_n10k) |
| n = 10,000 | honest | MIDIAN | 0.6678 3s (rivals_b_n10k) | 0.7850 3s (bars) | * |
| n = 10,000 | cartel | MIDIAN | * | 0.7522 3s (bars) | * |
| n = 10,000 | honest | flat probe argmax (online) | 0.6522 3s (rivals_b_n10k) | 0.7744 3s (bars) | * |
| n = 10,000 | cartel | flat probe argmax (online) | * | 0.7744 3s (bars) | * |
| n = 10,000 | honest | best learned router | 0.5700 3s, `knn_router_online` ×3 * | 0.7311 3s, `knn_router_online` ×3 * | * |
| n = 10,000 | cartel | best learned router | * | 0.7311 3s, `knn_router_online` ×3 * | * |
| n = 10,000 | honest | best bandit | 0.6989 3s, `warm_start_bandit` ×3 * | 0.7415 3s, `warm_start_bandit` ×3 * | * |
| n = 10,000 | cartel | best bandit | * | 0.7833 3s, `warm_start_bandit` ×3 * | * |
| n = 10,000 | honest | declared argmax | n/a | 0.6578 3s (bars) | n/a |
| n = 10,000 | cartel | declared argmax | n/a | 0.5322 3s (bars) | n/a |
| n = 10,000 | honest | random | n/a | 0.4167 3s (bars) | n/a |
| n = 10,000 | cartel | random | n/a | 0.4167 3s (bars) | n/a |
| n = 100,000 | honest | MIDIAN-VA | 0.6400 3s (va_b_n100k) | 0.8356 3s (bars) | * |
| n = 100,000 | cartel | MIDIAN-VA | * | 0.8278 3s (bars) | * |
| n = 100,000 | honest | MIDIAN | * | 0.7522 3s (bars) | * |
| n = 100,000 | cartel | MIDIAN | * | 0.7056 3s (bars) | * |
| n = 100,000 | honest | flat probe argmax (online) | * | 0.7489 3s (bars) | * |
| n = 100,000 | cartel | flat probe argmax (online) | * | 0.7489 3s (bars) | * |
| n = 100,000 | honest | best learned router | * | 0.7411 3s, `knn_router_online` ×3 * | * |
| n = 100,000 | cartel | best learned router | * | 0.7411 3s, `knn_router_online` ×3 * | * |
| n = 100,000 | honest | best bandit | * | 0.7378 3s, `warm_start_bandit` ×3 * | * |
| n = 100,000 | cartel | best bandit | * | 0.7778 3s, `warm_start_bandit` ×3 * | * |
| n = 100,000 | honest | declared argmax | n/a | 0.6222 3s (bars) | n/a |
| n = 100,000 | cartel | declared argmax | n/a | 0.5778 3s (bars) | n/a |
| n = 100,000 | honest | random | n/a | 0.4422 3s (bars) | n/a |
| n = 100,000 | cartel | random | n/a | 0.4422 3s (bars) | n/a |

Method and params behind each label:
- `midian_va` = method `midian_va`, params `{}`
- `midian` = `midian {}`
- `flat_probe_argmax_online` = `flat_probe_argmax {"online": true}`
- `knn_router_online` = `knn_router {"online": true}`
- `warm_start_bandit` = `warm_start_bandit {}` (n0 = 5)
- `declared_argmax {}`, `random {}`

---

### 4.2 Figure B: every family at its largest population, success / oracle

**Question answered.** Does the ordering in A hold across every experiment family (live LLM, calibrated Bernoulli, RouterBench replay, RouterEval real-LLM pool, LLMRouterBench) at each family's largest population, on a common oracle-normalised scale, honest versus cartel, at b = 1, 3, 5?

**Group selection** (`fig_B`, `:139-148`). For each family in `FAMILY` order (live, bernoulli, replay, routereval, llmrouterbench):
- n = the **largest** n whose primary group has **both** a cartel and a β=0 cell in `C`;
- primary group = `specialist` for live, `strong_to_weak` for routereval, and otherwise whatever group exists (`primary`, `:136`; `PRIMARY`, `:39`).

Result:

| x label | cell key |
|---|---|
| `live n = 100,000` | ("live","specialist",1e5) |
| `bernoulli n = 10,000,000` | ("bernoulli","specialist",1e7) |
| `RouterBench replay n = 1,000,000` | ("replay","all shapes pooled",1e6) |
| `RouterEval n = 5,000` | ("routereval","strong_to_weak",5000); really the leaderboard pool, `dist = all` (4.0.2) |
| `LLMRouterBench n = 20` | ("llmrouterbench","20 models",20) |

**Y axis**: `success / oracle`. Every bar is divided by that group's **honest b=3 oracle mean** from the bars CSV: live 0.8622, bernoulli 0.8462, replay 0.7900, RouterEval 0.9022, LLMRouterBench 0.7250. The dotted line is at 1.0, and the CIs are divided by the same scalar (4.0.5). Values slightly above 1 are possible; for example, the LLMRouterBench honest learned-router CI upper bound at b = 1 and 5 is 1.010.

**Bars, shades, hatches, stars, legend and stacking** all work as in A (4.0.5 and 4.1). In B, several rivals do have b=1 and b=5 bars:
- bernoulli b=1 (matrix for the single arms, `bernoulli_scale_v5` rows for the pooled arms);
- replay b=1 (same) and b=5 (`rivals_b_replay_1e6`);
- LLMRouterBench b=1 and b=5 (`rivals_b_llmrouterbench`).

Consequences for **B_stacked**:
- It shows real multi-level stacks for rivals.
- It shows **non-monotone stacks**, where negative gains appear as a lighter shade above a darker one:
  - LLMRouterBench honest best learned: b=1 0.9895 = b=5 0.9895 > b=3 0.9476. At b = 1 and 5 every seed picks `cluster_head_router`; at b = 3 that candidate has no rows yet, so every seed picks `mlp_router` and the bar carries the incomplete-pool `*`. See 4.5 #3.
  - LLMRouterBench cartel MIDIAN: b=1 0.8662 > b=5 0.8497 > b=3 0.8350.
  - LLMRouterBench cartel learned: b=3 0.9476 > b=5 0.9186 > b=1 0.7961.
- Hatched bars at alpha 0.8 overlap, so the shades blend.

**Provenance per family** (single arms; the pooled arms add the sources in §2.1.2).

| family (cell) | b=1 | b=3 (bars CSV) | b=5 |
|---|---|---|---|
| live 1e5 | `va_b_n100k`: MIDIAN-VA honest only, 3 seeds; everything else `*` | `live_n100k`, 3 seeds (4.0.2) | none yet: `va_b_n100k` b=5 running; `rivals_b_n100k` pending |
| bernoulli 1e7 (specialist, K16 Q1000, programmatic) | `bernoulli_scale_v5/matrix_success.csv` b=1, **100 seeds**; arms: `midian_va`, `midian`, `flat_online` | same matrix at b=3, 100 seeds | `va_b_bernoulli_1e7`: MIDIAN-VA only, **51 honest / 36 cartel seeds**; `rivals_b_bernoulli_1e7`: no rows at 02:06 (4 seeds now) |
| replay 1e6 (3 shapes, K64 Q1000) | `replay_scale_v5` matrix b=1, 100 seeds × 3 shapes | matrix b=3, 100 × 3 | `va_b_replay_1e6` MIDIAN-VA: **38 honest / 25 cartel** complete-shape seeds; `rivals_b_replay_1e6`: **20 honest / 12 cartel** complete-shape seeds at 02:06 |
| RouterEval 5k | `va_b_routereval5k` MIDIAN-VA, 3 seeds | `routereval_mmlu5k`, 3 seeds | `va_b_routereval5k`, 3 seeds; `rivals_b_routereval5k`: **0 rows** (running) |
| LLMRouterBench 20 (K15 Q1000) | `va_b_llmrouterbench` + `rivals_b_llmrouterbench`, 5 seeds, complete | `llmrouterbench_pool`, 5 seeds | same two grids, 5 seeds, complete |

**B value table (current CSV, 02:06).** All values are ÷ oracle. Same format as the A table; CIs and missing candidates are in the Appendix.

| group | regime | arm | b=1 | b=3 | b=5 |
|---|---|---|---|---|---|
| live n = 100,000 | honest | MIDIAN-VA | 0.7423 3s (va_b_n100k) | 0.9691 3s (bars) | * |
| live n = 100,000 | cartel | MIDIAN-VA | * | 0.9601 3s (bars) | * |
| live n = 100,000 | honest | MIDIAN | * | 0.8724 3s (bars) | * |
| live n = 100,000 | cartel | MIDIAN | * | 0.8183 3s (bars) | * |
| live n = 100,000 | honest | flat probe argmax (online) | * | 0.8686 3s (bars) | * |
| live n = 100,000 | cartel | flat probe argmax (online) | * | 0.8686 3s (bars) | * |
| live n = 100,000 | honest | best learned router | * | 0.8595 3s, `knn_router_online` ×3 * | * |
| live n = 100,000 | cartel | best learned router | * | 0.8595 3s, `knn_router_online` ×3 * | * |
| live n = 100,000 | honest | best bandit | * | 0.8557 3s, `warm_start_bandit` ×3 * | * |
| live n = 100,000 | cartel | best bandit | * | 0.9021 3s, `warm_start_bandit` ×3 * | * |
| live n = 100,000 | honest | declared argmax | n/a | 0.7216 3s (bars) | n/a |
| live n = 100,000 | cartel | declared argmax | n/a | 0.6701 3s (bars) | n/a |
| live n = 100,000 | honest | random | n/a | 0.5129 3s (bars) | n/a |
| live n = 100,000 | cartel | random | n/a | 0.5129 3s (bars) | n/a |
| bernoulli n = 10,000,000 | honest | MIDIAN-VA | 0.8015 100s (bernoulli_scale_v5) | 0.9469 100s (bars) | 0.9791 51s (va_b_bernoulli_1e7) |
| bernoulli n = 10,000,000 | cartel | MIDIAN-VA | 0.7977 100s (bernoulli_scale_v5) | 0.9386 100s (bars) | 0.9783 36s (va_b_bernoulli_1e7) |
| bernoulli n = 10,000,000 | honest | MIDIAN | 0.8015 100s (bernoulli_scale_v5) | 0.9110 100s (bars) | * |
| bernoulli n = 10,000,000 | cartel | MIDIAN | 0.7450 100s (bernoulli_scale_v5) | 0.8463 100s (bars) | * |
| bernoulli n = 10,000,000 | honest | flat probe argmax (online) | 0.8005 100s (bernoulli_scale_v5) | 0.9088 100s (bars) | * |
| bernoulli n = 10,000,000 | cartel | flat probe argmax (online) | 0.8005 100s (bernoulli_scale_v5) | 0.9088 100s (bars) | * |
| bernoulli n = 10,000,000 | honest | best learned router | 0.9909 100s, `cluster_head_router` ×100 * | 0.9909 100s, `cluster_head_router` ×100 | * |
| bernoulli n = 10,000,000 | cartel | best learned router | 0.8450 100s, `cluster_head_router` ×100 * | 0.8523 100s, `disrouter_cascade` ×100 | * |
| bernoulli n = 10,000,000 | honest | best bandit | 0.9890 100s, `warm_start_bandit` ×100 * | 0.9912 100s, `warm_start_bandit` ×100 * | * |
| bernoulli n = 10,000,000 | cartel | best bandit | 0.8934 100s, `warm_start_bandit` ×100 * | 0.9075 100s, `warm_start_bandit` ×100 * | * |
| bernoulli n = 10,000,000 | honest | declared argmax | n/a | 0.9942 100s (bars) | n/a |
| bernoulli n = 10,000,000 | cartel | declared argmax | n/a | 0.8551 100s (bars) | n/a |
| bernoulli n = 10,000,000 | honest | random | n/a | 0.4936 100s (bars) | n/a |
| bernoulli n = 10,000,000 | cartel | random | n/a | 0.4936 100s (bars) | n/a |
| RouterBench replay n = 1,000,000 | honest | MIDIAN-VA | 0.8272 100s (replay_scale_v5) | 0.9642 100s (bars) | 0.9815 38s (va_b_replay_1e6) |
| RouterBench replay n = 1,000,000 | cartel | MIDIAN-VA | 0.8216 100s (replay_scale_v5) | 0.9622 100s (bars) | 0.9845 25s (va_b_replay_1e6) |
| RouterBench replay n = 1,000,000 | honest | MIDIAN | 0.8272 100s (replay_scale_v5) | 0.8908 100s (bars) | 0.9246 20s (rivals_b_replay_1e6) |
| RouterBench replay n = 1,000,000 | cartel | MIDIAN | 0.5015 100s (replay_scale_v5) | 0.7095 100s (bars) | 0.8301 12s (rivals_b_replay_1e6) |
| RouterBench replay n = 1,000,000 | honest | flat probe argmax (online) | 0.8292 100s (replay_scale_v5) | 0.8946 100s (bars) | 0.9257 20s (rivals_b_replay_1e6) |
| RouterBench replay n = 1,000,000 | cartel | flat probe argmax (online) | 0.8292 100s (replay_scale_v5) | 0.8946 100s (bars) | 0.9249 12s (rivals_b_replay_1e6) |
| RouterBench replay n = 1,000,000 | honest | best learned router | 0.9980 100s, `cluster_head_router` ×100 * | 0.9980 100s, `cluster_head_router` ×100 | 0.9983 20s, `cluster_head_router` ×20 |
| RouterBench replay n = 1,000,000 | cartel | best learned router | 0.8619 100s, `cluster_head_router` ×100 * | 0.9022 100s, `flat_nsw_router` ×100 | 0.9372 12s, `flat_nsw_router` ×12 |
| RouterBench replay n = 1,000,000 | honest | best bandit | 0.9937 100s, `warm_start_bandit` ×100 * | 0.9948 100s, `warm_start_bandit` ×100 * | 0.9955 20s, `warm_start_bandit` ×20 * |
| RouterBench replay n = 1,000,000 | cartel | best bandit | 0.8719 100s, `warm_start_bandit` ×100 * | 0.8926 100s, `warm_start_bandit` ×100 * | 0.9114 12s, `warm_start_bandit` ×12 * |
| RouterBench replay n = 1,000,000 | honest | declared argmax | n/a | 0.9977 100s (bars) | n/a |
| RouterBench replay n = 1,000,000 | cartel | declared argmax | n/a | 0.8505 100s (bars) | n/a |
| RouterBench replay n = 1,000,000 | honest | random | n/a | 0.2421 100s (bars) | n/a |
| RouterBench replay n = 1,000,000 | cartel | random | n/a | 0.2421 100s (bars) | n/a |
| RouterEval n = 5,000 | honest | MIDIAN-VA | 0.6712 3s (va_b_routereval5k) | 0.7820 3s (bars) | 0.8399 3s (va_b_routereval5k) |
| RouterEval n = 5,000 | cartel | MIDIAN-VA | 0.6650 3s (va_b_routereval5k) | 0.7869 3s (bars) | 0.8313 3s (va_b_routereval5k) |
| RouterEval n = 5,000 | honest | MIDIAN | * | 0.7131 3s (bars) | * |
| RouterEval n = 5,000 | cartel | MIDIAN | * | 0.6638 3s (bars) | * |
| RouterEval n = 5,000 | honest | flat probe argmax (online) | * | 0.6884 3s (bars) | * |
| RouterEval n = 5,000 | cartel | flat probe argmax (online) | * | 0.6884 3s (bars) | * |
| RouterEval n = 5,000 | honest | best learned router | * | 0.6749 3s, `knn_router` ×3 * | * |
| RouterEval n = 5,000 | cartel | best learned router | * | 0.6749 3s, `knn_router` ×3 * | * |
| RouterEval n = 5,000 | honest | best bandit | * | 0.9113 3s, `warm_start_bandit` ×3 * | * |
| RouterEval n = 5,000 | cartel | best bandit | * | 0.6736 3s, `warm_start_bandit` ×2; `linucb_honest` ×1 * | * |
| RouterEval n = 5,000 | honest | declared argmax | n/a | 0.9581 3s (bars) | n/a |
| RouterEval n = 5,000 | cartel | declared argmax | n/a | 0.6810 3s (bars) | n/a |
| RouterEval n = 5,000 | honest | random | n/a | 0.6096 3s (bars) | n/a |
| RouterEval n = 5,000 | cartel | random | n/a | 0.6096 3s (bars) | n/a |
| LLMRouterBench n = 20 | honest | MIDIAN-VA | 0.8880 5s (va_b_llmrouterbench) | 0.9222 5s (bars) | 0.9539 5s (va_b_llmrouterbench) |
| LLMRouterBench n = 20 | cartel | MIDIAN-VA | 0.8850 5s (va_b_llmrouterbench) | 0.9189 5s (bars) | 0.9512 5s (va_b_llmrouterbench) |
| LLMRouterBench n = 20 | honest | MIDIAN | 0.8880 5s (rivals_b_llmrouterbench) | 0.9297 5s (bars) | 0.9553 5s (rivals_b_llmrouterbench) |
| LLMRouterBench n = 20 | cartel | MIDIAN | 0.8662 5s (rivals_b_llmrouterbench) | 0.8350 5s (bars) | 0.8497 5s (rivals_b_llmrouterbench) |
| LLMRouterBench n = 20 | honest | flat probe argmax (online) | 0.9040 5s (rivals_b_llmrouterbench) | 0.9476 5s (bars) | 0.9721 5s (rivals_b_llmrouterbench) |
| LLMRouterBench n = 20 | cartel | flat probe argmax (online) | 0.9040 5s (rivals_b_llmrouterbench) | 0.9476 5s (bars) | 0.9721 5s (rivals_b_llmrouterbench) |
| LLMRouterBench n = 20 | honest | best learned router | 0.9895 5s, `cluster_head_router` ×5 | 0.9476 5s, `mlp_router` ×5 * | 0.9895 5s, `cluster_head_router` ×5 |
| LLMRouterBench n = 20 | cartel | best learned router | 0.7961 5s, `mlp_router` ×2; `knn_router_online` ×2; `disrouter_cascade` ×1 | 0.9476 5s, `mlp_router` ×5 * | 0.9186 5s, `knn_router_online` ×2; `mlp_router` ×2; `flat_nsw_router` ×1 |
| LLMRouterBench n = 20 | honest | best bandit | 0.9310 5s, `warm_start_bandit` ×5 * | 0.9321 5s, `warm_start_bandit` ×3; `linucb_honest` ×2 * | 0.9462 5s, `linucb_honest` ×4; `warm_start_bandit` ×1 * |
| LLMRouterBench n = 20 | cartel | best bandit | 0.9012 5s, `linucb_honest` ×5 * | 0.9366 5s, `linucb_honest` ×5 * | 0.9559 5s, `linucb_honest` ×5 * |
| LLMRouterBench n = 20 | honest | declared argmax | n/a | 0.9898 5s (bars) | n/a |
| LLMRouterBench n = 20 | cartel | declared argmax | n/a | 0.7837 5s (bars) | n/a |
| LLMRouterBench n = 20 | honest | random | n/a | 0.6745 5s (bars) | n/a |
| LLMRouterBench n = 20 | cartel | random | n/a | 0.6745 5s (bars) | n/a |

---

### 4.3 Completeness right now, and how the figure shows it

The source grids are still being written, so a regeneration would change A and B as described below.

**What is missing, and why:**
- **Live n=1e5, MIDIAN-VA b=5 (both regimes) and cartel b=1**: `va_b_n100k` has 6 rows (b=1, β=0, seeds 1-3). The SLURM pack `rte_va_b_n100k_pack` is running. Shown as `*`.
- **Live rivals at b=1/5**:
  - n=100 and n=1000: `rivals_b_n100` and `rivals_b_n1000` (`configs/grid.yaml:1068-1069`) are queued in the focus pack plan and have no results directory yet.
  - n=10k: `rivals_b_n10k` has b = 1 honest (3 seeds); its pack is running.
  - n=1e5: `rivals_b_n100k` has 3 jobs PENDING.
- **Bernoulli rivals b=5**: `rivals_b_bernoulli_1e7` had no rows at 02:06 (4 seeds now; the rest pending as 20 seed-block jobs), so the figure shows `*`. **RouterEval rivals b=1/5**: `rivals_b_routereval5k`, 0 rows (running): `*`.
- **Pool candidates** (4.0.4 table): the `pool_fill_*`, `pool_seeds_n1000` and `tuned_wsb_*` grids are queued or pending, with no rows yet. Every pooled bar that misses a candidate carries the incomplete-pool `*`.
- **Partial-seed b=5 bars already drawn**:
  - bernoulli MIDIAN-VA: 51 / 36 of 100 seeds;
  - replay MIDIAN-VA: 38 / 25 complete of 100;
  - replay rivals: 20 honest / 12 cartel of 100 at 02:06 (24 / 13–16 now).

  These are drawn with no visual flag. Their seed sets are subsets of the b=1/b=3 seeds (100 each), so the b-steps in B for bernoulli and replay are **not seed-matched**.

**How missing data is marked:**
- A missing budget for an arm that has another budget in that (cell, regime) → `*` at y ≈ 0.205.
- A pooled bar whose pool misses a candidate at that b → `*` above the bar.
- An arm entirely absent from a (cell, regime) → empty slot, no mark.
- The CSV omits missing bars; the incomplete pool is flagged in `chosen`.
- Nothing distinguishes a partial-seed bar from a complete one.

---

### 4.4 Figures C and D
They are not drawn by `condensed_figs.py`. See Part 6.

### 4.5 Discrepancies, open questions and possible issues

1. **A pooled bar is not one arm's success.** "Best learned router" / "best bandit" are cross-fitted (§2.1.3): each seed is scored with the arm that is best on the other seeds, so the bar can mix arms (e.g. live 10^3 best bandit, `warm_start_bandit` ×5 and `linucb_honest` ×5; LLMRouterBench cartel learned b = 1, three arms). It is an honest estimate of "pick the best arm from past seeds, then deploy it", and it is at most the best single arm's mean. With 3 seeds (live 10^4 / 10^5, RouterEval) each pick rests on 2 seeds. The `chosen` column lists the picks.
2. **The pools mix channels.**
   - `LEARNED` includes `cluster_head_router` and `disrouter_cascade`, which METHODS.md:35-44 lists as *declared-channel-only* arms. They are also in `bar_figs.DECL`.
   - In bernoulli and replay (honest), "best learned router" is `cluster_head_router` on every seed, which is effectively declared argmax (0.9909 vs 0.9942; 0.9980 vs 0.9977) and is b-invariant.
   - `flat_nsw_router` is a verified-centralised arm, not a published learned router.
   - The legend label "best learned router" overstates what these bars are.
3. **Pool candidates still landing.** The candidate set is the same at every b of a cell (`POOLS` minus `NOT_RUNNABLE`), but many candidates have no rows yet at some b (4.0.4 table); those bars carry the incomplete-pool `*`. The effect is visible on LLMRouterBench: at b = 1 / 5 every honest seed picks `cluster_head_router` (0.9895), while at b = 3 that candidate has no rows yet (`pool_fill_llmrouterbench` is pending), so every seed picks `mlp_router` (0.9476 `*`). The apparent dip at b = 3 is a missing candidate, not a budget effect.
4. **b=1/5 and b=3 come from different runs and different snapshots.**
   - b=3 is the bars CSV (22:39 on 2026-09-22), pooled from several grids.
   - b=1/5 is `va_b_*` / `rivals_b_*` read live at 02:05–02:06; the pooled arms are read from the raw rows at every b at that time.
   - For live, RouterEval and LLMRouterBench the task streams match: the oracles are identical. But the b=3 bar for, e.g., MIDIAN-VA n=100 averages rows from 3 grids per seed, while its b=1/5 twin comes from 1 grid.
5. **Cross-grid averaging within a seed at b=3 (live).**
   - `stats_from_rows` averages every row of a (label, seed) across `LIVE_GRIDS`, with no dedup, and the duplicate rows are **not identical**. `midian` n=100 β=0 seed 1: 0.816 (`fw_live_n100`), 0.816 (`learned_n100`), 0.783 (`live_core_n100`). The maximum cross-grid spread is 0.076 at n=100, 0.020 at n=1000 and 0.017 at n=10k.
   - Seeds 1-5 at n=100 carry an extra `live_core_n100` row that seeds 6-10 lack, so seeds are weighted unevenly in content (though equally in count).
   - `live_core_n100` / `live_f1_n1000` may be older-code runs; I did not verify which code version produced them.
6. **Unequal seed counts inside one group.**
   - At live n=1000 b=3 the seven `live_f1_n1000`-only pool candidates (`warm_start_bandit`, `cluster_head`, `disrouter`, `flat_nsw`, `ucb`, `thompson`, `trueskill`) have **5 seeds**, so seeds 6–10 choose among the other candidates (§2.1.3); `pool_seeds_n1000` is queued to supply seeds 6–10.
   - n=1000 `flat_probe_argmax_online` mixes 5-seed `live_f1_n1000` rows with 10-seed `variants_f1` rows.
   - B's bernoulli and replay b=5 bars use 12–51 seeds against 100 for b=1/3, with no marker.
7. **Honest `liar_select` differs by source.** At b=3 honest uses `liar_select = random` rows (`bar_figs.py:81`); at b=1/5 the `va_b` / `rivals_b` β=0 rows are tagged `low_skill_first`. The pooled arms take β=0 rows under either tag (`seed_tables.py:44-45`). This is inert at β=0 (`n_liars = 0`), but a different cell tag.
8. **Normalisation in B uses one scalar**: the β=0, b=3 oracle from the bars CSV, for every bar in the group.
   - For bernoulli and replay, MIDIAN-VA's honest b=5 subsets have their own oracles 0.8456 and 0.7885 against 0.8462 and 0.7900. That is a small bias, 0.1–0.2 %.
   - The ratio CIs ignore oracle uncertainty and are not per-seed ratios.
9. **Replay pooling rules differ by b.** b=1/3 (matrix) average all (shape, seed) units; b=5 keeps only seeds with every shape (`:169-173`); the pooled arms keep only complete-shape seeds at every b (`seed_tables.py:55-58`). They are equivalent only when complete. The b=1/3 matrices are complete (100 × 3); b=5 is not.
10. **Live rivals at b = 1 / 5 are still to come.** `rivals_b_n100` and `rivals_b_n1000` (`configs/grid.yaml:1068-1069`) are queued in the focus pack plan and have no rows; `rivals_b_n100k` is pending; `rivals_b_n10k` has b = 1 honest only. Until they land, the A stars at those slots stay.
11. **Documentation tension about b=1.** README.md:344-345 says "never run b = 1 beside b = 3 in one table (verification is unfunded at b = 1)". A and B put them side by side, and MIDIAN-VA b=1 honest equals MIDIAN b=1 honest in bernoulli (0.8015), replay (0.8272) and LLMRouterBench (0.8880), consistent with verification being unfunded at b=1.
12. **Stacked presentation.**
    - Stacked bars are overlapping full-height bars, not additive segments.
    - Negative b-gains appear only as out-of-order shades; the legend and title do not say this, only the module docstring (`:9`).
    - The stacked CSV holds absolute values, not the gains its title describes.
    - Two missing budgets stack two stars at one x.
    - There are no CIs, by design.
13. **Dead code and a title that is easy to misread.**
    - The `nested` rendering path (`:88-89`, `:104`, `KEY_NEST` `:125`) is never called.
    - The `_allb` title says "light / mid / dark = b = 1 / 3 / 5", but budget-less arms are drawn in the mid (base) colour for their single bar. Correct, but easy to misread as b=3 only.
14. **Latent loader hazard (no effect today).** `fw_variant_numbers.load` deduplicates on `(n, b, dist, beta, liar_select, seed, method, params)` (`:32`), which omits `declared_source`, `K`, `Q`, `collude` and `lie_mode`. For a grid that mixes declared sources (e.g. `live_f1_n1000` has both `programmatic` and `self_described` rows with identical keys) it silently keeps one of the two. All `va_b_*` / `rivals_b_*` grids have a single value of each omitted axis today (verified), so A and B are unaffected. It would bite if a b-grid ever mixes them.
15. **`HIDE_HALVING` has no effect on A/B.** `extra_figs.HIDE_HALVING = True` (`:120`) and the do-not-add list (`excluded`, `:123-135`) are applied (`cells` `:56`, `add_budgets` `:168`), but no ARMS entry or pool member is a halving, SH/SHA or r≠10 arm, so A/B are unchanged by them.
    - Frameworks are dropped at `load()` (`:46`) and in `seed_tables.rows` (`seed_tables.py:39`).
    - **No erratum-28 filtering and no fallback-row filtering** happens anywhere in this path. Neither matters for A/B, since those issues affect framework rows only, and frameworks are not drawn.
16. **The "RouterEval strong_to_weak" key is a misnomer at n=5000.** `bar_figs.py:172` does not filter the 5000-LLM pool by `dist` (it is `all`), so B's RouterEval group is the leaderboard pool, not a strong-to-weak pool.
17. **Live n=100 cartel MIDIAN-VA b=5 equals honest (0.8121) seed for seed.** I checked this. It is genuine and not a duplicated-row bug: `n_liars` 0 vs 50, `misroute_to_liar` > 0 in the cartel rows, same oracle. It does warrant a sentence in any caption, because it looks like a copy error.
18. **Very small-seed CIs.** CIs from 3 seeds (live 10^4 and 10^5, RouterEval) are percentile bootstraps over at most 10 distinct resample means. They are coarse and tend to under-cover.

---

## 5. Figures E, F, G, H — the shortlist figures

Scope: `figures/condensed_sample/{E_shortlists_by_n, F_shortlists_1e5, G_shortlist_lift_1e5, H_routereval_shortlists}.{png,pdf,csv}`.
All claims below are traced to code (`path:line`, repo root `/n/home02/rsiegelmann/rte`) and, where numeric, recomputed from the raw
rows under `$RTE_DATA/results/<grid>/{rows.csv,rows.d/}` (`RTE_DATA=/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte`).
Verification status (2026-09-23 ~02:30): every E–H CSV row (mean, best, k, star) was recomputed from the per-framework rows
of `figures/shortlist/{live,routereval}.csv` and matches. Those CSVs and E–H were all written at 01:52. Rows that landed
after 01:52 are listed in §5.1.6.

## 5.1 The pipeline E–H share

### 5.1.1 Two stages, two scripts

```
$RTE_DATA/results/<grid>/rows.csv + rows.d/*.json
        │  scripts/shortlist_figs.py  (collect → draw → main)
        ▼
figures/shortlist/{live,routereval}.csv   (one row per framework × shortlist × (n, dist, regime), plus oracle / MIDIAN-VA rows)
        │  scripts/shortlist_condensed.py (load → summarise → fig_E / fig_F / fig_G)
        ▼
figures/condensed_sample/{E,F,G,H}_*.{png,pdf,csv}
```

- The condensed script never reads raw rows. It reads only `figures/shortlist/{family}.csv` (`shortlist_condensed.py:32`).
  That CSV is written by `shortlist_figs.main` (`shortlist_figs.py:164`). **The condensed figures are only as fresh as the
  last `shortlist_figs.py` run.** Right now `live.csv`/`routereval.csv` and E–H all date from 2026-09-23 01:52 (§5.1.6).
- Both scripts must run with `RTE_DATA` set. Otherwise `R` falls back to `/scratch/rte/results` (`shortlist_figs.py:24`,
  `fw_variant_numbers.py:11`). On the login node that directory exists but holds only an empty `extra_figs/`, so the run
  would find no grids and write empty CSVs.

### 5.1.2 Stage 1: `shortlist_figs.collect` (raw rows → per-framework seed means)

1. **Grids read** (`shortlist_figs.py:38-39`, `GRIDS`):
   - live: every directory whose name starts with `fw_live_n` and does not contain `lietext`, plus `live_n10k_v2` and
     `live_n100k`. Python precedence makes this `(A and B) or C`, which is the intended reading.
   - routereval: every directory starting with `fw_routereval_` or `re_sl_`.
   Grids are visited in sorted name order (`:82`).
2. **Row loading** (`rows`, `:65-71`): `fw_variant_numbers.load` (`fw_variant_numbers.py:24-32`) concatenates `rows.d/*.json` and
   `rows.csv`, drops duplicate `rid`, then drops duplicates on (n, b, dist, beta, liar_select, seed, method, params).
   Only **b = 3** is kept (`b.fillna(3) == 3`, `:69`). Every row in the plotted cells has b = 3.
3. **Framework rows**: `method` starts with `fw_` (`:85`). Each row gets a shortlist key from its params JSON via `source()`
   (`:42-53`). Rows that map to `None` are dropped (`:86`):

   | params | source key | display (SOURCES `:30-36`) |
   |---|---|---|
   | contains `supervisor` (14B Magentic arm) or `lie_text` | dropped | — |
   | no `retrieval`, no `dedup` | `tfidf` | hashed TF-IDF (pre-registered) |
   | no `retrieval`, `dedup: true` | dropped (dedup TF-IDF not drawn) | — |
   | `retrieval: midian_va`, `r: 10`, not `shuffle` | `va_cohort` | MIDIAN-VA leaf cohort |
   | `retrieval: midian_va` with shuffle or r ≠ 10 | dropped | — |
   | `retrieval: embed`, model contains "Qwen" | `dense` / `dense_icomp` / `dense_idemo` (instruction contains "competent" / "demonstrated") | Qwen3-8B dense [, I-competent / I-demonstrated] |
   | `retrieval: embed`, no Qwen model (MiniLM default) | `embed` | MiniLM |
   | `retrieval: sota` | `sota` / `sota_icomp` / `sota_idemo` | fusion + reranker [, I-…] |
   | `retrieval: bm25` / `declared` | `bm25` / `declared` | BM25 / declared-claim top-k |
   | `retrieval: hybrid` (fusion without reranker), `retrieval: midian` (V cohort) | dropped | — |

   `dedup` is ignored for every retrieval shortlist. In the data, every `embed`, `sota`, `declared`, `bm25` and `dense` row
   carries `"dedup": true` (verified on `fw_live_n100k_*`, `fw_live_n10k_em`, `fw_routereval_5k_em`). So only the `tfidf`
   bar is un-deduplicated: at 10^5 it is the clone-filled pre-registered adapter (erratum 25).
4. **Regime** (`tagged`, `:74-75` → `fw_variant_numbers.regime`, `:55-59`): β = 0 → `beta0` whatever the liar_select.
   β = 0.5 with `liar_select == "low_skill_first"` → `cartel`. Everything else → `beta<x>_<random|cartel>`. **"Hatched =
   β = 0.5 low-skill cartel" is verified**: only (β = 0.5, low_skill_first) maps to `cartel`.
5. **Liar-free de-duplication** (`:92`): when a β = 0 cell has rows under both liar_select values (grids with a liar_select
   axis, e.g. `live_n100k`, `fw_routereval_*`), only `random` is kept. The two copies are the same liar-free cell.
6. **Per-framework seed series** (`:93`): `q.groupby(["method","seed"]).success.mean().unstack(0)`. Each seed's value is the mean
   success of that framework's rows for that seed. Each plotted cell has exactly one params value per source (`nparams = 1`),
   so this is one row per seed.
7. **Do-not-add** (`:96`): `extra_figs.excluded` (`extra_figs.py:123-135`) removes nothing here (no framework name matches).
8. **One grid per (framework, shortlist, cell), with no pooling across grids** (`:98`): the grid with the most seeds wins.
   The comparison is strict `>`, so on a tie the alphabetically first grid wins.
9. **Erratum-28 star** (`:81`, `:87-90`, `:99`): `pending = fw_variant_numbers.pending_reruns()` (`fw_variant_numbers.py:35-42`)
   reads `results/quarantine_units.tsv` and returns every (grid, method, dist, regime) listed there, unless
   `logs/DONE_stage2` exists. With `DONE_stage1` present it keeps only the `fw_live_n(100|1000)(_lowskill)?_sota` grids.
   **Neither DONE file exists today**, so every quarantined unit counts as pending. Two consequences:
   - The star is keyed on the quarantine list, not on whether rerun rows have landed. It persists until `DONE_stage2`,
     even when every seed is back (e.g. n = 100 TF-IDF honest now has 10/10 seeds for every framework and is still starred).
   - A framework whose rows were **all** quarantined gets an empty, starred slot **in the per-condition figure only**
     (`:87-90`, `:135`). Nothing is written to `live.csv` for it (CSV rows are written only for series that exist, `:140`),
     so the condensed figures never learn it existed (§5.1.4, Discrepancy D1).
10. **Reference lines** (`:100-109`): for each (n, dist, regime) cell, the rows with `method ∈ {oracle, midian_va}` are taken
    from the grids in `REF_GRIDS[family][n]` (`:55-59`), filtered to the same n, dist and regime (same β = 0 random-only rule),
    and turned into per-seed means. Again the grid with the most seeds wins, and the first listed wins a tie. **There is no
    pooling and no seed matching to the framework bars**, despite the comment on `:54` ("pooled and matched on (dist, regime,
    seed)"). `midian_va` here is the whole-population MIDIAN-VA router arm, written to the CSV as
    `"MIDIAN-VA (whole population)"` (`:121`, `:126`).
11. **CSV row** (`draw`, `:113-154`; header at `:164`): `family, regime, dist, n, arm (ABBR name), shortlist, mean, ci_lo, ci_hi,
    seeds, rerun_outstanding`. Here `mean = s.mean()` over seeds, and the CI is a 95 % bootstrap over seeds
    (`extra_figs.ci`, `extra_figs.py:89-97`, B = 2000, rng seed 0). Reference rows get `shortlist = "-"`. `draw` returns early
    (`:117`) for a cell with fewer than two shortlists, but the reference rows for such a cell were already appended at
    `:126`. The fw rows are appended afterwards, so for such a cell nothing is written. This does not affect E–H: every
    plotted cell has at least 2 shortlists.

### 5.1.3 Stage 2: `shortlist_condensed` (per-framework means → bars)

- `load(family)` (`shortlist_condensed.py:31-34`): live keeps `dist == "specialist"`. routereval keeps
  `dist == "strong_to_weak"` **or** `n == 5000`. Both keep only `regime ∈ {beta0, cartel}` (`REG`, `:23`). The per-condition
  CSV also holds β = 0.1, 0.25 and the random-β = 0.5 regimes; E–H never draw them.
- `summarise(d)` (`:37-43`): for each (n, regime, shortlist) over framework rows (`shortlist != "-"`):
  - `mean` = **unweighted mean over frameworks** of each framework's seed mean. A framework with 3 seeds weighs the same as
    one with 10.
  - `best` = max over frameworks.
  - `k` = number of distinct frameworks (`arm` nunique).
  - `star` = any framework's `rerun_outstanding`.

  `ref` = a pivot of the `-` rows to (n, regime) × {oracle, MIDIAN-VA (whole population)}.
- `lines(ax, ref, n, x0, x1, first)` (`:46-50`): draws **only the (n, "beta0") reference values**: the oracle as a grey dotted
  `hlines` (#7f8c8d, lw 1.2) and MIDIAN-VA as a solid green `hlines` (#2ecc71, lw 1.4). The cartel-regime MIDIAN-VA value is in
  the CSV but is never drawn, so hatched bars are always read against the honest MIDIAN-VA line. The oracle is identical in
  both regimes in every plotted cell (e.g. 0.8449 / 0.8449 at live 10^2; lying changes declarations, not skill).
  MIDIAN-VA is not: 0.7816 vs 0.7680 at live 10^2, 0.8356 vs 0.8278 at 10^5, 0.7130 vs 0.6942 at RouterEval 10^3.
- `pair(ax, x, w, q, src, label, dots)` (`:53-68`), for h ∈ {0: beta0, 1: cartel}, at `x + (h − 0.5)·w`:
  - **Regime missing for this (n, shortlist)** → no bar; a `*` at y = 0.205 (the "empty slot / not in yet" star, `:61`).
  - **Otherwise** → a bar of height `mean` in the SOURCES colour. Cartel bars get `hatch="////"` and alpha 0.75. If `dots`, a
    black dot at `best` (ms 2.2). If `star`, a `*` at `mean + 0.008` (or `max(mean, best) + 0.008` with dots): the
    erratum-28 star (`:67`).
- `finish` (`:96-102`): `ylim(0.2, 0.95)`, y label "success", y grid. Legend above the axes (E/H, `ncol=6`) or inside at upper
  right (F, `ncol=3`). Saves png at dpi 250 plus pdf, and writes the summarised frame to `<name>.csv`.
- **Legend order**: importing `extra_figs` (`shortlist_condensed.py:16`) monkey-patches `Axes.legend` (`extra_figs.py:38-58`).
  Entries are ranked descending by the mean y of what each handle plots (`_plotted`, `:16-29`). Handles that plot nothing
  (F's "best single framework" dummy, `ax.plot([], [])`) go last. The list is then permuted row-major (`_rowmajor`, `:32-35`)
  so the best entry sits top-left.
  - In E and H a shortlist's handle is its **first drawn honest bar** (labels are spent once, `:78`). It is therefore ranked
    by its value at the smallest n where it exists. E.g. MiniLM is ranked by its 10^2 value (0.530) and declared top-k by
    its 10^5 value (0.620).
  - The E legend in the PNG reads: oracle, MIDIAN-VA, declared top-k, VA cohort, rerank, MiniLM / dense I-comp, TF-IDF,
    dense I-demo, rerank I-comp, rerank I-demo. This matches that rule.

### 5.1.4 Who contributes: frameworks, seeds, cell parameters

- **The ten frameworks** (`configs/grid.yaml` `_sets.frameworks`, ABBR in `scripts/paper_figs.py:26-27`): autogen,
  camel (CAMEL workforce), crewai, adk (Google ADK), langgraph, llama (LlamaIndex), maf, magentic (Magentic-One), openai
  (OpenAI Agents), smol (smolagents).
  - Magentic-One is **excluded by design** from `fw_live_n10k_cartel` (`grid.yaml:515-526`; "dropped for time … it was the TOP
    of the β = 0 range") and from every `fw_routereval_*` and `re_sl_*` grid (`grid.yaml:503-512`, `:528-538`). At the
    10^4 cartel it is queued separately (`fw_live_n10k_cartel_magentic`, `grid.yaml:1129-1132`; no rows yet).
  - **CrewAI and ADK are absent from most non-TF-IDF bars.** All their rows in the `_em`, `_sota`, `_verified_va`,
    `_declared`, `_dense_instruct` and `_sota_instruct` grids at 10^4–10^5 and in every `fw_routereval_*_em/_va` grid were
    moved to `quarantine/` (erratum 28, `CHANGES_AND_ERRATA.md:368-386`), and their reruns have not landed. The same holds
    at 10^2–10^3 for `_em` and `_sota`, except crewai's 10^3 MiniLM honest and VA honest, which are in the figure. The
    RouterEval declared top-k bars (`re_sl_declared_*`) do include both.
  - **ADK has no honest rows at live 10^2 / 10^3** (TF-IDF and MiniLM). Its Google ADK supervisor calls a non-existent
    tool named after the agent ("Tool 'agent_000030' not found"). Those units were refused as infrastructure errors past
    the erratum-28 threshold (`_common.py:379-389`), so they wrote no row. The current adapter treats this error as a
    supervisor non-pick instead: declared argmax inside the shortlist, counted in `invalid_action` and `fallback_rate`,
    never an infrastructure error (erratum 29, `CHANGES_AND_ERRATA.md` §8e; `INVALID_ACTION`, `_common.py:26-29, 371-378`;
    §3.2.6). The ADK reruns are queued in the focus pack plan. Whether ADK
    belongs in these means is an author decision. ADK is also absent from the 10^4 TF-IDF bars.
- **Probes.** A framework reads only the declared/self-described channel. `FrameworkMethod.needs = {"declared"}`
  (`rte/methods/frameworks/_common.py:111`). **The one exception is the VA-cohort shortlist**, which adds
  `{"probe", "reports"}` (`:160-161`) and builds an internal `MidianVA(r=10)` with the cell's budget (`:290-292`). Its
  shortlist is that router's pick followed by the rest of its leaf cohort (`:312-319`). So the VA-cohort bars spend the
  same b = 3 probe budget as MIDIAN-VA; all other bars spend none. Every shortlist has k = 10 by default (`:116`, `:331`).
- **Cell parameters** (`configs/grid.yaml`; the K = 16 default is at `:8`). Grids named `_em/_sota/_va/_declared/…` are
  `mirror_of` the base grid on the same row, so the cells match.

  | family / n | base grid (line) | backend | dist | K | b | Q | seeds | declared channel |
  |---|---|---|---|---|---|---|---|---|
  | live 10^2 | `fw_live_n100` (`:112`, mirror of `fw_live_n1000`) and `fw_live_n100_lowskill` (`:146`) | llm | specialist | 16 | 3 | 1000 | 1–10 | self_described |
  | live 10^3 | `fw_live_n1000` (`:107-109`), `fw_live_n1000_lowskill` (`:145`) | llm | specialist | 16 | 3 | 1000 | 1–10 | self_described |
  | live 10^4 | `live_n10k_v2` (`:130`, β ∈ {0, 0.25}, random liars), `fw_live_n10k_cartel` (`:515`, β = 0.5 low_skill_first) | llm | specialist | 16 | 3 | 300 | 1–3 | self_described |
  | live 10^5 | `live_n100k` (`:460-481`, β ∈ {0, 0.25, 0.5} × both liar_select) | llm | specialist | 16 | 3 | 300 | 1–3 | self_described |
  | RouterEval m = 10, 100 | `fw_routereval_small` (`:970`, mirror of `fw_routereval_1k`) | routereval (mmlu) | strong_to_weak (plotted) | 16 | 3 | 1000 | 1–5 | programmatic, rendered to text |
  | RouterEval m = 1000 | `fw_routereval_1k` (`:503`, mirror of `routereval_mmlu` `:316`, β ∈ {0, 0.5}) | routereval (mmlu) | strong_to_weak | 16 | 3 | 1000 | 1–5 | programmatic |
  | RouterEval 5000 | `fw_routereval_5k` (`:528`, mirror of `routereval_mmlu5k` `:359`) | routereval (leaderboard_mmlu) | all | 16 | 3 | 300 | 1–3 | programmatic |

### 5.1.5 Reference-line provenance (all reproduced exactly)

| family, n | oracle (grid, seeds, β = 0 value) | MIDIAN-VA whole population (grid, seeds, β = 0 value) | why this grid |
|---|---|---|---|
| live 10^2 | `fw_live_n100`, 1–10, 0.8449 | `fw_live_n100`, 1–10, 0.7816 | first in list; `learned_n100` ties with identical values |
| live 10^3 | `fw_live_n1000`, 1–10, 0.8612 | `fw_live_n1000`, 1–10, 0.8134 | first; `variants_f1`/`learned_f1` tie (identical) |
| live 10^4 | `learned_n10k`, 1–3, 0.8589 | `learned_n10k`, 1–3, 0.8111 | first; `live_n10k_v2` ties with identical values |
| live 10^5 | `live_n100k`, 1–3, 0.8622 | `live_n100k`, 1–3, 0.8356 | `midian_va` exists only there |
| RouterEval 10 / 100 / 1000 | `routereval_mmlu`, 1–5: 0.8622 / 0.7920 / 0.8722 | `routereval_mmlu`, 1–5: 0.7658 / 0.6694 / 0.7130 | only grid listed |
| RouterEval 5000 | `routereval_mmlu5k`, 1–3, 0.9022 | `routereval_mmlu5k`, 1–3, 0.7056 | only grid listed |

At every n the reference seeds are the same seed set as the framework grids (1–10, 1–3 or 1–5), and the cells match by
`mirror_of`. But the line is not restricted to the seeds each framework actually has. That matters where a framework is
missing seeds (e.g. RouterEval camel cartel with 1 seed, `fw_live_n1000_sota` with 3–6 seeds).

### 5.1.6 Freshness: what the figures contain and what landed since

`figures/shortlist/{live,routereval}.csv` and the four condensed CSVs were written together at 01:52.
- **Checked:** every E and H CSV row (mean over frameworks, best, k, star) equals `summarise` applied to the per-framework
  rows of those CSVs, and every framework × seed count in the tables below is taken from them.
- **Rows newer than `live.csv`** (`find -newer`, ~02:25): `fw_live_n1000` 2, `re_sl_declared_small` 42,
  `re_sl_declared_1k` 86 and `re_sl_declared_5k` 58. A rerun of `shortlist_figs.py` and then `shortlist_condensed.py`
  (with `RTE_DATA` set) would add H's declared top-k bars at m = 1,000 (8 frameworks) and 5,000 (9 frameworks), and more
  seeds to the declared bars at m = 10 / 100.

## 5.2 Figure E: `E_shortlists_by_n` (live, specialist, n = 10^2 … 10^5)

**Question.** On the live LLM backend (specialist population), how well do deployed frameworks route under each shortlist
source, and how does that change with population size n? It also shows how far each is from the oracle and from MIDIAN-VA
routing the whole population, honest and under the β = 0.5 low-skill cartel.

**Code.** `fig_E(s, ref)` (`shortlist_condensed.py:71-81`, called at `:133`).
- **x axis**: one group per n, sorted (`:73`; 100, 1,000, 10,000, 100,000), tick "n = …".
- **Within a group**: one fixed slot per source in `MAIN` order (`:28`): TF-IDF, MiniLM, dense I-comp, dense I-demo, rerank,
  rerank I-comp, rerank I-demo, declared top-k, VA cohort. BM25 and plain Qwen dense are excluded from E; they exist only
  at 10^3.
- **Slot geometry**: slot width w = 0.86/18 ≈ 0.048. Source j sits at `i + (2j+1−9)·w`, honest at −w/2 and cartel at +w/2
  (`:75-78`, `pair`).
- **Marks**:
  - Bars are **mean over frameworks** (`dots=False`, `:78`), so **E has no black dots**.
  - The reference lines span `i ± 0.46` and use the β = 0 values (`:79`).
  - y axis 0.2–0.95.
  - Colours come from SOURCES; legend labels come from `SHORT` (`:25-27`).

**Two kinds of `*` in E, easy to confuse.** The title says "* = not in yet".
- A row of `*` at the baseline (y ≈ 0.205) is an **empty slot**: that (n, source, regime) has no framework row in `live.csv`.
- A `*` just above a bar is the **erratum-28 star**: at least one contributing framework is on the quarantine list.
  Currently TF-IDF at n = 100 and n = 1,000 (both regimes) and MiniLM at n = 1,000 honest.

**Empty slots.** At n = 100, 1,000 and 10,000, the four slots for dense I-comp and dense I-demo (both regimes) and the six
slots for rerank I-comp, rerank I-demo and declared are empty: 4 + 6 `*` per group, as in the PNG. These shortlists are
defined below 10^5 by the backfill grids `fw_live_n{100,1000}_backfill`, `fw_live_n{100,1000}_lowskill_backfill`,
`fw_live_n10k_backfill` and `fw_live_n10k_cartel_backfill` (`grid.yaml:1014-1023`, method list `*fw10_backfill`: dense
I-comp / I-demo, rerank I-comp / I-demo, declared; all ten frameworks, seeds 1–3). They are queued in the focus pack plan
(30 units each) and have no rows yet, which is what the `MAIN` comment "run (or queued) at every n"
(`shortlist_condensed.py:28`) refers to. Nothing is starred-empty at 10^5.

**Provenance per bar.** Winning grids:

| n | TF-IDF | MiniLM | rerank | VA cohort | declared / dense-I / rerank-I |
|---|---|---|---|---|---|
| 100 | `fw_live_n100` / `fw_live_n100_lowskill` | `fw_live_n100_em` / `fw_live_n100_lowskill_em` | `fw_live_n100_sota` / `…_lowskill_sota` | `fw_live_n100_verified_va` / `…_verified_va_lowskill` | queued: `fw_live_n100_backfill` / `fw_live_n100_lowskill_backfill` |
| 1,000 | `fw_live_n1000` / `fw_live_n1000_lowskill` | `fw_live_n1000_em` / `…_lowskill_em` | `fw_live_n1000_sota` / `…_lowskill_sota` | `fw_live_n1000_verified_va` / `…_verified_va_lowskill` | queued: `fw_live_n1000_backfill` / `fw_live_n1000_lowskill_backfill` |
| 10,000 | `live_n10k_v2` / `fw_live_n10k_cartel` | `fw_live_n10k_em` / `fw_live_n10k_cartel_em` | `fw_live_n10k_sota` / `fw_live_n10k_cartel_sota` | `fw_live_n10k_verified_va` / `fw_live_n10k_cartel_verified_va` | queued: `fw_live_n10k_backfill` / `fw_live_n10k_cartel_backfill` |
| 100,000 | `live_n100k` (both) | `fw_live_n100k_em` | `fw_live_n100k_sota` | `fw_live_n100k_verified_va` | `fw_live_n100k_declared`, `fw_live_n100k_dense_instruct`, `fw_live_n100k_sota_instruct` |

Framework sets per bar:
- **n = 10^2, 10^3 (10 seeds)**: MiniLM, rerank and VA use the 8 frameworks without crewai and adk, except 10^3 MiniLM
  honest and 10^3 VA honest, which include crewai (9). TF-IDF uses 9 (with crewai, without adk). **Exception: 10^3 rerank
  honest has only 6 frameworks** (autogen, langgraph, llama, maf, openai, smol) **with 3–10 seeds each** (autogen 6,
  langgraph 3, llama 3, maf 10, openai 4, smol 10). Its cartel twin has the full 8 with 10 seeds, including magentic, the
  usual best.
- **n = 10^4 (3 seeds)**: honest uses the 8 (or 9 for TF-IDF, which includes crewai). Cartel uses 7 (or 8), **because
  Magentic-One is not in any 10^4 cartel grid that has landed.**
- **n = 10^5 (3 seeds)**: every non-TF-IDF bar uses the same 8 (no crewai, no adk). TF-IDF uses all 10.

| x slot | shortlist | regime | mean | best | #fw | star (csv) | frameworks:seeds |
|---|---|---|---|---|---|---|---|
| n = 100 | tfidf | honest (solid) | 0.4772 | 0.5559 | 9 | * above bar | autogen:10 camel:10 crewai:10 langgraph:10 llama:10 maf:10 magentic:10 openai:10 smol:10 |
| n = 100 | tfidf | cartel (hatched) | 0.4632 | 0.4831 | 9 | * above bar | autogen:10 camel:10 crewai:6 langgraph:9 llama:10 maf:10 magentic:10 openai:10 smol:10 |
| n = 100 | embed | honest (solid) | 0.5303 | 0.6111 | 8 |  | autogen:10 camel:10 langgraph:10 llama:10 maf:10 magentic:10 openai:10 smol:10 |
| n = 100 | embed | cartel (hatched) | 0.5205 | 0.5532 | 8 |  | autogen:10 camel:10 langgraph:10 llama:10 maf:10 magentic:10 openai:10 smol:10 |
| n = 100 | dense_icomp | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| n = 100 | dense_icomp | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| n = 100 | dense_idemo | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| n = 100 | dense_idemo | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| n = 100 | sota | honest (solid) | 0.5382 | 0.6037 | 8 |  | autogen:10 camel:10 langgraph:10 llama:10 maf:10 magentic:10 openai:10 smol:10 |
| n = 100 | sota | cartel (hatched) | 0.5346 | 0.5661 | 8 |  | autogen:10 camel:10 langgraph:10 llama:10 maf:10 magentic:10 openai:10 smol:10 |
| n = 100 | sota_icomp | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| n = 100 | sota_icomp | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| n = 100 | sota_idemo | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| n = 100 | sota_idemo | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| n = 100 | declared | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| n = 100 | declared | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| n = 100 | va_cohort | honest (solid) | 0.5658 | 0.6482 | 8 |  | autogen:10 camel:10 langgraph:10 llama:10 maf:10 magentic:10 openai:10 smol:10 |
| n = 100 | va_cohort | cartel (hatched) | 0.5585 | 0.6103 | 8 |  | autogen:10 camel:10 langgraph:10 llama:10 maf:10 magentic:10 openai:10 smol:10 |
| n = 1,000 | tfidf | honest (solid) | 0.3806 | 0.4313 | 9 | * above bar | autogen:10 camel:10 crewai:10 langgraph:10 llama:10 maf:10 magentic:10 openai:10 smol:10 |
| n = 1,000 | tfidf | cartel (hatched) | 0.3738 | 0.4109 | 9 | * above bar | autogen:10 camel:10 crewai:10 langgraph:10 llama:10 maf:10 magentic:10 openai:10 smol:10 |
| n = 1,000 | embed | honest (solid) | 0.5037 | 0.6315 | 9 | * above bar | autogen:10 camel:10 crewai:10 langgraph:10 llama:10 maf:10 magentic:10 openai:10 smol:10 |
| n = 1,000 | embed | cartel (hatched) | 0.4987 | 0.5598 | 8 |  | autogen:10 camel:10 langgraph:10 llama:10 maf:10 magentic:10 openai:10 smol:10 |
| n = 1,000 | dense_icomp | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| n = 1,000 | dense_icomp | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| n = 1,000 | dense_idemo | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| n = 1,000 | dense_idemo | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| n = 1,000 | sota | honest (solid) | 0.4702 | 0.4883 | 6 |  | autogen:6 langgraph:3 llama:3 maf:10 openai:4 smol:10 |
| n = 1,000 | sota | cartel (hatched) | 0.4801 | 0.5398 | 8 |  | autogen:10 camel:10 langgraph:10 llama:10 maf:10 magentic:10 openai:10 smol:10 |
| n = 1,000 | sota_icomp | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| n = 1,000 | sota_icomp | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| n = 1,000 | sota_idemo | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| n = 1,000 | sota_idemo | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| n = 1,000 | declared | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| n = 1,000 | declared | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| n = 1,000 | va_cohort | honest (solid) | 0.5704 | 0.6593 | 9 |  | autogen:10 camel:10 crewai:10 langgraph:10 llama:10 maf:10 magentic:10 openai:10 smol:10 |
| n = 1,000 | va_cohort | cartel (hatched) | 0.5663 | 0.6211 | 8 |  | autogen:10 camel:10 langgraph:10 llama:10 maf:10 magentic:10 openai:10 smol:10 |
| n = 10,000 | tfidf | honest (solid) | 0.2951 | 0.3667 | 9 |  | autogen:3 camel:3 crewai:3 langgraph:3 llama:3 maf:3 magentic:3 openai:3 smol:3 |
| n = 10,000 | tfidf | cartel (hatched) | 0.2875 | 0.3189 | 8 |  | autogen:3 camel:3 crewai:3 langgraph:3 llama:3 maf:3 openai:3 smol:3 |
| n = 10,000 | embed | honest (solid) | 0.4365 | 0.5711 | 8 |  | autogen:3 camel:3 langgraph:3 llama:3 maf:3 magentic:3 openai:3 smol:3 |
| n = 10,000 | embed | cartel (hatched) | 0.4173 | 0.4544 | 7 |  | autogen:3 camel:3 langgraph:3 llama:3 maf:3 openai:3 smol:3 |
| n = 10,000 | dense_icomp | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| n = 10,000 | dense_icomp | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| n = 10,000 | dense_idemo | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| n = 10,000 | dense_idemo | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| n = 10,000 | sota | honest (solid) | 0.4140 | 0.5156 | 8 |  | autogen:3 camel:3 langgraph:3 llama:3 maf:3 magentic:3 openai:3 smol:3 |
| n = 10,000 | sota | cartel (hatched) | 0.4051 | 0.4767 | 7 |  | autogen:3 camel:3 langgraph:3 llama:3 maf:3 openai:3 smol:3 |
| n = 10,000 | sota_icomp | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| n = 10,000 | sota_icomp | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| n = 10,000 | sota_idemo | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| n = 10,000 | sota_idemo | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| n = 10,000 | declared | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| n = 10,000 | declared | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| n = 10,000 | va_cohort | honest (solid) | 0.5899 | 0.6778 | 8 |  | autogen:3 camel:3 langgraph:3 llama:3 maf:3 magentic:3 openai:3 smol:3 |
| n = 10,000 | va_cohort | cartel (hatched) | 0.5462 | 0.6211 | 7 |  | autogen:3 camel:3 langgraph:3 llama:3 maf:3 openai:3 smol:3 |
| n = 100,000 | tfidf | honest (solid) | 0.3789 | 0.3789 | 10 |  | adk:3 autogen:3 camel:3 crewai:3 langgraph:3 llama:3 maf:3 magentic:3 openai:3 smol:3 |
| n = 100,000 | tfidf | cartel (hatched) | 0.3789 | 0.3789 | 10 |  | adk:3 autogen:3 camel:3 crewai:3 langgraph:3 llama:3 maf:3 magentic:3 openai:3 smol:3 |
| n = 100,000 | embed | honest (solid) | 0.4603 | 0.6011 | 8 |  | autogen:3 camel:3 langgraph:3 llama:3 maf:3 magentic:3 openai:3 smol:3 |
| n = 100,000 | embed | cartel (hatched) | 0.4485 | 0.4978 | 8 |  | autogen:3 camel:3 langgraph:3 llama:3 maf:3 magentic:3 openai:3 smol:3 |
| n = 100,000 | dense_icomp | honest (solid) | 0.4864 | 0.6267 | 8 |  | autogen:3 camel:3 langgraph:3 llama:3 maf:3 magentic:3 openai:3 smol:3 |
| n = 100,000 | dense_icomp | cartel (hatched) | 0.4882 | 0.6089 | 8 |  | autogen:3 camel:3 langgraph:3 llama:3 maf:3 magentic:3 openai:3 smol:3 |
| n = 100,000 | dense_idemo | honest (solid) | 0.4649 | 0.6189 | 8 |  | autogen:3 camel:3 langgraph:3 llama:3 maf:3 magentic:3 openai:3 smol:3 |
| n = 100,000 | dense_idemo | cartel (hatched) | 0.4583 | 0.5711 | 8 |  | autogen:3 camel:3 langgraph:3 llama:3 maf:3 magentic:3 openai:3 smol:3 |
| n = 100,000 | sota | honest (solid) | 0.4028 | 0.5122 | 8 |  | autogen:3 camel:3 langgraph:3 llama:3 maf:3 magentic:3 openai:3 smol:3 |
| n = 100,000 | sota | cartel (hatched) | 0.4047 | 0.4856 | 8 |  | autogen:3 camel:3 langgraph:3 llama:3 maf:3 magentic:3 openai:3 smol:3 |
| n = 100,000 | sota_icomp | honest (solid) | 0.4318 | 0.5533 | 8 |  | autogen:3 camel:3 langgraph:3 llama:3 maf:3 magentic:3 openai:3 smol:3 |
| n = 100,000 | sota_icomp | cartel (hatched) | 0.4279 | 0.5278 | 8 |  | autogen:3 camel:3 langgraph:3 llama:3 maf:3 magentic:3 openai:3 smol:3 |
| n = 100,000 | sota_idemo | honest (solid) | 0.4231 | 0.5211 | 8 |  | autogen:3 camel:3 langgraph:3 llama:3 maf:3 magentic:3 openai:3 smol:3 |
| n = 100,000 | sota_idemo | cartel (hatched) | 0.4164 | 0.5233 | 8 |  | autogen:3 camel:3 langgraph:3 llama:3 maf:3 magentic:3 openai:3 smol:3 |
| n = 100,000 | declared | honest (solid) | 0.6203 | 0.6400 | 8 |  | autogen:3 camel:3 langgraph:3 llama:3 maf:3 magentic:3 openai:3 smol:3 |
| n = 100,000 | declared | cartel (hatched) | 0.5444 | 0.5700 | 8 |  | autogen:3 camel:3 langgraph:3 llama:3 maf:3 magentic:3 openai:3 smol:3 |
| n = 100,000 | va_cohort | honest (solid) | 0.5682 | 0.6656 | 8 |  | autogen:3 camel:3 langgraph:3 llama:3 maf:3 magentic:3 openai:3 smol:3 |
| n = 100,000 | va_cohort | cartel (hatched) | 0.5519 | 0.6667 | 8 |  | autogen:3 camel:3 langgraph:3 llama:3 maf:3 magentic:3 openai:3 smol:3 |

**Reading notes (verified numerically).**
- The honest-vs-hatched gap at 10^4 is largely a framework-set artefact. Magentic-One is the best framework at 10^4 honest
  for every shortlist (MiniLM 0.571, rerank 0.516, TF-IDF 0.367, VA 0.678). Dropping it from the honest means gives MiniLM
  0.4173 and rerank 0.3995, i.e. the cartel values (0.4173, 0.4051) or nearly so.
- The 10^3 rerank pair (honest 0.470, k = 6, no magentic/camel; cartel 0.480, k = 8) is also not a like-for-like comparison.
- 10^3 MiniLM honest (0.5037, k = 9) includes crewai; its cartel twin (0.4987, k = 8) does not.

**Current completeness (E).** Of the 9 × 2 × 4 = 72 slots, 18 are filled at 10^5 and 8 at each of 10^2, 10^3 and 10^4:
42 bars and 30 empty `*` in total. Five bars carry an erratum-28 star: TF-IDF at 10^2 and 10^3 (both regimes) and MiniLM
at 10^3 honest. The crewai/adk absences from every other bar are **not** starred (D1).
- The CSV also carries two rows E does not draw: BM25 and plain dense at 10^3 honest (6 frameworks each).
- There are no cartel rows for BM25 or plain dense at all: `fw_live_n1000_lowskill_sota` holds only sota (+hybrid) arms.

## 5.3 Figure F: `F_shortlists_1e5` (live, specialist, n = 100,000)

**Question.** At the one scale where all nine shortlist sources ran (live 10^5), which shortlist makes deployed
frameworks route best? What does the best single framework reach? How far are both from the oracle and from MIDIAN-VA
routing the whole population?

**Code.** `fig_F(s, ref, n=100000)` (`shortlist_condensed.py:84-93`).
- **x axis**: shortlists at n = 10^5, **sorted by the honest mean, descending** (`:86`). Tick labels are the long
  SOURCES names (`NAME`, `:24`), rotated 28°.
- **Bars**: at each x, a solid honest bar at x − 0.19 and a hatched cartel bar at x + 0.19 (w = 0.38), height = mean over
  frameworks.
- **Black dot**: on each bar, **the best single framework's seed mean** for that (shortlist, regime). This is `best` =
  max over frameworks (`summarise`, `:40`; `pair(..., dots=True)`, `:66`).
- **Reference lines**: dotted oracle 0.8622 and solid MIDIAN-VA 0.8356 across the full width, both the β = 0 values from
  `live_n100k` seeds 1–3 (`:89`). MIDIAN-VA's own cartel value (0.8278) is not drawn.
- **Axes and legend**: y range 0.2–0.95. The legend sits inside at upper right (oracle, MIDIAN-VA, "best single framework").
  Stars would sit at `max(mean, best) + 0.008`; none now.

**Provenance.**
- Every non-TF-IDF bar uses the **same 8 frameworks** (autogen, camel, langgraph, llama, maf, magentic, openai, smol) × seeds
  1–3.
- TF-IDF uses all **10** (adds crewai, adk). Its grid is `live_n100k`; the others are `fw_live_n100k_{declared,
  dense_instruct, em, sota, sota_instruct, verified_va}`. Every one is a `mirror_of: live_n100k` (or copies its cells), so
  the cells are identical: K = 16, b = 3, Q = 300, specialist, self-described, seeds 1–3.
- Regimes: honest = β 0 with liar_select `random` (the `low_skill_first` duplicate is dropped); cartel = β 0.5 with
  `low_skill_first`.
- The TF-IDF "best" dot equals its bar: every framework scores exactly 0.378889 = mean(0.3467, 0.3667, 0.4233) over the
  three seeds, in every regime (the clone shortlist, §5.4).

| x slot | shortlist | regime | mean (bar) | best (black dot) | #fw | star | best framework |
|---|---|---|---|---|---|---|---|
| 0 | declared | honest | 0.6203 | 0.6400 | 8 |  | maf |
| 0 | declared | cartel | 0.5444 | 0.5700 | 8 |  | magentic |
| 1 | va_cohort | honest | 0.5682 | 0.6656 | 8 |  | autogen |
| 1 | va_cohort | cartel | 0.5519 | 0.6667 | 8 |  | autogen |
| 2 | dense_icomp | honest | 0.4864 | 0.6267 | 8 |  | magentic |
| 2 | dense_icomp | cartel | 0.4882 | 0.6089 | 8 |  | magentic |
| 3 | dense_idemo | honest | 0.4649 | 0.6189 | 8 |  | magentic |
| 3 | dense_idemo | cartel | 0.4583 | 0.5711 | 8 |  | magentic |
| 4 | embed | honest | 0.4603 | 0.6011 | 8 |  | magentic |
| 4 | embed | cartel | 0.4485 | 0.4978 | 8 |  | magentic |
| 5 | sota_icomp | honest | 0.4318 | 0.5533 | 8 |  | magentic |
| 5 | sota_icomp | cartel | 0.4279 | 0.5278 | 8 |  | magentic |
| 6 | sota_idemo | honest | 0.4231 | 0.5211 | 8 |  | magentic |
| 6 | sota_idemo | cartel | 0.4164 | 0.5233 | 8 |  | magentic |
| 7 | sota | honest | 0.4028 | 0.5122 | 8 |  | magentic |
| 7 | sota | cartel | 0.4047 | 0.4856 | 8 |  | magentic |
| 8 | tfidf | honest | 0.3789 | 0.3789 | 10 |  | all 10 tied |
| 8 | tfidf | cartel | 0.3789 | 0.3789 | 10 |  | all 10 tied |
**Current completeness (F).** Every slot is filled and none is starred. crewai and adk are missing from every non-TF-IDF
bar (all their 10^5 shortlist rows were quarantined), and nothing on the figure says so (D1).

## 5.4 Figure G: `G_shortlist_lift_1e5` (gain over TF-IDF at n = 100,000, paired within framework)

**Question.** How much does each shortlist source improve a framework over the pre-registered hashed TF-IDF shortlist? The
comparison is made framework by framework, honest and under the cartel.

**Code.** `fig_G(d, n=100000)` (`shortlist_condensed.py:105-128`). This is the only figure computed from the
per-framework rows `d` rather than from the summary `s`.
1. `fw` = the framework rows at n = 10^5 (`:106`). `base` = TF-IDF means indexed by (regime, framework) (`:107`).
2. For each (regime, shortlist ≠ tfidf): `diff = mean[shortlist] − base` aligned on (regime, framework), with `.dropna()`
   (`:110`). **"Paired within framework"** means each framework is compared with its own TF-IDF number in the same regime.
   Only frameworks present under both shortlists enter. Currently that is the same 8 for every bar, because TF-IDF has all
   10 and the others have the 8.
3. `lift = diff.mean()` and `half = 1.96 · sd(diff, ddof=1) / √k`, with k = number of paired frameworks (`:111`). **The
   whisker is a normal-approximation 95 % interval across frameworks, not a t-interval.** The module docstring (`:7-8`)
   says "95% t-interval", which is wrong. With k = 8, t₀.₉₇₅,₇ = 2.365, so the whiskers are 17 % too short (table below).
   The spread is across frameworks. Seed-to-seed noise enters only through each framework's 3-seed mean.
4. `star` = any rerun_outstanding among the shortlist's rows (`:112`). TF-IDF's own star is not consulted.
5. **Order**: by honest lift, descending (`:113`). Bars are drawn like F without dots, plus an error bar and a zero line
   (`:115-123`). **The y range is automatic**; there is no `finish()` call, so no 0.2 floor. An empty regime would draw `*` at
   0.002 (`:118`).

**Erratum 25, and why G is a gain over a floor.**
- At 10^5 the pre-registered adapter ranks agents by hashed TF-IDF of their self-descriptions and takes the top 10 with a
  stable sort, lowest ids first.
- Specialist descriptions are memoized per prompt, and at 10^5 the top 10 is **exactly one prompt signature**: ten clones
  (`CHANGES_AND_ERRATA.md:241-258`). The framework's choice among clones cannot matter.
- So every framework scores the same 0.378889 in every seed and regime. I checked this on `live_n100k`: `success.nunique()
  == 1` per (β, liar_select, seed), 0.3467 / 0.3667 / 0.4233.
- Consequently `base` is a constant, and `diff` is just the shortlist's per-framework mean minus 0.378889. The pairing
  removes nothing; its variance is the between-framework spread of the new shortlist.
- G is therefore "how far above the clone floor each shortlist lifts a framework". It is **not** a gain over a working
  TF-IDF retriever.
- For contrast: the de-duplicated TF-IDF variant (`*_dd` grids; `dedup` without `retrieval` is dropped by `source()`, so it
  is never drawn) gives 0.363 honest / 0.344 cartel at 10^5 (erratum 25). That is *below* the clone floor.

| x slot | shortlist | regime | lift (bar) | ± half (whisker, 1.96·sd/√k) | t-based half (t₀.₉₇₅,₇) | #fw paired | star |
|---|---|---|---|---|---|---|---|
| 0 | declared | honest | +0.2414 | 0.0096 | 0.0115 | 8 |  |
| 0 | declared | cartel | +0.1656 | 0.0105 | 0.0127 | 8 |  |
| 1 | va_cohort | honest | +0.1893 | 0.0501 | 0.0604 | 8 |  |
| 1 | va_cohort | cartel | +0.1731 | 0.0487 | 0.0588 | 8 |  |
| 2 | dense_icomp | honest | +0.1075 | 0.0471 | 0.0568 | 8 |  |
| 2 | dense_icomp | cartel | +0.1093 | 0.0387 | 0.0466 | 8 |  |
| 3 | dense_idemo | honest | +0.0860 | 0.0440 | 0.0531 | 8 |  |
| 3 | dense_idemo | cartel | +0.0794 | 0.0332 | 0.0400 | 8 |  |
| 4 | embed | honest | +0.0814 | 0.0467 | 0.0563 | 8 |  |
| 4 | embed | cartel | +0.0696 | 0.0283 | 0.0341 | 8 |  |
| 5 | sota_icomp | honest | +0.0529 | 0.0352 | 0.0425 | 8 |  |
| 5 | sota_icomp | cartel | +0.0490 | 0.0291 | 0.0351 | 8 |  |
| 6 | sota_idemo | honest | +0.0442 | 0.0337 | 0.0406 | 8 |  |
| 6 | sota_idemo | cartel | +0.0375 | 0.0365 | 0.0441 | 8 |  |
| 7 | sota | honest | +0.0239 | 0.0462 | 0.0557 | 8 |  |
| 7 | sota | cartel | +0.0258 | 0.0378 | 0.0456 | 8 |  |
(Lifts and normal half-widths reproduced to 1e-6 from the raw rows. The "t-based" column is what the docstring promises:
the normal half-width × 2.365/1.96.)

**Reading notes.**
- The declared top-k whisker is very narrow (±0.010) because every framework does about the same once the top 10 by
  declared claim is handed over. The 10^5 declared-shortlist mean, 0.620 honest, is essentially the 0.619 honest
  score that erratum 28 quotes for declared argmax (`CHANGES_AND_ERRATA.md:377`).
- The plain rerank bar (+0.024 honest, +0.026 cartel) has an interval that crosses 0 under either the normal or the t
  rule.

**Current completeness (G).** All 8 × 2 bars are present, none starred; each difference is over the same 8 frameworks.

## 5.5 Figure H: `H_routereval_shortlists` (RouterEval, strong-to-weak pools m = 10/100/1000 + leaderboard 5000)

**Question.** Is the live-backend shortlist ranking reproduced on real LLM pools (RouterEval, MMLU)? Here "agents" are real
LLMs with real per-question correctness, and self-descriptions are rendered from the declaration vector.

**Code.** The same `fig_E` function, called as `fig_E(s, ref, "H_routereval_shortlists", title, MAIN, "pool m")`
(`shortlist_condensed.py:134-135`), on `summarise(load("routereval"))`.
- **Sources**: `MAIN`, the same nine slots as E (TF-IDF, MiniLM, dense I-comp, dense I-demo, rerank, rerank I-comp,
  rerank I-demo, declared top-k, VA cohort). Slot width w = 0.86/18 ≈ 0.048. A slot with no rows is an empty `*` at the
  baseline.
- **x axis**: groups at m = 10, 100, 1,000, 5,000 ("pool m"). **Pool m** is the backend's `n`: the number of LLMs
  (agents) in the RouterEval pool the task stream is routed over.
- No dots, y range 0.2–0.95, and the reference lines are the β = 0 values.
- **Legend** (PNG): oracle, MIDIAN-VA, declared top-k, VA cohort, TF-IDF, MiniLM, ranked by each shortlist's first honest
  bar (m = 10: 0.668, 0.628, 0.619, 0.608).

**Why 5,000 is dist "all".**
- The m ≤ 1,000 pools come from `routereval_mmlu` (`grid.yaml:316-330`), whose `dist` axis is a *pool configuration*:
  `strong_to_weak`, `all_strong`, `all_weak`. `load` keeps the mixed `strong_to_weak` pool (`shortlist_condensed.py:33`).
- The 5,000 point is the separate leaderboard backend (`routereval_mmlu5k`, `grid.yaml:359-371`,
  `dataset: leaderboard_mmlu`). It has a single pool, "ALL 5,000 leaderboard LLMs", and so a single `dist: [all]`.
  `load` admits it by `n == 5000`.
- Its model names are synthetic, so descriptions are rendered declaration vectors (`grid.yaml:528-534`).

**Provenance.**
- Grids: `fw_routereval_small` (m = 10, 100), `fw_routereval_1k`, `fw_routereval_5k` (TF-IDF), each with `_em` (MiniLM) and
  `_va` (VA cohort) mirrors; `re_sl_declared_{small,1k,5k}` (declared top-k) and `re_sl_embed_{small,1k,5k}` (Qwen3 dense
  I-comp / I-demo, rerank stock / I-comp / I-demo) (`grid.yaml:1088-1097`). The `re_sl_*` cells are
  `dist: [strong_to_weak]` (the 5k grid inherits `all`), `liar_select: [low_skill_first]`, β ∈ {0, 0.5}. Their β = 0 rows
  have only `low_skill_first`, so they pass the liar-free rule (`nunique == 1`).
- `source()` classifies the `re_sl_*` rows as `declared`, `dense_icomp/idemo` and `sota/_icomp/_idemo` with no special
  case.
- In the figure (01:52): `re_sl_declared_small` only. The other `re_sl_declared_*` rows landed after 01:52 (§5.1.6). The
  `re_sl_embed_*` units are queued behind the cache pre-warm (`scripts/embed_routereval.py`) and have no rows.
- Magentic-One is excluded by design, so at most 9 frameworks; crewai and adk are quarantined in `_small` and in every
  `_em` and `_va` grid, but present in `re_sl_declared_*`.
- Seeds: 1–5 at m ≤ 1,000 and 1–3 at 5,000. The declared cartel bars at m = 10 / 100 have 3 seeds per framework so far
  (camel 1 at m = 100).
- Regimes: honest = β 0 (random copy where both exist) and cartel = β 0.5 low_skill_first.
- Reference lines come from `routereval_mmlu` / `routereval_mmlu5k` (§5.1.5). Note the oracle at m = 100 (0.792) is lower
  than at m = 10 or 1,000.

| x slot | shortlist | regime | mean | best | #fw | star (csv) | frameworks:seeds |
|---|---|---|---|---|---|---|---|
| pool m = 10 | tfidf | honest (solid) | 0.6195 | 0.7102 | 7 |  | autogen:5 camel:4 langgraph:5 llama:5 maf:5 openai:5 smol:5 |
| pool m = 10 | tfidf | cartel (hatched) | 0.5792 | 0.5970 | 6 |  | autogen:5 langgraph:5 llama:5 maf:5 openai:5 smol:5 |
| pool m = 10 | embed | honest (solid) | 0.6083 | 0.7104 | 7 |  | autogen:5 camel:5 langgraph:5 llama:5 maf:5 openai:5 smol:5 |
| pool m = 10 | embed | cartel (hatched) | 0.5579 | 0.5914 | 7 |  | autogen:5 camel:1 langgraph:5 llama:5 maf:5 openai:5 smol:5 |
| pool m = 10 | dense_icomp | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 10 | dense_icomp | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 10 | dense_idemo | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 10 | dense_idemo | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 10 | sota | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 10 | sota | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 10 | sota_icomp | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 10 | sota_icomp | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 10 | sota_idemo | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 10 | sota_idemo | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 10 | declared | honest (solid) | 0.6680 | 0.7368 | 9 |  | adk:5 autogen:5 camel:5 crewai:5 langgraph:5 llama:5 maf:5 openai:5 smol:5 |
| pool m = 10 | declared | cartel (hatched) | 0.5984 | 0.6113 | 9 |  | adk:3 autogen:3 camel:3 crewai:3 langgraph:3 llama:3 maf:3 openai:3 smol:3 |
| pool m = 10 | va_cohort | honest (solid) | 0.6277 | 0.6776 | 7 |  | autogen:5 camel:5 langgraph:5 llama:5 maf:5 openai:5 smol:5 |
| pool m = 10 | va_cohort | cartel (hatched) | 0.5534 | 0.5900 | 6 |  | autogen:5 langgraph:5 llama:5 maf:5 openai:5 smol:5 |
| pool m = 100 | tfidf | honest (solid) | 0.5819 | 0.6130 | 6 |  | autogen:5 langgraph:5 llama:5 maf:5 openai:5 smol:5 |
| pool m = 100 | tfidf | cartel (hatched) | 0.5674 | 0.5882 | 5 |  | autogen:5 langgraph:5 maf:5 openai:5 smol:5 |
| pool m = 100 | embed | honest (solid) | 0.5394 | 0.6208 | 6 |  | autogen:5 langgraph:5 llama:5 maf:5 openai:5 smol:5 |
| pool m = 100 | embed | cartel (hatched) | 0.5064 | 0.5404 | 6 |  | autogen:5 langgraph:5 llama:5 maf:5 openai:5 smol:5 |
| pool m = 100 | dense_icomp | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 100 | dense_icomp | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 100 | dense_idemo | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 100 | dense_idemo | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 100 | sota | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 100 | sota | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 100 | sota_icomp | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 100 | sota_icomp | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 100 | sota_idemo | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 100 | sota_idemo | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 100 | declared | honest (solid) | 0.7514 | 0.7757 | 9 |  | adk:5 autogen:5 camel:3 crewai:5 langgraph:5 llama:5 maf:5 openai:5 smol:5 |
| pool m = 100 | declared | cartel (hatched) | 0.5361 | 0.5610 | 9 |  | adk:3 autogen:3 camel:1 crewai:3 langgraph:3 llama:3 maf:3 openai:3 smol:3 |
| pool m = 100 | va_cohort | honest (solid) | 0.6259 | 0.6560 | 6 |  | autogen:5 langgraph:5 llama:5 maf:5 openai:5 smol:5 |
| pool m = 100 | va_cohort | cartel (hatched) | 0.5545 | 0.5650 | 6 |  | autogen:5 langgraph:5 llama:5 maf:5 openai:5 smol:5 |
| pool m = 1,000 | tfidf | honest (solid) | 0.5801 | 0.5970 | 9 |  | adk:5 autogen:5 camel:4 crewai:5 langgraph:5 llama:5 maf:5 openai:5 smol:5 |
| pool m = 1,000 | tfidf | cartel (hatched) | 0.5400 | 0.5830 | 9 |  | adk:5 autogen:5 camel:4 crewai:5 langgraph:5 llama:5 maf:5 openai:5 smol:5 |
| pool m = 1,000 | embed | honest (solid) | 0.5425 | 0.5890 | 7 |  | autogen:5 camel:5 langgraph:5 llama:5 maf:5 openai:5 smol:5 |
| pool m = 1,000 | embed | cartel (hatched) | 0.5320 | 0.5650 | 7 |  | autogen:5 camel:1 langgraph:5 llama:5 maf:5 openai:5 smol:5 |
| pool m = 1,000 | dense_icomp | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 1,000 | dense_icomp | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 1,000 | dense_idemo | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 1,000 | dense_idemo | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 1,000 | sota | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 1,000 | sota | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 1,000 | sota_icomp | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 1,000 | sota_icomp | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 1,000 | sota_idemo | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 1,000 | sota_idemo | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 1,000 | declared | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 1,000 | declared | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 1,000 | va_cohort | honest (solid) | 0.6548 | 0.6892 | 7 |  | autogen:5 camel:5 langgraph:5 llama:5 maf:5 openai:5 smol:5 |
| pool m = 1,000 | va_cohort | cartel (hatched) | 0.5759 | 0.6104 | 7 |  | autogen:5 camel:1 langgraph:5 llama:5 maf:5 openai:5 smol:5 |
| pool m = 5,000 | tfidf | honest (solid) | 0.5922 | 0.6156 | 9 |  | adk:3 autogen:3 camel:3 crewai:3 langgraph:3 llama:3 maf:3 openai:3 smol:3 |
| pool m = 5,000 | tfidf | cartel (hatched) | 0.5330 | 0.5478 | 9 |  | adk:3 autogen:3 camel:3 crewai:3 langgraph:3 llama:3 maf:3 openai:3 smol:3 |
| pool m = 5,000 | embed | honest (solid) | 0.4813 | 0.5467 | 7 |  | autogen:3 camel:3 langgraph:3 llama:3 maf:3 openai:3 smol:3 |
| pool m = 5,000 | embed | cartel (hatched) | 0.4741 | 0.5289 | 7 |  | autogen:3 camel:3 langgraph:3 llama:3 maf:3 openai:3 smol:3 |
| pool m = 5,000 | dense_icomp | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 5,000 | dense_icomp | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 5,000 | dense_idemo | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 5,000 | dense_idemo | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 5,000 | sota | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 5,000 | sota | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 5,000 | sota_icomp | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 5,000 | sota_icomp | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 5,000 | sota_idemo | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 5,000 | sota_idemo | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 5,000 | declared | honest (solid) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 5,000 | declared | cartel (hatched) | — | — | — | EMPTY slot, * at 0.205 | — |
| pool m = 5,000 | va_cohort | honest (solid) | 0.6389 | 0.6822 | 7 |  | autogen:3 camel:3 langgraph:3 llama:3 maf:3 openai:3 smol:3 |
| pool m = 5,000 | va_cohort | cartel (hatched) | 0.5670 | 0.6089 | 7 |  | autogen:3 camel:3 langgraph:3 llama:3 maf:3 openai:3 smol:3 |

**Framework sets are uneven here, and nothing is starred:**
- m = 10: 7 frameworks honest for TF-IDF / MiniLM / VA. Cartel TF-IDF and VA have 6 (camel missing); cartel MiniLM has
  camel with **1 seed**. Honest TF-IDF camel has 4 seeds. Declared has 9.
- m = 100: 6 frameworks for TF-IDF / MiniLM / VA (no camel). Cartel TF-IDF has 5 (no llama either). Declared has 9.
- m = 1,000: TF-IDF has **9** (incl. crewai, adk; camel 4 seeds). MiniLM and VA have 7, and their cartel camel has
  **1 seed**.
- m = 5,000: TF-IDF has 9; MiniLM and VA have 7.

So the TF-IDF bars at m = 1,000 / 5,000 and the declared bars at m = 10 / 100 average over crewai and adk, and the MiniLM
and VA bars do not. These gaps are not on the quarantine list, except crewai/adk, which are listed but absent from the CSV,
so no star is drawn (D1, D5).

**At m = 10 the shortlist is the whole pool.** With k = 10 (`_common.py:116`) and a pool of 10 unique descriptions
(erratum 25: RouterEval descriptions are all unique), TF-IDF, MiniLM and declared top-k all return all 10 agents (the deduplicated ones fewer when
liars' rendered texts coincide, §3.9 #1); only the order differs (`_common.py:328-331`). The VA cohort is MIDIAN-VA(r = 10)'s pick plus its leaf cohort. I did not verify
whether that is the whole 10-agent pool. Differences at m = 10 therefore reflect ordering and the supervisor, not
retrieval quality.

**Current completeness (H).** Of the 9 × 2 × 4 = 72 slots, 28 are filled: TF-IDF, MiniLM and VA cohort at every m, and
declared top-k at m = 10 and 100. The other 44 are empty `*`. None is starred.

## 5.6 Discrepancies, open questions, possible issues

Severity: **[H]** can change a reader's conclusion; **[M]** is misleading labelling or a comparability caveat; **[L]** is
cosmetic or documentation only.

- **D1 [H]: Wholly-quarantined frameworks vanish from E–H without a star.**
  - `collect` gives a framework whose rows were all quarantined an empty starred slot only inside the per-condition figure
    (`shortlist_figs.py:87-90`, `:135`). No CSV row is written for it (`:140`).
  - `summarise` therefore sees k = 8 (or 7) instead of 10 (or 9), and `star` stays False (`shortlist_condensed.py:40-41`).
  - Result: crewai and adk are silently missing from most non-TF-IDF bars at every n in E and F, from all of G, and from
    MiniLM/VA (and TF-IDF at m ≤ 100) in H. ADK is also missing from the TF-IDF bars at 10^2–10^4; at 10^2 / 10^3 honest
    because its units were refused over the "Tool … not found" error (§5.1.4).
  - By erratum 28 these two were biased *upward* before quarantine, so their honest reruns could move means either way.
  - E's title promises "* = not in yet". For these bars that promise is not kept.
- **D2 [H]: Means over different framework sets are compared side by side.**
  - TF-IDF bars average 9–10 frameworks while most others average 8 (live 10^5: 10 vs 8; RouterEval 1k/5k: 9 vs 7).
    RouterEval declared top-k averages 9 at m = 10 / 100, against 6–7 for MiniLM and VA.
  - Honest vs cartel at 10^4 differs by Magentic-One, excluded from `fw_live_n10k_cartel` by design and the best framework
    at 10^4 honest. Without it, honest MiniLM is 0.4173 = the cartel value. `fw_live_n10k_cartel_magentic` is queued to
    close this gap.
  - 10^3 rerank honest has 6 frameworks (no magentic or camel) vs 8 for cartel; 10^3 MiniLM honest has 9 (with crewai)
    vs 8 for cartel.
  - RouterEval m = 100 has 6 frameworks, with 5 for cartel TF-IDF.
  - G is immune by construction (paired `dropna`), apart from reducing to the common 8.
- **D4 [M]: G's whisker is a normal interval, and the docstring says t.** `1.96·sd/√k` (`shortlist_condensed.py:111`) vs
  "95% t-interval" (`:7-8`). With k = 8 the whiskers are 17 % narrower than a t-interval. They are also a
  between-framework spread, not seed uncertainty.
- **D5 [M]: Unequal seeds are averaged with equal weight.** `summarise` takes an unweighted mean of per-framework seed means.
  Examples:
  - 10^3 rerank / BM25 / dense honest: 3–10 seeds per framework (`fw_live_n1000_sota`).
  - n = 100 TF-IDF cartel: crewai 6, langgraph 9 seeds.
  - RouterEval cartel MiniLM/VA: camel has **1 seed** at m = 10 and m = 1,000; TF-IDF camel has 4.
  - RouterEval declared cartel at m = 100: camel 1 seed, the others 3.

  Only the erratum-28 subset gets a star. Incomplete runs that are not on the quarantine list (camel, llama and magentic
  gaps at RouterEval small and at 10^3 sota) carry no mark at all.
- **D6 [M]: The figures are only as fresh as the last `shortlist_figs.py` run.** E–H and the per-framework CSVs are from
  01:52. Since then `re_sl_declared_1k` / `_5k` rows have landed that would add H's declared bars at m = 1,000 and 5,000
  (§5.1.6). The condensed script never checks freshness.
- **D7 [M]: The reference lines are honest-only and not seed-matched.**
  - `lines()` draws only the (n, beta0) oracle and MIDIAN-VA (`shortlist_condensed.py:47-50`). Hatched bars are compared
    with the honest MIDIAN-VA line: 0.8356 vs its own cartel 0.8278 at 10^5, 0.7130 vs 0.6942 at RouterEval 1k. The oracle
    is regime-invariant in every plotted cell, so it is unaffected.
  - Each line comes from one grid chosen by most seeds, first listed on a tie (`shortlist_figs.py:101-109`). The comment
    at `:54` says every grid is "pooled and matched on (dist, regime, seed)". The code does neither.
  - In practice the seed set matches the framework grids (1–10, 1–3, 1–5) and the cells are `mirror_of` twins. At 10^4 the
    line comes from `learned_n10k`, not the frameworks' `live_n10k_v2`, but the values are identical.
- **D8 [M]: E's empty slots at n ≤ 10^4 are queued, not landed.** Dense I-comp / I-demo, rerank I-comp / I-demo and
  declared at 10^2–10^4 come from `fw_live_n{100,1000}_backfill`, `fw_live_n{100,1000}_lowskill_backfill`,
  `fw_live_n10k_backfill` and `fw_live_n10k_cartel_backfill` (`grid.yaml:1014-1023`, anchor `*fw10_backfill`), which
  are in the focus pack plan with no rows yet. They run seeds 1–3, so at 10^2 / 10^3 these bars will have 3 seeds per
  framework against 10 for the other shortlists, and all ten frameworks, including Magentic-One in the 10^4 cartel.
- **D9 [M]: The `*` mark means two things.** In E and H a baseline `*` means an empty slot and an above-bar `*` means the
  erratum-28 rerun is outstanding. E's title explains only the first. The erratum star also persists after the rerun seeds
  land: it is keyed on `quarantine_units.tsv` until `logs/DONE_stage2` exists (`fw_variant_numbers.py:38-42`). For example,
  n = 100 TF-IDF honest is now complete at 10/10 seeds and is still starred.
- **D10 [M]: G is a lift over a degenerate floor.** At 10^5 TF-IDF = ten clones, 0.378889 for every framework in every seed
  and regime (erratum 25; verified on the raw rows). So the "paired" difference has no pairing benefit. The docstring says
  this (`:8-9`), but the figure title only says "gain over the pre-registered TF-IDF shortlist". The de-duplicated TF-IDF
  (0.363 / 0.344) is below this floor and is never drawn.
- **D11 [M]: Only the TF-IDF bar is un-deduplicated.** `source()` ignores `dedup` when `retrieval` is set
  (`shortlist_figs.py:49-53`), and every MiniLM / Qwen / rerank / declared row carries `dedup: true`. The shortlists
  therefore differ in two ways at once: the retriever, and clones-allowed vs one-per-text. At 10^5 this conflates "better
  retriever" with "no clones".
- **D12 [L]: At RouterEval m = 10, k = 10 = m.** TF-IDF and MiniLM hand the supervisor the same 10 agents, in different
  order. Whether the VA cohort is also the full pool at m = 10 is unverified (open question).
- **D13 [L]: Legend ranking mixes n.** In E and H each shortlist's legend entry is ranked by the height of its first drawn
  honest bar, i.e. at the smallest n where it exists (`extra_figs.py:16-29` on the handle labelled at `:78`).
- **D14 [L]: The bars start at y = 0.2 in E, F and H** (`finish`, `:97`). Bar heights exaggerate ratios; G has an automatic
  range with a zero line.
- **D15 [L]: Documentation drift in `shortlist_condensed.py`.**
  - The header says it writes `{E_shortlists_by_n,F_shortlists_1e5}` (`:2`), but it also writes G and H.
  - `pair`'s docstring says "a black dot on each = the best framework" (`:54`), but E and H call it with `dots=False`.
- **D16 [L]: The `RTE_DATA` default is a trap.** Without `RTE_DATA` both scripts read `/scratch/rte/results`
  (`shortlist_figs.py:24`, `fw_variant_numbers.py:11`), which exists on this login node but holds no grids. The run would
  produce empty CSVs and figures silently.
- **Open question: is the per-framework "best" dot comparable across bars?** It is the max over frameworks of a 3-seed mean
  (10^5), so it is upward-biased by selection. More frameworks means more selection: for TF-IDF all 10 tie, so no issue
  there.

---

## 6. Figures C and D: routing work and energy per query

Scope: `figures/condensed_sample/C_routing_work_vs_n.{png,pdf,csv}` and `D_energy_per_query.{png,pdf,csv}`, both written
at **2026-09-23 12:16** by `python scripts/efficiency_figs.py` (120 lines). That script imports `scripts/energy.py`
(130 lines) for the energy model and the framework numbers. It also takes `ARMS`, `OUT` and `save` from
`scripts/condensed_figs.py` (`efficiency_figs.py:18`), `label` from `scripts/seed_tables.py` (`:17`) and the legend rule
from `scripts/extra_figs.py` (`:16`). Importing `condensed_figs` also applies its rcParams (`condensed_figs.py:27-28`), so C
and D share A and B's fonts. Line citations in this part are to `efficiency_figs.py` unless another file is named.

**How this part was checked (read-only).**
- The C CSV was compared with `cost_by_n.csv`, and the cache itself was re-derived from `bernoulli_scale_v5/rows.csv`: the
  same medians for all 51 (arm, n, b) rows.
- Every D row was re-derived from the cache and the energy constants (`build_J`, `marginal_J` to 1e-6 J).
- The `va_build` inputs were re-read from the five grids, with row counts per source, and `va_build` itself was called on
  the current cache. It returns the nine values of D's CSV, all tagged `ledger`.
- `energy.table()` was re-run with its row loader swapped for a read-only one (rows.csv plus rows.d, deduplicated on
  `rid`). The stock loader, `rte.analyze.load`, runs `consolidate`, which writes rows.csv (`rte/analyze.py:57`). The band
  edges came out the same: 20.600235 and 219.754123 J.

---

### 6.1 Shared inputs

#### 6.1.1 The ledger cache `cost_by_n.csv` (`costs`, `:29-36`)
- **Source.** `$RTE_DATA/results/bernoulli_scale_v5/rows.csv`, read in chunks of 500,000 rows. Only the columns in
  `COLS` are kept (`:22`). The grid has no `rows.d` files, so rows.csv is complete.
- **Filter** (`:31-32`). `beta == 0`, and either `b == 3` or `method == "midian_va"`. So the SHOW arms are kept at b = 3, and
  MIDIAN-VA at every b the grid ran: b = 3 at every n, plus b = 1 at 10^6 and 10^7 (the grid's only b = 1 cells, §1.6.3).
  The b = 1 rows are there for D's build ledger. There is no filter on `dist`, `liar_select` or `declared_source`. It
  needs none: every kept row is `dist = specialist`, `liar_select = random`, `declared_source = programmatic` (checked).
- **Labels.** `seed_tables.label(method, params)` (`:34`), which applies `ALIAS`, so `flat_probe_argmax{"online":true}`
  becomes `flat_probe_argmax_online`. Only the seven labels in `SHOW` are kept (`:23-25`, `:35`).
- **Aggregation.** The median over rows per (label, n, b) of `messages_per_task`, `comparisons_per_task`, `build_probes`
  and `build_messages` (`:35`). There is one row per seed. The file has 51 rows: 7 arms × 7 n at b = 3, plus MIDIAN-VA
  b = 1 at 10^6 (16,799,988.5 build probes) and 10^7 (167,999,523.5). It has a `b` column.

  | n | 10, 100, 1,000, 10^4 | 10^5 | 10^6 | 10^7 |
  |---|---|---|---|---|
  | seeds per (arm, n, b) | 1,000 | 500 | 200 | 100 |

  The seed counts are the same for all seven arms, and for MIDIAN-VA b = 1. The per-query counts are identical across seeds for every arm, so the
  median is the only value. Only MIDIAN-VA's `build_probes` varies (e.g. 49,325–49,586 at n = 10^3; 495,989,791–496,008,863
  at 10^7).
- **Caching.** When the file exists it is read back and the rows are not touched (`:30`). The docstring says to delete it
  to re-read (`:10`). The current file was written at 12:16, in the same run as the figures. It covers n = 10 … 10^7; C
  draws n ≥ 100 at b = 3 (`:73`, `:75`).
- **Arm labels in the figures** (`SHOW`, `:23-25`; colours `COL`, `:26`):

  | cache label | figure label | colour | also in A/B? |
  |---|---|---|---|
  | `midian_va` | MIDIAN-VA | `#2ecc71` (A/B's) | yes |
  | `midian` | MIDIAN | `#c0392b` | yes |
  | `flat_probe_argmax_online` | flat probe argmax (online) | `#3498db` | yes |
  | `warm_start_bandit` | warm-start bandit | `#9467bd` (A/B's "best bandit" purple) | as a pool candidate |
  | `ucb_per_family` | UCB bandit | `#c5b0d5` | as a pool candidate |
  | `flat_nsw_router` | flat NSW index router | `#ff7f0e` (A/B's "best learned" orange) | as a pool candidate |
  | `declared_argmax` | declared argmax | `#5d6d7e` | yes |

  Random, the oracle, the cross-fitted pooled arms and the frameworks are not in C. D uses only the `midian_va` rows.

#### 6.1.2 What "routing work" counts
C's y value is `messages_per_task + comparisons_per_task` at b = 3 (`:73`). These are the ledger's per-task counts (§1.1.7). Hops
and reports are not counted, and neither is the build. Per arm (§2.11), with depth = ⌈log₁₀ n⌉ at r = 10:
- MIDIAN-VA: 2 + depth messages and 1 + 10·depth comparisons, so work = 3 + 11·depth.
- MIDIAN: 3·depth messages and 20·depth comparisons, so work = 23·depth.
- Flat probe argmax, the bandits and declared argmax: 0 messages and n comparisons.
- Flat-NSW router: 0 messages and ef = 50 comparisons. Its ⌈log₂ n⌉ hops are not in C.

The cache confirms each formula at every n. MIDIAN's and MIDIAN-VA's work is therefore linear in tree depth, which is
log₁₀ n rounded up: it grows like log n, not like a power of n. MIDIAN-VA's b = 1 rows at 10^6 / 10^7 have the same
per-query counts as its b = 3 rows. Messages and comparisons are added one for one, although they cost very different
amounts (§6.3.2: 1e-3 J against 1e-8 J).

---

### 6.2 Figure C: `C_routing_work_vs_n`

**Question answered.** How does the communication and computation a router spends on each query grow with the population
n, for MIDIAN and MIDIAN-VA against flat arms that scan every agent?

**What is drawn** (`draw_I`, `:72-83`):
- One line per arm in `SHOW` order. Circle markers (ms 3), lw 1.6, at n = 10², 10³ … 10⁷, b = 3 rows only (`:73`, `:75`, `:78`). There are no
  error bars: the counts do not vary across seeds.
- Both axes are log (`:79`). x = "population n (agents)", y = "messages + comparisons per query". The major grid is lw
  0.3, α 0.4 (`:80`).
- **Four lines coincide.** Flat probe argmax, warm-start bandit, UCB bandit and declared argmax are all exactly n.
  Declared argmax is drawn last, so only its grey shows. The blue, purple and lilac lines are underneath it.
- **Legend** (`:82`): two columns, top left, font 6. Each entry is `name  (growth)` (`:78`). `growth(s)` (`:67-69`) turns
  the fitted slope s into words: "constant" if |s| < 0.02, "∝ log n" if s < 0.3, otherwise "∝ n^s" with two decimals.
  The current entries are: flat NSW index router (constant), MIDIAN-VA (∝ log n), MIDIAN (∝ log n), and "∝ n^1.00" for
  the four flat arms.
  It goes through the `extra_figs` wrapper with `rank="asc"`: entries are sorted by the mean of the plotted y values,
  lowest first, and laid out row-major (`extra_figs.py:16-54`). The four tied arms keep `SHOW` order. The legend reads:
  flat NSW index router, MIDIAN-VA / MIDIAN, flat probe argmax (online) / warm-start bandit, UCB bandit / declared argmax.
- **Title** (`:81`): "C  routing work per query: MIDIAN grows like log n, flat methods like n (calibrated bernoulli, b = 3,
  exact ledger)".
- Figure size 7.2 × 2.8 in; PNG at 250 dpi, plus a PDF (`:113-116`, `condensed_figs.py:78-79`).

**The slope** (`slope`, `:62-64`). An ordinary least-squares line through (log₁₀ n, log₁₀ work) for n ≥ 10³ and work > 0,
via `np.polyfit(..., 1)`. That is five points, n = 10³ … 10⁷. n = 10² is drawn but not fitted. The slope is written to the
CSV but is only a classifier for the legend. MIDIAN's and MIDIAN-VA's work is 23·depth and 3 + 11·depth (§6.1.2), linear
in log n. A power law fitted to such a curve has a small slope (0.09) that falls as n grows, so `growth` labels it
"∝ log n". The thresholds 0.02 and 0.3 are fixed in the code. They separate the three classes here with wide margins
(|−2.6e-17|, 0.086–0.091, 1.000).

**Values** (`C_routing_work_vs_n.csv`: columns `arm, n, work, slope`; one row per drawn point, the slope repeated on
each row):

| n | MIDIAN-VA | MIDIAN | flat probe argmax, warm-start, UCB, declared argmax | flat NSW index router |
|---|---|---|---|---|
| 10² | 25 | 46 | 100 | 50 |
| 10³ | 36 | 69 | 1,000 | 50 |
| 10⁴ | 47 | 92 | 10,000 | 50 |
| 10⁵ | 58 | 115 | 100,000 | 50 |
| 10⁶ | 69 | 138 | 1,000,000 | 50 |
| 10⁷ | 80 | 161 | 10,000,000 | 50 |
| fitted slope (n ≥ 10³) | 0.0860 | 0.0912 | 1.0000 | −2.6e-17 |

**Reading.** At n = 10² the flat-NSW router (50) and MIDIAN-VA (25) are both below the flat arms (100). MIDIAN (46) is
below flat as well. MIDIAN-VA is under the flat-NSW router up to 10⁴ (47 vs 50) and over it from 10⁵ (58).

---

### 6.3 Figure D: `D_energy_per_query`

**Question answered.** Counting LLM calls, messages and comparisons in joules, after how many routed queries does
MIDIAN-VA's one-off probing build cost less than a framework that calls a supervisor LLM on every query? How does that
point move with n and b?

**What is drawn** (`draw_J`, `:86-110`):
- **x** = queries served T, 200 log-spaced points from 10² to 10⁹ (`:91`). **y** = energy per query in J, with the build
  amortised (`:106`). Both axes are log.
- **MIDIAN-VA lines.** Nine lines, one per (n, b) with n ∈ {10³, 10⁵, 10⁷} and b ∈ {1, 3, 5} (`:96-102`):
  `y(T) = build_J / T + marginal_J`.
  - The shade is the base green `#2ecc71` times (1.35 − s), with s = 0.55 / 0.8 / 1.0 for n = 10³ / 10⁵ / 10⁷. That is ×0.80,
    ×0.55 and ×0.35: darker means a larger n (`:96`, `:101`).
  - The line style gives b: dotted `:` for b = 1, solid for b = 3, dashed `--` for b = 5 (`:98`). lw 1.4.
  - As T grows each line falls with slope −1 and flattens at `marginal_J`. At T = 10⁹ only the n = 10³ lines have
    reached their floor, about 0.005 J.
- **The grey band** (`:93-94`) spans all T, from the cheapest to the costliest framework's per-query energy: 20.60 J
  (AutoGen) to 219.75 J (Magentic-One, 7B). `#bbbbbb`, α 0.5. Legend: "frameworks: 1-10.7 supervisor-call equivalents per
  query (21-220 J)". The call-equivalent range is the min and max of `sup_call_equiv` over the drawn frameworks (`:92`),
  printed as `.0f` and `.1f`: 1.000 (AutoGen) and 10.673 (Magentic-One). §6.3.3 says what a call-equivalent is.
- **The ×s** (`:103-104`): black, ms 3, drawn above the lines. Each is at (break-even T, framework J), where a MIDIAN-VA line
  crosses the lower edge (vs the cheapest framework) or the upper edge (vs the costliest). There are 18: 9 lines × 2 edges.
- **Legend** (`:109`): four columns above the axes, font 5.5, `rank="asc"`. The lines are ranked by the mean of their y
  values, which is set by build energy, so the order runs from n = 10³, b = 1 to n = 10⁷, b = 5. The band has no y data
  (`extra_figs._plotted` returns None for it), so it comes last. The legend reads, row by row: 10³ b1, 10³ b3, 10³ b5,
  10⁵ b1 / 10⁵ b3, 10⁵ b5, 10⁷ b1, 10⁷ b3 / 10⁷ b5, band.
- **Title** (`:108`): "D  energy per query (\*): MIDIAN-VA pays one probing build, then ~0.01 J / query; frameworks pay
  supervisor LLM calls every query". The `*` marks an estimate, as in `energy.py`'s docstring ("ESTIMATE (\*)", `energy.py:1`). It is
  not an erratum-28 star.

#### 6.3.1 MIDIAN-VA's build probes (`va_build`, `:42-59`)
- **Grids read** (`VA_GRIDS`, `:39`): `va_b_bernoulli_1e7`, `va_b_n1000`, `va_b_n100k`, `fw_live_n1000`, `live_n100k`. For
  each, every `rows.d/*.json` and the `rows.csv` are read (`:48-52`). rows.csv and rows.d hold the same rows in four of
  the five grids. Rows are deduplicated on `rid`; rows without a `rid` are all kept (`:54`). Every rows.csv here has a
  `rid` column.
- **Aggregation** (`:55`): `method == "midian_va"`, then the median of `build_probes` per (n, b). There is no filter on
  β, `dist`, `liar_select` or channel.
- **Bernoulli fill-in** (`:56`): for every (n, b) of MIDIAN-VA in the cache, the grid value is kept if there is one, else the
  cache's value (the `bernoulli_scale_v5` median) is used. So the live and `va_b` ledgers come first, then the bernoulli
  sweep. D uses the sweep for (10⁷, 1) and (10⁷, 3). The cache values at 10³ (49,440) and 10⁵ (4,959,820) are overridden
  by the live grids.
- **Ratio fallback** (`:57-58`): for each b, `ratio[b]` = the median over the n available at that b of
  `build_probes / (n · 16 · b)`. Any (n, b) with no value would become `ratio[b] · n · 16 · b` and be tagged
  `ratio r x nKb` in the CSV's `build_source` column. K = 16 is hard-coded. No D cell uses it now; the docstring says so
  (`:44-45`).

| n | b | build probes | ÷ n·K·b | CSV `build_source` | where the number comes from |
|---|---|---|---|---|---|
| 10³ | 1 | 16,805 | 1.0503 | ledger | `va_b_n1000`: 10 seeds × β {0, 0.5}, live specialist (20 rows) |
| 10³ | 3 | 49,427.5 | 1.0297 | ledger | `fw_live_n1000`: 10 seeds × β {0, 0.1, 0.25, 0.5} × 3 shapes (120 rows) |
| 10³ | 5 | 83,007.5 | 1.0376 | ledger | `va_b_n1000` (20 rows) |
| 10⁵ | 1 | 1,680,005 | 1.0500 | ledger | `va_b_n100k`: 3 seeds × β {0, 0.5} (6 rows, rows.d only) |
| 10⁵ | 3 | 4,959,662 | 1.0333 | ledger | `live_n100k`: 3 seeds, β {0, 0.25, 0.5} (18 rows) |
| 10⁵ | 5 | 8,319,642 | 1.0400 | ledger | `va_b_n100k`: 3 seeds, β = 0 only (3 rows; the cartel b = 5 rows had not landed) |
| 10⁷ | 1 | 167,999,523.5 | 1.0500 | ledger | the cache: `bernoulli_scale_v5`, 100 seeds, β = 0 |
| 10⁷ | 3 | 495,999,294.5 | 1.0333 | ledger | the cache: `bernoulli_scale_v5`, 100 seeds, β = 0 |
| 10⁷ | 5 | 831,998,807 | 1.0400 | ledger | `va_b_bernoulli_1e7`: 100 seeds × β {0, 0.5} (200 rows) |

Every build overspends n·K·b by 3–5 % (§2.3.4). "ledger" means a median of measured `build_probes`, from a live or a
bernoulli run: the column does not say which backend.

#### 6.3.2 The energy model (`scripts/energy.py`)
Everything is estimated from call counts, not measured on a power meter. The module docstring gives the model
(`energy.py:1-4`).
- **GPU-seconds per LLM call** (`call`, `energy.py:17`): `params_b · (A · prompt_tokens + B · gen_tokens)`. `params_b` comes
  from `configs/models.yaml:14-20` (0.5, 1.5, 2, 3, 7, 9 and 14 B; `energy.py:12`).
- **B = 5A** (`energy.py:8`). A decode token is taken as 5× a prefill token on H100 / vLLM (docstring). This ratio is
  assumed, not measured.
- **A is calibrated to one measurement** (`energy.py:7-8`). A 7B supervisor call has 1,900 prompt and 65 generated tokens
  (`SUP_TOK`). It is set to cost 1/34 GPU-s: 34 req/s is the throughput of a saturated 1-GPU 7B replica, measured on
  2026-09-03 from 4 samples of 32–36 req/s (docstring). So A = (1/34) / (7 · (1,900 + 5·65)) = 1.888e-6 and B = 9.442e-6.
- **A supervisor call** (`SUP`, `energy.py:19`) = 1/34 = 0.029412 GPU-s. At 700 W that is **20.588 J**.
- **A probe** is one call by the probed agent's own model, using that model's measured token means (`PROBE_TOK`,
  `energy.py:13-14`, "measured per-server lifetime means"). 7B and 14B use the 3B means as a proxy (294 prompt / 86
  generated tokens, the tool-enabled prompt). The expected cost per probe is the mix-weighted mean over the population's
  models (`MIX`, `energy.py:15-16`; `probe_cost`, `energy.py:18`). The mixes follow `draw_profiles` (`rte/backends/population.py:60-70`):

  | shape | model mix | GPU-s per probe | J at 700 W |
  |---|---|---|---|
  | specialist (used in D) | all 7 models, 1/7 each | 0.0057697 | **4.0388** |
  | heavy_tail | 90 % split over 0.5B / 1.5B, 10 % over 7B / 9B / 14B | 0.0017557 | 1.2290 |
  | bimodal | 80 % 0.5B, 20 % 7B | 0.0020319 | 1.4223 |

  Per model at 700 W: 0.5B 0.10 J, 1.5B 0.86 J, 2B 0.50 J, 3B 2.87 J, 7B 6.70 J, 9B 3.84 J, 14B 13.40 J.
- **Watts.** 700 W, the H100 TDP (`WATTS`, `energy.py:9`; `table(watts=700)`, `energy.py:28`). `draw_J` multiplies by 700 itself
  (`:90`). The 400 W "typical draw" entry is used only in the RESULTS_energy.md table (`energy.py:72`).
- **Message = 1e-3 J** (`J_MSG`, `energy.py:10`): one RPC handled in about 100 µs on a 10 W core (`energy.py:44`).
- **Comparison = 1e-8 J** (`J_CMP`, `energy.py:10`): one float compare.
- **Pessimistic messages.** `table()` also computes `*_pess` columns with 10 × J_MSG = 1e-2 J (`energy.py:45-48`). D
  does not use them. With them, every break-even in §6.3.4 moves by less than 0.31 %.
- **Excluded.** The routed task's own execution. `exec_gpu_s_per_task` is computed (`energy.py:38`) but left out of
  `per_task_gpu_s` (`energy.py:37`). It is the same for every method: 0.0058 GPU-s per task on specialist (`energy.py:68`).
  CPU-side routing work (tree descent, TF-IDF) is left out of the LLM term (`energy.py:67`). It enters only through the
  ledger's message and comparison counts.

**MIDIAN-VA in D** (`:95-100`):
- `build_J = build_probes · 4.0388 + build_messages · 1e-3`. The build messages are the b = 3 cache value at every b
  (1,010 / 101,110 / 10,111,110 at n = 10³ / 10⁵ / 10⁷), a rounding term (`:95`, `:100`). Build comparisons are not charged.
- `marginal_J = messages_per_task · 1e-3 + comparisons_per_task · 1e-8` from the cache (b = 3). That is 5 msgs + 31 cmp =
  0.00500031 J at 10³, 7 + 51 = 0.00700051 J at 10⁵ and 9 + 71 = 0.00900071 J at 10⁷. It is the same at every b, as the
  ledger's per-query counts are.
- No LLM term per query: MIDIAN-VA routes without an LLM call and without run-time probes (§2.3.6).

#### 6.3.3 The framework band (`energy.table()`, `energy.py:28-54`; filter at `:88`)
- **Rows** (`rows`, `energy.py:21-26`): grids `live_f1_n1000`, `variants_f1` and `fw_live_n1000` at n = 1,000, filtered to
  `declared_source == "self_described"`. The per-method **mean** of each ledger column is taken over every such row. β,
  `dist`, seed and quarantine are not filtered. Every `fw_` row comes from `fw_live_n1000`: 10 seeds × 3 shapes ×
  β {0, 0.1, 0.25, 0.5}, `liar_select = random`. That is 120 rows per framework (CAMEL 114, LlamaIndex 119, Magentic-One
  204 over its 7B and 14B arms). No erratum-28 quarantine is applied, so CrewAI and ADK, which E–H drop, are in the band.
- **Supervisor call-equivalents** (`energy.py:29, 35`) = the framework's mean `wall_clock_per_task` divided by
  AutoGen's (1.8987 s). AutoGen makes one supervisor call per task, so it is 1.0 by construction. The code comments call
  the latency a "measured median" (`energy.py:49-50`), but it is the mean over rows. This assumes latency is proportional
  to GPU work. It was measured on a shared fleet.
- **The 14B Magentic-One arm** would be scaled by 14/7 (`energy.py:36`). D excludes it: `~fw.index.str.contains("supervisor")`
  (`:88`) drops the `fw_magentic_one{"supervisor":"Qwen/Qwen2.5-14B-Instruct"}` row (8.93 call-equivalents, 367.87 J).
  The band is also restricted to `fw_*` rows, so `llm_supervisor` is not in it.
- **`per_task_J`** (`energy.py:37, 47`) = `(probes_per_task · probe_cost + call-equivalents · SUP) · 700 +
  messages_per_task · 1e-3 + comparisons_per_task · 1e-8`. For every framework, probes_per_task = 0, messages = 12
  (k + 2) and comparisons = 10 (k) per task (`_common.py:364-365`, k = 10).

| framework | mean latency, s | call-equivalents | `per_task_J` |
|---|---|---|---|
| AutoGen (band floor) | 1.899 | 1.000 | **20.600** |
| MAF | 2.309 | 1.216 | 25.051 |
| Google ADK | 2.327 | 1.226 | 25.250 |
| smolagents | 2.873 | 1.513 | 31.166 |
| LangGraph | 3.772 | 1.987 | 40.911 |
| OpenAI Agents | 3.995 | 2.104 | 43.327 |
| CrewAI | 6.015 | 3.168 | 65.230 |
| LlamaIndex | 8.978 | 4.728 | 97.361 |
| CAMEL Workforce | 9.668 | 5.092 | 104.843 |
| Magentic-One, 7B (band ceiling) | 20.265 | 10.673 | **219.754** |
| *Magentic-One, 14B (excluded)* | 16.962 | 8.934 | 367.870 |

Only the two edges are drawn. The per-framework values are not in D's CSV. They are in the `fw_J` column only as the
edges.

#### 6.3.4 Break-even points
A MIDIAN-VA line meets a framework level F where build/T + marginal = F, so (`:104`)

  **T\* = build_J / (F − marginal_J)**.

The framework build (n registry messages, 1 J at n = 10³) is not in the band, so it does not enter T\*. T\*/n is given
as queries per agent.

| n | b | build probes | build J | marginal J | T\* vs AutoGen (20.60 J) | per agent | T\* vs Magentic-One (219.75 J) | per agent |
|---|---|---|---|---|---|---|---|---|
| 10³ | 1 | 16,805 | 6.787e4 | 0.00500 | 3,296 | 3.30 | 309 | 0.31 |
| 10³ | 3 | 49,427.5 | 1.996e5 | 0.00500 | 9,693 | 9.69 | 908 | 0.91 |
| 10³ | 5 | 83,007.5 | 3.353e5 | 0.00500 | 16,278 | 16.28 | 1,526 | 1.53 |
| 10⁵ | 1 | 1,680,005 | 6.785e6 | 0.00700 | 329,492 | 3.29 | 30,878 | 0.31 |
| 10⁵ | 3 | 4,959,662 | 2.003e7 | 0.00700 | 972,709 | 9.73 | 91,156 | 0.91 |
| 10⁵ | 5 | 8,319,642 | 3.360e7 | 0.00700 | 1,631,678 | 16.32 | 152,910 | 1.53 |
| 10⁷ | 1 | 167,999,523.5 | 6.785e8 | 0.00900 | 32,952,258 | 3.30 | 3,087,794 | 0.31 |
| 10⁷ | 3 | 495,999,294.5 | 2.003e9 | 0.00900 | 97,286,799 | 9.73 | 9,116,267 | 0.91 |
| 10⁷ | 5 | 831,998,807 | 3.360e9 | 0.00900 | 163,190,424 | 16.32 | 15,291,772 | 1.53 |

`D_energy_per_query.csv` has two rows per (n, b), one per edge. Its columns are `n, b, build_probes, build_source,
build_J, marginal_J, vs` ("cheapest" / "costliest"), `fw_J, break_even_queries`. Since the build is about 1.03–1.05 ·
n·K·b probes and the marginal is tiny, T\* ≈ 16·b·1.04·4.04 J · n / F. It is linear in n and in b: about 3.3 / 9.7 / 16.3
queries per agent at b = 1 / 3 / 5 against AutoGen, and 0.31 / 0.91 / 1.53 against Magentic-One.

---

### 6.4 Caveats, discrepancies and open questions

1. **Bernoulli counts, live prices.** At n = 10⁷ the build probe counts come from bernoulli ledgers (`bernoulli_scale_v5`
   via the cache for b = 1 and 3, `va_b_bernoulli_1e7` for b = 5), but every probe is priced at the live specialist per-probe energy (4.04 J).
   No live run exists at 10⁷. Probe counts do not depend on the backend (the 10³ / 10⁵ live counts match the bernoulli
   cache to 0.03 %), so the price is the assumption, not the count.
2. **`build_source` does not name the backend.** All nine D rows say "ledger". Six are live or `va_b` ledgers at 10³ / 10⁵
   (live) and 10⁷ b = 5 (`va_b_bernoulli_1e7`), and two are the `bernoulli_scale_v5` sweep at 10⁷ b = 1 and 3. The table
   in §6.3.1 gives each one's source. The ratio fallback (`:57-58`) is still in the code but unused.
3. **The flat-NSW router is constant in C because it is centralised.** It is an ANN index over flat probe means held in one
   place (§2.8.3; METHODS.md calls it a verified-centralised arm). It shows that sublinear routing is possible centrally.
   What MIDIAN adds is routing that is sublinear, decentralised and verified. C does not show that difference, and C
   leaves out the router's ⌈log₂ n⌉ hops.
4. **D's advantage is over the frameworks, not over flat probing.** Flat probe argmax builds with exactly n·K·b probes,
   slightly fewer than MIDIAN-VA's 1.03–1.05 × n·K·b. Its per-query energy is n comparisons: 1e-5 J at 10³, 1e-3 J at 10⁵,
   0.1 J at 10⁷. Its D curve is not drawn. It would lie within a few percent of MIDIAN-VA's: its build is 3–5 % smaller,
   and its per-query floor is lower at 10³ and higher at 10⁷. The asymptotic advantage over flat methods is C's routing work, which D prices at near zero.
5. **The framework band is a live n = 1,000 measurement, applied at every n.**
   - Retrieval over the n descriptions (TF-IDF or embedding scoring) is not charged. The ledger charges each framework
     query `compare(k)` and `message(k + 2)` (`_common.py:364-365`), which is 10 comparisons and 12 messages whatever n
     is. `energy.py` says CPU-side routing is "microseconds per task and omitted" (`energy.py:67`).
   - The framework build is not drawn: `message(n)` for registration (`_common.py:289`), 1 J at n = 10³ and 1e4 J at 10⁷.
     Neither is any description-embedding cost.
   - The supervisor prompt holds k = 10 descriptions, so its token count, and hence the call-equivalents, is taken as
     constant in n. It was measured only at n = 1,000.
   
   Every omission favours the frameworks, so the band is a lower bound on their cost at large n under this model.
6. **The band pools regimes and shapes.** It averages over β ∈ {0, 0.1, 0.25, 0.5} and all three population shapes. Per-β
   means of AutoGen's latency range from 0.92 s to 3.13 s. It includes CrewAI and ADK rows that E–H drop (no quarantine
   filter). Only the edges matter for D: AutoGen and Magentic-One.
7. **Call-equivalents are a latency ratio, not a count of calls.** The legend says "1-10.7 supervisor-call equivalents per
   query" (`:93-94`). The floor is AutoGen, one call by construction. The 10.7 at the ceiling is Magentic-One's mean latency
   divided by AutoGen's (§6.3.3). No call count was recorded.
8. **Energy is an estimate** (the `*`). Its uncertain parts: B = 5A is assumed; A rests on one 34 req/s throughput figure;
   7B / 14B probe tokens are 3B proxies; latency stands in for GPU work; 700 W is the TDP. A 400 W draw would scale every
   LLM term by 4/7. That lowers every y value but leaves T\* almost unchanged, because the build and the band scale
   together. Messages at 1e-3 J are a guess. The 1e-2 J variant moves T\* by under 0.31 %.
9. **MIDIAN-VA's build is pooled over regimes and shapes.** `va_build` does not filter β or `dist`. The 10³ b = 3 value
   mixes 3 live shapes and 4 β values. The spread is small (49,368–49,468 at 10³ b = 3), so the effect is under 0.2 %.
10. **The 10⁵, b = 5 build rests on 3 honest rows.** Those were the rows present at 12:16, and still at 12:19. The cartel b = 5 rows of
    `va_b_n100k` were still running (§4.3).
11. **The cache does not refresh itself.** `cost_by_n.csv` is reused until it is deleted (`:30`). The `va_build` grids and
    the framework rows are re-read on every run. A rerun can therefore mix a new build with an old cache. For the counts in
    C this does not matter, because they are deterministic. A cache written before the `b` column was added would fail in
    `draw_I` (`d.b`, `:73`), so an old file has to be deleted. The current one was rebuilt at 12:16.
12. **Running `efficiency_figs.py` writes to result directories.** `energy.table()` loads through `rte.analyze.load`,
    which consolidates rows.d into rows.csv for `live_f1_n1000`, `variants_f1` and `fw_live_n1000` (`rte/analyze.py:57`),
    unless a grid has a `.merge_owner` file (`rte/run.py:190-191`). Without `RTE_DATA`, the default
    `/scratch/rte/results` (`:20`) holds no grids. The script then stops with an error (`rte/analyze.py:60`) and draws
    nothing.
13. **C adds unlike units.** A message and a comparison each count 1 in C, although the energy model prices them 1e5 apart.
    In messages alone, the flat arms send 0 per query and MIDIAN-VA sends 2 + depth.

---

## Appendix: figures A and B, per row (seed counts, source grid)

Values as in `A_live_allb.csv` (02:05) and `B_families_allb.csv` (02:06); B is ÷ oracle. Format: value [95 % CI], seeds,
source. For the pooled arms: value [CI], scored seeds, pick counts, then the candidates with no rows yet ("missing") and
a bold **\*** when the pool is incomplete. A bold **\*** alone = budget not in yet. The seed counts of the six replay b=5 rival bars are
those at 02:06 (§4 intro).

### App. A_live_allb

| group | regime | arm | b=1 | b=3 | b=5 |
|---|---|---|---|---|---|
| n = 100 | honest | MIDIAN-VA | 0.7105 [0.690,0.732] 10s (va_b_n100) | 0.7816 [0.758,0.802] 10s (bars CSV) | 0.8121 [0.796,0.827] 10s (va_b_n100) |
| n = 100 | β=0.5 cartel | MIDIAN-VA | 0.6972 [0.676,0.717] 10s (va_b_n100) | 0.7680 [0.746,0.790] 10s (bars CSV) | 0.8121 [0.796,0.827] 10s (va_b_n100) |
| n = 100 | honest | MIDIAN | **\*** | 0.7747 [0.757,0.792] 10s (bars CSV) | **\*** |
| n = 100 | β=0.5 cartel | MIDIAN | **\*** | 0.7359 [0.722,0.751] 10s (bars CSV) | **\*** |
| n = 100 | honest | flat probe argmax (online) | **\*** | 0.7805 [0.766,0.793] 10s (bars CSV) | **\*** |
| n = 100 | β=0.5 cartel | flat probe argmax (online) | **\*** | 0.7805 [0.766,0.793] 10s (bars CSV) | **\*** |
| n = 100 | honest | best learned router | **\*** | 0.7376 [0.716,0.758] 10s, picks `knn_router_online` ×10; missing: flat_nsw_router, cluster_head_router, disrouter_cascade **\*** | **\*** |
| n = 100 | β=0.5 cartel | best learned router | **\*** | 0.7376 [0.716,0.758] 10s, picks `knn_router_online` ×10; missing: flat_nsw_router, cluster_head_router, disrouter_cascade **\*** | **\*** |
| n = 100 | honest | best bandit | **\*** | 0.7300 [0.715,0.744] 10s, picks `warm_start_bandit` ×10; missing: ucb_per_family, thompson_per_family, linucb_honest, trueskill_per_family, warm_start_bandit[n0=0.5] **\*** | **\*** |
| n = 100 | β=0.5 cartel | best bandit | **\*** | 0.7183 [0.704,0.732] 10s, picks `warm_start_bandit` ×10; missing: ucb_per_family, thompson_per_family, linucb_honest, trueskill_per_family, warm_start_bandit[n0=0.5] **\*** | **\*** |
| n = 100 | honest | declared argmax | n/a (budgetless) | 0.5992 [0.580,0.618] 10s (bars CSV) | n/a (budgetless) |
| n = 100 | β=0.5 cartel | declared argmax | n/a (budgetless) | 0.5181 [0.483,0.550] 10s (bars CSV) | n/a (budgetless) |
| n = 100 | honest | random | n/a (budgetless) | 0.4258 [0.408,0.442] 10s (bars CSV) | n/a (budgetless) |
| n = 100 | β=0.5 cartel | random | n/a (budgetless) | 0.4258 [0.408,0.442] 10s (bars CSV) | n/a (budgetless) |
| n = 1,000 | honest | MIDIAN-VA | 0.6913 [0.681,0.702] 10s (va_b_n1000) | 0.8134 [0.805,0.821] 10s (bars CSV) | 0.8377 [0.831,0.845] 10s (va_b_n1000) |
| n = 1,000 | β=0.5 cartel | MIDIAN-VA | 0.6914 [0.679,0.704] 10s (va_b_n1000) | 0.8074 [0.797,0.817] 10s (bars CSV) | 0.8375 [0.831,0.844] 10s (va_b_n1000) |
| n = 1,000 | honest | MIDIAN | **\*** | 0.7890 [0.785,0.792] 10s (bars CSV) | **\*** |
| n = 1,000 | β=0.5 cartel | MIDIAN | **\*** | 0.7379 [0.724,0.750] 10s (bars CSV) | **\*** |
| n = 1,000 | honest | flat probe argmax (online) | **\*** | 0.7871 [0.781,0.793] 10s (bars CSV) | **\*** |
| n = 1,000 | β=0.5 cartel | flat probe argmax (online) | **\*** | 0.7869 [0.781,0.793] 10s (bars CSV) | **\*** |
| n = 1,000 | honest | best learned router | **\*** | 0.7656 [0.753,0.776] 10s, picks `knn_router_online` ×10 | **\*** |
| n = 1,000 | β=0.5 cartel | best learned router | **\*** | 0.7656 [0.753,0.776] 10s, picks `knn_router_online` ×10 | **\*** |
| n = 1,000 | honest | best bandit | **\*** | 0.7501 [0.737,0.761] 10s, picks `warm_start_bandit` ×5; `linucb_honest` ×5; missing: warm_start_bandit[n0=0.5] **\*** | **\*** |
| n = 1,000 | β=0.5 cartel | best bandit | **\*** | 0.7518 [0.738,0.764] 10s, picks `warm_start_bandit` ×5; `linucb_honest` ×5; missing: warm_start_bandit[n0=0.5] **\*** | **\*** |
| n = 1,000 | honest | declared argmax | n/a (budgetless) | 0.6154 [0.593,0.636] 10s (bars CSV) | n/a (budgetless) |
| n = 1,000 | β=0.5 cartel | declared argmax | n/a (budgetless) | 0.5202 [0.493,0.546] 10s (bars CSV) | n/a (budgetless) |
| n = 1,000 | honest | random | n/a (budgetless) | 0.4321 [0.423,0.440] 10s (bars CSV) | n/a (budgetless) |
| n = 1,000 | β=0.5 cartel | random | n/a (budgetless) | 0.4321 [0.423,0.440] 10s (bars CSV) | n/a (budgetless) |
| n = 10,000 | honest | MIDIAN-VA | 0.6678 [0.643,0.680] 3s (va_b_n10k) | 0.8111 [0.787,0.857] 3s (bars CSV) | 0.8356 [0.807,0.877] 3s (va_b_n10k) |
| n = 10,000 | β=0.5 cartel | MIDIAN-VA | 0.6700 [0.647,0.690] 3s (va_b_n10k) | 0.8100 [0.787,0.853] 3s (bars CSV) | 0.8311 [0.807,0.860] 3s (va_b_n10k) |
| n = 10,000 | honest | MIDIAN | 0.6678 [0.643,0.680] 3s (rivals_b_n10k) | 0.7850 [0.757,0.825] 3s (bars CSV) | **\*** |
| n = 10,000 | β=0.5 cartel | MIDIAN | **\*** | 0.7522 [0.707,0.803] 3s (bars CSV) | **\*** |
| n = 10,000 | honest | flat probe argmax (online) | 0.6522 [0.600,0.687] 3s (rivals_b_n10k) | 0.7744 [0.715,0.830] 3s (bars CSV) | **\*** |
| n = 10,000 | β=0.5 cartel | flat probe argmax (online) | **\*** | 0.7744 [0.717,0.830] 3s (bars CSV) | **\*** |
| n = 10,000 | honest | best learned router | 0.5700 [0.470,0.683] 3s, picks `knn_router_online` ×3; missing: flat_nsw_router, cluster_head_router, disrouter_cascade **\*** | 0.7311 [0.683,0.787] 3s, picks `knn_router_online` ×3; missing: flat_nsw_router, cluster_head_router, disrouter_cascade **\*** | **\*** |
| n = 10,000 | β=0.5 cartel | best learned router | **\*** | 0.7311 [0.683,0.787] 3s, picks `knn_router_online` ×3; missing: flat_nsw_router, cluster_head_router, disrouter_cascade **\*** | **\*** |
| n = 10,000 | honest | best bandit | 0.6989 [0.647,0.727] 3s, picks `warm_start_bandit` ×3; missing: ucb_per_family, thompson_per_family, trueskill_per_family **\*** | 0.7415 [0.682,0.773] 3s, picks `warm_start_bandit` ×3; missing: ucb_per_family, thompson_per_family, trueskill_per_family, warm_start_bandit[n0=0.5] **\*** | **\*** |
| n = 10,000 | β=0.5 cartel | best bandit | **\*** | 0.7833 [0.723,0.840] 3s, picks `warm_start_bandit` ×3; missing: ucb_per_family, thompson_per_family, linucb_honest, trueskill_per_family, warm_start_bandit[n0=0.5] **\*** | **\*** |
| n = 10,000 | honest | declared argmax | n/a (budgetless) | 0.6578 [0.587,0.707] 3s (bars CSV) | n/a (budgetless) |
| n = 10,000 | β=0.5 cartel | declared argmax | n/a (budgetless) | 0.5322 [0.470,0.607] 3s (bars CSV) | n/a (budgetless) |
| n = 10,000 | honest | random | n/a (budgetless) | 0.4167 [0.383,0.467] 3s (bars CSV) | n/a (budgetless) |
| n = 10,000 | β=0.5 cartel | random | n/a (budgetless) | 0.4167 [0.383,0.467] 3s (bars CSV) | n/a (budgetless) |
| n = 100,000 | honest | MIDIAN-VA | 0.6400 [0.590,0.690] 3s (va_b_n100k) | 0.8356 [0.810,0.857] 3s (bars CSV) | **\*** |
| n = 100,000 | β=0.5 cartel | MIDIAN-VA | **\*** | 0.8278 [0.813,0.837] 3s (bars CSV) | **\*** |
| n = 100,000 | honest | MIDIAN | **\*** | 0.7522 [0.713,0.790] 3s (bars CSV) | **\*** |
| n = 100,000 | β=0.5 cartel | MIDIAN | **\*** | 0.7056 [0.680,0.747] 3s (bars CSV) | **\*** |
| n = 100,000 | honest | flat probe argmax (online) | **\*** | 0.7489 [0.703,0.807] 3s (bars CSV) | **\*** |
| n = 100,000 | β=0.5 cartel | flat probe argmax (online) | **\*** | 0.7489 [0.703,0.807] 3s (bars CSV) | **\*** |
| n = 100,000 | honest | best learned router | **\*** | 0.7411 [0.710,0.777] 3s, picks `knn_router_online` ×3; missing: flat_nsw_router, cluster_head_router, disrouter_cascade **\*** | **\*** |
| n = 100,000 | β=0.5 cartel | best learned router | **\*** | 0.7411 [0.710,0.777] 3s, picks `knn_router_online` ×3; missing: flat_nsw_router, cluster_head_router, disrouter_cascade **\*** | **\*** |
| n = 100,000 | honest | best bandit | **\*** | 0.7378 [0.680,0.783] 3s, picks `warm_start_bandit` ×3; missing: ucb_per_family, thompson_per_family, warm_start_bandit[n0=0.5] **\*** | **\*** |
| n = 100,000 | β=0.5 cartel | best bandit | **\*** | 0.7778 [0.740,0.817] 3s, picks `warm_start_bandit` ×3; missing: ucb_per_family, thompson_per_family, warm_start_bandit[n0=0.5] **\*** | **\*** |
| n = 100,000 | honest | declared argmax | n/a (budgetless) | 0.6222 [0.593,0.673] 3s (bars CSV) | n/a (budgetless) |
| n = 100,000 | β=0.5 cartel | declared argmax | n/a (budgetless) | 0.5778 [0.543,0.613] 3s (bars CSV) | n/a (budgetless) |
| n = 100,000 | honest | random | n/a (budgetless) | 0.4422 [0.420,0.473] 3s (bars CSV) | n/a (budgetless) |
| n = 100,000 | β=0.5 cartel | random | n/a (budgetless) | 0.4422 [0.420,0.473] 3s (bars CSV) | n/a (budgetless) |

### App. B_families_allb

| group | regime | arm | b=1 | b=3 | b=5 |
|---|---|---|---|---|---|
| live n = 100,000 | honest | MIDIAN-VA | 0.7423 [0.684,0.800] 3s (va_b_n100k) | 0.9691 [0.939,0.994] 3s (bars CSV) | **\*** |
| live n = 100,000 | β=0.5 cartel | MIDIAN-VA | **\*** | 0.9601 [0.943,0.970] 3s (bars CSV) | **\*** |
| live n = 100,000 | honest | MIDIAN | **\*** | 0.8724 [0.827,0.916] 3s (bars CSV) | **\*** |
| live n = 100,000 | β=0.5 cartel | MIDIAN | **\*** | 0.8183 [0.789,0.866] 3s (bars CSV) | **\*** |
| live n = 100,000 | honest | flat probe argmax (online) | **\*** | 0.8686 [0.816,0.936] 3s (bars CSV) | **\*** |
| live n = 100,000 | β=0.5 cartel | flat probe argmax (online) | **\*** | 0.8686 [0.816,0.936] 3s (bars CSV) | **\*** |
| live n = 100,000 | honest | best learned router | **\*** | 0.8595 [0.823,0.901] 3s, picks `knn_router_online` ×3; missing: flat_nsw_router, cluster_head_router, disrouter_cascade **\*** | **\*** |
| live n = 100,000 | β=0.5 cartel | best learned router | **\*** | 0.8595 [0.823,0.901] 3s, picks `knn_router_online` ×3; missing: flat_nsw_router, cluster_head_router, disrouter_cascade **\*** | **\*** |
| live n = 100,000 | honest | best bandit | **\*** | 0.8557 [0.789,0.909] 3s, picks `warm_start_bandit` ×3; missing: ucb_per_family, thompson_per_family, warm_start_bandit[n0=0.5] **\*** | **\*** |
| live n = 100,000 | β=0.5 cartel | best bandit | **\*** | 0.9021 [0.858,0.947] 3s, picks `warm_start_bandit` ×3; missing: ucb_per_family, thompson_per_family, warm_start_bandit[n0=0.5] **\*** | **\*** |
| live n = 100,000 | honest | declared argmax | n/a (budgetless) | 0.7216 [0.688,0.781] 3s (bars CSV) | n/a (budgetless) |
| live n = 100,000 | β=0.5 cartel | declared argmax | n/a (budgetless) | 0.6701 [0.630,0.711] 3s (bars CSV) | n/a (budgetless) |
| live n = 100,000 | honest | random | n/a (budgetless) | 0.5129 [0.487,0.549] 3s (bars CSV) | n/a (budgetless) |
| live n = 100,000 | β=0.5 cartel | random | n/a (budgetless) | 0.5129 [0.487,0.549] 3s (bars CSV) | n/a (budgetless) |
| bernoulli n = 10,000,000 | honest | MIDIAN-VA | 0.8015 [0.796,0.807] 100s (bernoulli_scale_v5 matrix) | 0.9469 [0.944,0.950] 100s (bars CSV) | 0.9791 [0.975,0.984] 51s (va_b_bernoulli_1e7) |
| bernoulli n = 10,000,000 | β=0.5 cartel | MIDIAN-VA | 0.7977 [0.792,0.803] 100s (bernoulli_scale_v5 matrix) | 0.9386 [0.935,0.943] 100s (bars CSV) | 0.9783 [0.972,0.984] 36s (va_b_bernoulli_1e7) |
| bernoulli n = 10,000,000 | honest | MIDIAN | 0.8015 [0.796,0.807] 100s (bernoulli_scale_v5 matrix) | 0.9110 [0.907,0.915] 100s (bars CSV) | **\*** |
| bernoulli n = 10,000,000 | β=0.5 cartel | MIDIAN | 0.7450 [0.735,0.754] 100s (bernoulli_scale_v5 matrix) | 0.8463 [0.840,0.853] 100s (bars CSV) | **\*** |
| bernoulli n = 10,000,000 | honest | flat probe argmax (online) | 0.8005 [0.795,0.806] 100s (bernoulli_scale_v5 matrix) | 0.9088 [0.905,0.913] 100s (bars CSV) | **\*** |
| bernoulli n = 10,000,000 | β=0.5 cartel | flat probe argmax (online) | 0.8005 [0.795,0.806] 100s (bernoulli_scale_v5 matrix) | 0.9088 [0.905,0.913] 100s (bars CSV) | **\*** |
| bernoulli n = 10,000,000 | honest | best learned router | 0.9909 [0.988,0.994] 100s, picks `cluster_head_router` ×100; missing: disrouter_cascade **\*** | 0.9909 [0.988,0.994] 100s, picks `cluster_head_router` ×100 | **\*** |
| bernoulli n = 10,000,000 | β=0.5 cartel | best learned router | 0.8450 [0.840,0.851] 100s, picks `cluster_head_router` ×100; missing: disrouter_cascade **\*** | 0.8523 [0.848,0.857] 100s, picks `disrouter_cascade` ×100 | **\*** |
| bernoulli n = 10,000,000 | honest | best bandit | 0.9890 [0.986,0.992] 100s, picks `warm_start_bandit` ×100; missing: linucb_honest, warm_start_bandit[n0=0.5] **\*** | 0.9912 [0.988,0.994] 100s, picks `warm_start_bandit` ×100; missing: warm_start_bandit[n0=0.5] **\*** | **\*** |
| bernoulli n = 10,000,000 | β=0.5 cartel | best bandit | 0.8934 [0.890,0.897] 100s, picks `warm_start_bandit` ×100; missing: linucb_honest, warm_start_bandit[n0=0.5] **\*** | 0.9075 [0.904,0.911] 100s, picks `warm_start_bandit` ×100; missing: warm_start_bandit[n0=0.5] **\*** | **\*** |
| bernoulli n = 10,000,000 | honest | declared argmax | n/a (budgetless) | 0.9942 [0.991,0.997] 100s (bars CSV) | n/a (budgetless) |
| bernoulli n = 10,000,000 | β=0.5 cartel | declared argmax | n/a (budgetless) | 0.8551 [0.849,0.862] 100s (bars CSV) | n/a (budgetless) |
| bernoulli n = 10,000,000 | honest | random | n/a (budgetless) | 0.4936 [0.490,0.497] 100s (bars CSV) | n/a (budgetless) |
| bernoulli n = 10,000,000 | β=0.5 cartel | random | n/a (budgetless) | 0.4936 [0.490,0.497] 100s (bars CSV) | n/a (budgetless) |
| RouterBench replay n = 1,000,000 | honest | MIDIAN-VA | 0.8272 [0.823,0.831] 100s (replay_scale_v5 matrix) | 0.9642 [0.961,0.968] 100s (bars CSV) | 0.9815 [0.976,0.987] 38s (va_b_replay_1e6) |
| RouterBench replay n = 1,000,000 | β=0.5 cartel | MIDIAN-VA | 0.8216 [0.816,0.827] 100s (replay_scale_v5 matrix) | 0.9622 [0.959,0.966] 100s (bars CSV) | 0.9845 [0.977,0.992] 25s (va_b_replay_1e6) |
| RouterBench replay n = 1,000,000 | honest | MIDIAN | 0.8272 [0.823,0.831] 100s (replay_scale_v5 matrix) | 0.8908 [0.887,0.894] 100s (bars CSV) | 0.9246 [0.916,0.933] 20s (rivals_b_replay_1e6) |
| RouterBench replay n = 1,000,000 | β=0.5 cartel | MIDIAN | 0.5015 [0.481,0.521] 100s (replay_scale_v5 matrix) | 0.7095 [0.694,0.725] 100s (bars CSV) | 0.8301 [0.817,0.844] 12s (rivals_b_replay_1e6) |
| RouterBench replay n = 1,000,000 | honest | flat probe argmax (online) | 0.8292 [0.825,0.834] 100s (replay_scale_v5 matrix) | 0.8946 [0.891,0.898] 100s (bars CSV) | 0.9257 [0.917,0.934] 20s (rivals_b_replay_1e6) |
| RouterBench replay n = 1,000,000 | β=0.5 cartel | flat probe argmax (online) | 0.8292 [0.825,0.834] 100s (replay_scale_v5 matrix) | 0.8946 [0.891,0.898] 100s (bars CSV) | 0.9249 [0.914,0.936] 12s (rivals_b_replay_1e6) |
| RouterBench replay n = 1,000,000 | honest | best learned router | 0.9980 [0.995,1.001] 100s, picks `cluster_head_router` ×100; missing: disrouter_cascade **\*** | 0.9980 [0.995,1.001] 100s, picks `cluster_head_router` ×100 | 0.9983 [0.990,1.006] 20s, picks `cluster_head_router` ×20 |
| RouterBench replay n = 1,000,000 | β=0.5 cartel | best learned router | 0.8619 [0.858,0.865] 100s, picks `cluster_head_router` ×100; missing: disrouter_cascade **\*** | 0.9022 [0.898,0.906] 100s, picks `flat_nsw_router` ×100 | 0.9372 [0.925,0.949] 12s, picks `flat_nsw_router` ×12 |
| RouterBench replay n = 1,000,000 | honest | best bandit | 0.9937 [0.990,0.997] 100s, picks `warm_start_bandit` ×100; missing: linucb_honest, warm_start_bandit[n0=0.5] **\*** | 0.9948 [0.991,0.998] 100s, picks `warm_start_bandit` ×100; missing: warm_start_bandit[n0=0.5] **\*** | 0.9955 [0.988,1.003] 20s, picks `warm_start_bandit` ×20; missing: warm_start_bandit[n0=0.5] **\*** |
| RouterBench replay n = 1,000,000 | β=0.5 cartel | best bandit | 0.8719 [0.869,0.875] 100s, picks `warm_start_bandit` ×100; missing: linucb_honest, warm_start_bandit[n0=0.5] **\*** | 0.8926 [0.890,0.896] 100s, picks `warm_start_bandit` ×100; missing: warm_start_bandit[n0=0.5] **\*** | 0.9114 [0.902,0.920] 12s, picks `warm_start_bandit` ×12; missing: warm_start_bandit[n0=0.5] **\*** |
| RouterBench replay n = 1,000,000 | honest | declared argmax | n/a (budgetless) | 0.9977 [0.994,1.001] 100s (bars CSV) | n/a (budgetless) |
| RouterBench replay n = 1,000,000 | β=0.5 cartel | declared argmax | n/a (budgetless) | 0.8505 [0.847,0.854] 100s (bars CSV) | n/a (budgetless) |
| RouterBench replay n = 1,000,000 | honest | random | n/a (budgetless) | 0.2421 [0.240,0.244] 100s (bars CSV) | n/a (budgetless) |
| RouterBench replay n = 1,000,000 | β=0.5 cartel | random | n/a (budgetless) | 0.2421 [0.240,0.244] 100s (bars CSV) | n/a (budgetless) |
| RouterEval n = 5,000 | honest | MIDIAN-VA | 0.6712 [0.606,0.735] 3s (va_b_routereval5k) | 0.7820 [0.724,0.817] 3s (bars CSV) | 0.8399 [0.813,0.857] 3s (va_b_routereval5k) |
| RouterEval n = 5,000 | β=0.5 cartel | MIDIAN-VA | 0.6650 [0.591,0.732] 3s (va_b_routereval5k) | 0.7869 [0.739,0.813] 3s (bars CSV) | 0.8313 [0.813,0.857] 3s (va_b_routereval5k) |
| RouterEval n = 5,000 | honest | MIDIAN | **\*** | 0.7131 [0.676,0.743] 3s (bars CSV) | **\*** |
| RouterEval n = 5,000 | β=0.5 cartel | MIDIAN | **\*** | 0.6638 [0.647,0.691] 3s (bars CSV) | **\*** |
| RouterEval n = 5,000 | honest | flat probe argmax (online) | **\*** | 0.6884 [0.650,0.717] 3s (bars CSV) | **\*** |
| RouterEval n = 5,000 | β=0.5 cartel | flat probe argmax (online) | **\*** | 0.6884 [0.650,0.717] 3s (bars CSV) | **\*** |
| RouterEval n = 5,000 | honest | best learned router | **\*** | 0.6749 [0.643,0.709] 3s, picks `knn_router` ×3; missing: knn_router_online, flat_nsw_router, cluster_head_router, disrouter_cascade **\*** | **\*** |
| RouterEval n = 5,000 | β=0.5 cartel | best learned router | **\*** | 0.6749 [0.643,0.709] 3s, picks `knn_router` ×3; missing: knn_router_online, flat_nsw_router, cluster_head_router, disrouter_cascade **\*** | **\*** |
| RouterEval n = 5,000 | honest | best bandit | **\*** | 0.9113 [0.890,0.924] 3s, picks `warm_start_bandit` ×3; missing: ucb_per_family, thompson_per_family, trueskill_per_family, warm_start_bandit[n0=0.5] **\*** | **\*** |
| RouterEval n = 5,000 | β=0.5 cartel | best bandit | **\*** | 0.6736 [0.661,0.691] 3s, picks `warm_start_bandit` ×2; `linucb_honest` ×1; missing: ucb_per_family, thompson_per_family, trueskill_per_family, warm_start_bandit[n0=0.5] **\*** | **\*** |
| RouterEval n = 5,000 | honest | declared argmax | n/a (budgetless) | 0.9581 [0.905,0.986] 3s (bars CSV) | n/a (budgetless) |
| RouterEval n = 5,000 | β=0.5 cartel | declared argmax | n/a (budgetless) | 0.6810 [0.632,0.717] 3s (bars CSV) | n/a (budgetless) |
| RouterEval n = 5,000 | honest | random | n/a (budgetless) | 0.6096 [0.591,0.621] 3s (bars CSV) | n/a (budgetless) |
| RouterEval n = 5,000 | β=0.5 cartel | random | n/a (budgetless) | 0.6096 [0.591,0.621] 3s (bars CSV) | n/a (budgetless) |
| LLMRouterBench n = 20 | honest | MIDIAN-VA | 0.8880 [0.876,0.900] 5s (va_b_llmrouterbench) | 0.9222 [0.895,0.953] 5s (bars CSV) | 0.9539 [0.922,0.986] 5s (va_b_llmrouterbench) |
| LLMRouterBench n = 20 | β=0.5 cartel | MIDIAN-VA | 0.8850 [0.865,0.901] 5s (va_b_llmrouterbench) | 0.9189 [0.887,0.952] 5s (bars CSV) | 0.9512 [0.920,0.982] 5s (va_b_llmrouterbench) |
| LLMRouterBench n = 20 | honest | MIDIAN | 0.8880 [0.876,0.900] 5s (rivals_b_llmrouterbench) | 0.9297 [0.901,0.960] 5s (bars CSV) | 0.9553 [0.927,0.984] 5s (rivals_b_llmrouterbench) |
| LLMRouterBench n = 20 | β=0.5 cartel | MIDIAN | 0.8662 [0.842,0.890] 5s (rivals_b_llmrouterbench) | 0.8350 [0.811,0.868] 5s (bars CSV) | 0.8497 [0.832,0.867] 5s (rivals_b_llmrouterbench) |
| LLMRouterBench n = 20 | honest | flat probe argmax (online) | 0.9040 [0.895,0.913] 5s (rivals_b_llmrouterbench) | 0.9476 [0.908,0.976] 5s (bars CSV) | 0.9721 [0.946,0.998] 5s (rivals_b_llmrouterbench) |
| LLMRouterBench n = 20 | β=0.5 cartel | flat probe argmax (online) | 0.9040 [0.895,0.913] 5s (rivals_b_llmrouterbench) | 0.9476 [0.908,0.976] 5s (bars CSV) | 0.9721 [0.946,0.998] 5s (rivals_b_llmrouterbench) |
| LLMRouterBench n = 20 | honest | best learned router | 0.9895 [0.971,1.010] 5s, picks `cluster_head_router` ×5 | 0.9476 [0.923,0.964] 5s, picks `mlp_router` ×5; missing: flat_nsw_router, cluster_head_router, disrouter_cascade **\*** | 0.9895 [0.971,1.010] 5s, picks `cluster_head_router` ×5 |
| LLMRouterBench n = 20 | β=0.5 cartel | best learned router | 0.7961 [0.705,0.860] 5s, picks `mlp_router` ×2; `knn_router_online` ×2; `disrouter_cascade` ×1 | 0.9476 [0.923,0.964] 5s, picks `mlp_router` ×5; missing: flat_nsw_router, cluster_head_router, disrouter_cascade **\*** | 0.9186 [0.900,0.937] 5s, picks `knn_router_online` ×2; `mlp_router` ×2; `flat_nsw_router` ×1 |
| LLMRouterBench n = 20 | honest | best bandit | 0.9310 [0.905,0.958] 5s, picks `warm_start_bandit` ×5; missing: warm_start_bandit[n0=0.5] **\*** | 0.9321 [0.910,0.954] 5s, picks `warm_start_bandit` ×3; `linucb_honest` ×2; missing: ucb_per_family, thompson_per_family, trueskill_per_family, warm_start_bandit[n0=0.5] **\*** | 0.9462 [0.923,0.969] 5s, picks `linucb_honest` ×4; `warm_start_bandit` ×1; missing: warm_start_bandit[n0=0.5] **\*** |
| LLMRouterBench n = 20 | β=0.5 cartel | best bandit | 0.9012 [0.880,0.921] 5s, picks `linucb_honest` ×5; missing: warm_start_bandit[n0=0.5] **\*** | 0.9366 [0.911,0.960] 5s, picks `linucb_honest` ×5; missing: ucb_per_family, thompson_per_family, trueskill_per_family, warm_start_bandit[n0=0.5] **\*** | 0.9559 [0.927,0.980] 5s, picks `linucb_honest` ×5; missing: warm_start_bandit[n0=0.5] **\*** |
| LLMRouterBench n = 20 | honest | declared argmax | n/a (budgetless) | 0.9898 [0.970,1.010] 5s (bars CSV) | n/a (budgetless) |
| LLMRouterBench n = 20 | β=0.5 cartel | declared argmax | n/a (budgetless) | 0.7837 [0.743,0.827] 5s (bars CSV) | n/a (budgetless) |
| LLMRouterBench n = 20 | honest | random | n/a (budgetless) | 0.6745 [0.650,0.702] 5s (bars CSV) | n/a (budgetless) |
| LLMRouterBench n = 20 | β=0.5 cartel | random | n/a (budgetless) | 0.6745 [0.650,0.702] 5s (bars CSV) | n/a (budgetless) |
