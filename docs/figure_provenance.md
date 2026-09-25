# Figure provenance: where every plotted number comes from

A reference for anyone writing about, or checking, the paper figures A-I in `figures/paper/`. For
every bar, line and marker it says what the number is, which grids, cells and seeds it comes from, how it is averaged,
and how each method and backend behind it is implemented. It was written from the code; where code and prose documents
disagreed, the code was taken as correct and the disagreement is listed in the part's "Discrepancies" section. Code is
cited by file and function (`rte/world.py`, `Midian.build`), not by line. Value tables are read from the figure CSVs
(`figures/paper/*.csv`), and seed counts and grid provenance from the figure scripts' own table builders run read-only on
the same rows. Every slot of A-I is filled and no pool is incomplete (`figure_status.py` reports no `INCOMPLETE` in any
CSV).

**Code references.** File and function names were written against the code at tag `pre-refactor-2026-09-24`; the
refactor moved the files without changing any plotted value (every figure CSV value-identical). Paths below are the
current ones: the figure scripts (`condensed_figs.py`, `shortlist_figs.py`, `shortlist_condensed.py`,
`efficiency_figs.py`, `bar_figs.py`, `lie_max_fig.py`) are in `scripts/figures/`; the shared helpers `figspec.py`,
`seed_tables.py`, `stats.py` (`se`, `ci`, formerly in `extra_figs.py`), `exclusions.py` (`excluded`, formerly in
`extra_figs.py`) and `rows.py` (`label`) in `scripts/figures/lib/`; `energy.py`, `fw_variant_numbers.py` and
`scale_matrix.py` in `scripts/analysis/`; `figure_status.py` in `cluster/ops/`. Quoted line numbers and a few helper
names (`fw_variant_numbers.load` / `regime` / `ci`) are those of the tag. `figures/bars/` and `figures/shortlist/` are
the exploratory per-family figures (generated, not tracked); their CSVs are in `results/aggregates/{bars,shortlist}/`.

**Names.** Method names follow the 2026-09-24 rename ([errata.md](errata.md)): **MIDIAN** is the full method with both
defenses (formerly MIDIAN-VA, key `midian_va`); **MIDIAN w/o defenses** is the plain, pre-registered tree (formerly
MIDIAN); **MIDIAN w/o audits** and **MIDIAN w/o verification** are the single-defense ablations (formerly MIDIAN-V and
MIDIAN-A). All four are one class in `rte/methods/midian.py`. Grid and directory names (`va_b_*`, `*_verified_va*`, the
`va_cohort` shortlist key, ...) are identifiers and keep their old spelling. The figures follow
[figures.md](figures.md) through `figspec.py`: serif fonts, fixed legend order, the specified names, no titles and no
incompleteness marks. The rename changed no value: the figure CSVs regenerated on the renamed rows match those of tag
`submission-2026-09-24` (737 rows matched, 0 differing).

| Part | Covers |
|---|---|
| 1 Foundations | world, true skill S vs declared D, tasks, probes, the ledger, skill distributions, liars / β / cartel / lie modes, the four backends (live, bernoulli, replay, RouterEval + LLMRouterBench), the probe budget b, grid mechanics, seeds, pairing, aggregation and CIs, errata and incompleteness reporting, glossary |
| 2 Methods | every non-framework arm in A/B: MIDIAN w/o defenses, MIDIAN and its ablations (deepest), oracle, flat probe argmax, declared argmax, random, the bandit pool, the learned-router pool, how "best learned/declared router" / "best bandit" is picked |
| 3 Frameworks and shortlists | the framework adapters, supervisor LLM, fallback, prefetch, every shortlist variant (TF-IDF, BM25, MiniLM, Qwen3 dense ± instructions, fusion + reranker, declared top-k, MIDIAN cohort), caches, which combinations exist |
| 4 Figures A and B | every bar, shade, error bar in A (live, n = 10^2…10^5) and B (all families, / oracle), A's "best framework, best text shortlist" arm, the erratum-30 rows B's four non-live families read (and the leaky rows they replaced), with value tables |
| 5 Figures E, F, G, H | the two-stage shortlist pipeline, each figure's bars, dots and lines, the full-seed and MIN_FW rules, the body figures' best-instruction variants, the G lift, H on RouterEval's erratum-30 (calibrated, no-repeat, shuffled) reruns, with value tables |
| 6 Figures C and D | C (routing work per query vs n: MIDIAN, MIDIAN w/o defenses, any flat scan, the learned routers as a band) and D (estimated energy per query vs queries served): the ledger cache, pool members' work, MIDIAN's build probes, the energy model of `energy.py`, the framework band, slopes and break-even tables |
| 7 Figure I | the appendix panel `I_max_lie`: the strongest declared lie against the standard lie |
| Appendix | per-row A/B table with seed counts and source grid |

---

## 0.1 The figures in one paragraph each

- **A — live RTE, specialist population, n = 10², 10³, 10⁴, 10⁵.** Mean task success per arm, honest (β = 0, solid) and
  β = 0.5 low-skill cartel (hatched `////`) side by side. The arms, in the fixed legend order (`figspec.ORDER`): MIDIAN,
  MIDIAN w/o defenses, flat probe argmax, best learned/declared router, best bandit, declared argmax, random, and "best
  framework, best text shortlist"; the oracle is a grey dotted line over each group. There are two versions.
  - `A_live_allb` (appendix, 5.5 × 2.4 in): one bar per (arm, regime, b), b = 1 / 3 / 5 shaded light, as is and dark
    (`figspec.shade`), with a three-swatch budget legend. 144 bars: five budgeted arms × 3 b × 2 regimes × 4 n, plus one
    bar per regime and n for declared argmax, random and the framework arm (b does not apply to the first two; the
    framework arm is b = 3 only).
  - `A_live_stacked` (Figure 1, 5.5 × 1.9 in): **every arm at b = 3**, one bar per (arm, regime); **MIDIAN alone carries
    the budget overlay**, its b = 5, 3 and 1 bars drawn at their true heights tallest first, so b = 1 (light) is in front
    and b = 5 (dark) behind. No whiskers. 80 CSV rows (MIDIAN 3 per slot).
  - Every whisker (on `_allb`) is **±1 standard error over seeds** (sd / √seeds of the per-seed values; `stats.se`,
    applied by `condensed_figs.narrow`), a ~68 % interval, **not** a 95 % CI; over 10 seeds at n = 10² / 10³ and **3
    seeds** at 10⁴ / 10⁵.
  - For the five single arms, b = 3 values come from the per-cell bar CSVs and b = 1 / 5 values from the `va_b_*`
    (MIDIAN) and `rivals_b_*` grids. "Best learned/declared router" and "best bandit" are cross-fitted from per-seed
    tables at every b: each seed's arm is chosen on the other seeds (§2.1.3).
  - **"Best framework, best text shortlist"** (`add_best_framework`, `condensed_figs.py`): per n and regime, the
    seed × (framework | text shortlist) table of every pair with the full seed count (framework rows at b = 3 from
    `shortlist_figs.collect`, specialist; the nine text shortlists, not declared top-k or the MIDIAN cohort), cross-fitted
    per seed over the pairs like the pooled arms. Its CSV `chosen` column names the pair picked, e.g.
    `fw_magentic_one | embed x10`. Drawn in A only. See §4.1.
  - **No titles and no markers.** Incompleteness (a missing bar, or a pool that lacks a member or has a member missing
    some seeds) is printed by the script (`[A_live_allb] INCOMPLETE …`) and listed by `cluster/ops/figure_status.py`;
    today there is none.
- **Arms that do not change with b.** Four arms never probe, so b cannot change them: **declared argmax, random,
  `cluster_head_router` and `disrouter_cascade`** (`B_INVARIANT`, `condensed_figs.py`). Declared argmax and random are
  drawn once per regime; the other two are "best learned/declared router" candidates, and a pooled bar whose every seed
  picked one of them is the same number at b = 1, 3 and 5. The CSV column `b_invariant` flags these rows (§4.0.5).
- **B — every family at its largest pool, success / oracle.** The same arms (without the framework arm), order and b
  layout (`B_families_allb` 170 bars, `B_families_stacked` Figure 2), for live 10⁵, bernoulli 10⁷, replay 10⁶
  (RouterBench models replayed on RouterBench categories), RouterEval (the 5,000-LLM leaderboard pool) and the
  LLMRouterBench pool; x ticks are the backend (`live`, `Bernoulli`, `RouterBench replay`, `RouterEval`,
  `LLMRouterBench`) over $n = 10^k$. Every bar and its interval is divided by one number, the honest b = 3 oracle of that
  group; the oracle line sits at 1. See §4.2.
  **Which data B reads (erratum 30).** Each non-live family switches by itself to its erratum-30 rerun once that grid has
  every planned row (`seed_tables.complete` / `switched`). **All four have switched** (`switched()` returns bernoulli,
  replay, RouterEval and LLMRouterBench). So B draws: bernoulli 10⁷ with the claim-reading arms on calibrated claims
  (`bernoulli_1e7_cal`) and every other arm on its unchanged rows; replay 10⁶ from `replay_1e6_split_cal` (calibrated
  claims, probe/task split, 30 seeds); RouterEval 5,000 from `routereval5k_norep_cal` and LLMRouterBench from
  `llmrouterbench_norep_cal` (calibrated claims, each test prompt at most once, Q = 300, agent order shuffled per seed).
  The rows with the three leaks are **no longer drawn**: declared claims equal to true skill + N(0, 0.05) (an answer key
  that inflated declared argmax, the declared-only "learned" routers and the warm-start bandits), replay probes and routed
  tasks drawn from the same prompts with S the in-sample accuracy, and RouterEval / LLMRouterBench test prompts repeating
  in the stream (§4.0.7).
- **C — routing work per query vs n (appendix).** Messages plus comparisons that the ledger charges per routed query, for
  n = 10² … 10⁷ on calibrated bernoulli (`bernoulli_scale_v5`, b = 3, honest), log-log, 5.5 × 2.4 in
  (`scripts/figures/efficiency_figs.py`).
  - Three lines: MIDIAN, MIDIAN w/o defenses and **"any flat scan"**, one line for flat probe argmax, declared argmax and
    every bandit (`FLAT_SCAN`), which all scan the n agents; their counts coincide (the script prints the largest
    relative spread and draws flat probe argmax's values). The bandits therefore no longer get a band of their own.
  - "Best learned/declared router" is a band from the cheapest to the costliest pool member that can run at that n.
    A member whose own work is constant in n is left out of the band; only the flat-NSW router is. Learned runs from
    cluster head (24.6 at 10², about n/10 at 10⁷) to DisRouter (about 0.37·n at 10⁷), legend "∝ n^0.97 to n^0.99".
    Flat-NSW is still a candidate in A and B's cross-fit, and it is the majority pick in 21 of the 54 "best
    learned/declared router" bars of the A and B `_allb` CSVs (every bernoulli 10⁷ b = 3 / 5 bar among them), so the
    learned band is not the whole pool (§6.2).
  - Random does no routing work and is not drawn.
  - Each legend entry carries its fitted log-log slope over n ≥ 10³, classified as "constant", "∝ log n" or "∝ n^s"
    (`growth`).
  - MIDIAN goes 25 → 80 and MIDIAN w/o defenses 46 → 161, a fixed step per tree level: linear in depth, ∝ log n. Any
    flat scan is exactly n.

  It is a count (medians over seeds), not an estimate, with no error bars. A band is not the cost of the member B's
  cross-fit picks. See §6.2.
- **D — estimated energy per query vs queries served T (Figure 5, 5.5 × 1.8 in).** MIDIAN's probing build divided by T
  plus its per-query messages and comparisons, at n = 10³ / 10⁵ / 10⁷ (MIDIAN's green, light to dark) and b = 1 / 3 / 5
  (dotted, solid, dashed), against a light-grey band: the per-query supervisor energy of the ten agent frameworks measured
  live at n = 1,000 (21–220 J, i.e. 1 to 10.7 supervisor-call equivalents, a latency ratio to AutoGen). Both band edges
  are drawn and named at the right margin ("AutoGen, 20.6 J", "Magentic-One, 220 J"). Each black × is a break-even point.
  The legend (inside, bottom left, two columns) has three colour entries ($n = 10^3$ / $10^5$ / $10^7$) and three line
  styles (b = 1 / 3 / 5). Energies come from `scripts/analysis/energy.py` (a probe 4.04 J, a 7B supervisor call 20.6 J, a message
  1e-3 J, a comparison 1e-8 J, 700 W). See §6.3.
- **E — frameworks by shortlist across n (appendix; live, specialist).** For each n, one bar per shortlist: the
  unweighted mean over frameworks of each framework's seed mean. A framework counts only if it has the full seed count
  **of that (n, shortlist)** (the most seeds any framework has there, in either regime: 10 for the pre-registered-era
  shortlists at 10² / 10³, 3 for the backfill shortlists there and for everything at 10⁴ / 10⁵), and a bar is drawn only
  if at least **6** such frameworks exist (`MIN_FW`); otherwise the slot stays empty. Solid is honest, hatched is cartel.
  Three reference lines over each group, honest values only: the oracle and random (grey dotted) and MIDIAN on the whole
  population ("MIDIAN, no framework", green solid). Nine shortlists with the appendix names (both instruction variants of
  dense and fusion + reranker apart; BM25 and plain dense ran at 10³ only and are not in E's slots). All **72** slots
  (9 shortlists × 4 n × 2 regimes) are drawn and none has a rerun outstanding; the CSV has 74 rows (plus BM25 and plain
  dense at 10³ honest, not drawn). Declared top-k here ranks by real claims: every live framework grid has
  `declared_source = self_described`, the agent model's own 0–1 self-rating (§1.3.3, §3.4.9). See §5.2.
- **F — n = 10⁵, the shortlists best to worst (Figure 3).** Same bars and rules; a black dot marks the best single
  framework; the three reference lines as in E. The body figure `F_shortlists_1e5` draws six shortlists with the body
  names: dense (Qwen3-8B) and fusion + reranker are each their **best instruction variant** at 10⁵ by honest mean
  (`body`, `shortlist_condensed.py`; today the competence instruction for both, the CSV's `variant` column names
  it); BM25 has no rows at 10⁵. `F_shortlists_1e5_appendix` draws all nine variants (18 bars). See §5.3.
- **G — gain over the pre-registered TF-IDF shortlist at n = 10⁵ (appendix).** Per framework, shortlist minus TF-IDF,
  averaged over the full-seed frameworks both have (at least 6, else the bar is left out). The whisker is ±1 standard
  error across the paired frameworks, sd/√k (a ~68 % interval, not a 95 % CI). All 16 bars are drawn (9–10 paired
  frameworks each). At 10⁵ TF-IDF is a floor: 0.378889 for every framework, seed and regime (erratum 25). See §5.4.
- **H — RouterEval shortlists (Figure 4).** E's layout and rules with the body shortlists (hashed TF-IDF, MiniLM, dense
  and fusion + reranker at their best instruction variant per m, declared top-k, MIDIAN cohort), on RouterEval, pool
  m = 10 / 100 / 1,000 (strong-to-weak) plus the 5,000-LLM leaderboard. **H reads the erratum-30 reruns**
  (`shortlist_figs.h30()` is True): the `*_norep_cal` framework mirrors, with calibrated (live-like) claims, from which
  both the declared top-k ranking and the framework description text are rendered, no repeated test prompts, Q = 300 and
  a per-seed agent order; the reference lines (oracle, random, MIDIAN) come from `routereval_mmlu_norep_cal` (m ≤ 1,000)
  and `routereval5k_norep_cal` (5,000). The old rows (programmatic claims, repeated prompts, weak-to-strong agent order)
  are not drawn; the switch is all-or-nothing, never a mix. All 48 slots are drawn (9 frameworks each). See §5.5.

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
  live / bernoulli / replay). Exceeding the budget only logs a warning (§1.5). MIDIAN can spend a few % over;
  see 0.3.
- **Seed.** A new population, liar set and task stream, all drawn together. Every method in a (cell, seed) unit sees the
  same world, the same tasks and the same probe instances. That makes comparisons paired within a unit (§1.1, §1.6), with
  the RouterEval exception in §1.6.4 (pool fixed across seeds; only its order is shuffled per seed).
- **Declared claims D.** What each agent says about its own skill per family, before lying. On live it is the model's
  own self-rating (`self_described`: over-confident, correlation with S about 0.36). On bernoulli, replay, RouterEval and
  LLMRouterBench every drawn claim is `calibrated`: drawn from the live self-rating's distribution given S (correlation
  about 0.36; §1.3.3). The older `programmatic` claims, clip(S + N(0, 0.05)) (correlation about 0.99, an answer key),
  sit in rows that B and H no longer read.
- **Methods never see true skill S or the liar set.** Only the oracle and offline diagnostics read S.

## 0.3 Issues that can change what a reader concludes (with where they are explained)

Severity: **H** = can change a conclusion, **M** = comparability or labelling caveat that belongs in a caption.

| # | Issue | Figures | Sev. | Where |
|---|---|---|---|---|
| 1 | **Peer-reported halving is hidden, and only two regimes are shown.** The do-not-add filter removes every halving arm (trusted-observer halving is withdrawn by erratum 26 and never reported). Peer-reported halving, a pre-registered rival on the same report channel and budget, beats MIDIAN paired on every seed in every honest b = 3 cell of A, and loses under the cartel. A and B show only honest and the β = 0.5 low-skill cartel, the one liar regime (5 of 24 live regimes) in which MIDIAN beats it. Captions must say both. | A, B | H | §1.8.2, §4.5 #14, #17 |
| 2 | **B's non-live groups are erratum-30 reruns, with their own caveats.** All four families read them now (the leaky rows, with claims = true skill + N(0, 0.05), replay probing and routing on the same prompts, and repeated RouterEval / LLMRouterBench test prompts, are not drawn). What a caption still needs: calibrated claims are a post-hoc model of live self-ratings (i.i.d. per agent and family, fitted on live specialist populations and applied to every shape and backend, so less informative than live on the heavy_tail / bimodal shapes that replay pools); replay has 30 seeds, not 100; LLMRouterBench now runs Q = 300, not 1,000; the split-replay and RouterEval oracles are not exact ceilings. | B | M | §4.0.7 |
| 3 | **H's agent descriptions are rendered claim vectors.** On RouterEval every framework reads "Self-rated competence: <top-5 families by declared, two decimals>" and the supervisor sees only "A task of family X", never the prompt. H now uses calibrated claims (live-like, corr ≈ 0.36), so declared top-k is no longer an oracle ranking: the pre-registered TF-IDF is the best honest H bar at m = 100 / 1,000 / 5,000. Only live (E–G) has real self-descriptions and real prompts. | H | M | §5.5 |
| 4 | **The "best learned/declared router" label covers a constant-cost ANN index and two declared-only arms.** `flat_nsw_router` (an hnswlib index over flat probe means, not a published learned router) is the majority pick in 19 of the 54 bars (live 10⁴ b = 1 / 3, bernoulli 10⁷ b = 3 / 5, all six replay 10⁶ bars, RouterEval 5,000 except cartel b = 1) and 610 of the 1,002 seed-level picks. The declared-only `cluster_head_router` / `disrouter_cascade` (no probes) are the majority pick in 3, all at b = 1 (bernoulli 10⁷ both regimes, RouterEval 5,000 cartel); such bars do not change with b. | A, B | H | §2.1.3, §4.5 #2 |
| 5 | **Four arms never change with b**: declared argmax, random, `cluster_head_router`, `disrouter_cascade` (`B_INVARIANT`; CSV column `b_invariant`). | A, B | M | §4.0.5 |
| 6 | **Post-hoc choices, none marked on the figures**: MIDIAN as the headline (its pre-registered target V2-11 was a miss), the tuned warm-start bandit (n0 = 0.5, seeds 11-15), the pool edits (fixed-bonus LinUCB replaces the pre-registered one; TrueSkill only from post-probe-fix rows), the erratum-30 reruns themselves (calibrated-claim model, replay split, no-repeat, agent shuffle), hidden halving, and the regime choice. The pre-registered warm-start bandit (n0 = 5) is in every drawn pool (live and all four switched families). | A, B, E–H | H | §4.5 #17 |
| 7 | **Every pool is complete.** Every bandit pool (`linucb_honest[bonus=own]` and post-fix TrueSkill landed) and every learned pool (the last live 10⁵ kNN seeds landed 2026-09-24 11:20) has every allowed member on every scored seed. Figures carry no incompleteness marks at all; `cluster/ops/figure_status.py` reports missing slots and incomplete pools, and reports none. | A, B | M | §4.0.4, §4.3 |
| 8 | **E's bars at one n can rest on different seed counts and slightly different framework sets.** A framework counts only with the full seed count of that (n, shortlist); at 10² / 10³ the five backfill shortlists have 3 seeds per framework and the others 10. Four E bars drop one framework for partial seeds (10³ rerank honest: Magentic-One; 10⁵ MiniLM honest: CrewAI; 10⁵ dense I-comp and declared honest: ADK); H drops none. | E–H | M | §5.1.3, §5.6 D1–D2 |
| 9 | **MIDIAN overspends the build budget**: 1.03–1.07 × n·K·b (audit re-probes; max 1.070 on LLMRouterBench b = 1) while every rival spends ≤ 1.00 ×. "Same budget" is not literally true; the MIDIAN-cohort shortlist and the E–H reference line spend it too, the frameworks none. | A, B, E–H | M | §1.5, §2.3.4 |
| 10 | **At b = 1 MIDIAN verifies nothing** (no probes are left), so its b = 1 bar is MIDIAN w/o verification with a cached root; honest, it equals MIDIAN w/o defenses at b = 1 in every cell of A and B. | A, B | M | §2.3, §4.0.6 |
| 11 | **Whiskers are ±1 standard error** (~68 %), never 95 % CIs: over seeds in A / B (3 seeds at live 10⁴ / 10⁵ and RouterEval, 5 LLMRouterBench, 30 replay, 100 bernoulli), over frameworks in G. | A, B, G | M | §1.7, §4.0.5, §5.4 |
| 12 | **Live b = 3 bars average rows from several grids per seed, and those rows disagree** by up to 0.020 at n = 1,000 (old per-method probe instances in early `live_f1_n1000`, averaged into MIDIAN w/o defenses). `live_core_n100` (pre-09-02 instances, spread 0.076) was dropped from n = 100 on 2026-09-24: MIDIAN w/o defenses there is 0.777 / 0.741 (was 0.774 / 0.736). | A | M | §4.5 #5, §1.10 #2 |
| 13 | **Only TF-IDF is not deduplicated.** Every other shortlist carries `dedup: true`. At 10⁵ TF-IDF is ten clones of one agent, so G's "gain over TF-IDF" includes the dedup gain. | E–G | M | §3.4.1, §5.6 D9–D10 |
| 14 | **Dedup is not a no-op on RouterEval.** Calibrated claims take 11 values, so rendered descriptions coincide: at seed 1, 931 / 1,000 and 4,134 / 5,000 are distinct honest, 681 / 1,000, 88 / 100 and 2,551 / 5,000 under the β = 0.5 cartel. Every dedup shortlist in H (all but TF-IDF) drops 7–49 % of the pool. | H | M | §3.9 #1 |
| 15 | **RouterEval TF-IDF is not "degenerate".** It shortlists agents whose rendered top-5 claims name the family, and under the cartel liars fill it (7 of 10 on family 0 at m = 1,000, seed 1). It is the best honest H bar at m ≥ 100. | H | M | §3.9 #2–3 |
| 16 | **The reference lines in E–H are honest-only.** Hatched cartel bars are drawn against the honest MIDIAN line (10⁵: 0.8356 honest vs 0.8278 cartel). | E–H | M | §5.6 D7 |
| 17 | **"Cartel" in the figures means `liar_select = low_skill_first`.** `collude = true` is on in every grid, including the random-liar ones. | all | M | §1.10 #13 |
| 18 | **Live declarations are the model's own self-rating** (mean 0.686 vs true 0.419, correlation 0.36). On live, the description text includes an honest "Declared areas" clause that lying does not change (erratum 27), which helps every text shortlist. | E, F | M | §1.3, §3.9 #15 |
| 19 | **Supervisor picks are not bit-reproducible on the real fleet:** random replica per call, no seed sent, batched vLLM. | E–H | M | §3.9 #8 |
| 20 | **The split-replay and RouterEval oracles are not exact ceilings.** They pick by probe-row / train-prompt S and execute on task / test rows, so an arm can in principle beat them; B's replay and RouterEval bars are ratios to such an oracle. | B, H | M | §1.4.3, §1.4.4 |
| 21 | **C's learned band leaves out the constant-cost flat-NSW router**, which is still a candidate in A and B's cross-fit and the majority pick in 19 of their 54 "best learned/declared router" bars (including bernoulli 10⁷ at b = 3, C's budget); for those the reported arm does 50 comparisons per query, below MIDIAN w/o defenses at every n and below MIDIAN from 10⁵ up. D's advantage is over the frameworks, not over flat probing. | C, D | H | §6.2, §6.4 #3–4 |
| 22 | **D's framework band is a live n = 1,000 measurement applied at every n**, pooled over all β and all three population shapes; its upper edge is a latency ratio, not a count of calls. | D | M | §6.4 #5–7 |
| 23 | **D mixes sources.** Build probes at n = 10⁷ are bernoulli ledger counts priced at the live specialist probe energy (4.04 J). | D | M | §6.4 #1–2 |
| 24 | **Bars start at y = 0.2** in A, B, E, F and H, which exaggerates ratios between tall and short bars. | A, B, E, F, H | M | §4.0.5, §5.6 D13 |

Lower-severity documentation drift (stale docstrings, stale grid comments, docs/methods.md / README.md vs code) is listed in each
part's final section.

---

## 1. Foundations: the world model, backends, experimental axes and aggregation

Everything below was checked against the code in the repository root (paths are relative to that repo root unless they start with
`$RTE_DATA`, which is `$RTE_DATA`). Citations are `path:line`. Where a `.md`
document says something different from the code, the code is taken as correct and the mismatch is listed in §1.10.
Numbers quoted "from data" were computed read-only from the files named next to them on 2026-09-23 (population and skill statistics); the grid, row and erratum-30 facts were re-checked on 2026-09-24.

---

### 1.1 The world (`rte/world.py`)

#### 1.1.1 Objects

| Object | What it is | Code |
|---|---|---|
| **agents** | integer ids `0..n-1`. What an agent *is* depends on the backend (§1.4). | `rte/world.py` |
| **families** | `K` task categories, indexed `0..K-1`, with a name list `world.families`. A task always belongs to exactly one family. | `rte/world.py` |
| **true skill `S[n,K]`** | float32 in [0,1]: the probability (bernoulli), measured accuracy (live), recorded accuracy (replay, RouterEval) of agent *a* on family *f*. Supplied by the backend's `true_skill()`. **Runner-only**: used for the oracle, liar selection (`low_skill_first`) and offline diagnostics; no method can read it (§1.1.6). | `rte/world.py` |
| **declared `D[n,K]`** | what agents *claim* about themselves. `D_honest = backend.declared(declared_source)`, then liars' rows are overwritten by `apply_lying`. Read-only copy `D_view` is what methods see. | `rte/world.py` |
| **liars** | bool mask `[n]`, chosen once per world by `select_liars` (§1.3). Runner-only. | `rte/world.py` |
| **demand** | probability vector over families used to draw the task stream: `uniform` (1/K) or `skewed` (Zipf(1)). Every grid behind figures A–H uses `uniform`. | `rte/world.py` |
| **ledger** | six counters `probes, reports, messages, hops, comparisons, tasks`; each has exactly one increment method. | `rte/ledger.py` |

#### 1.1.2 Construction and seeding

`World(n, K, dist, beta, liar_select, collude, seed, backend, lie_mode, declared_source, demand, backend_kwargs)`
(`rte/world.py`). All randomness goes through `stable_seed_32(*parts)` (blake2b of the `repr` of the parts,
masked to 32 bits; `rte/stable_hash.py`), so it is identical across processes. The streams:

| RNG | seeded by | used for |
|---|---|---|
| world rng | `(seed, "world", n, K, dist, backend)` `rte/world.py` | handed to the backend: bernoulli skill draws / calibrated row resampling, replay profile draws. The llm and routereval backends ignore it. |
| liar rng | `(seed, "liars")` `rte/world.py` | `liar_select="random"` only |
| honest-declaration noise | `(seed, "declared")` `rte/backends/__init__.py`, `rte/backends/llm.py` | `D = clip(S + N(0, 0.05))` (programmatic) |
| calibrated declarations | `(seed, "declared_calibrated")` `rte/backends/__init__.py` | non-live `declared_source: calibrated` only (erratum 30) |
| RouterEval agent order | `(seed, "agent_order")` `rte/backends/routereval.py` (`shuffle: true` only) | per-seed permutation of the pool (erratum 30) |
| task stream | `(seed, "stream", K, demand)` `rte/world.py` | which family each task is from |
| task instance | `(seed, "inst", i, f)` `rte/world.py` | the concrete instance of task *i* |
| probe salt | `(seed, "probes")` `rte/world.py` | index-seeded probe instances (§1.5) |
| method view rng | `(seed, "view", sorted(needs))` `rte/world.py` | a method's own randomness (two methods with the same `needs` get the same stream) |
| llm population | `(seed, "profiles", n, K, dist)` `rte/backends/population.py` | live agent profiles |

Consequences worth knowing:
- The **task stream does not depend on n, dist, beta, liar_select or declared_source** — only on (seed, K, demand, Q),
  except under `no_repeat`, where it also depends on the backend's per-family test-pool sizes (§1.1.3).
  For a given seed and K, the live n = 100, 1,000, 10^4 and 10^5 cells route the *same* sequence of families and the same
  instance seeds. `rng.choice(size=Q)` is prefix-consistent (verified: the first 300 draws of a Q = 1000 stream equal a
  Q = 300 stream), so Q = 300 cells see the first 300 tasks of the Q = 1000 stream.
- The **liar set, lies, probes and tasks do not depend on the order in which methods run** (§1.1.5).
- Because `select_liars` returns an empty mask for `round(beta*n) = 0` (`rte/world.py`) and no other RNG uses
  `liar_select`, **β = 0 cells with `random` and `low_skill_first` are the same world**. The grid comment at
  `configs/grids/` states they were verified bit-identical over 690 cells.

#### 1.1.3 Tasks (`rte/world.py`)

`Task(id, family, instance)`. With `no_repeat` (RouterEval / LLMRouterBench erratum-30 grids) `World._tasks_no_repeat`
(`rte/world.py`) instead visits each (family, test prompt) at most once: families are drawn by the demand vector
renormalised over the families with prompts left, each family's prompts in a per-stream random order, and the task
instance is the prompt's index in the family's test rows; Q above the test pool raises. Otherwise `World.tasks(Q)` draws `Q` family indices from `demand` and gives task *i* the instance
seed `stable_seed_32(seed, "inst", i, f)`. The backend regenerates the concrete problem from `(family, instance)`
(live: a Reasoning-Gym instance; replay / RouterEval: a recorded prompt row, see §1.4). **Methods receive the whole
`Task` in `fetch(task)`**, so they always know the task's family index (and its instance seed, and can read its text,
§1.1.6). Routing is therefore "pick an agent for a task of known family f".

#### 1.1.4 `execute` and the oracle (`rte/world.py`)

- `execute(a, task)` charges `ledger.tasks += 1` and returns the backend's 0/1 outcome of agent *a* on that task
  (plus the churn epoch rule, irrelevant for A–H because no grid behind A–H has churn — all resolved `churn=None`).
  **Liars execute at their true skill**; lying only touches declarations and reports (`rte/world.py`).
- `oracle(task) = argmax_a S[a, task.family]` (numpy `argmax`: ties go to the lowest agent id). The oracle row is a
  *realised* success on the same stream (`rte/run.py`), not `max S`. It is computed once per unit, before any
  method. Because S is an expectation (bernoulli), a measurement on a different instance set (live) or a train-split
  mean (RouterEval), a method can occasionally beat the oracle on a finite stream (negative regret).

#### 1.1.5 Pairing: one world, one stream, every method (`rte/run.py`, `rte/world.py`)

`run_unit(cell, seed, specs)` builds **one** `World` and **one** stream (`world.tasks(Q)`), runs the oracle line, then
for each method: `world.reset()` (zero the ledger, forget reporters' memories, set every probe index back to 0, undo
churn), `m.build(view, Budget(b))`, then for each task `m.fetch(task)` → `execute` → `m.observe(task, a, outcome)`.
So within a (cell, seed) unit every method faces the same agents, the same liars, the same declarations, the same task
sequence, and — because probe index k of (a, f) always maps to the same instance (§1.5) — the same probe outcomes.
Differences between methods in a unit are therefore paired. Caveats on pairing *across* units/grids are in §1.10
(RouterEval family-order and live drift items).

#### 1.1.6 What a method can see (`rte/world.py`, `rte/methods/base.py`)

A method gets a `View` constructed from its declared `needs ⊆ {"declared", "probe", "reports", "bus"}`
(`rte/world.py`). Any access outside `needs` raises `AccessError`; any attribute not defined on `View` raises
("S and liars are never exposed", `rte/world.py`).

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
`rte.backends.llm.current_backend()` (`rte/methods/frameworks/_common.py`) to read the agents'
self-descriptions, family descriptions and task text — the only out-of-View access in `rte/methods/`.

#### 1.1.7 Ledger and what "communication" means (`rte/ledger.py`, `rte/run.py`)

`probes` and `tasks` are charged by the world on every probe/execution, `reports` by the report channel, and
`messages / hops / comparisons` by methods themselves. "Total communication" is defined as
`probes + reports + messages + tasks` (`rte/run.py`); `hops` and `comparisons` are recorded but not summed into it.

#### 1.1.8 Churn (brief; not used by A–H)

`World.churn(frac)` replaces `round(frac*n)` agents in place, re-draws their liar status at rate β, recomputes D, and
the first task routed to a replaced, never-re-observed agent scores 0 (`rte/world.py`).
`reset()` restores the original population for the next method. Only `churn_n1000*` grids use it.

---

### 1.2 Skill distributions (`dist`)

The `dist` axis names the *population shape*. Its meaning is backend-specific:

#### 1.2.1 bernoulli, uncalibrated (`rte/world.py`)

| dist | S draw |
|---|---|
| `specialist` | each agent has exactly 3 random "good" families: U(0.70, 0.95) there, U(0.05, 0.30) elsewhere (`_profiles.pick_k_per_agent`, `rte/backends/_profiles.py`) |
| `heavy_tail` | `0.05 + 0.90 * Beta(0.5, 3)` i.i.d. per cell |
| `bimodal` | 20 % of agents U(0.75, 0.95) everywhere, 80 % U(0.20, 0.40) everywhere |
| `correlated` | 4 family groups (`f % 4`): one U(0.15, 0.90) per (agent, group) + N(0, 0.05), clipped |
| `iid_uniform` | U(0.20, 0.90) i.i.d. |

**Every bernoulli grid behind the figures passes `calibrate_from`, which overrides `dist` entirely**: each agent's
S-row is a uniformly resampled (with replacement) row of the measured live matrix
`$RTE_DATA/populations/specialist_n1000_K16_seed1/S.npy` (`rte/backends/bernoulli.py`). The `dist` column then
still reads `specialist` but is only a label. That source matrix (1000 × 16) has 887 distinct rows, mean 0.419 and a
mean-over-families of the per-family max of 0.845 (its `summary.json`), which is why the bernoulli oracle sits at
≈0.845 for n ≥ 10^3 (`bernoulli_scale_v5/matrix_success.csv`: 0.845–0.846).

#### 1.2.2 live / llm (`rte/backends/population.py`, `configs/models.yaml`)

The shape is realised by how **profiles** `{model, specialty, tool}` are drawn; S is then *measured* (§1.4.1). The
ladder (`configs/models.yaml`, ascending size): Qwen2.5-0.5B, Qwen2.5-1.5B, Gemma-2-2B, Qwen2.5-3B, Qwen2.5-7B,
Gemma-2-9B, Qwen2.5-14B. Bands (`configs/models.yaml`, `population.py`): small = ≤ 1.5B {0.5B, 1.5B};
big = ≥ 7B {7B, 9B, 14B}; "large" (measured with 60 instead of 200 probes) = ≥ 9B {Gemma-9B, Qwen-14B}; the `python`
tool only for models ≥ 3B (`population.py`).

| dist | profile draw |
|---|---|
| `specialist` | model uniform over all 7; 3 random specialty families |
| `heavy_tail` | 10 %: a random big model, specialty = all families; 90 %: a random small model, specialty = none |
| `bimodal` | 20 %: the smallest big model (Qwen2.5-7B), specialty = all; 80 %: the smallest small model (Qwen2.5-0.5B), specialty = none |
| `correlated` | model uniform; each of 4 family groups is a specialty with prob 0.5 |
| `iid_uniform` | model uniform; each family a specialty with prob 0.5 |

On a non-specialty ("handicapped") family the agent's prompt has no family description, no worked example and no tool
(`rte/backends/prompts.py`, `population.py`). An agent's behaviour on family f is fully determined by
its **signature** `(model, handicapped?, tool, max_tokens)`: at most 7 × 2 = 14 distinct behaviours per family
whatever n is. Measured on `seed1`, n = 1000 populations: specialist 887 distinct S-rows (mean S 0.419),
heavy_tail 5 distinct rows (mean 0.261), bimodal 2 distinct rows (mean 0.237).

#### 1.2.3 replay (`rte/backends/replay.py`; docs/archive/DEVIATIONS.md 2026-09-02)

An agent = one of the 11 RouterBench models + a per-category mask; a masked category executes as the category's
*weakest* model. Models are ranked by mean accuracy over the K used categories; "strong half" = top 6, "weak half" =
bottom 5 (`half = (M+1)//2`, `replay.py`).

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

Here `dist` is not a shape but the **name of the pool**: `pool = pool or dist` (`routereval.py`). For
`dataset: mmlu` it selects RouterEval's hard-setting pool config `strong_to_weak | all_strong | all_weak`
(`routereval.py`); for `leaderboard_mmlu` and `llmrouterbench` the value `all` is only a label (the pool is fixed).
The population is the real pool itself and is **the same for every seed** (§1.6.4).

---

### 1.3 Liars

#### 1.3.1 How many and who (`rte/world.py`)

`m = round(beta * n)` liars.
- `liar_select = random`: `m` agents uniformly without replacement (rng `(seed,"liars")`).
- `liar_select = low_skill_first`: the `m` agents with the **lowest mean true skill across families**
  (`argsort(S.mean(1), kind="stable")[:m]`; ties → lowest ids). Deterministic given S. At β = 0.5 on the live
  specialist n = 1000 seed-1 population the liar half has mean S 0.274 vs 0.565 for the honest half.

#### 1.3.2 The declared-channel lie (`rte/world.py`)

`apply_lying(D_honest, liars, lie_mode)`; honest rows are untouched.
- `inflate` (**every grid behind A–H**): `D[liar] = clip(D_honest[liar] + 0.4, 0, 1)` (`DELTA_INFLATE = 0.4`,
  `rte/world.py`). Preserves the liar's ranking of its own families and adds 0.4.
- `max`: `D[liar] = 1` everywhere (only the `lie_max_*` grids, 2026-09-23).
- `squat`: `D[liar, top-3-demand families] = 1` (unused by the figure grids).

#### 1.3.3 What honest declarations look like (`declared_source`)

- `programmatic`: `clip(S + N(0, 0.05))` — an honest, slightly noisy copy of true skill
  (`rte/backends/__init__.py`; llm version `rte/backends/llm.py`). bernoulli, replay, RouterEval and
  LLMRouterBench use this for both `programmatic` and `self_described` (`declared_for`, `rte/backends/__init__.py`; called at
  `bernoulli.py`, `replay.py`, `routereval.py`). On these backends it is an answer key: corr(S, D)
  about 0.99 (erratum 30). The older bernoulli / replay / RouterEval / LLMRouterBench grids (`bernoulli_scale_v5`,
  `replay_scale_v5`, `routereval_mmlu*`, `llmrouterbench_pool`, the `va_b_` / `rivals_b_` / `pool_fill_` twins, the old
  `fw_routereval_*` / `re_sl_*` framework grids) use it; A–H no longer draw a claim-reading arm from those rows.
- `calibrated` (non-live only, erratum 30; the llm backend rejects it): D is drawn i.i.d. per (agent, family) from
  P(D | decile of S), the live self-rating's empirical distribution over its 11 values (`CAL_VALUES`, `CAL_P`,
  `rte/backends/__init__.py`, fitted by `scripts/data/fit_declared_calibration.py` on the 40 live specialist n = 100 /
  1,000 populations). It reproduces live mean 0.688 and corr 0.354–0.357 (CHANGES §8f). The draws ignore the per-agent
  component of live self-ratings (25 % of their residual variance) and the specialist fit is applied to every shape, so
  on heavy_tail / bimodal the claims are less informative than live (DEVIATIONS "Erratum 30"). Used by the `*_cal`,
  `*_split_cal` and `*_norep_cal` grids, which are what B's four non-live groups and H draw.
- `self_described` (llm backend only; all live cells in A, B, E–H): each signature's own model is asked to rate its
  competence on the family 0–1 given the family description and an example problem (`prompts.py`), parsed by
  `parse_rating` (`prompts.py`; unparseable → 0.5). Cached per population in `D_self_described.npy`.
  It is coarse and over-confident: on specialist n = 1000 seed 1 it takes 11 distinct values
  {0, .2, .5, .6, .7, .75, .8, .85, .9, .95, 1}, mean 0.686 vs mean S 0.419, corr(S, D) = 0.36
  (heavy_tail: mean 0.526 vs S 0.261; bimodal 0.581 vs 0.237). After `inflate` a specialist liar's mean D is ≈ 0.882.
- Self-description *text* (llm): one ≤ 70-word paragraph per agent written by its own model, plus the appended clause
  "Declared areas: <its TRUE specialty list>" (`rte/backends/llm.py`). It is generated once per population
  (`descriptions.json`, path has no β) and **is not touched by the lie** — erratum 27 (§1.8).

#### 1.3.4 The report channel and collusion (`rte/world.py`)

Decentralised methods learn through reports: a method passes `(reporter j, agent a, outcome)` and gets back the value
j reports. If `collude` is **False**, or j is honest, the true outcome comes back. If `collude` is True and j is a liar:
1. liar j reports **1** for any liar a (vouching);
2. liar j reports **0** for the top `ceil(20 %)` of *honest* agents ranked by j's observed mean (in `report_many`, the
   mean within that batch per reporter; in the scalar path, j's cumulative observations);
3. otherwise the true outcome.

Every grid behind A–H has `collude = true` (the default, `configs/grids/`) — **random liars collude too**. The
only thing that distinguishes the regimes called "cartel" is `liar_select`.

#### 1.3.5 Definition: "β = 0.5 cartel"

In code and in every figure script, a *cartel* is `liar_select = low_skill_first` (`fw_variant_numbers.regime`,
`scripts/analysis/fw_variant_numbers.py`). Concretely, "β = 0.5 cartel" =
`beta = 0.5, liar_select = low_skill_first, collude = true, lie_mode = inflate`: the half of the population with the
lowest mean true skill (i) declares `D + 0.4` (clipped) and (ii) as reporters vouches 1 for each other and reports 0 on
the best-observed 20 % of honest agents; (iii) executes tasks honestly at its true skill. Regime labels used downstream:
`beta0` (no liars, one cell), `beta{β}_random`, `beta{β}_cartel`, and `cartel` ≡ β = 0.5 low_skill_first
(`fw_variant_numbers.py`; `bar_figs.py`).

#### 1.3.6 `lie_text` (framework parameter, erratum 27)

Not a world axis: a framework-adapter flag (`rte/methods/frameworks/_common.py`). When on, the
trailing "Declared areas:" clause of *every* agent's description is re-derived from `view.declared` (top 3 families, or
all above `claim_threshold`), so liars' text claims match their inflated D. The LLM prose is unchanged (partial text
lie). It appears only in `fw_live_n100k_lietext*` grids, which `shortlist_figs.GRIDS` (`shortlist_figs.py`) explicitly excludes.

---

### 1.4 Backends

A backend supplies `n, K, families, true_skill(), declared(source), execute(a, task), execute_many(agents, fams,
inst), stats()` (+ churn hooks) (`rte/backends/__init__.py`); `make()` dispatches on
`bernoulli | replay | llm | routereval` (`rte/backends/__init__.py`). The grid axis value for the live backend
is `llm`; figure scripts call that family "live".

#### 1.4.1 live (`backend: llm`) — `rte/backends/llm.py`, `prompts.py`, `families.py`, `tools.py`, `rte/llm_client.py`

- **Agent**: a profile (model from the 7-model ladder, specialty set, tool) — §1.2.2. Real vLLM-served models,
  temperature 0, `seed=0` (`rte/llm_client.py`). n = 10^5 means 10^5 agent ids mapped onto ≤ 14 behaviours per
  family and (specialist) at most 7 × C(16,3) = 3,920 distinct self-description prompts (erratum 25's count).
- **Families / tasks**: K = 16 Reasoning-Gym generators (`families.py`: basic_arithmetic, chain_sum,
  letter_counting, syllogism, family_relationships, gcd, lcm, prime_factorization, number_sorting, simple_equations,
  count_bits, number_format, time_intervals, word_sequence_reversal, binary_alternation, calendar_arithmetic);
  basic_arithmetic and chain_sum use eased settings (`families.py`). A task is one generated problem, the
  instance seed being the dataset seed.
- **Success**: the agent's answer (last `<answer>…</answer>`, else last line; one optional python-tool round for tool
  holders) scored by the Reasoning-Gym verifier, `correct = score >= 0.99` (`families.py`, `llm.py`).
- **S is measured, per signature**, on a fixed project-wide instance set `stable_seed_32("measure", fam, r)`,
  r < 200 (60 for Gemma-9B / Qwen-14B), not on task instances (`llm.py`). Cached as
  `$RTE_DATA/populations/<dist>_n<n>_K<K>_seed<seed>/S.npy`. Available populations (dir count): specialist n = 100 and
  1,000 seeds 1–20, 10^4 seeds 1–5, 10^5 seeds 1–3; heavy_tail / bimodal n = 100, 1,000 seeds 1–20 and 10^4 seeds 1–3;
  correlated / iid_uniform n = 1,000 seeds 1–5.
- **LLM cache / memo** (`rte/llm_client.py`): every request is keyed by blake2b of
  `(model, messages, max_tokens)`; answers are stored in per-process SQLite shards under `$RTE_DATA/cache/` and every
  process reads all shards. Agents sharing a signature send byte-identical prompts, so they get identical answers on
  the same instance: **same-signature agents are exact clones**. This is what makes n = 10^5 affordable (the probe
  build for a (n, dist, seed) costs `n·K·b` lookups, not new generations, once warmed) and is also the root of
  erratum 25.
- **Supervisor**: framework arms (not agents) call a supervisor LLM, `SUPERVISOR = "Qwen/Qwen2.5-7B-Instruct"`
  (`rte/methods/frameworks/_common.py`), over a top-k shortlist; covered in the frameworks part.

#### 1.4.2 bernoulli — `rte/backends/bernoulli.py`

Synthetic: outcome ~ Bernoulli(S[a, f]) from a hash of (seed, agent, family, instance) (`bernoulli.py`). Agents
are rows of S; with `calibrate_from` (all figure grids) each row is resampled from the live specialist n = 1000 seed-1
S (§1.2.1), so a bernoulli agent is "a clone of a random live specialist agent's skill profile with i.i.d. Bernoulli
outcomes". Declared = `clip(S + N(0, 0.05))` for `programmatic` / `self_described`, the live-like draw for `calibrated`. n goes up to 10^7 (`bernoulli_scale_v5`, K = 16).
Note the module docstring's own policy: "Never a headline number" (`bernoulli.py`).

#### 1.4.3 replay (RouterBench) — `rte/backends/replay.py`

Data: `$RTE_DATA/data/routerbench_cells.npz`, built by `scripts/data/02_download_routerbench.py` from RouterBench's 0-shot
pickle: category = `eval_name`, categories with ≥ 60 prompts kept (67 survive), score binarised at ≥ 0.5; 11 models,
36,093 prompts. With K = 64 the 64 largest categories are used (`replay.py`). An agent = (model, mask) (§1.2.3);
`execute` = the recorded outcome of the agent's model (or the category's weakest model if masked) on prompt
`instance % n_prompts[f]` (`replay.py`). By default probes and tasks draw from the **same** prompt pool and S = the
(model or weakest-model) accuracy over the whole category. n up to 10^6 in the figures (`replay_scale_v5`).
Declared = `clip(S + N(0, 0.05))` (or `calibrated`).
**`split: true` (erratum 30, `replay.py`)**: each category's prompts are permuted once (rng 0) and cut 70 / 30
into probe rows and task rows. Probes (`execute_many`), S, the weakest-model mask and the model ranking read only probe
rows; routed tasks (`execute`) read only task rows. The oracle therefore picks by probe-row S and executes on task
rows, so it is not an exact ceiling (a method can beat it; none does in B today). Only `replay_1e6_split_cal` sets it, and
B's replay group reads that grid.

#### 1.4.4 RouterEval (`dataset: mmlu`, `leaderboard_mmlu`) and LLMRouterBench (`dataset: llmrouterbench`) — `rte/backends/routereval.py`

All three are recorded per-prompt 0/1 scores of **real, distinct LLMs**; n must equal the pool size (asserted); no
resampling of agents.

| dataset | pool (n) | families (K) | train / test split | S |
|---|---|---|---|---|
| `mmlu` | RouterEval "hard" pools m ∈ {10, 100, 1000} × config {strong_to_weak, all_strong, all_weak} (Open-LLM-Leaderboard models) | the K = 16 MMLU subjects with the most train prompts, parsed from the prompt text (`routereval.py`) | RouterEval's own (11,215 train / 1,430 test prompts; 6,253 / 790 fall in the 16 subjects) | mean train score |
| `leaderboard_mmlu` | all 5,000 `leaderboard_old` LLMs, synthetic names `llm0..llm4999` | 16 largest `harness_hendrycksTest_*` subjects | fixed 80/20 per subject, rng 0 (`routereval.py`) | mean train score |
| `llmrouterbench` | 20 models (7–9B open models, e.g. Qwen3-8B, DeepSeek-R1-0528-Qwen3-8B, gemma-2-9b-it, …) | 15 datasets (aime, math500, mathbench, humaneval, mbpp, livecodebench, bbh, korbench, kandk, mmlupro, gpqa, finqa, medqa, emorynlp, meld) | global 70/30, rng 0; score binarised ≥ 0.5 (`routereval.py`) | mean train score |

**Probes read TRAIN prompts, tasks read TEST prompts** of the family (`routereval.py`), so the oracle (argmax of
train accuracy) is not guaranteed to be the test-best agent. By default a task reads test row `instance % pool`, so
test prompts repeat within a stream (repeat share 0.48 on mmlu, 0.20 on LLMRouterBench, CHANGES §8f); `no_repeat: true`
removes that (§1.1.3). `shuffle: true` permutes the agents per seed (`stable_seed_32(seed, "agent_order")`): the stored
pools run weak → strong, so without it every lowest-index tie-break picks the weakest tied agent. Every erratum-30
RouterEval / LLMRouterBench grid sets both (unshuffled rows that landed first were moved to
`$RTE_DATA/results/_quarantine_unshuffled_2026-09-23` and are not read), and those are the only RouterEval / LLMRouterBench rows
B and H draw. Declared = `clip(S + N(0, 0.05))` (or `calibrated`); no self-descriptions
exist (frameworks get the declaration vector rendered as text, `_common.py`). Churn is a no-op.
From the backend (seed 1): mmlu strong_to_weak mean S 0.551 / 0.545 / 0.545 at m = 10 / 100 / 1000, per-family max
mean 0.876 / 0.799 / 0.890; leaderboard 5,000: mean 0.542, per-family max mean 0.917; LLMRouterBench: mean 0.481,
per-family max mean 0.725.

---

### 1.5 The probe budget `b`

- **Formula**: `Budget(b).total_probes(n, K) = n · K · b` (`rte/budget.py`) — "b probes per (agent, family)" on
  average. Every probing method receives the same `Budget(b)`; `b` is the grid's `b` axis (default 3,
  `configs/grids/`). Examples: live 10^5 at b = 3 → 4.8 M probes; bernoulli 10^7 at b = 1 → 1.6 × 10^8.
- **What a probe is**: executing agent *a* on a *fresh* instance of family *f* and returning 0/1, charged
  `ledger.probes += 1` (`rte/world.py`). The k-th probe of (a, f) uses instance seed
  `probe_seed(salt, a, f, k)` (`rte/world.py`); `reset()` puts every k back to 0 per method
  (`rte/world.py`), so **every method sees the same outcome for the k-th probe of (a, f)** (and on live they share
  memoised generations). `probe_text` also returns the instance seeds (so a router can read the probe prompt via
  `view.text(f, inst, probe=True)`); `probe_at(a, f, k)` re-runs a past instance for audits, charged as a probe, index
  untouched (`rte/world.py`). Live probe instances, measurement instances and task instances are three
  different seed families; RouterEval probes are train prompts.
- **How it is checked**: only in `run_method`, after `build`: if `build_probes > n·K·b` a `[WARNING]` is **logged**;
  nothing is enforced and the row is written (`rte/run.py`). Run-time probes (during `fetch/observe`) are not
  checked at all; they appear in `probes_per_task`. From data (build_probes / nKb): flat probe argmax, KNN/MLP routers,
  LinUCB, warm-start bandit, MIDIAN w/o defenses and MIDIAN w/o audits spend exactly 1.000; peer/trusted sequential halving 0.833 on
  LLMRouterBench; **MIDIAN w/o verification (and the since-withdrawn audited successive-halving variant) ≈ 1.04–1.06; MIDIAN 1.03–1.07: per-cell means 1.048–1.050 / 1.029–1.033 / 1.038–1.041 at b = 1 / 3 / 5 on live, RouterEval 5k and LLMRouterBench, maximum 1.070 (LLMRouterBench, b = 1; protocol audit F3), but a mean of 0.92–0.96 at b = 3 on the scale sweeps and RouterEval-mmlu (max ≈ 1.04)**. On the erratum-30 rows B and H draw: `replay_1e6_split_cal` 1.050 / 1.033 / 1.040, `routereval5k_norep_cal` 1.049 / 1.033 / 1.039, `llmrouterbench_norep_cal` 1.050 / 1.033 / 1.041 (max 1.070 at b = 1), and `routereval_mmlu_norep_cal` (H's m ≤ 1,000 line, b = 3) a mean of 0.92 (0.69–1.03, small pools) (the 5 % audit re-probes, documented in
  the old `rte/methods/midian_a.py` as "≤ 1.05×"; observed up to 1.0625 on `routereval_mmlu`); `verify_on_claim` spends 0 at
  build and ≈ 2.4 probes per task at run time on live 10^5 (range 1.4–4.1 across cells); declared argmax and random spend 0.
- **What changing b does**: more probes per cell → sharper estimates for probe users; declaration-only arms are
  unaffected (the condensed figures draw declared argmax / random once, at b = 3, `BUDGETLESS`, `scripts/figures/condensed_figs.py`; the four never-probing arms are `B_INVARIANT`).
  For the verified MIDIAN variants (MIDIAN and MIDIAN w/o audits) the build splits b into `b0 = b − 1` level-0 probes and verification probes
  `e = (b − b0)·n / C` (`rte/methods/midian.py`); at **b = 1, b0 = 1 and e = 0, so verification is unfunded:
  MIDIAN w/o audits ≡ MIDIAN w/o defenses and MIDIAN ≡ MIDIAN w/o verification** (also stated at `configs/grids/`). The condensed A/B
  b = 1 bars inherit this.
- Budget-less arms still carry a `b` value in their row and a distinct row id per b.

---

### 1.6 The experimental unit and grid mechanics (`rte/run.py`, `configs/grids/`)

#### 1.6.1 Cell, unit, row

- **CELL axes** (`rte/run.py`): `backend, n, K, dist, beta, liar_select, collude, declared_source, lie_mode,
  demand, b, Q`. A **cell** is one point of their Cartesian product (`rte/run.py`); `backend_kwargs` (e.g.
  `calibrate_from`, `dataset`) and `churn` ride along.
- A **unit** = (cell, seed): one `World`, one stream, the oracle line, then every requested method (§1.1.5)
  (`rte/run.py`). A method that raises is logged and skipped; the unit continues.
- **row_id** = blake2b-128 of the CELL values + backend_kwargs + method + params + seed (+ churn) (`rte/run.py`).
  It does not include the grid name; results are separated by directory `$RTE_DATA/results/<grid>/`. Each row is
  written atomically as `rows.d/<rid>.json` (`rte/run.py`); a rerun of the same unit overwrites the same files.
- **Resume**: a (cell, seed, method) is skipped if its rid is in `rows.d` or `rows.csv` (`rte/run.py`).
- **Consolidation** (`rte/run.py`): `rows.d/*.json` are merged into `rows.csv` (additive, dedup on `rid`,
  keep last), optional prune; skipped if `.merge_owner` exists unless forced. Note `rte.analyze.load` calls
  `consolidate` (`rte/analyze.py`), so figure scripts that use it rewrite `rows.csv`;
  `fw_variant_numbers.load` (`scripts/analysis/fw_variant_numbers.py`) instead reads `rows.d` + `rows.csv` directly and dedups on `rid` and then on
  `(n, b, dist, beta, liar_select, seed, method, params)` (`fw_variant_numbers.py`).

#### 1.6.2 Grid resolution

`blocks(cfg, grid)` (`rte/run.py`): `defaults` < `mirror_of` source (chains allowed) < the grid's own keys <
each `blocks:` entry. Defaults (`configs/grids/`): backend bernoulli, K 16, liar_select random, collude true,
declared_source programmatic, lie_mode inflate, demand uniform, b 3, Q 1000, seeds 1–5, methods `all` + six paired
variants. `methods: all` = every method file except LLM-only ones off the llm backend (`rte/run.py`);
`allow_llm_methods: true` lets framework arms run on routereval. Seeds spec `"1-10"` → `[1..10]` (`rte/run.py`).

#### 1.6.3 Resolved grids behind figures A–H (exact values from `rte.run.blocks`)

All rows: `K = 16` unless noted, `collude = true`, `lie_mode = inflate`, `demand = uniform`, no churn.

| grid | backend | n | dist / pool | β | liar_select | decl. source | b | Q | seeds |
|---|---|---|---|---|---|---|---|---|---|
| `live_core_n100` | llm | 100 | spec, heavy, bimodal | 0, .1, .25, .5 | random, lsf | prog, self | 3 | 1000 | 1–5 |
| `live_f1_n1000` | llm | 1000 | spec, heavy, bimodal | 0, .1, .25, .5 | random, lsf | prog, self | 3 | 1000 | 1–5 |
| `fw_live_n100` / `fw_live_n1000` | llm | 100 / 1000 | spec, heavy, bimodal | 0, .1, .25, .5 | random | self | 3 | 1000 | 1–10 |
| `fw_live_n100_lowskill` / `_n1000_lowskill` | llm | 100 / 1000 | spec, heavy, bimodal | .5 | lsf | self | 3 | 1000 | 1–10 |
| `learned_n100`, `variants_f1`, `learned_f1` | llm | 100 / 1000 / 1000 | spec, heavy, bimodal | 0, .1, .25, .5 | random, lsf | self | 3 | 1000 | 1–10 |
| `learned_n10k` | llm | 10^4 | spec | 0, .25, .5 | random, lsf | self | 3 | 300 | 1–3 |
| `live_n10k_v2` | llm | 10^4 | spec | 0, .25 | random | self | 3 | 300 | 1–3 |
| `fw_live_n10k_cartel`, `live_n10k_cartel_random` | llm | 10^4 | spec | .5 | lsf | self | 3 | 300 | 1–3 |
| `learned_n10k_beta01` | llm | 10^4 | spec | .1 | random, lsf | self | 3 | 300 | 1–3 |
| `live_n100k`, `live_n100k_fill` | llm | 10^5 | spec | 0, .25, .5 | random, lsf | self | 3 | 300 | 1–3 |
| `live_n100k_beta01` | llm | 10^5 | spec | .1 | random, lsf | self | 3 | 300 | 1–3 |
| `va_b_n100` / `va_b_n1000` | llm | 100 / 1000 | spec | 0, .5 | lsf | self | **1, 5** | 1000 | 1–10 |
| `va_b_n10k` / `va_b_n100k` | llm | 10^4 / 10^5 | spec | 0, .5 | lsf | self | **1, 5** | 300 | 1–3 |
| `rivals_b_n100/n1000/n10k/n100k` | as their `va_b_*` twin | | | | | | 1, 5 | | |
| `routereval_mmlu` | routereval, `mmlu` | 10, 100, 1000 | s2w, all_strong, all_weak | 0, .25, .5 | random, lsf | prog | 3 | 1000 | 1–5 |
| `routereval_mmlu5k` | routereval, `leaderboard_mmlu` | 5000 | all | 0, .25, .5 | random, lsf | prog | 3 | 300 | 1–3 |
| `va_b_routereval5k`, `rivals_b_routereval5k` | routereval, `leaderboard_mmlu` | 5000 | all | 0, .5 | lsf | prog | 1, 5 | 300 | 1–3 |
| `llmrouterbench_pool` | routereval, `llmrouterbench` | 20 | all | 0, .25, .5 | random, lsf | prog | 3 | 1000 | 1–5 (K = 15) |
| `va_b_llmrouterbench`, `rivals_b_llmrouterbench` | same | 20 | all | 0, .5 | lsf | prog | 1, 5 | 1000 | 1–5 (K = 15) |
| `bernoulli_scale_v5` | bernoulli, calibrated | 10, 100, 1000, 10^4 / 10^5 / 10^6 / 10^7 | spec (label) | 0; .25, .5 | random; random, lsf | prog | 3 / 3 / 1, 3 / 1, 3 | 1000 | 1–1000 / 1–500 / 1–200 / 1–100 |
| `va_b_bernoulli_1e7`, `rivals_b_bernoulli_1e7` | bernoulli, calibrated | 10^7 | spec | 0, .5 | lsf | prog | **5** | 1000 | 1–100 |
| `replay_scale_v5` | replay | 10–10^4 / 10^5 / 10^6 | spec, heavy, bimodal | 0; .25, .5 | random; random, lsf | prog | 3 / 3 / 1, 3 | 1000 | 1–1000 / 1–200 / 1–100 (K = 64) |
| `va_b_replay_1e6`, `rivals_b_replay_1e6` | replay | 10^6 | spec, heavy, bimodal | 0, .5 | lsf | prog | **5** | 1000 | 1–100 (K = 64) |
| `fw_routereval_small` / `_1k` / `_5k` | routereval | 10, 100 / 1000 / 5000 | 3 configs / 3 configs / all | 0, .5 | random, lsf | prog | 3 | 1000 / 1000 / 300 | 1–5 / 1–5 / 1–3 |
| `tuned_wsb_n100/n1000/n10k/n100k` | as their `va_b_*` twin | | | 0, .5 | lsf | self | 3 | | |
| `pool_fill_n100` / `_n10k` / `_n100k` | llm | 100 / 10^4 / 10^5 | spec | 0, .5 | lsf | self | 1, 3, 5 | 1000 / 300 / 300 | 1–10 / 1–3 / 1–3 |
| `pool_fill_routereval5k`, `pool_fill_llmrouterbench` | as their `va_b_*` twin | 5000 / 20 | all | 0, .5 | lsf | prog | 1, 3, 5 | 300 / 1000 | 1–3 / 1–5 |
| `pool_fill_bernoulli_1e7`, `pool_fill_replay_1e6` | bernoulli / replay | 10^7 / 10^6 | spec / 3 shapes | 0, .5 | lsf | prog | 1, 3, 5 | 1000 | 1–100 |
| `pool_seeds_n1000` | llm | 1000 | spec | 0, .5 | lsf | self | 3 | 1000 | **6–10** |
| `linucb_fix_n100/n1000/n10k/n100k`, `_routereval5k`, `_llmrouterbench` | as their `va_b_*` twin | | | 0, .5 | lsf | | 1, 3, 5 | | |
| `linucb_fix_bernoulli_1e7` | bernoulli, calibrated | 10^7 | spec | 0, .5 | lsf | prog | 1, 3, 5 | 1000 | 1–100 |
| `trueskill_fix_n100/n1000/n10k` | as their `va_b_*` twin | | | 0, .5 | lsf | self | 1, 3, 5 | | |
| `bernoulli_1e7_cal` | bernoulli, calibrated | 10^7 | spec | 0, .5 | lsf | **calibrated** | 1, 3, 5 | 1000 | 1–100 (claim readers only) |
| `replay_1e6_split_cal` | replay, `split: true` | 10^6 | spec, heavy, bimodal | 0, .5 | lsf | **calibrated** | 1, 3, 5 | 1000 | 1–30 |
| `routereval5k_norep_cal` | routereval, `leaderboard_mmlu`, `no_repeat`, `shuffle` | 5000 | all | 0, .5 | lsf | **calibrated** | 1, 3, 5 | 300 | 1–3 |
| `llmrouterbench_norep_cal` | routereval, `llmrouterbench`, `no_repeat`, `shuffle` | 20 | all | 0, .5 | lsf | **calibrated** | 1, 3, 5 | 300 | 1–5 (K = 15) |
| `routereval_mmlu_norep_cal` | routereval, `mmlu`, `no_repeat`, `shuffle` | 10, 100, 1000 | s2w | 0, .5 | lsf | **calibrated** | 3 | 300 | 1–5 (MIDIAN, random + oracle: H's reference lines; random added 2026-09-24) |
| `fw_routereval_{small,1k,5k}{,_em,_va}_norep_cal`, `re_sl_{declared,embed}_{small,1k,5k}_norep_cal` | routereval, `mmlu` / `leaderboard_mmlu`, `no_repeat`, `shuffle` | 10, 100 / 1000 / 5000 | s2w / s2w / all | 0, .5 | lsf | **calibrated** | 3 | 300 | 1–5 / 1–5 / 1–3 (H's framework bars) |

(lsf = low_skill_first; spec = specialist; prog = programmatic; self = self_described.) The live framework-shortlist
grids (`fw_live_n*_{dd,em,sota,verified_va,…}`) mirror their source grid cell for cell (same n, β, liar_select,
seeds, Q); they are resolved in the frameworks part. For bernoulli and replay, β = 0 is run with `random` only in the
scale grids. Because all four non-live families have switched (§4.0.7), B takes every bernoulli / replay / RouterEval /
LLMRouterBench bar from the per-seed tables of the grids listed there, not from the scale matrices or the old
RouterEval / LLMRouterBench grids. `routereval5k_cal` and `llmrouterbench_cal` (calibrated claims with repeated prompts)
are superseded by the `_norep_cal` grids and read by no figure. The `routereval5k_norep_cal` rows B reads are the shuffled
ones (the unshuffled first attempt is quarantined, §1.4.4).

The `pool_fill_*` grids hold only the candidates of "best learned/declared router" / "best bandit" that no other grid ran in
that cell at that b, so that every b of a cell has the same pool (comment at `configs/grids/`). Their
method lists differ per grid and per b block. `pool_seeds_n1000` gives the seven live 10^3 b = 3
candidates that `live_f1_n1000` ran on seeds 1–5 their seeds 6–10. `tuned_wsb_*` runs
`warm_start_bandit[n0=0.5]` at b = 3; `linucb_fix_*` runs `linucb_honest[bonus=own]` and `trueskill_fix_*` post-fix
TrueSkill at b = 1 / 3 / 5. `seed_tables.tables` (`scripts/figures/lib/seed_tables.py`) reads all of them for the
cross-fitted pools (§2.1.3); `bar_figs.py` does not.

Rows on disk: every grid in the table has rows. `linucb_fix_*` and `trueskill_fix_*` have landed (their arms have every
seed in every A / B pool table); the five erratum-30 grids B and H use are complete by `seed_tables.complete`, and so
are the 15 H framework mirrors (`shortlist_figs.h30()` is True). `rivals_b_n100k` has both regimes at b = 1 and 5 for
every rival; its kNN units, the cartel kNN units of `rivals_b_n10k` and the cartel kNN units of `routereval5k_norep_cal`
were run with the speed-only opt-ins of §2.8.1. The live 10^5 cartel kNN seeds landed on 2026-09-24 11:20; every kNN row is complete in the pool tables. In the
condensed A / B figures an absent bar is an empty slot, reported by the script and `cluster/ops/figure_status.py`
(§1.8.4); no A / B slot is empty today.

#### 1.6.4 What a seed is

A seed re-draws everything that is random in the world: the population (live profiles; bernoulli row resampling;
replay profiles), the honest-declaration noise, the random liar set, the task stream and task instances, the probe
instances, and each method's view rng (§1.1.2). Exceptions: on **RouterEval / LLMRouterBench the pool, S, and (for
`low_skill_first`) the liar set are identical in every seed** — a seed there only re-draws declaration noise, random
liars, tasks and probes (and, with `shuffle: true`, the agents' index order, which moves lowest-index tie-breaks but not
the pool). On live, the self-rating and the measured S of a *signature* are the same in every
population (fixed measurement set), so seeds change which signatures are present and in what proportion. Seeds with the
same number but different n are different populations (the profile seed includes n).

#### 1.6.5 The metrics row (`rte/run.py`)

Cell columns + `seed, grid, method, params, backend_kwargs, n_agents, n_liars`, world summary columns prefixed
`skill_` (`skill_mean`, `skill_per_agent_std`, `skill_p90_p10`, `skill_family_max_mean`, `skill_excess_ratio`,
`skill_excess_ratio_family`, plus backend stats), and:

| column | definition |
|---|---|
| `success` | mean 0/1 outcome over the Q routed tasks. Route-to-many (a list from `fetch`): majority of the executed agents' outcomes, ties → 0 (`rte/run.py`) |
| `success_late`, `n_late` | mean over the last `min(500, max(1, Q//4))` tasks (250 at Q = 1000, 75 at Q = 300) |
| `success_by_block` | per-100-task means |
| `oracle_success` | the unit's oracle-line success; `regret = oracle_success − success` |
| `misroute_to_liar` | fraction of tasks whose (first) routed agent is a liar |
| `build_{probes,reports,messages,hops,comparisons}`, `build_total_comm` | ledger after `build` (`total_comm` = probes + reports + messages) |
| `{probes,reports,messages,hops,comparisons,tasks}_per_task`, `total_comm_per_task` | run-phase ledger / Q |
| `wall_clock_build`, `wall_clock_per_task`, `wall_clock_per_task_total` | seconds; memo-dependent, not used as costs (`rte/analyze.py`) |
| `method_stats` | JSON of the method's `stats` (e.g. framework picks / fallbacks) |

The oracle row has zero build/run communication except `tasks_per_task = 1`.

---

### 1.7 Aggregation conventions (general machinery)

- **Labels** (`rte/analyze.py`): `label = method` if params are `{}`, else `method[k=v,…]` (sorted, floats
  `%.3g`), then `ALIAS`: `flat_probe_argmax → flat_probe_argmax_frozen`, `flat_probe_argmax[online=True] →
  flat_probe_argmax_online`, `knn_router[online=True] → knn_router_online`, `midian[verify=False] →
  midian_wo_verify`, `midian[audit=False] → midian_wo_audit`, `midian[audit=False,verify=False] → midian_wo_defenses`,
  `midian[audit=False,r=5] → midian_wo_audit_r5`, `sequential_halving[peer_reported=True] → sequential_halving_peer`,
  `midian[audit=False,stratify=True,verify=False] → midian_stratified`, and the two churn-mode halving labels (the full
  method, `midian` with params `{}`, is labelled `midian`). Rows stored under the old keys are translated on read
  (`keys.normalize`).
  `scripts/figures/lib/seed_tables.py` builds the same label for the b = 1/5 rows and the per-seed tables.
- **Unit of replication = the seed.** The figure CSVs (`scripts/figures/bar_figs.py`, `condensed_figs.py`, `seed_tables.py`,
  `scale_matrix.py`) first average each label's rows **per seed** (over shapes where a family pools them, and
  over duplicate rows of the same cell from different grids), then report the mean of the per-seed means.
- **Condensed-figure whiskers are ±1 standard error, never 95 % CIs.** `stats.se` (`scripts/figures/lib/stats.py`)
  returns mean ∓ sd(ddof = 1)/√m over the m units (per-seed means when the series is seed-indexed; 0 half-width for one
  unit), a ~68 % interval. A and B use it over seeds for every bar (`condensed_figs.narrow`, 4.0.5); G over the paired
  frameworks; the per-condition shortlist figures (`figures/shortlist/`, bars and the reference-line bands) over seeds.
  With 3 seeds (live 10^4 / 10^5, leaderboard 5k) it rests on 3 values; LLMRouterBench has 5, replay 30, bernoulli 100.
- **Other intervals in the pipeline are 95 % percentile bootstraps over seeds, B = 2000**: `stats.ci`
  (`scripts/figures/lib/stats.py`, rng 0; resamples seeds when the series is seed-indexed) in the bar CSVs
  (`results/aggregates/bars/`) and scale matrices, `fw_variant_numbers.ci` (`fw_variant_numbers.py`, rng 0) in the paper
  numbers, `rte.analyze.boot` (`rte/analyze.py`, rng 12345). With 3 seeds the bootstrap has only 10 distinct
  resamples, and its 2.5 / 97.5 percentiles are the smallest and largest seed mean (about 73 % coverage for normal data;
  figures audit F5).
- **Regimes** (`fw_variant_numbers.regime`, `bar_figs.REGIMES`): β = 0 is one regime (`beta0`); for bar CSVs the β = 0
  row set is filtered to `liar_select = random` when present (`bar_figs.py`), because the two liar selections are
  the same world at β = 0.
- **Paired comparisons**: `rte.analyze.paired` (`rte/analyze.py`) — per cell, pivot seed × label, delta
  `ref − rival` per seed, bootstrap CI + sign test, `WITHIN_FLOOR` if |mean delta| ≤ the reference's seed envelope (the reference `REF` is MIDIAN w/o defenses, `rte/analyze.py`).
  `paired_gaps.py` (pre-refactor, since removed; b = 3 only, not a condensed figure): per condition, `midian − rival` (MIDIAN minus the rival) per unit (unit =
  seed, or seed|shape where shapes are not pooled), 95 % bootstrap CI of that difference, "win/loss/tie" by CI sign;
  requires ≥ 2 shared units.
- **Figure filters**: `exclusions.excluded` (`scripts/figures/lib/exclusions.py`) drops MIDIAN variants with r ≠ 10 or
  δ ≠ 1/3, trusted-observer `sequential_halving`, `route_to_k_majority`, the LLM-descent ablation,
  online-off and cohort/churn variants, and every
  label containing "halving", including peer-reported halving (added 2026-09-22 as the temporary `HIDE_HALVING`
  switch, a permanent exclusion since 2026-09-24).

---

### 1.8 Errata and incompleteness reporting (`docs/archive/CHANGES_AND_ERRATA.md`)

#### 1.8.1 Erratum 25 — clone-filled framework shortlist (`docs/archive/CHANGES_AND_ERRATA.md`)

Same-signature agents have identical, memoised self-descriptions and answers (§1.4.1), so the pre-registered TF-IDF
top-10 (stable sort, lowest ids) fills with copies of the best-matching text: bimodal has 2 and heavy_tail 5 distinct
descriptions at any n; specialist has 3,920, giving ~4.4 distinct signatures in the top-10 at 10^3, ~2.6 at 10^4 and
exactly one at 10^5. Those cells measured the retriever, not the framework. Fix: labeled variant `dedup: true` (one
agent per distinct text), rerun into separate `*_dd` grids; RouterEval descriptions are all unique and unaffected.
The pre-registered numbers stay as quoted. The document's statement that the 10^5 TF-IDF row is a single clone agrees
with the code path (`descriptions()` is per agent but identical prompts produce one memoised text).

#### 1.8.2 Erratum 26 — trusted-observer halving withdrawn (`docs/archive/CHANGES_AND_ERRATA.md`)

`sequential_halving` (no params) is scored by a trusted observer the setting does not provide; only
`sequential_halving[peer_reported=True]` is a legitimate rival. Rows remain in the grids; `scale_matrix.py` and the
`DO_NOT_ADD` set drop the label, so trusted-observer halving is never reported. Peer-reported halving is hidden too:
`exclusions.excluded` drops every label containing "halving" (since 2026-09-22); no condensed
figure draws either (protocol audit F1: peer halving beats MIDIAN in every honest b = 3 cell of A).

#### 1.8.3 Erratum 27 — the lie does not touch description text (`docs/archive/CHANGES_AND_ERRATA.md`)

`apply_lying` edits only the D matrix; `descriptions.json` is per population with no β in its path, and its
"Declared areas:" clause lists the agent's true specialty (`rte/backends/llm.py`). Therefore every
text-retrieval shortlist (TF-IDF, dedup, BM25, MiniLM, dense, hybrid, sota) is **identical across all liar regimes**;
its only β-dependence is the declared-argmax fallback when a framework fails to pick. Cross-regime robustness of text
arms is an artefact of the threat model; only declared-channel arms are attacked by the lie. `lie_text` (§1.3.6) is
the partial remedy.

#### 1.8.4 Erratum 28 — framework rows that measured declared argmax, and how incompleteness is reported (`docs/archive/CHANGES_AND_ERRATA.md`)

Broken library symlinks in 7 framework conda envs made workers crash on some nodes; the adapter then fell back to
declared argmax (`_common.py`) and wrote normal-looking rows. CrewAI and ADK were hit hardest (~2/3 of units)
and were biased *upward* (declared argmax scores 0.619 honest). Fix: envs restored, 4,269 rows moved to
`results/<grid>/quarantine/`, their units rerun, and `fetch` now fails the unit past 2 % infra errors.

**Incompleteness reporting.** No condensed figure carries a title, a marker on a bar or any other incompleteness
mark (`docs/figures.md` §1). Instead each script prints `[<name>] INCOMPLETE: …` and
`cluster/ops/figure_status.py` reads the figures' own CSVs and lists what is missing (C and D have no such check):
1. *A / B* (`budget_bars`, `scripts/figures/condensed_figs.py`; `figure_status.ab` on the `_allb` CSVs): any
   (arm, regime, b) slot without a bar (declared argmax, random and A's framework arm: one `b = "-"` slot per regime), or
   any pooled bar whose pool lacks an allowed member or has a member missing some scored seeds (`INCOMPLETE POOL, missing
   or partial …` in the CSV `chosen` column). A missing-data flag, unrelated to erratum 28.
2. *E–H* (`shortlist_condensed.pair` / `finish` / `fig_G`;
   `figure_status.shortlist`): any empty slot (a bar needs at least `MIN_FW = 6` frameworks with the full seed count; G
   needs 6 paired frameworks) or any drawn bar that averages a framework × shortlist series with `rerun_outstanding` (the
   CSV's `star` column). That flag comes from `pending_reruns()` (`fw_variant_numbers.py`): every (grid, framework, dist,
   regime) listed in `$RTE_DATA/results/quarantine_units.tsv` whose rerun has **not landed** — `landed()` checks that
   every param variant of that method in that grid has a row for that (dist, β, liar_select, seed) — unless
   `$RTE_DATA/logs/DONE_stage2` exists (then none; with only `DONE_stage1`, just the `fw_live_n{100,1000}[_lowskill]_sota`
   grids). Today `logs/DONE_stage1` exists and `DONE_stage2` does not, so only those four `*_sota` grids are checked; the
   tsv has 3,572 unit rows, and 51 (grid, framework, dist, regime) keys are still outstanding, all in
   `fw_live_n{100,1000}[_lowskill]_sota`. None touches a drawn E–H bar: the three `rerun_outstanding` rows of
   `results/aggregates/shortlist/live.csv` are heavy_tail / bimodal cartel cells.
Today nothing in A–H is incomplete: no missing slot, no incomplete pool, no drawn bar with a rerun outstanding.

#### 1.8.5 Rival note 8d (`docs/archive/CHANGES_AND_ERRATA.md`)

The warm-start bandit (prior Beta(n0·D, n0·(1−D)), n0 = 5, on self-described claims) scores higher under the cartel
than honest at live 10^4/10^5 (0.743 → 0.783, 0.738 → 0.778) because liars' D ≈ 1 removes their prior failures; the
tuned n0 = 0.5 arm (`tune_wsb_n1000`, seeds 11–15) competes beside it in the best-bandit pool.

#### 1.8.6 Erratum 29 — supervisor invalid actions are non-picks (`docs/archive/CHANGES_AND_ERRATA.md` §8e)

A framework's own invalid action (ADK / OpenAI Agents calling a tool named after the agent, MAF "next_speaker must be
provided", and MAF's orchestrator naming a participant that is not among its candidates, which surfaces as the worker's
name lookup failing with "KeyError: 'agent_005044'" — 14 units at 10^2–10^4) used to count as an infrastructure error
and fail the unit past the erratum-28 threshold, so ADK's honest 10^2 / 10^3 units never landed. `FrameworkMethod.fetch`
now treats these (`INVALID_ACTION`, `_common.py`) as a non-pick: declared argmax inside the shortlist, counted as
`invalid_action` in `fallback_rate`, 0 under `success_strict`, not retried (§3.2.6). The failed units were rerun.

#### 1.8.7 Erratum 30 — leaks on the non-live backends (`docs/archive/CHANGES_AND_ERRATA.md` §8f)

Programmatic claims were an answer key (corr ≈ 0.99), replay probed and routed on the same prompts, and RouterEval /
LLMRouterBench streams repeated test prompts; `World._probe` also gave a repeated (agent, family) in one call the same
instance (TrueSkill). Fixes: `declared_source: calibrated` (§1.3.3), replay `split: true` (§1.4.3), `no_repeat` and
`shuffle` (§1.4.4), and the `_probe` fix (not opt-in; §2.7.5). Old rows are unchanged on disk. Each B family and H moved
to its reruns only once those grids were complete; all have, so B's four non-live groups and H draw only erratum-30 rows,
and TrueSkill is drawn only from post-fix rows (§4.0.7, §5.5). The kNN reruns also used speed-only opt-ins that do not
change outcomes (§2.8.1).

---

### 1.9 Glossary

| term | meaning in code |
|---|---|
| **n** | number of agents in the world (`World.n`). live: agent profiles; bernoulli: resampled skill rows; replay: (model, mask) agents; RouterEval/LLMRouterBench: the real pool size m |
| **K** | number of task families (16 live, bernoulli, RouterEval; 64 replay; 15 LLMRouterBench) |
| **family** | a task category: a Reasoning-Gym generator (live), a RouterBench `eval_name` (replay), an MMLU subject (RouterEval), a dataset (LLMRouterBench), or `famNN` (bernoulli) |
| **task / Q** | one routed request of known family; Q = tasks per unit (1000, or 300 for live 10^4/10^5 and every RouterEval / LLMRouterBench grid B and H draw) |
| **stream** | the Q tasks of a unit; same for every method in the unit, and for every n/dist/β at a given seed |
| **true skill S** | per-(agent, family) success probability / measured accuracy; hidden from methods |
| **declared D** | per-(agent, family) claim, honest = S + N(0, .05) (programmatic), the model's self-rating (self_described) or a draw from the live self-rating's distribution given S (calibrated); liars +0.4 |
| **declared_source** | `programmatic` (S + N(0, .05)), `self_described` (live only) or `calibrated` (non-live erratum-30 grids, live-like) |
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
| **grid** | a named experiment in `configs/grids/`; results in `$RTE_DATA/results/<grid>/` |
| **row / rid** | one (cell, method, params, seed) result; rid = blake2b of those |
| **regime** | beta0, beta{β}_random, beta{β}_cartel, cartel (= β .5 lsf) |
| **b = 1 / 3 / 5 bars** | same arm at different probe budgets; never pooled |
| **incomplete** | a figure slot without a bar or a pooled bar with an incomplete pool; never marked in the figure (no titles, no markers), reported by the scripts and `cluster/ops/figure_status.py` (§1.8.4) |

---

### 1.10 Discrepancies / open questions

1. **Older RouterEval MMLU rows come from one of two family orders (affects `dataset: mmlu` only).**
   `RouterEvalBackend._families` ranks the subjects by train-prompt count and breaks ties by subject name
   (`rte/backends/routereval.py`; test `tests/test_routereval_family_order.py`). "High school biology" and
   "philosophy" tie at 248 train prompts, at family indices 9 / 10. Rows written before the tie-break took their order
   from Python set iteration, which varies with `PYTHONHASHSEED` (under hash seeds 0–11 the order flips for 6, 8, 9 and
   11), so about half of them come from the other order. The subject set is the same either way; only the two indices
   swap. Task instances, probes and declaration noise are keyed by family index, so rows of the same (cell, seed) from
   different processes can come from two slightly different worlds. Observed: in `routereval_mmlu/rows.csv` 240 of 270
   units have rows whose `oracle_success` disagrees (spread up to 0.013), and `fw_routereval_1k` 28 of 30 units.
   Averages are unbiased. Within one unit (one process) pairing holds; pairing *across* grids or reruns on the mmlu
   pools (e.g. `fw_routereval_*` vs `routereval_mmlu`, `re_sl_*`) is noisier than it looks for rows written before the
   tie-break. `leaderboard_mmlu`, `llmrouterbench`, live, bernoulli and replay are deterministic (`routereval_mmlu5k`,
   `llmrouterbench_pool`: 0 units with disagreement). **No drawn figure is affected any more**: B and H read only the
   erratum-30 grids, written after the tie-break; `routereval_mmlu_norep_cal`, `fw_routereval_{small,1k}_norep_cal` and
   `re_sl_embed_1k_norep_cal` have 0 units with two `oracle_success` values.
2. **Live n = 1000 rows are not bit-reproducible across reruns.** `fw_live_n1000`: 54/120 units, `variants_f1`:
   64/240 units carry more than one `oracle_success` value (spread ≤ 0.006); duplicate cells across grids differ
   (plain `midian` in `fw_live_n1000` vs `variants_f1`: 39 % of seed-pairs differ, max 0.011; vs `live_f1_n1000` up
   to 0.075 over all shared methods). n = 100, 10^4, 10^5 and the va_b grids show no oracle drift. Cause not determined
   (candidates: memo entries regenerated non-deterministically, code changes between runs); figure CSVs average such
   duplicates per seed.
3. **Build-budget contract vs code.** `docs/architecture.md` says probing methods spend "AT MOST" n·K·b in build; MIDIAN w/o verification /
   -SHA spend up to 1.0625× and MIDIAN up to ≈ 1.04× per cell (audit re-probes; mean 1.033× at b = 3 on LLMRouterBench), and `run.py` only logs a warning. The
   old `midian_a.py` docstring's "≤ 1.05×" was also exceeded on small pools (1.0625 on `routereval_mmlu`).
4. **README / docstrings say three backends** (`README.md`, `rte/world.py`); there are four (`routereval`).
5. **README: "bernoulli: synthetic S with five population shapes, calibrated to the measured live S"** — with
   `calibrate_from` the shape is ignored; every figure grid resamples rows of one live specialist population.
6. **README: "S is measured per signature (200 probes)"** — 60 for the ≥ 9B models (`llm.py`,
   `models.yaml`).
7. **`View` docstring says `dist` is always available** (`rte/world.py`); `View.__init__` never sets it and
   `view.dist` would raise `AccessError`.
8. **`_common.py` comment** says honest agents' declared "is true skill plus 0.05 noise"; on the live
   `self_described` channel (where `lie_text` is used) honest D is the model's coarse self-rating (mean 0.686 vs S
   0.419), not S + noise.
9. **Stale docstrings about markers and intervals**: `condensed_figs.py`'s module docstring  still says an
   incomplete pool "puts the one * in the figure title", and `add_budgets`'s docstring  says a missing cell is
   marked by "a *"; the code draws no title and no marker (§1.8.4). `shortlist_figs.py`'s docstring  and the
   `INDEX.md` text it writes  say an asterisk marks a bar with an outstanding rerun and that the lines carry a
   "95% seed-bootstrap" band; `draw` collects the starred positions (`stars`) but never plots them, and every
   interval there is ±1 s.e. (`stats.se`, imported as `_ci`, `shortlist_figs.py`).
10. **Erratum 28 counts vs the quarantine file**: the erratum says 2,904 units were rerun; `quarantine_units.tsv` has
    3,572 unit rows (it also lists `*_sota`, `*_em`, `*_verified*` and RouterEval-framework grids). Whether the tsv
    grew after the erratum was written is not determined. `pending_reruns()` drops every listed unit whose rerun rows are
    on disk, so the tsv's size does not matter for the figures.
11. **Stale grid comment**: `fw_live_n1000` comment "(v2 0.4: 5 seeds, Q=1000 (was 3/300))" (`configs/grids/`)
    vs actual `seeds: 1-10`.
12. **`bernoulli.py` says bernoulli is "never a headline number"**, while the condensed figure B includes a
    bernoulli 10^7 bar and C/D include bernoulli cells. A policy question for the author, not a code error.
13. **"Cartel" naming**: random liars also collude (`collude = true` everywhere), so "cartel" in figures means
    *low-skill-first selection*, not "collusion on vs off". Readers may assume the latter.
14. **Open**: live oracle is argmax of S measured on a fixed 200/60-instance set; how often the oracle agent is not
    the best on the actual task instances was not quantified here.

---

## 2. The non-framework methods in figures A and B, plus the oracle and MIDIAN reference lines in E, F and H

All paths are relative to the repository root. Every claim below was checked against the code on 2026-09-24. Where
docs/methods.md, docs/archive/DEVIATIONS.md or docs/archive/CHANGES_AND_ERRATA.md disagree with the code, the code is described here and the disagreement
is listed in §2.12.

---

## 2.1 Which methods appear in A and B, and how they get there

### 2.1.1 The arms that are drawn

`scripts/figures/condensed_figs.py` draws the arms in `ARMS` (`scripts/figures/condensed_figs.py`: `figspec.ORDER` without the oracle
and the framework arm), in that legend order, plus A's framework arm. Names and colours come from `scripts/figures/lib/figspec.py`
(`NAME`, `COLOR`, `ORDER`):

| key in code | legend label | colour | what the key resolves to |
|---|---|---|---|
| `midian` | MIDIAN | green `#2ecc71` | method `midian`, params `{}` (both defenses; formerly `midian_va`; §2.3) |
| `midian_wo_defenses` | MIDIAN w/o defenses | red `#c0392b` | method `midian`, params `{"audit": false, "verify": false}` (formerly the plain `midian`; §2.2). The label comes from `rte/analyze.py` (`ALIAS`) |
| `flat_probe_argmax_online` | flat probe argmax | blue `#3498db` | method `flat_probe_argmax`, params `{online: true}`. The alias comes from `rte/analyze.py` (`FLAT_ON`, `ALIAS`) |
| `best_learned` | best learned/declared router | orange `#ff7f0e` | the cross-fitted best of the learned-router pool in this cell, regime and b (§2.1.3); "declared" because two of its members read only declarations |
| `best_bandit` | best bandit | purple `#9467bd` | the cross-fitted best of the bandit pool in this cell, regime and b (§2.1.3) |
| `declared_argmax` | declared argmax | slate `#5d6d7e` | method `declared_argmax`, params `{}` (not cached) |
| `random` | random | light grey `#bbbbbb` | method `random` |
| `best_framework` (A only) | best framework (caption: best text shortlist) | brown `#8e6c3a` | the cross-fitted best (framework, text shortlist) pair at b = 3 (`add_best_framework`, §4.1) |

The **oracle** is not an arm. It is the grey dotted horizontal line over each group (`figspec.oracle`,
`scripts/figures/condensed_figs.py`). In B every bar is divided by the oracle's mean (`norm=True`).

**Arms that do not change with b.** `B_INVARIANT = {declared_argmax, random, cluster_head_router, disrouter_cascade}`
(`condensed_figs.py`): these never probe, so b cannot change them. Declared argmax and random are drawn once per regime;
a pooled bar whose picks are all `cluster_head_router` / `disrouter_cascade` is identical at b = 1, 3 and 5. The CSV
column `b_invariant` marks both cases (§4.0.5).

Rows whose label starts with `fw_` (frameworks) are removed at load time (`scripts/figures/condensed_figs.py`; `seed_tables.py` for the pooled arms). Every label also
goes through the do-not-add filter `exclusions.excluded` (`scripts/figures/condensed_figs.py`). Because `ARMS` and the two
pools list every drawable label, anything outside them is never drawn in A or B, whether or not it is excluded. That covers
MIDIAN w/o audits, MIDIAN w/o verification, verify_on_claim, cnp_self_bid, declared_softmax, gossip, referral, sequential halving and the rest.
No condensed figure draws them; they are in the per-family bar figures (`figures/bars/`, `scripts/figures/bar_figs.py`), where
`exclusions.excluded` also removes every halving arm (§1.7).

### 2.1.2 Where each budget's numbers come from

- **Live, the five single arms at b = 3** (MIDIAN, MIDIAN w/o defenses, flat probe argmax, declared argmax, random). From
  `results/aggregates/bars/live.csv` (`scripts/figures/condensed_figs.py`), which `scripts/figures/bar_figs.py` writes from the grids in
  `LIVE_GRIDS` (`scripts/figures/bar_figs.py`).
- **Live, the same arms at b = 1 and b = 5.** From the `va_b_n*` grids (MIDIAN only) and the `rivals_b_n*` grids (the
  budget-matched rivals) (`add_budgets`, `scripts/figures/condensed_figs.py`; grids in `configs/grids/`).
- **bernoulli 10^7, replay 10^6, RouterEval 5,000 and LLMRouterBench: every arm at every b, and the oracle.** All four
  have switched to their erratum-30 grids (§4.0.7), so `add_budgets` skips them  and `from_tables`
  (`condensed_figs.py`) fills each cell from its per-seed tables: mean and ±1 s.e. over seeds. The bar CSVs and
  the scale matrices (`bernoulli_scale_v5` / `replay_scale_v5` `matrix_success.csv`) are not used for these groups.
- **"Best learned/declared router" and "best bandit" at every b.** From per-seed tables built straight from the raw rows by
  `seed_tables.tables()` (`scripts/figures/lib/seed_tables.py`, attached to each cell at `condensed_figs.py`). Per cell
  it reads: live, `LIVE_GRIDS[n]` + `va_b_*` + `rivals_b_*` + `pool_fill_*` + `linucb_fix_*` + `trueskill_fix_*` +
  `tuned_wsb_*` + `pool_seeds_*`; RouterEval 5,000, `routereval5k_norep_cal` only; LLMRouterBench,
  `llmrouterbench_norep_cal` only; replay 10^6, `replay_1e6_split_cal` only; bernoulli 10^7, `bernoulli_scale_v5` + its
  b-grids (`va_b_`, `rivals_b_`, `pool_fill_`, `linucb_fix_bernoulli_1e7`; the code also names `trueskill_fix_bernoulli_1e7`,
  which is not a grid) + `bernoulli_1e7_cal`, with
  the claim-reading arms restricted to the calibrated rows. TrueSkill rows count only from `trueskill_fix_*`
  or erratum-30 grids.
- **Budgetless arms.** `declared_argmax` and `random` are drawn only from the b = 3 data, as one bar each (`BUDGETLESS`,
  `scripts/figures/condensed_figs.py`).

### 2.1.3 How "best learned/declared router" and "best bandit" are chosen (`arms_at`, `crossfit`)

Both are **cross-fitted**: no bar is the maximum of noisy means over the seeds it reports.

- **Pools** (`POOLS`, `scripts/figures/condensed_figs.py`):
  - best learned/declared router = `LEARNED = [knn_router, knn_router_online, mlp_router, flat_nsw_router, cluster_head_router,
    disrouter_cascade]`;
  - best bandit = `ucb_per_family, thompson_per_family, warm_start_bandit` (n0 = 5), `trueskill_per_family`,
    **`linucb_honest[bonus=own]`** (the fixed-bonus LinUCB, which replaces the pre-registered `linucb_honest`; §2.7.4) and
    `warm_start_bandit[n0=0.5]` (the tuned warm-start bandit).
- **Who may compete in a cell** (`want`, `arms_at`): the pool minus
  - `NOT_RUNNABLE(family, n)` : `trueskill_per_family` at n ≥ 10^5 (it raises), `mlp_router` at n ≥ 5,000, and
    `knn_router`, `knn_router_online`, `mlp_router` on bernoulli and replay (no prompt text);
  - the pre-registered `warm_start_bandit` (n0 = 5) on a non-live family that has **not** switched to its
    calibrated-claims erratum-30 rows (`CLAIM_KEY - SW`), because on the old rows its prior is built on
    claims equal to true skill + 5 % noise. All four non-live families have switched, so this set is empty today: the
    n0 = 5 arm competes in every A and B cell, live and non-live.

  The set is the same at every b of a cell.
- **Per-seed tables.** `seed_tables.tables()` gives, for each (cell, regime, b), a seed × arm table of mean success
  (`_table`, `seed_tables.py`). Rows of one (seed, arm) from several grids are averaged, as `bar_figs` does. β = 0
  counts as honest whatever its `liar_select` tag; the cartel is β = 0.5 with `low_skill_first`. Live keeps
  `dist = specialist` and the self-described channel. Replay pools the three shapes per seed and keeps only
  seeds where the arm ran on all three. `rows()`  reads `rows.csv` + `rows.d` with `rid` = the
  `rows.d` file name, so rows not yet merged into `rows.csv` keep distinct ids and are not collapsed by the `rid` dedup.
- **Cross-fitting** (`crossfit`, `seed_tables.py`). For each seed s, among the arms that ran on s, the arm with
  the highest mean over the **other** seeds is picked, and its success **on s** is s's score. The bar is the mean of
  those per-seed scores. The whisker is ±1 standard error over the same per-seed scores (`stats.se`,
  `condensed_figs.py`). A cell with fewer than 2 scored seeds gets no bar.
- **The CSV `chosen` column holds the pick counts**, most-picked first, e.g. `warm_start_bandit x7; warm_start_bandit[n0=0.5] x3`. The picked arm can differ between seeds, regimes and b values.
- **Incomplete pools**. If an allowed member has no column in the table at that b, **or has no value on some
  scored seed**, `chosen` ends with `| INCOMPLETE POOL, missing or partial <arms>`; the script prints `INCOMPLETE` for the figure and
  `cluster/ops/figure_status.py` lists the bar (nothing is drawn on the figure). None is incomplete today. A member that ran on some seeds still competes only on those seeds.
- **The "learned router" pool includes two declaration-only methods and a constant-cost index.** `cluster_head_router`
  and `disrouter_cascade` (§2.8) spend no probes and read only the declared channel; `flat_nsw_router` is an ANN index
  over flat probe means. With the answer-key claims gone (every drawn non-live claim is calibrated), the declaration-only
  arms are the majority pick in only 3 of the 54 "best learned/declared router" bars, all at b = 1 (bernoulli 10^7 both regimes,
  RouterEval 5,000 cartel); `flat_nsw_router` is the majority pick in 19 (§6.2).

What is picked in the current A / B CSVs (full tables in §4.1, §4.2):

| figure / cell | best learned/declared router | best bandit |
|---|---|---|
| A, live n = 10^2 / 10^3 | b = 1 `mlp_router` ×8–9 (+ `knn_router_online`); b = 3, 5 `knn_router_online` ×10 | 10^2: b = 1 honest `warm_start_bandit[n0=0.5]` ×10, every other bar `linucb_honest[bonus=own]` ×10; 10^3: honest `warm_start_bandit[n0=0.5]` ×10; cartel b = 1 `warm_start_bandit` ×8 + `trueskill_per_family` ×2, b = 3 `warm_start_bandit` ×7 + n0 = 0.5 ×3, b = 5 n0 = 0.5 ×10 |
| A, live n = 10^4 | `flat_nsw_router` ×2–3 at b = 1, 3; `knn_router_online` ×2 + `flat_nsw_router` ×1 at b = 5 | b = 1 `trueskill_per_family` ×3 (honest) / ×2 + `warm_start_bandit` ×1 (cartel); b = 3 honest `trueskill_per_family` ×2 + n0 = 0.5 ×1, cartel `warm_start_bandit` ×3; b = 5 n0 = 0.5 ×3 (honest), `warm_start_bandit` ×2 + n0 = 0.5 ×1 (cartel) |
| A / B, live n = 10^5 | `knn_router_online` majority at every b in both regimes (×3 at b = 3 and cartel b = 1; honest b = 1 + `cluster_head_router` ×1; b = 5 + `flat_nsw_router` ×1) | honest `warm_start_bandit[n0=0.5]` (b = 1 ×2 + n0 = 5 ×1; b = 3, 5 ×3); cartel `warm_start_bandit` ×3 at b = 1, 3, ×2 + n0 = 0.5 ×1 at b = 5 |
| B, bernoulli 10^7 | b = 1 `cluster_head_router` ×100 (honest) / `disrouter_cascade` ×100 (cartel); b = 3, 5 `flat_nsw_router` ×100 | `warm_start_bandit` (n0 = 5) ×100, every b and regime |
| B, replay 10^6 | `flat_nsw_router` ×30, every b and regime | honest b = 1 `warm_start_bandit` ×30; every other bar `linucb_honest[bonus=own]` ×30 |
| B, RouterEval 5,000 | `flat_nsw_router` (honest b = 1 ×2 + `cluster_head_router` ×1; b = 3, 5 ×3); cartel b = 1 `disrouter_cascade` ×3 | `linucb_honest[bonus=own]` ×3, every b and regime |
| B, LLMRouterBench 20 | `mlp_router` ×5, every b and regime | cartel `linucb_honest[bonus=own]` ×5; honest ×4 + `warm_start_bandit` ×1 (b = 1) / n0 = 0.5 ×1 (b = 3, 5) |

### 2.1.4 Shared machinery every method uses

- **Interface** (`rte/methods/base.py`):
  - `needs` is a subset of {declared, probe, reports, bus}. The `View` raises `AccessError` on any other access
    (`rte/world.py`), and it never exposes S or the liar set.
  - `build(view, budget)` is the pre-emptive phase.
  - `fetch(task)` returns an agent id.
  - `observe(task, agent, outcome)` is the online update. The runner passes the **true executed outcome** directly
    (`rte/run.py`), so every online method learns from trusted outcomes, not from reports.
- **Budget** (`rte/budget.py`): `total_probes(n, K) = n·K·b`, with b = `probes_per_agent_family`. The runner does
  **not** enforce it. It only logs a warning when build probes exceed it (`rte/run.py`).
- **Probes are index-seeded** (`rte/world.py`). The k-th probe of (agent, family) is the same instance for every
  method. `World.reset` zeroes the probe index before each method (`rte/world.py`), so two methods probing the
  same cell see identical outcomes. The runner resets the world with the method's **pre-rename** key
  (`keys.legacy`, `rte/run.py`), so a MIDIAN arm reruns bit-identically to the rows stored under its old key.
- **Method randomness** comes from `view.rng = default_rng(stable_seed_32(seed, "view", sorted(needs)))`
  (`rte/world.py`). It depends only on the world seed and the method's `needs`. Methods with the same `needs`
  therefore start from the same RNG stream. Checked: MIDIAN w/o defenses, MIDIAN w/o audits and MIDIAN build **identical leaf cohorts** at
  a given seed (n = 100 and 1,000, b = 1, 3, 5).
- **Declared-channel helpers** (`rte/methods/_decl.py`):
  - `declared(view)` charges n messages once, for collecting the registry.
  - `scan` charges n comparisons per read of one family's column.
- **What liars do.** The world layer is described in Part 1; summarised here so each method's liar behaviour reads
  on its own.
  - Declared lie (`rte/world.py`): `inflate` sets D = clip(D_honest + 0.4). `max` sets D = 1 everywhere.
    `squat` sets D = 1 on the top-3 demand families.
  - Report lie, when `collude` is on (`rte/world.py`): a liar reporter says 1 about any liar, and 0 about
    the top 20% (by its observed mean in this batch) of the honest agents it reports on. Otherwise it reports the
    truth.
  - Liars **execute at their true skill**.

---

## 2.2 MIDIAN w/o defenses (`midian{"audit": false, "verify": false}`, label "MIDIAN w/o defenses")

**File:** `rte/methods/midian.py`, class `Midian` (one class for MIDIAN and every ablation; renamed 2026-09-24,
CHANGES_AND_ERRATA §8g). **Needs:** `{probe, reports}`. **Params used in A/B:** `audit=False, verify=False`, and
the defaults `r=10, delta=1/3, online=True, top=1, cohort="random"`; `cached` defaults to `verify`, so it is off
. Before the rename this was the plain `midian` with params `{}` (the pre-registered tree, SPEC §5); stored rows
were rewritten to the new key and reproduce bit for bit (the world is seeded with the legacy key, `rte/run.py`,
`rte/methods/keys.py`). The parameters r = 10 and δ = 1/3 are defaults, not tuned. `exclusions.excluded` drops any
MIDIAN with r ≠ 10 or δ ≠ 1/3 (`scripts/figures/lib/exclusions.py`).

### 2.2.1 The idea

Put agents in small random groups ("cohorts") of r. Each agent is probed, and its cohort peers, not a trusted central
observer, report what they saw. A trimmed mean over those reports absorbs a few lying peers. The cohorts are then stacked
into an r-ary tree. Each tree node remembers, per task family, the best estimate anywhere below it and which child holds
it. Routing a task walks down the tree from the root, about log_r n steps, instead of scanning all n agents.

### 2.2.2 Tree construction (`_cohorts`, `_structure`)

- **Leaf cohorts.** `c = rng.permutation(n)`, padded with −1 to a multiple of r and reshaped to `(ceil(n/r), r)`. A "leaf cohort" is one row of `self.leaves`: r agent ids (only the last row can be short). `self.leaf_of[a]`
  maps each agent to its row.
- **Upper levels.** At each level the m nodes are randomly permuted and grouped r at a time into parents,
  until one node is left. `self.children[l]` holds, for each node at level l, its r children (agent ids at level 0, node
  ids above). `self.parent[l]` maps a node to its parent.
- **Depth.** `depth = len(children)`. Checked: depth = 2 at n = 100 (10 leaf cohorts, 1 root) and 3 at n = 1,000.

### 2.2.3 Level-0 estimation (`peer_reported_estimates`, `rte/methods/_est.py`)

- **Probes.** Every agent is probed b times on every family. That is n·K·b probes, exactly `Budget.total_probes`.
- **Reports.** Every probe outcome is reported by **every other member of the agent's cohort** (s − 1 reporters). That is
  Σ_c s_c(s_c − 1)·K·b reports. At r = 10 this is 9 × the probes. Checked: 43,200 reports for 4,800 probes at n = 100,
  b = 3.
- **Aggregation without defenses** (`by_reporter=self.verify`, i.e. False; `_level0`, `midian.py`):
  - The (s − 1)·b individual reports about (member, family) are pooled, and a trimmed mean drops
    `trim_k(delta, s, b) = max(0, min(floor(delta·(s−1)), ((s−1)·b − 1)//2))` **single reports** from each end
    (`_est.py`).
  - At r = 10, δ = 1/3 that is **3 reports per side** at b = 1, 3 and 5, out of 9, 27 and 45 reports.
  - A colluding peer contributes all b of its reports. So at b = 3 the trim removes about one liar peer's worth of
    reports per side, and at b = 5 about 0.6. The code's own comment on `trimmed_by_reporter` (`_est.py`) says
    per-report trimming "under-trims" for this reason. The by-peer trim is what every variant with a defense uses
    (MIDIAN w/o audits through `by_reporter=True`; MIDIAN and MIDIAN w/o verification through the audited level 0, §2.3.3).
- **Singleton cohort.** A cohort of size 1 has no reporters, so its estimate is its own probe mean (`_est.py`).
- **Build messages** (`midian.py`): (n − #leaves) member-to-leader messages plus (#nodes − 1) node-to-parent messages.
  Checked: 100 at n = 100 and 1,010 at n = 1,000. No comparisons are charged at build.

### 2.2.4 Summaries (`build`)

- At every node and for every family, the children's candidate values are sorted with a stable sort.
- `best[l][node, f]` is the child slot holding the maximum.
- `summary[l][node, f]` is that maximum value.
- `cand[l][node, f]` is the agent holding it.
- `topc` holds the top `top` (= 1) forwarded agents.
- **Ties** go to the lowest slot index. Slots are randomly permuted, so this amounts to a random tie-break fixed at
  build time.

### 2.2.5 Routing (`fetch`)

- Start at the root and descend `depth` levels. At each level take `child = best[l][node, f]`, and return the level-0
  member reached.
- **Cost per task:** 1 hop, r comparisons and 2 messages per level, so depth hops, r·depth comparisons and 2·depth
  messages.

### 2.2.6 Online update (`observe`, with `online=True`)

- The routed agent's estimate gets a running-mean update: `est += (outcome − est)/k`. The count k starts at
  `w0 = max(1, (r−1)·b − 2·trim_k)` : 3 at b = 1, 21 at b = 3, 39 at b = 5. So the build estimate carries
  the weight of that many observations, even though only b independent probe outcomes lie behind it.
- The family's summaries are then recomputed up the agent's path (`_recompute`). This charges r comparisons
  and 1 message per level.
- **Only the routed agent is ever updated.** The update is greedy exploitation: a routed agent that keeps failing sinks
  until another subtree's summary is higher.

### 2.2.7 Churn and budget

- **Churn**  re-probes arrivals b times per family through their cohort peers. It is not relevant to A/B,
  whose cells have no churn.
- **Probe spend** is exactly n·K·b, and nothing at run time.
- **Declarations are never read** (with `cohort=random`).

### 2.2.8 Behaviour with liars

- Liars can corrupt the level-0 estimates of every member of any cohort they sit in, through the report channel.
- Per-report trimming of 3 reports per side absorbs only a few colluding reports.
- Online updates use true outcomes, so a liar that was over-estimated and gets routed is corrected after its failures.

---

## 2.3 MIDIAN and its single-defense ablations (MIDIAN is drawn in A and B; it is the reference line "MIDIAN, no framework" in E, F and H)

### 2.3.1 One class, two defenses

- `Midian(audit=True, verify=True)` is the default (`midian.py`): **MIDIAN** is `midian` with params `{}`. Each
  defense is a flag (module docstring):
  - `audit` (True = rate 0.05, `AUDIT_RATE`): report audits with reporter exclusion (§2.3.3 steps 4–5, §2.3.6).
  - `verify`: verification at promotion with b0 = b − 1 level-0 probes (§2.3.3 steps 1, 3, 6); `cached` defaults to
    `verify`, so the root caches its pick (§2.3.5).
- The ablations switch a flag off: `midian{"verify": false}` = **MIDIAN w/o verification** (label `midian_wo_verify`),
  `midian{"audit": false}` = **MIDIAN w/o audits** (`midian_wo_audit`), both off = **MIDIAN w/o defenses** (§2.2).
  Labels come from `rte.analyze.ALIAS` (`rte/analyze.py`).
- **Level 0 keeps the engine each variant had before the rename** (`_level0`): with audits,
  `_level0_audited` (per-peer means, trimmed over peers, audits); without audits,
  `_est.peer_reported_estimates` with `by_reporter=verify` (per-peer trim for MIDIAN w/o audits, per-report trim for MIDIAN
  w/o defenses). So every stored row is reproduced (216/216 parity rows bit-identical, CHANGES_AND_ERRATA §8g).
- Former names: MIDIAN was MIDIAN-VA (`midian_va`, class chain `MidianVA(MidianA(MidianSH(Midian)))` with the halving
  engine off), MIDIAN w/o verification was MIDIAN-A, MIDIAN w/o audits was MIDIAN-V. MIDIAN was pre-registered as V2-11 in
  docs/archive/preregistration/TARGETS_rte_v2.md and is a labelled variant added 2026-09-03.
- **Needs:** `{probe, reports}`. It **never reads declarations** in A/B (`cohort="random"`).

### 2.3.2 MIDIAN w/o audits for reference

- It is `Midian(audit=False)`: verification and the cached root pick, no audits. The ALIAS `midian[audit=False]` maps to
  `midian_wo_audit` (`rte/analyze.py`).
- It is not drawn in A or B (it is not in `ARMS`); it is in the per-family bar figures (`figures/bars/`).
- It differs from MIDIAN only in level 0: `peer_reported_estimates` with `by_reporter=True` (per-peer trim), and no
  audits.

### 2.3.3 Build, step by step (`Midian.build`, `midian.py`, both defenses on)

1. **Split the budget.** `b0 = max(1, min(b, self.b0 or b − 1))` when `verify` (without verification b0 = b).
   With `b0=None` this gives b0 = b − 1 for b ≥ 2 and **b0 = 1 at b = 1**. Level 0 spends b0 probes per (agent, family).
   The rest, n·K·(b − b0), is kept for verification.
2. **Tree.** Identical to MIDIAN w/o defenses (`_structure`). It uses the same `view.rng`, so at the same seed it builds
   the same leaf cohorts (§2.1.4).
3. **Verification probes per candidate**.
   - `C = top · Σ_{l≥1} (#valid child slots at level l)` is the number of forwarded candidates across all upper levels.
   - `e = floor((b − b0)·n / C)` fresh probes are spent per (candidate, family).
   - For r = 10, C ≈ n/9, so **e ≈ 9·(b − b0) = 9 at both b = 3 and b = 5**. Checked: e = 10 at n = 100 and e = 9 at
     n = 1,000.
   - **At b = 1, b − b0 = 0, so e = 0 and there is no verification at all.** MIDIAN at b = 1 is MIDIAN w/o verification
     with a cached root (DEVIATIONS "Erratum 22").
4. **Level 0, audited** (`_level0_audited`, `midian.py`): one round of b0 probes per (member, family).
   - Each member is probed b0 times per family. Each of the s − 1 cohort peers reports every outcome
     (`_est.peer_estimate`, `_est.py`).
   - Per-peer report sums and counts are stored in `rsum` / `rcnt` (accumulated), and each member's
     reporter ids in `peer_of`.
   - The estimate is `_estimates` :
     - First average each peer's reports about the member (the per-peer mean).
     - Mask the excluded peers (and peers with no reports), unless every peer is excluded.
     - Then apply a **trimmed mean over peers**, dropping `t = min(floor(δ·(s−1)), (v−2)//2)` peers from each end,
       where v is the number of peers left (`_est.py`). With 9 peers and none excluded, t = 3: the middle 3 of
       9 per-peer means are averaged.
5. **Audit during the build** (`_audit`, `midian.py`).
   - For every level-0 probe instance, the auditor draws uniformly with rate 0.05.
   - Each drawn instance is **re-run on the same index-seeded instance** with `view.probe_at` (`rte/world.py`,
     charged as a probe, probe index untouched). The true outcome is compared with **every** peer's report about that
     instance.
   - Each mismatch is a strike (`_strike`). A reporter with `STRIKES = 2` mismatches  is excluded from
     every later aggregation. Its later reports are still charged but never used.
   - An honest reporter reports the observed outcome verbatim, so it can only be struck if re-running the same instance
     gives a different outcome. That depends on backend determinism (see Part 1). A liar is struck only when its lie
     actually flips an outcome.
6. **Verification at promotion** (`_verify`, `midian.py`, called at every level l ≥ 1).
   - Every candidate forwarded by a child (its per-family best) is re-probed `e` times per family.
   - **Who reports.** The reporters are `self.rep[l−1]` of the **other** r − 1 children of the same parent node. `rep` is
     **one uniformly random member of each child's subtree** (comment: "reporters: RANDOM members of
     the sibling"). The docstring's "representatives lead[M, r]" is a parameter name. The `self.lead` array is not used for reporting.
   - With the default `observers = r − 1` all 9 report. Each report is trimmed by reporter exactly as at level 0. With
     audits on, excluded reporters are masked.
   - The new mean is folded into the candidate's running estimate, weighted by probe count:
     `est = (est·k + m_new·e)/(k + e)`, where k starts at b0. At b = 3, e = 9 outweighs the b0 = 2
     level-0 probes. A candidate forwarded again at the next level is verified again.
   - The child's summary value is rewritten with the verified estimate  before the parent compares its
     children.
7. **Build messages** are the same as MIDIAN w/o defenses.

### 2.3.4 Probe spend (checked by instantiating the classes on a bernoulli world, K = 16, β = 0.5 cartel, seed 1; measured before the rename, reproduced by the parity check)

| n | b | MIDIAN w/o defenses probes / n·K·b | MIDIAN w/o audits | MIDIAN |
|---|---|---|---|---|
| 100 | 1 | 1.000 | 1.000 | **1.045** |
| 100 | 3 | 1.000 | 1.000 | **1.031** |
| 100 | 5 | 1.000 | 1.000 | **1.037** |
| 1,000 | 1 | 1.000 | 1.000 | **1.050** |
| 1,000 | 3 | 1.000 | 0.997 | **1.029** (49,383 vs 48,000) |
| 1,000 | 5 | 1.000 | 0.998 | **1.037** |

- MIDIAN overspends by the audit re-runs: about 0.05·b0/b of the budget, which is ≈5% at b = 1, 3.3% at b = 3 and
  4% at b = 5.
- The floor in `e` makes the verification slightly under-spend.
- This matches DEVIATIONS: "Audited builds exceed the n*K*b cap by the audit rate by design" (2026-09-03, variants fork),
  and the "build spent 49380 probes > budget 48000" warning (2026-09-16 entry, recorded, not corrected).
- The runner only warns (`rte/run.py`, `run_method`), so these rows are kept.

### 2.3.5 Routing (`fetch` with `cached=True`, `midian.py`)

- The route returns `cand[-1][0, f]`, the agent the root holds for family f.
- It is charged 1 comparison, 2 messages and 0 hops.
- The choice is the same agent a full descent would reach. Only the charged cost differs.

### 2.3.6 Online updates (`Midian.observe`, `midian.py`)

- It first runs the running-mean update with pseudo-count `w0` computed from the full b, then a path
  recompute. With `cached=True` this also refreshes the cached root pick.
- Then, with audits on and probability 0.05, it runs an **online audit** :
  - The routed outcome is put to the routed agent's cohort peers through the report channel, charged s − 1 reports.
  - Each peer's claim is compared with the true outcome, adding strikes.
  - If a reporter is newly excluded, every member of that reporter's cohort is re-aggregated from the stored level-0
    per-peer means, and the cohort's path is recomputed for all K families.
- That re-aggregation **overwrites** those members' online running-mean updates and their promotion-verification folds.
  DEVIATIONS notes the first effect ("rare event; noted, not fixed", 2026-09-03 variants-fork entry). The loss of the
  verification fold is not mentioned there.
- **Per-task run cost:**
  - 1 comparison and 2 messages to route.
  - r·depth comparisons and depth messages for the path recompute.
  - On average 0.05·(r − 1) ≈ 0.45 audit reports.
  - **No probes at run time.**

### 2.3.7 Behaviour with liars

- MIDIAN never reads declarations, so declared lies have no effect on it.
- Report lies are fought three ways:
  1. **By-peer trimming** removes 3 of 9 peers per side, which is robust to a colluding peer's b correlated reports.
  2. **Audits** exclude a reporter caught twice.
  3. **Promotion re-probes** use reporters drawn from **other subtrees**, so a cohort-local cartel cannot vouch for its
     own member at the higher levels.
- **At b = 1 defence 3 is absent** (e = 0).

### 2.3.8 Hyperparameters

- r = 10 and δ = 1/3 are the tree's defaults.
- The audit rate 0.05 is `AUDIT_RATE` (`midian.py`, used when `audit=True`), and `STRIKES = 2` comes from the v2 plan (item 1.2).
- `b0 = b − 1` and `observers = r − 1` are defaults.
- None of these was tuned on reported seeds, as far as the code and grid comments show.

### 2.3.9 The "leaf cohort" and the MIDIAN-cohort shortlist (used by the frameworks' `va_cohort` source)

- With `retrieval="midian"` (before the rename `"midian_va"`), `FrameworkMethod.build` builds its own `Midian(r=self.r)`
  inside the framework unit (`rte/methods/frameworks/_common.py`). That instance spends its own probes, including
  the audit overspend. (`retrieval="midian_wo_audit"`, formerly `"midian"`, builds `Midian(audit=False)` instead; that
  cohort is not drawn.)
- `retrieve` then returns MIDIAN's pick first, followed by the other members of **that pick's leaf cohort**,
  `leaves[leaf_of[a]]` (`_common.py`). That is at most r = 10 agents, all probed and peer-reported together at
  level 0.
- The framework's `observe` forwards the outcome to the embedded MIDIAN (`_common.py`).
- The framework's `needs` gain probe and reports (`_common.py`) on top of declared. The embedded MIDIAN's
  `view.rng` is therefore seeded with a **different `needs` set** than a standalone `midian`, so its cohorts need not
  match the standalone arm's cohorts at the same seed.
- The source key stays `va_cohort` in code and CSVs (`shortlist_figs.source`, `figspec.SHORTLIST_*`); its display name is
  "MIDIAN cohort".

### 2.3.10 MIDIAN as the reference line in E, F and H ("MIDIAN, no framework")

- `shortlist_figs.collect` takes rows with `method == "midian"` and params `{}` (the full method, `shortlist_figs.py`)
  at b = 3 (`rows`) from `REF_GRIDS`, matched on (n, dist, regime). The grid with the most seeds wins.
  Its label in `results/aggregates/shortlist/*.csv` is `figspec.NAME["midian_ref"]` = "MIDIAN, no framework".
- The condensed E, F and H figures draw the **honest (β = 0)** line only, across both the honest and the cartel bars
  (`lines`, `scripts/figures/shortlist_condensed.py`: `ref.loc[(n, "beta0")]`), green solid (`figspec.ref`).
- In H the line comes from `routereval_mmlu_norep_cal` (m = 10, 100, 1,000) and `routereval5k_norep_cal` (5,000)
  (`H30_REF`, `shortlist_figs.py`), because every H erratum-30 grid is complete (`h30()`, §5.5): calibrated claims,
  no repeated test prompts (so MIDIAN can no longer memorise answers, the +0.02 of leakage audit L10 on the old
  `routereval_mmlu` line), shuffled agent order, Q = 300. Honest values 0.7080 / 0.6653 / 0.6860 / 0.7389 at m = 10 /
  100 / 1,000 / 5,000. It is drawn over every H shortlist.

---

## 2.4 The oracle (dotted line in A, B, E, F and H; the normaliser in B)

- `World.oracle(task) = argmax_a S[a, task.family]` (`rte/world.py`). It is the **only** routing rule that reads
  true skill S. Ties go to the lowest agent id (`np.argmax`).
- The runner runs it once per (cell, seed) in `oracle_line` (`rte/run.py`). It executes the chosen agent on the
  same task stream as every method (`world.execute`), so **oracle success is the realised success** of the best-S agent
  on these task instances, not max S itself.
- After a churn event it re-picks using the new S (`rte/run.py`); A/B cells have no churn.
- It is written as the row `method="oracle"`, with a zero build ledger (`rte/run.py`), and used as the regret
  baseline.
- **Liars do not affect it.** Liars execute at true skill, S and the task stream do not depend on β, and the liar RNG is
  separate (`rte/world.py`). The world seed does not include b either, so the oracle is also
  independent of b.
- In B, each group divides **both regimes'** bars by the **honest** cell's oracle mean
  (`scripts/figures/condensed_figs.py`), for all b. For live it comes from the b = 3 bar CSV; for the four switched families
  from the erratum-30 per-seed table at b = 3 (bernoulli 0.8462, replay 0.7919, RouterEval 0.9200, LLMRouterBench
  0.7500; live 10^5 0.8622).
- On split replay and on RouterEval / LLMRouterBench the oracle picks by probe-row / train-prompt S and executes on
  task / test rows, so it is not an exact ceiling there; no B bar exceeds it today.

---

## 2.5 Flat probe argmax, online (label "flat probe argmax")

**File:** `rte/methods/flat_probe_argmax.py`. **Needs:** `{probe}`. **Params:** `online=True`, `cached=False`
(ALIAS `rte/analyze.py`). docs/methods.md (SPEC §6) calls it "the key control": MIDIAN's probes without the tree or the
reports.

- **Build** : `probe_successes(view, b)` (`_est.py`), i.e. n·K·b probes, observed directly by the router
  (a **trusted observer**). The estimate is the probe mean, and the per-family argmax is stored.
- **Fetch** : `argmax(est[:, f])`, charged n comparisons (not cached). **Ties go to the lowest agent id.**
  - After b probes the estimates lie on the grid {0, 1/b, ..., 1}, so ties at the maximum are large.
  - DEVIATIONS reports about 116 agents tied at n = 10^3, b = 3, specialist, and about 30,000 at n = 10^5, b = 1
    (`docs/archive/DEVIATIONS.md`).
  - DEVIATIONS calls lowest-id tie-breaking "a different arbitrary rule, not a better one".
- **Observe** (`online=True`): running mean with count starting at b, then re-argmax of that family. The
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

- **Build:** collect D, charged n messages (`_decl.py`).
- **Fetch:** `argmax_a D[a, f]`, with an O(n) comparison scan. Ties go to the lowest id.
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

All of them spend exactly n·K·b warm-up probes as trusted observations (checked: ratio 1.000, except TrueSkill), learn online,
and charge n comparisons per fetch.

### 2.7.1 `ucb_per_family` (UCB1; `rte/methods/ucb_per_family.py`)

- **Needs:** `{probe}`. **Default:** c = √2.
- **Build:** probe means, `cnt = b`, and per-family pull count `t[f] = n·b`.
- **Fetch:** `argmax mean[:, f] + c·sqrt(ln t[f] / cnt[:, f])`. It is deterministic, and ties go to the
  lowest id.
- **Observe:** running mean, with cnt and t incremented.
- **Declarations:** ignored.

### 2.7.2 `thompson_per_family` (`rte/methods/thompson_per_family.py`, via `_est.BetaBandit`, `_est.py`)

- **Needs:** `{probe}`.
- **Prior:** Beta(1, 1) + b probes, so α = 1 + s and β = 1 + b − s.
- **Fetch:** one Beta sample per agent from `view.rng`, then argmax. It is random per fetch.
- **Observe:** α or β += 1.

### 2.7.3 `warm_start_bandit` (label `warm_start_bandit`, n0 = 5; and `warm_start_bandit[n0=0.5]`)

**File:** `rte/methods/warm_start_bandit.py`. **Needs:** `{declared, probe}`.

- **Prior** :
  - Charges n messages, for collecting the declarations.
  - D is clipped to [1e-3, 1 − 1e-3].
  - The prior is α0 = n0·D and β0 = n0·(1 − D).
- **Build:** the prior plus b probes (BetaBandit).
- **Fetch and observe:** Thompson sampling, as in 2.7.2.
- **It trusts declarations as a prior** worth n0 pseudo-observations: 5 against b = 3 real probes at the default.
- **Liars:** an inflated liar has D ≈ 1, so its prior failure count is ≈ 0 and its first observed probe failures
  dominate. CHANGES_AND_ERRATA §8d (`docs/archive/CHANGES_AND_ERRATA.md`) uses this to explain why this arm scores **higher
  under the cartel** than honest (live 10^4: 0.743 → 0.783; 10^5: 0.738 → 0.778). §8d calls the honest arm
  "under-tuned" and states that reported numbers keep the pre-registered n0 = 5.
- **Tuned variant `n0 = 0.5`:**
  - Chosen by `tune_wsb_n1000` (`configs/grids/`): live, n = 1,000, specialist, β = 0, self-described,
    b = 3, **seeds 11-15**, n0 ∈ {0.5, 1, 2, 5}.
  - The grid comment (`configs/grids/`) says seeds 11-15 are never reported (reported live 10^3 cells use
    seeds 1-10) and that the result was 0.762 for n0 = 0.5 against 0.745-0.750 for the others.
  - The tuned arm is run "beside" the pre-registered one in `tuned_wsb_n{100,1000,10k,100k}` at b = 3
     and in the live `rivals_b_n*` grids at b = 1 and 5.
  - In the condensed figures it only enters the best-bandit pool (`POOLS`, `scripts/figures/condensed_figs.py`), at every
    b and in every family, beside the pre-registered n0 = 5 arm. Its rows come from `tuned_wsb_*` (live b = 3), the live
    `rivals_b_n*` and `pool_fill_*` grids, `bernoulli_1e7_cal` (calibrated claims) and the other erratum-30 grids. Of the
    54 best-bandit bars in A and B it is the majority pick in 12 (the n0 = 5 arm in 17, `linucb_honest[bonus=own]` in 22,
    post-fix TrueSkill in 3).
  - The pre-registered n0 = 5 arm competes on live everywhere, and on bernoulli, replay, RouterEval and LLMRouterBench
    because all four have switched to their calibrated-claims rows (`CLAIM_KEY - SW` is empty, `condensed_figs.py`). On the old rows it was withheld, because its prior was then built on claims equal to true skill + 5 % noise.
    On calibrated claims it is the pick in every bernoulli 10^7 bandit bar (×100).
  - §8d as written does **not** mention the tuning run. The "CHANGES 8d" citation in `configs/grids/` points at the
    motivation, not at a record of the tuning.

### 2.7.4 `linucb_honest` and `linucb_honest[bonus=own]` (`rte/methods/linucb_honest.py`)

- **In the bandit pool** is only the fixed-bonus variant `linucb_honest[bonus=own]` (`POOLS`, `condensed_figs.py`), a
  post-hoc fix (rivals audit F1, 2026-09-23). Its rows come from the `linucb_fix_*` grids (`configs/grids/`: live, RouterEval-5k and LLMRouterBench cells and bernoulli 10^7, b = 1 / 3 / 5) and the erratum-30 grids
  (replay 10^6 gets it only from `replay_1e6_split_cal`). They have landed: it has every seed in every A / B pool table,
  and it is the majority best-bandit pick in 22 of the 54 bars (live 10^2 except honest b = 1, replay 10^6 except honest
  b = 1, every RouterEval 5,000 and LLMRouterBench bar).
- **Needs:** `{probe}`. **Default:** α = 1.0, `bonus="context"` (the pre-registered behaviour; `bonus` is recorded in
  params only when it is not `context`).
- It is a labelled v2 rival (2026-09-03). The code cites no paper. "Honest" means its context never uses model
  identity, specialty or declarations.
- **Context of agent a for family f:** `x = [1, mean_af, sqrt(cnt_af), mean over families of a]`.
- **Model:** a ridge model per family with `A_f = I + Σ x xᵀ` and `b_f = Σ x y`.
- **Warm-up:** each arm's b probes are added as b copies of (x, mean). This is equivalent to adding the b
  individual outcomes, because x is constant during the warm-up.
- **Fetch** : `argmax xᵀA⁻¹b + α·bonus`. With `bonus="context"` the bonus is `sqrt(xᵀA⁻¹x)` over the full
  context; agents tied at the top estimate are then separated by how atypical their cross-family mean is, which favours
  weak agents that got lucky, and LinUCB fell to random at 10^4–10^5 (code comment). With `bonus="own"` it is
  `sqrt(x_oᵀ A_o⁻¹ x_o) / sqrt(cnt_af)` over the agent's own (intercept, count) evidence only.
- **Observe:** updates A, b, the count and the mean.
- **Declarations:** ignored.

### 2.7.5 `trueskill_per_family` (`rte/methods/trueskill_per_family.py`)

- **Needs:** `{probe}`. Uses the `trueskill` package's default environment.
- **Build**, for each family:
  - Draw n·b//2 random pairs, with replacement, and drop pairs where a1 = a2. It therefore spends slightly **less** than
    the budget. Checked: ratio 0.989-0.999.
  - Probe each agent of a pair once and apply a 1-vs-1 update: a win, a loss, or a draw when the outcomes are equal.
  - The docstring says both are probed "on the same instance". **The code probes each agent at its own index-seeded
    instance**, because instance seeds depend on the agent id (`rte/world.py`).
- **Fetch:** argmax μ.
- **No online update**.
- It raises `NotImplementedError` at n ≥ 100,000, so it is absent from live 10^5, bernoulli 10^7 and
  replay 10^6.
- **Probe-duplicate fix (erratum 30).** Its random pairs draw agents with replacement, so one probe call often held the
  same (agent, family) several times, and `World._probe` used to give every occurrence the same instance. `World._probe`
  now gives each occurrence the next instances (`_occurrence`, `rte/world.py`); TrueSkill is the only
  arm that passes repeated cells, so only its rows change. `seed_tables.tables` counts TrueSkill only from
  `trueskill_fix_n{100,1000,10k}` (`configs/grids/`) and the erratum-30 grids (`seed_tables.py`).
  Those rows have landed (every seed at live 10^2–10^4, RouterEval 5,000 and LLMRouterBench, b = 1 / 3 / 5); the
  pre-fix rows in `LIVE_GRIDS`, `pool_fill_*` and the old RouterEval / LLMRouterBench grids are never read. Post-fix
  TrueSkill is the majority best-bandit pick in 3 bars (live 10^4 honest b = 1, 3 and cartel b = 1).

---

## 2.8 The "learned router" pool (candidates for "best learned/declared router")

### 2.8.1 `knn_router` and `knn_router_online` (`rte/methods/knn_router.py`)

- **Needs:** `{probe}`. The code docstring calls it "RouterBench's KNN predictive router (Hu et al. 2024) on our terms".
- **Build:** `probe_set` (`_learned.py`) probes every agent b times per family, n·K·b probes, **keeping the prompt
  of every probe**.
  - Each probe prompt is embedded with the backend's own vectors when it ships them (RouterEval RoBERTa), otherwise with
    all-MiniLM-L6-v2 on the prompt text (`_learned.py`).
  - Bernoulli has no real text: its "text" is `"A task of family X (instance i)."` (`rte/world.py`).
  - The embedding arithmetic is not charged in the ledger (`_learned.py`).
- **Predicted success of agent a** = the mean outcome of a's **k nearest** probes by cosine to the task's embedding.
  k defaults to b, so k = 1 at b = 1.
- **Fetch:** argmax over agents, charged n comparisons. Ties go to the lowest id.
- **Online variant** (`online=True`, ALIAS `knn_router_online`): appends every routed (task embedding, outcome) to the
  agent's store.
- **Liars:** it reads no declarations and no reports, so liars have no effect. Checked: the honest and cartel values are
  identical in A wherever the pool is complete.
- **Speed-only opt-ins** (CHANGES §8f, last bullet; all off by default, so default cells are unchanged). The kNN units of
  `rivals_b_n10k` (cartel), `rivals_b_n100k` and the cartel kNN units of `routereval5k_norep_cal` were run with:
  `RTE_EMBED_BATCH=1` (one batched MiniLM encode of the whole probe set instead of one call per prompt, the last
  embedding cached in-process keyed by the probe-instance matrix, read-only, so kNN / kNN-online and both regimes at
  one b share one encode; `_learned.py`), `RTE_MINILM_CUDA=1` (MiniLM on a GPU in fp32, `_learned.py`),
  `RTE_TEXT_PROCS=N` (probe prompts generated in N forked processes, identical texts, `_learned.py`),
  `RTE_OUTCOME_CACHE=1` (each (population, agent, family, instance) probe scored once per process, forked over
  `RTE_TEXT_PROCS`; identical outcomes, only the `llm_executions` diagnostic counts fewer calls,
  `rte/backends/llm.py`) and `RTE_RG_CACHE=1` (Reasoning-Gym's word-corpus read and regex scan cached for
  letter_counting / word_sequence_reversal; identical problems and scores, `rte/backends/families.py`). Outcomes and
  texts are identical; batched vs per-prompt embeddings differ at float level (≤ 2.4e-7) and gave identical top-10
  neighbour sets on 300 queries.

### 2.8.2 `mlp_router` (`rte/methods/mlp_router.py`)

- **Needs:** `{probe}`. The code cites RouterBench's MLP router (Hu et al. 2024).
- **Model:** one sklearn `MLPRegressor(hidden=128, max_iter=30, random_state=0)` on the input
  [agent one-hot (n dims) ⊕ prompt embedding], fit on the n·K·b probes.
  - `random_state=0` is fixed, so it does not vary with the world seed.
- **Fetch:** predict for all n agents, then argmax (n comparisons).
- **No online update.**
- The one-hot input makes memory O(n²·K·b). The grids exclude it at n = 10,000 and at the 5,000-LLM pool
  (`docs/archive/DEVIATIONS.md`; `configs/grids/` comment). There is no guard in the code itself.

### 2.8.3 `flat_nsw_router` (`rte/methods/flat_nsw_router.py`)

- **Needs:** `{probe}`. docs/methods.md calls it the "E7 flat rival".
- **Build:** the same trusted probe means as `flat_probe_argmax`, inserted into an hnswlib inner-product index
  (M = 16, ef = 50, ef_construction = 200). The index seed comes from `view.rng`.
- **Fetch:** a one-hot(f) query, i.e. an approximate argmax of `est[:, f]`, charged ⌈log2 n⌉ hops and ef comparisons. Which of the tied top agents it returns depends on the index seed (`docs/archive/DEVIATIONS.md`).
- **No online update.**

### 2.8.4 `cluster_head_router` (`rte/methods/cluster_head_router.py`): declared-only, no probes

- **Needs:** `{declared}`. docs/methods.md describes it as "AgentNet++-style prior art".
- **Build** :
  - Split the agents into random buckets of 20,000 (DEVIATIONS).
  - Run k-means on D inside each bucket, with k = ceil(size/10) and 5 iterations (+1 assignment pass).
  - The **head** of each cluster is the member with the highest **mean over families** of D.
- **Fetch** :
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
- **Fallback:** if nobody qualifies, route to the highest declarer (DEVIATIONS).
- It picks the **lowest-mean declarer that clears the threshold**, not the best one.
- **Liars:** it trusts declarations. **No online update.**

---

## 2.9 Methods that are run but excluded from A and B (brief)

| label | file | why it is not in A/B | one-line mechanism |
|---|---|---|---|
| `midian_wo_audit` (MIDIAN w/o audits, formerly `midian_v`) | `midian.py` | not in `ARMS`; in `figures/bars/` | §2.3.2 |
| `midian_wo_verify` (MIDIAN w/o verification, formerly `midian_a`) | `midian.py` | not in `ARMS`; in `figures/bars/` | MIDIAN without promotion verification and without the cached pick. At b = 1, MIDIAN ≡ MIDIAN w/o verification in its picks |
| successive-halving MIDIAN variants (formerly `midian_sh`, `midian_sha`) | deleted 2026-09-24 | withdrawn; rows moved to `_premigration_v2` backups (CHANGES_AND_ERRATA §8g) | successive halving inside each cohort, spending exactly s·b probes |
| `midian[r=5]`, `midian_wo_audit_r5`, r = 20 | `midian.py` | `excluded`: r ≠ 10 | same tree, different r |
| `midian[audit=False,online=False,verify=False]` | `midian.py` | `DO_NOT_ADD` | frozen internals ablation |
| `midian[cohort=...]`, `stratify=True` | `midian.py` | `_VARIANT` regex | budget-neutral cohort modes (`block`, `specialty`, `declared`; `declared` adds needs `declared`) |
| `midian_llm_descent` | `midian_llm_descent.py` | `DO_NOT_ADD` | an LLM chooses among children at each level; argmax fallback |
| `sequential_halving`, `sequential_halving_peer` | `sequential_halving.py` | every halving label is excluded (`scripts/figures/lib/exclusions.py`, since 2026-09-22); the plain trusted-observer arm is also in `DO_NOT_ADD` | per-family fixed-budget best-arm identification, n·b probes per family. `peer_reported` scores through r − 1 random peer reports, per-reporter trimmed |
| `route_to_k_majority` | `route_to_k_majority.py` | `DO_NOT_ADD`: executes 3 agents per task | top-3 by D; the runner majority-votes (`rte/run.py`) |
| `verify_on_claim` | `verify_on_claim.py` | not in any A/B pool; appears in D | rank by D, then at **fetch time** probe the top unverdicted candidate k = 3 times, accept if mean ≥ D − 0.15, up to 5 new verifications per fetch; verdicts cached. It spends **run-time** probes and no build probes |
| `declared_softmax` | `declared_softmax.py` | not in `ARMS`; appears in D | sample ∝ exp(D/0.1) |
| `cnp_self_bid` | `cnp_self_bid.py` | not in `ARMS`; appears in D | Contract Net: broadcast, bids D + N(0, 0.02), argmax; 2n messages |
| `gossip_reputation_greedy` | `gossip_reputation_greedy.py` | not in `ARMS`; appears in D | one random peer reports each probe; seedless EigenTrust; T-Man overlay; greedy walk |
| `referral_network` | `referral_network.py` | not in `ARMS`; appears in D | d = 10 regular graph; neighbour beliefs from single-observer reports; greedy referral walk |
| `llm_supervisor` | `llm_supervisor.py` | live only; not in A/B pools | framework adapter's retrieval + one direct supervisor call. Code default k = 20  |
| `flat_probe_argmax_frozen` | `flat_probe_argmax.py` | only the online version is in `ARMS` | same as §2.5 without `observe` |

---

## 2.10 Summary table

"Build probes" is relative to `Budget.total_probes = n·K·b`. "Trusts declarations" means the routing decision uses D.

| label in A/B | code name + params | reads (`needs`) | build probes | run-time probes | trusts declarations? | tuned? |
|---|---|---|---|---|---|---|
| MIDIAN | `midian` {} (audit and verify on; r = 10, δ = 1/3, b0 = b − 1, audit 0.05, 2 strikes, cached) | probe, reports | n·K·b·(1 + 0.05·b0/b): **+5% / +3.3% / +4%** at b = 1 / 3 / 5 (the floor in e trims slightly) | 0 (≈0.45 audit **reports** per task) | no | no (pre-registered V2-11; defaults) |
| MIDIAN w/o defenses | `midian` {audit: false, verify: false} (r = 10, δ = 1/3, online) | probe, reports | exactly n·K·b | 0 | no | no (SPEC §5) |
| flat probe argmax | `flat_probe_argmax` {online: true} | probe | exactly n·K·b | 0 | no | no |
| best learned/declared router ∋ | `knn_router` / `knn_router[online=True]` (k = b) | probe | n·K·b | 0 | no | no |
| | `mlp_router` (128 hidden, 30 iter, random_state 0) | probe | n·K·b | 0 | no | no |
| | `flat_nsw_router` (M 16, ef 50, efc 200) | probe | n·K·b | 0 | no | no |
| | `cluster_head_router` (r 10, 5 iter, bucket 20k) | declared | **0** | 0 | **yes** | no |
| | `disrouter_cascade` (τ 0.7) | declared, bus | **0** | 0 | **yes** | no |
| best bandit ∋ | `ucb_per_family` (c = √2) | probe | n·K·b | 0 | no | no |
| | `thompson_per_family` (Beta(1, 1)) | probe | n·K·b | 0 | no | no |
| | `warm_start_bandit` (n0 = 5) | declared, probe | n·K·b | 0 | **yes, as a prior** worth n0 | no (pre-registered) |
| | `warm_start_bandit[n0=0.5]` | declared, probe | n·K·b | 0 | yes, weak prior | **yes**: n0 picked on unreported seeds 11-15, live 10^3 honest |
| | `linucb_honest[bonus=own]` (α = 1; replaces the pre-registered context bonus) | probe | n·K·b | 0 | no | **post-hoc fix** |
| | `trueskill_per_family` (post-fix rows only) | probe | ≈0.99·n·K·b (pairs with a1 = a2 dropped) | 0 | no | no |
| declared argmax | `declared_argmax` {} | declared | 0 | 0 | **yes** | no |
| random | `random` | nothing | 0 | 0 | no | no |
| oracle (line) | `World.oracle` | **true S** (runner only) | 0 | 0 | no | n/a |

---

## 2.11 Cost per routed task (what the ledger charges; not drawn in A/B, but it defines the arms)

| arm | comparisons | hops | messages | reports |
|---|---|---|---|---|
| MIDIAN w/o defenses | r·depth (fetch) + r·depth (observe) | depth | 2·depth + depth | 0 |
| MIDIAN | 1 + r·depth | 0 | 2 + depth | ≈0.05·(r − 1) |
| flat probe argmax, bandits, knn, mlp | n | 0 | 0 | 0 |
| flat_nsw_router | ef = 50 | ⌈log2 n⌉ | 0 | 0 |
| cluster_head_router | #heads + cluster size | 2 | 4 | 0 |
| disrouter_cascade | 0 | pos | pos | 0 |
| declared_argmax | n | 0 | 0 | 0 |
| random | 0 | 0 | 0 | 0 |

---

## 2.12 Discrepancies and open questions

1. **MIDIAN w/o defenses trims reports, not reporters.** docs/methods.md §2 says plain MIDIAN (now MIDIAN w/o defenses) takes "a trimmed mean over reporters
   (drop ⌊δ(r−1)⌋ from each end) ... so up to that many liars per cohort are absorbed". The code trims `trim_k` = 3
   **single reports** per side out of (r − 1)·b (`_est.py`). DEVIATIONS states this correctly.
   docs/methods.md overstates MIDIAN w/o defenses' robustness at b ≥ 2.
2. **MIDIAN w/o verification / MIDIAN per-task cost.** docs/methods.md §2 says MIDIAN w/o verification costs "1.05× build probes, nothing per task". The code's
   online audit charges s − 1 reports on 5% of routed outcomes (`midian.py`), and the observe path recompute is
   charged too.
3. **MIDIAN overspend** is 5% of the level-0 probes, i.e. 5% / 3.3% / 4% of the budget at b = 1 / 3 / 5. It is not a
   flat "+5%" and not "≤ 5% of build" in the usual sense. The runner only warns (`rte/run.py`). A/B bars compare
   MIDIAN at up to 1.05 n·K·b against rivals at ≤ n·K·b.
4. **At b = 1, MIDIAN has no promotion verification** (e = 0), so its b = 1 bar is "MIDIAN w/o verification + cached root". This
   matches DEVIATIONS Erratum 22. Paper text should not describe MIDIAN's b = 1 bars as "verified".
5. **The bandit pool was edited post hoc.** `warm_start_bandit[n0=0.5]` (tuned on seeds 11-15) is in the pool at every b;
   `linucb_honest` was replaced by `linucb_honest[bonus=own]`, which is now the most frequent best-bandit pick (22 of 54
   bars); TrueSkill counts only from post-probe-fix rows. The pre-registered n0 = 5 arm would be withheld on unswitched
   non-live rows (`condensed_figs.py`); no family is unswitched, so it competes everywhere. None of this is
   marked on the figures (protocol audit F7).
6. **CHANGES §8d does not record the tuning.** §8d (`docs/archive/CHANGES_AND_ERRATA.md`) says "Reported numbers are
   unchanged (the rival is as pre-registered)" and does not mention the n0 = 0.5 tuning or its 0.762 result. The only
   record is the grid comment (`configs/grids/`). I found no results file confirming 0.762.
7. **The "best learned/declared router" pool contains declaration-only and non-learned methods.** It includes
   `cluster_head_router` and `disrouter_cascade` (no probes, declared-only) and `flat_nsw_router` (an ANN over probe
   means). `flat_nsw_router` is the majority pick in 19 of the 54 A / B bars (every replay 10^6 bar, bernoulli 10^7 at
   b = 3 / 5, most RouterEval 5,000 bars, live 10^4 b = 1 / 3); the declared-only pair in 3, all at b = 1. The legend
   label "best learned/declared router" names the declared-only pair but not the ANN index; the CSV's `chosen` column says which arm was picked.
8. **Candidates that ran on only some seeds.** `crossfit` lets a candidate compete only on the seeds it has
   (`seed_tables.py`). `arms_at` flags this: a member missing on any scored seed is listed as "missing or
   partial" in `chosen`, and the script and `cluster/ops/figure_status.py` report it (`condensed_figs.py`, §1.8.4).
9. **The whiskers of the pooled bars are ±1 s.e. of the cross-fitted per-seed scores.** They include the variation of
   the pick across seeds, but with 3 seeds (live 10^4 / 10^5, RouterEval) each seed's pick rests on 2 others (4 on
   LLMRouterBench, 29 on replay, 99 on bernoulli).
10. **TrueSkill docstring vs code.** The docstring says pairs are probed "on the same instance". The code probes each
    agent at its own index-seeded instance (`trueskill_per_family.py`, `world.py`).
11. **`_verify` docstring vs code.** The docstring says reporters are the "representatives lead[M, r]". The call passes
    `self.rep` (a random subtree member, `midian.py`). `self.lead` is computed  but never used
    for reporting.
12. **Padded upper-level nodes.** In `_verify` (`midian.py`), empty slots are filled with slot 0's representative.
    If the candidate's own child is slot 0, a reporter from the candidate's own subtree (possibly the candidate itself)
    can be used. Could not determine how often this occurs; it only affects short nodes.
13. **A new exclusion discards the verification fold.** When an online audit excludes a reporter, the cohort's estimates
    are rebuilt from level-0 reports only (`midian.py`), discarding promotion-verification folds as well as
    online updates. DEVIATIONS mentions only the online updates.
14. **Embedded MIDIAN vs standalone MIDIAN.** The framework adapter's embedded `Midian(r)` uses a `view.rng` seeded with
    the framework's `needs` ({declared, probe, reports}), so its cohorts differ from the standalone `midian` at the same
    seed. The E/F/H "MIDIAN, no framework" line and the MIDIAN-cohort (`va_cohort`) shortlist therefore do **not** share a
    tree.
15. **Reference-line filter (resolved by the rename).** Before the rename `shortlist_figs.collect` filtered reference rows by
    `method == "midian_va"` without checking params, so a REF_GRID row with non-default params (for example
    `cohort=block`, `r=20`) would have been pooled into the line. Since `midian` also names the ablations, the filter now
    requires params `{}` (`shortlist_figs.py`), which excludes every variant.
16. **Honest-only reference lines.** The E/F/H lines use the honest regime's MIDIAN and oracle for the cartel bars
    too (`shortlist_condensed.py`). B also normalises cartel bars by the honest oracle. The oracle is identical
    across regimes by construction, but MIDIAN is not.
17. **docs/methods.md minor points.**
    - `llm_supervisor` is listed with k = 10; the code default is k = 20 (`llm_supervisor.py`). The grids may
      override it; not checked.
    - `mlp_router` "does not fit above 10^4": it is in fact excluded **at** 10^4 by the grids (`docs/archive/DEVIATIONS.md`),
      and the code has no guard.
18. **MIDIAN's online pseudo-count overstates information (every variant).** MIDIAN's online `w0` counts reports (21 at b = 3),
    not independent probes (3), so a build estimate carries 21 pseudo-observations. For the variants with verification (MIDIAN, MIDIAN w/o
    audits), `w0` uses the full b while level 0 used b0 (`midian.py`). This is a design choice, not documented anywhere I found.

---

## 3. Agent frameworks and shortlist (retrieval) variants: how they are implemented

Scope: every framework arm (`fw_*`) and every shortlist that sits in front of the frameworks in condensed Figures E/F/G/H.
Everything here was checked against code on 2026-09-24. Paths are relative to the repository root unless they
start with `$RTE_DATA` (= `$RTE_DATA`). Where the code and the docs disagree,
the code is described here and the conflict is listed in 3.9.

---

## 3.1 What a framework arm is, in one paragraph

A framework arm does not route over the whole population. Per task it does four things:
1. It builds a **shortlist** of about k = 10 agents with a retriever (`FrameworkMethod.retrieve`).
2. It sends the **task text** plus the shortlist's `{name, description}` pairs to a real agent-framework library. That library runs in its own venv, as a JSON-lines subprocess.
3. It lets that library's own delegation primitive (the supervisor, manager, router or triage agent) name one agent. The run stops before the named agent executes.
4. It maps the name back to an agent id (`rte/methods/frameworks/_common.py`).

The supervisor LLM is `Qwen/Qwen2.5-7B-Instruct` (`_common.py`), served by the vLLM fleet. The arm never probes, never
reads reports, and never learns from outcomes, except when the shortlist itself is a MIDIAN cohort (3.4.10). When the
framework names no valid candidate, the adapter falls back to declared argmax within the shortlist (3.2.6).

---

## 3.2 The shared pipeline (`rte/methods/frameworks/_common.py`)

### 3.2.1 Class and constructor
- `FrameworkMethod(Method)` (`_common.py`) has `needs = {"declared"}`  and `requires_llm = True`. The View
  therefore raises `AccessError` on any probe, report or bus access (`rte/world.py`). The one exception is the MIDIAN
  retrieval modes, which add `{"probe", "reports"}` (`_common.py`).
- Constructor defaults (`_common.py`):
  - `k = 10`
  - `supervisor = Qwen/Qwen2.5-7B-Instruct`
  - `retrieval = "tfidf"`
  - `r = 10`
  - `dedup = False`
  - `embed_model = all-MiniLM-L6-v2` (`MINILM`, `rte/methods/_learned.py`)
  - `rerank_model = Qwen/Qwen3-Reranker-4B`
  - `rerank_pool = 50`
  - `embed_instruct = ""`
  - `shuffle = False`
  - `lie_text = False`
  - `claim_threshold = 0.0`
- Only `k, supervisor, retrieval, r` and any extra `**params` go into `self.params` (`rte/methods/base.py`).
  Those are what the worker receives in `req["params"]` (`_common.py`), which is how `mode=` reaches the MAF and
  LlamaIndex workers. The retrieval settings (`dedup`, `embed_model`, ...) are not forwarded to the worker. They are still
  recorded in the row's `params` JSON, because the runner writes the grid spec's params.
- Each subclass sets only `name`, `env` (venv name) and `worker` (script in `workers/`); see the table in 3.3.

### 3.2.2 `build(view, budget)` (`_common.py`)
1. `self._index(view)` computes the texts, retrieval vectors and shortlist tables (3.2.3).
2. `self.bridge = Bridge(self.env, self.worker)` sets up one worker subprocess, which is started lazily. `self._pre = {}`.
3. `self.base_url = _endpoint(supervisor)` reads `$RTE_DATA/endpoints.json`. It accepts the key `model` and any replica keys `model#<jobid>`,
   and picks one URL uniformly at random with an unseeded `np.random.default_rng()` (`_common.py`). The pick is
   repeated on every call, so replicas that join mid-run get traffic.
4. `view.ledger.message(view.n)`: every agent sends its description to the registry once.
5. MIDIAN modes only: it builds `Midian(r)` (retrieval `midian`, both defenses) or `Midian(audit=False, r)` (retrieval `midian_wo_audit`, MIDIAN w/o audits) on the same view and budget. **This spends the probe budget** (3.4.10).

### 3.2.3 `_index(view)` (`_common.py`)
`scripts/embed_routereval.py` calls exactly this function, on a GPU, to pre-warm caches.
- `self.desc, self.fdesc, self._task_text = self._texts(view)` (3.2.4). If `lie_text` is set, `self.desc = self._relabel(...)` (3.4.12).
- Agent names are `agent_{a:06d}`. `_name2id` inverts them.
- Vectors:
  - Dense modes (`embed`, `hybrid`, `sota`; `DENSE`): `self._Xa, self._Xf = self._embeddings(view)` (3.6).
  - Every other mode, including `declared` and `bm25` whose rankings never use it: one hashed TF-IDF matrix over `desc + fdesc`.
- BM25 matrix `self._B = _bm25(desc, fdesc)` (K × n) is built only for `bm25`, `hybrid` and `sota` (`LEXICAL`).
- Dedup pool: `self._pool` holds the lowest agent id per distinct description string, in sorted order. It is always
  computed and used only when `dedup=True`, and always for `sota`, whose cached table was built with dedup (3.6).
- `self._sota = self._sota_table(view)` for `sota` only.

### 3.2.4 What text the supervisor and the retriever see (`_texts`, `_common.py`)

**Live backend (`backend: llm`).** `_texts` returns the backend's own texts when `rte.backends.llm.current_backend()` is set
and has the same n :
- **Agent descriptions** come from `LLMBackend.descriptions()` (`rte/backends/llm.py`) and are cached in
  `$RTE_DATA/populations/<dist>_n<n>_K16_seed<s>/descriptions.json`.
  - Each is written by the agent's own base model. The system prompt asks for "one short paragraph (<=70 words) describing yourself as a service in an agent marketplace". The user prompt is `Your base model is {model}. Your tool is: {tool}. You are equipped for these task families: {specialty}...` (`rte/backends/prompts.py`).
  - The adapter then appends ` Declared areas: {true specialty list}.` (`llm.py`).
  - The text is generated per agent from its profile (true model, tool, specialty). **It does not depend on beta or on who lies.**
  - Descriptions can name the base model, for example "As an agent in the Qwen/Qwen2.5-0.5B-Instruct model ...", as seen in `specialist_n1000_K16_seed1/descriptions.json`. The text therefore carries a model-size signal.
- **Family descriptions** (the retrieval queries) are `families.describe(f)`, which returns `"<family name> problems. Example question: <one example question, ≤240 chars>"` (`rte/backends/families.py`).
- **Task text** (what the supervisor is asked to route) is the real task question, `families.question(family, instance)` (`llm.py`).

**Every other backend, including RouterEval.** `_texts` falls back to text rendered from the declaration matrix (`_common.py`):
- Agent: `"Self-rated competence: <fam> 0.xx, <fam> 0.xx, ..."`, listing the agent's **top-5 families by `view.declared`**, with two decimals.
- Family query: `"Tasks of family <fam>"`.
- Task text: `"A task of family <fam> (instance <i>)."`

**Consequences on RouterEval:**
- The text **is** the declared channel. `view.declared` already includes the lie (`world.py/281`, `apply_lying`, `world.py`: `inflate` adds `DELTA_INFLATE = 0.4`, clipped to 1, `world.py`). A liar's inflated numbers therefore appear in its description, and in which families make its top 5.
- **The supervisor never sees the actual MMLU prompt, only the subject name.**
- Example printed at n = 1,000, strong_to_weak, seed 1, on the old programmatic world (the calibrated world H reads renders the same format with the 11 calibrated values): desc `"Self-rated competence: high school biology 0.70, elementary mathematics 0.70, philosophy 0.69, ..."`, query `"Tasks of family professional law"`, task `"A task of family professional psychology (instance 7)."`

### 3.2.5 `retrieve(task)` (`_common.py`)
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
  - Any unrecognised `retrieval` string falls silently into the hashed-TF-IDF branch ( and).

### 3.2.6 `fetch(task)`: calling the framework, parsing, fallback (`_common.py`)
1. `cand = retrieve(task)`. Ledger: `compare(len(cand))`, `hop(1)`, `message(len(cand)+2)` for k descriptions read plus request and reply.
2. Payload: `[{"name": "agent_000123", "description": desc[a]} for a in cand]`, in shortlist order.
3. `resp` is the prefetched response if one exists (3.2.8), otherwise `bridge.select(...)`. If `resp["error"]` is set and is not a supervisor invalid action, the call is retried once.
4. **Supervisor invalid action** (`INVALID_ACTION`; erratum 29, `docs/archive/CHANGES_AND_ERRATA.md` §8e). An error matching `Tool '…' not found` (ADK, OpenAI Agents: a tool named after the agent instead of the routing tool), `ModelBehaviorError`, `next_speaker must be provided` (MAF) or `KeyError: 'agent_NNNNNN'` (MAF's group-chat orchestrator naming a participant that is not among its candidates, so the worker's `back[...]` name lookup fails; the pattern is not tied to one framework, so the same KeyError from the OpenAI Agents or LlamaIndex handoff lookup would also match) is the framework failing to route, not infrastructure:
   - it increments `stats["invalid_action"]`, sets `_picked = False` and routes by declared argmax over `cand`;
   - it is **not retried** (no other non-pick gets a second sample) and never counts as an infrastructure error;
   - it enters `fallback_rate` and scores 0 under `success_strict`.
   Rows written before this rule are unchanged; the units it had failed (31 ADK, `re_sl_declared_1k` MAF seed 3, `re_sl_declared_5k` OpenAI Agents seed 1, and the 14 MAF units at 10^2–10^4 that failed with the KeyError) are rerun.
5. **Infrastructure error, still failing after the retry** :
   - It increments `stats["infra_errors"]`, sets `_picked = False` and routes by declared argmax over `cand`.
   - Once `infra_errors > max(3, 0.02 × calls)`, it raises `RuntimeError("... refusing to write a fallback-contaminated row")`, so the unit writes **no row**.
   - This guard was added after erratum 28 (`docs/archive/CHANGES_AND_ERRATA.md` §8c; `docs/operations.md` D1).
6. **Valid pick:** `choice` is a known name whose id is in `cand`. Then `stats["picks"] += 1` and the arm routes to it.
7. **Otherwise it is a genuine framework non-pick.** It is classified as follows :
   - `failures` if the worker's `raw` begins with `"FAILURE"`, meaning the supervisor answered the task itself instead of delegating. The CrewAI, ADK and Magentic-One workers emit this.
   - `fallbacks` if `choice is None`.
   - `bad_name` if the model named a non-candidate or a non-agent, for example smolagents calling its `final_answer` tool.
   - In all three cases the adapter routes to **declared argmax within the shortlist**.
8. **Scoring:**
   - The row's `success` is **lenient**: the fallback agent is executed and scored.
   - `success_strict` counts an outcome only when `_picked` is true (`observe`).
   - `fallback_rate = 1 - picks/(picks+fallbacks+failures+bad_name+invalid_action)`. `infra_errors` is excluded from the denominator.
   - All of these go in the row's `method_stats` JSON (`rte/run.py`).
9. **What "fallback rows" or "contamination" means (erratum 28).**
   - Before 2026-09-22, broken venvs made workers crash. `choice=None` then fell through to declared argmax, and the row looked normal while actually measuring declared argmax under the framework's name.
   - `cluster/archive/quarantine_fallback_rows.py` moved such rows to `results/<grid>/quarantine/`. The criterion was fallback > 0.05 for CrewAI and ADK, and ≥ 0.9 for every other framework. The units were then rerun.
   - Genuine non-picks are still routed by declared argmax. So even clean rows contain a declared-argmax component equal to `fallback_rate`, and that component is **the only way the inflate lie reaches a text-retrieval arm on live** (erratum 27).

### 3.2.7 What the framework does NOT get
- **No probes, no reports, and a build probe budget of 0.** The View denies access (`needs={"declared"}`), except in the MIDIAN modes.
- **No outcomes.** `observe` only updates `stats`, except in the MIDIAN modes, where it forwards the outcome to MIDIAN.
- **No declared numbers in its prompt on live.** Only name and description are sent. Declared numbers are used by the adapter for `retrieval="declared"` and for the fallback. On RouterEval, though, the description *is* the rendered declared vector.
- **No true skill, no liar labels, and no family label on live.** On live the supervisor sees the real question text. On RouterEval it sees the family name and nothing else.
- **A fresh team per request, with no memory.** Every worker constructs a new team inside `select()`.

### 3.2.8 Parallel prefetch (`RTE_FW_PARALLEL`, `_common.py`)
- If `RTE_FW_PARALLEL > 1` and the retrieval is not a MIDIAN mode:
  - `prefetch(stream)` opens N extra Bridges (one worker process each).
  - It sends every task's request concurrently through a `ThreadPoolExecutor`.
  - It stores the responses by `id(task)`.
  - `fetch` then consumes them in order and falls back to a live call on an error.
- `rte/run.py` calls it before the task loop, except on churn grids. Its wall time is charged to routing.
- It is sound because each request is stateless and uses temperature 0.
- `tests/test_fw_sota_retrieval.py` shows identical picks, stats and ledger at 1 and 8 for tfidf, bm25 and declared. That test uses a mock endpoint.
- Against real vLLM, results are only as reproducible as greedy decoding:
  - Replicas are chosen at random.
  - Batch composition varies.
  - No `seed` is sent by any worker.
- So parallelism does not change results *systematically*, but bit-identity with a sequential run is not guaranteed on the real fleet, and it is not guaranteed between two sequential runs either.
- `launch_units.sh` passes `RTE_FW_PARALLEL` (default 1). `docs/operations.md` T1 prescribes 8, with `--mem=40G`.

### 3.2.9 The supervisor LLM and serving
- **Model.** `Qwen/Qwen2.5-7B-Instruct` (`_common.py`) for every framework. The exception is one labeled arm, `fw_magentic_one` with `supervisor: Qwen/Qwen2.5-14B-Instruct`, which runs in `fw_live_n1000`, `fw_live_n100` and their `_lowskill`, `_dd` and `_em` mirrors. That arm is dropped from the shortlist figures (`source()`, `scripts/figures/shortlist_figs.py`).
- **Endpoints.** `$RTE_DATA/endpoints.json`: keys `model#<slurm job>` are replicas.
- **Supervisor-only replicas.** `configs/fleet_supervisor.yaml` places just the 7B at `gpu_share 0.90` on one GPU. `serve_fleet.sbatch` selects the models through `RTE_FLEET_PLACEMENT` and notes that "the 7B supervisor took 487,222 requests in 16 h".
- **vLLM flags** (`scripts/_serve_env.sh`):
  - `--generation-config vllm` ignores Qwen's own repetition_penalty and temperature defaults.
  - `--enable-auto-tool-choice --tool-call-parser hermes` is needed by the tool-calling frameworks.
  - `--enable-prefix-caching`, max-model-len 8192, max-num-seqs 64.
  - `VLLM_USE_FLASHINFER_SAMPLER=0`.
- **Temperature 0 is set client-side in every worker**: LangGraph `ChatOpenAI(temperature=0)`, CrewAI `LLM(temperature=0)`, AutoGen `OpenAIChatCompletionClient(temperature=0)`, MAF `default_options.temperature=0.0`, OpenAI Agents `ModelSettings(temperature=0.0)`, ADK `LiteLlm(temperature=0.0)`, LlamaIndex `OpenAILike(temperature=0.0)`, smolagents `OpenAIServerModel(temperature=0.0)`, CAMEL `model_config_dict={"temperature": 0.0}`, AgentScope `Parameters(temperature=0.0)`.

### 3.2.10 The Bridge subprocess protocol (`rte/methods/frameworks/_bridge.py`)
- **Protocol**, one JSON object per line.
  - Request: `{"id", "task", "candidates": [{"name","description"}], "model", "base_url", "api_key", "params"}`.
  - Response: `{"id", "choice": str|null, "error": str|null, "raw"}`.
- **Interpreter.** `$RTE_DATA/env/<env>/bin/python`.
- **Worker environment** : `PYTHONPATH=workers/`, proxies stripped, `OPENAI_API_KEY=EMPTY` by default, `PYTHONNOUSERSITE=1`.
- **Reads.** Per-request timeout of 120 s. Non-JSON stdout lines are skipped.
- **Failures.** On a timeout, dead worker or bad JSON, the worker is killed, `{"error": ...}` is returned, and the worker restarts on the next call.
- **Worker side.** `serve_worker(select_fn)` never dies on one bad request: an exception becomes `resp["error"]`.
  - **An exception raised inside a worker is therefore reported as an infrastructure error, not as a framework non-pick.** Examples:
    - LlamaIndex's selector failing to parse the model output.
    - AgentScope `json.loads` failing.
    - a `back[...]` KeyError in MAF, OpenAI Agents or LlamaIndex handoff, **unless** its message is `KeyError: 'agent_NNNNNN'` (a named non-candidate), which `INVALID_ACTION` reclassifies as a non-pick (erratum 29, §3.2.6).
  - Such errors count toward the 2 % unit-failure guard.

---

## 3.3 The frameworks

The code lists 12 `fw_*` classes (`rte/methods/frameworks/fw_*.py`).
- The **ten grid frameworks** are the YAML set `frameworks` (`configs/grids/`): LangGraph, CrewAI, AutoGen, Magentic-One, MAF, OpenAI Agents SDK, Google ADK, LlamaIndex, smolagents and CAMEL Workforce.
- **AgentScope** appears only in the appendix grids `fw_appendix` and `fw_appendix_dd` (n = 100, specialist, beta ∈ {0, 0.25}, seeds 1-3, Q = 300; `configs/grids/`).
- **Echo** is a protocol check that picks the first candidate (`fw_echo.py`, `workers/echo_worker.py`) and is never in a grid.

Versions are the pins in `requirements-frameworks/<fw>.txt`. I confirmed each against the installed `*.dist-info` in `$RTE_DATA/env/<env>/lib/python3.12/site-packages`, and they match.

| arm (`name`) | venv (`env`) | library, pinned = installed | worker | selection primitive driven, and how the pick is captured | what the supervisor reads |
|---|---|---|---|---|---|
| `fw_langgraph` | fw_langgraph | langgraph 1.2.11, langgraph-supervisor 0.0.31, langchain-openai 1.6.0 | `langgraph_worker.py` | `create_supervisor(agents, prompt=...)`, with one `create_react_agent(model, tools=[], name=...)` per candidate. The pick is the first tool call named `transfer_to_<name>`; the stream is then closed, aborting the graph  | Its prompt holds the roster `- name: description`, "Assign the user's task to exactly one agent by calling its transfer tool. Do not do any work yourself.", plus the task as the user message |
| `fw_crewai` | fw_crewai | crewai 1.15.18 | `crewai_worker.py` | `Process.hierarchical` crew with an explicit `manager_agent` "Dispatcher". The coworkers are `Agent(role="<name>: <description>", backstory=description)`. `BaseAgentTool._execute` is monkey-patched to raise `Picked(coworker)`, and the name is taken with the regex `agent_\d{6}`. A kickoff with no delegate call returns `"FAILURE: manager answered itself"`  | The manager's delegate-tool description lists the coworker roles; the task is "Delegate this to exactly one coworker: {task}" |
| `fw_autogen` | fw_autogen | autogen-agentchat 0.7.5, autogen-ext[openai] 0.7.5 | `autogen_worker.py` | `SelectorGroupChat(Idle participants, allow_repeated_speaker=True, MaxMessageTermination(2))`. The pick is the first `SelectSpeakerEvent`. Participants are `Idle` agents, so none executes. **Library-internal fallback:** after 3 failed selector attempts, AutoGen itself returns the previous speaker, or else `participants[0]`, which is shortlist position 1 (`autogen_agentchat/.../_selector_group_chat.py` in the venv). The adapter counts that as a valid pick | The selector prompt's roster `"<name>: <description>"` |
| `fw_magentic_one` | fw_autogen (shared) | same as AutoGen | `magentic_one_worker.py` | `MagenticOneGroupChat(max_turns=1)`, with the ledger requested as structured output. `_log_message` is patched to catch `"Next Speaker: <name>"`. With `robust=True`, the default and never overridden in any grid, a rejected ledger is read by mention: the first `agent_\d{6}` after "next_speaker". A satisfied ledger, or one naming nobody, is a `FAILURE`  | The orchestrator's `{team}` roster built from the descriptions |
| `fw_maf` | fw_maf | agent-framework-core 1.16.0, -openai 1.14.1, -orchestrations 1.1.1 | `maf_worker.py` | Default `mode="groupchat"` (`fw_maf.py`; no grid sets `handoff`): `GroupChatBuilder(participants, orchestrator_agent=router, max_rounds=1)`, and the pick is the first `GroupChatRequestSentEvent.participant_name`. `handoff` mode, never run, would use `HandoffBuilder` and `HandoffSentEvent.target`. Names are sanitised to identifiers and mapped back | Router instruction "You are a router. Pick the single participant best suited to solve the task. Do not solve it yourself." plus the participants' `name` and `description` |
| `fw_openai_agents` | fw_openai_agents | openai-agents 0.22.0 | `openai_agents_worker.py` | Triage `Agent(handoffs=specialists)`. Each specialist has `handoff_description=description`, which becomes a `transfer_to_<name>` tool. `RunHooks.on_handoff` raises `Picked(to_agent.name)`. `max_turns=1`, `use_responses=False`. If the run ends without a handoff, the result is `None` → counted as `fallbacks` | "You are a triage agent. Hand the task off to the single specialist best suited to solve it." |
| `fw_google_adk` | fw_google_adk | google-adk 2.8.0, litellm 1.99.0 | `google_adk_worker.py` | Router `LlmAgent(sub_agents=...)` with auto-delegation via `transfer_to_agent`. The pick is the first `event.actions.transfer_to_agent`. No transfer returns `"FAILURE: router answered itself"`  | ADK's roster of `Agent name:` / `Agent description:`, plus "You are a router. Transfer the task to the single sub-agent best suited to solve it." |
| `fw_llamaindex` | fw_llamaindex | llama-index-core 0.14.24, llama-index-llms-openai-like 0.8.0 | `llamaindex_worker.py` | Default `mode="selector"` (`fw_llamaindex.py`; no grid sets `handoff`): one `LLMSingleSelector.aselect([ToolMetadata(name, description)], task)` call, taking `selections[0].index`. An out-of-range index gives `None` → `fallbacks`. A parser exception becomes a worker error  | The selector prompt listing the numbered descriptions |
| `fw_smolagents` | fw_smolagents | smolagents 1.26.0 | `smolagents_worker.py` | Manager `ToolCallingAgent(managed_agents=...)`, `max_steps=1`, `run(stream=True)`. The pick is the first `ToolCall` event's name, taken before execution. A `final_answer` call → `bad_name` | Managed agents as tools, `name` plus `description` |
| `fw_camel_workforce` | fw_camel | camel-ai 0.2.90 | `camel_worker.py` | `Workforce` with one `add_single_agent_worker(description, ...)` per candidate. `node_id` is overwritten with the candidate name. A `WorkforceCallback.log_task_assigned` raises `Picked(worker_id)`  | The coordinator's `<node_id>:<description>:<toolkits>` list |
| `fw_agentscope` (appendix) | fw_agentscope | agentscope 2.0.7 | `agentscope_worker.py` | No selection primitive, so this is a DIY router: one `OpenAIChatModel` call json_schema-constrained to `{"agent": name}`  | Roster plus task in a hand-written prompt |

Notes that apply to all of them:
- Every worker stops **before the chosen agent runs**, so the only LLM traffic is the supervisor's. The routed task is then executed by the benchmark world, not by the framework.
- The agents inside the framework are placeholders; only the manager or router makes model calls.
- Names such as `agent_000123` are already identifier-safe, so `sanitize` (`workers/_wk.py`) is a no-op on them.
- **Magentic-One exclusions.** It is left out of every RouterEval framework grid and of `fw_live_n10k_cartel` and its `_dd`, `_em`, `_sota`, `_shapes` and `_verified_va` mirrors, "for time (18 s/task)" (`configs/grids/`; `docs/archive/DEVIATIONS.md` §"Frameworks on RouterEval's 5,000-LLM pool"). It **is** included in:
  - `fw_live_n10k_cartel_backfill` (it inherits `*fw10_backfill`, all ten frameworks);
  - `fw_live_n10k_cartel_magentic` (`configs/grids/`): Magentic-One alone at the 10^4 cartel, under TF-IDF, MiniLM,
    fusion + reranker and the MIDIAN cohort, with the honest grids' params;
  - `lie_max_fw_n10k` (`*fw10_declared`);
  - every other live grid.
- The CAMEL rows of `fw_routereval_1k` are partial: 8/30 units, excluded from the published ranges according to DEVIATIONS.

---

## 3.4 Every shortlist variant

The source key comes from `source(params)` in `scripts/figures/shortlist_figs.py`; its display name from `figspec.SHORTLIST_BODY` / `SHORTLIST_APPENDIX` (condensed figures) or `shortlist_figs._LONG` (per-condition figures). It works as follows:
- Rows with `supervisor` or `lie_text` in their params → not drawn.
- `retrieval` absent: `dedup` → not drawn; otherwise `tfidf`.
- `midian` (the full method's cohort; `midian_va` before the rename) with `r == 10` and no `shuffle` → `va_cohort`; `midian_wo_audit` (formerly `midian`) → not drawn.
- `embed` → `dense[_icomp|_idemo]` if the `embed_model` contains "Qwen", else `embed` (MiniLM).
- `sota` → `sota[_icomp|_idemo]`. The tag comes from the substring "competent" or "demonstrated" in `embed_instruct`.
- `bm25` and `declared` keep their names.
- Everything else (`hybrid`, `midian`, `tfidf` + dedup) → not drawn.

Display names come from `SOURCES` (`shortlist_figs.py`); short legend names come from `SHORT` (`scripts/figures/shortlist_condensed.py`).
- Fig E draws `MAIN`, which is every source except `bm25` and `dense`.
- Fig H draws `MAIN` too, on the RouterEval rows. Every RouterEval grid named in 3.4.1–3.4.10 (`fw_routereval_*`, `re_sl_*`) reaches H only through its `_norep_cal` mirror (calibrated claims, no repeated prompts, shuffled agents; §5.5); the rows of the grids themselves are no longer drawn.

### 3.4.1 `tfidf`: pre-registered hashed TF-IDF (condensed label "hashed TF-IDF"; "hashed TF-IDF (pre-registered)" in `figures/shortlist/`)
- **Params:** none, i.e. the defaults `retrieval="tfidf"`, `dedup=False`, `k=10`.
- **Grids:**
  - `fw_live_n1000` and `fw_live_n100` (+ `_lowskill`)
  - `live_n10k_v2` and `live_n100k` (their `*frameworks` arms)
  - `fw_live_n10k_cartel`, `fw_live_n10k_beta01`, `fw_live_n10k_shapes`, `fw_live_n10k_cartel_shapes`, `live_n100k_beta01`
  - `fw_routereval_small`, `_1k`, `_5k`
- **Algorithm** (`_hash_tfidf`, `_common.py`):
  1. Tokens are `[a-z0-9]+` on lowercased text, so `basic_arithmetic` splits into `basic` and `arithmetic`.
  2. Each token is hashed with `blake2b(digest_size=4) mod 4096`, giving counts in 4,096 buckets.
  3. IDF is `log((N+1)/(df+1))` with N = n + K. It is fitted on the agent descriptions **and** the 16 family queries together.
  4. Rows are L2-normalised, and similarity is cosine against the family description.
- **Erratum 25: clone shortlists** (`docs/archive/CHANGES_AND_ERRATA.md` item 25).
  - Agents that share a prompt signature have byte-identical memoized descriptions and identical answers.
  - The stable sort therefore fills the top 10 with clones of the best-matching text.
  - bimodal has 2 distinct texts and heavy_tail 5, at any n. specialist has ~3,920 distinct texts, giving ~4.4 signatures per top 10 at 10^3, ~2.6 at 10^4, and **exactly 1 at 10^5**.
  - At 10^5 every framework therefore gets ten copies of one agent, and its choice cannot matter. `shortlist_condensed.py` states that TF-IDF is "0.379 for every framework" at 10^5.
  - **Fig G's baseline is this clone floor.** Every other drawn text arm uses `dedup: true`, so Fig G's "gain over TF-IDF" includes the dedup gain.
- **Liars:**
  - Live: they cannot move the ranking, because the text is independent of lying (erratum 27). They reach the row only through the fallback.
  - RouterEval: they **can**. The query `"Tasks of family X"` matches every agent whose declared top 5 contains X. Inflation (+0.4, clipped to 1) pushes liars into many top-5s and, with ties broken by id, to the front. Check (routereval mmlu n = 1,000, strong_to_weak, beta 0.5 low_skill_first, seed 1, family 0): on the calibrated, shuffled world H reads, the TF-IDF top 10 holds **7 liars** (mean true skill 0.466 against a population mean of 0.428); on the old programmatic world it was 10 of 10 (mean 0.276). See 3.9 on the "degenerate" wording.

### 3.4.2 `dedup` (TF-IDF + `dedup: true`): computed but not drawn
- **Params:** `{dedup: true}` (`&fw10_dd`, `configs/grids/`).
- **Grids:** `fw_live_n1000_dd`, `fw_live_n100_dd`, the `_lowskill_dd` pair, `fw_live_n10k_dd`, `fw_live_n100k_dd`, `fw_live_n10k_cartel_dd`, `fw_k_sensitivity_dd`, `fw_appendix_dd`, `budget_b10_fw_dd` and `churn_n1000_fw_dd`.
- **Mechanism:** the same ranking, restricted to `_pool`, the lowest id per distinct description string (`_common.py`).
- **Figures:** excluded by `source()`.
- **Liars:** the same as TF-IDF. The one difference is which member of a clone group is offered: the lowest id, whatever its liar status.

### 3.4.3 `bm25` (label "BM25")
- **Params:** `{retrieval: bm25, dedup: true}` (`&fw10_bm25`, `configs/grids/`).
- **Grids:**
  - `fw_live_n1000_sota` only, at n = 10^3, all 3 shapes × 4 random-liar betas, seeds 1-10. This is the "four-arm ablation".
  - Also `fw_live_n100k_lietext` (honest and lied text). The `GRIDS` filter excludes `lietext` grids, so these never reach the figures (`shortlist_figs.py`).
- **Algorithm:** Okapi BM25 with k1 = 1.5 and b = 0.75 over an inverted index; IDF is `log(1 + (N − df + 0.5)/(df + 0.5))`. Documents are the n agent descriptions and queries are the K family descriptions, including the example question's tokens, numbers too (`_common.py`).
- **Figures:** Fig E leaves it out (`MAIN`).
- **Liars:** text only, so the same as TF-IDF.

### 3.4.4 `embed`: MiniLM (label "MiniLM")
- **Params:** `{retrieval: embed, dedup: true}`, with `embed_model` at its default `all-MiniLM-L6-v2` (`&fw10_em`, `configs/grids/`; `&fw9_em`, `configs/grids/`).
- **Grids:**
  - `fw_live_n1000_em`, `fw_live_n100_em`, the `_lowskill_em` pair, `fw_live_n10k_em`, `fw_live_n100k_em`, `fw_live_n10k_cartel_em`
  - `fw_routereval_1k_em`, `_5k_em`, `_small_em`
- **Algorithm:** cosine between unit-norm MiniLM embeddings of the descriptions and the family descriptions. There is no query prompt (`_common.py`: `qp=None` for MiniLM), and it runs on CPU (`_learned.py`).
- **Cache:**
  - Live: `descriptions_minilm.npy` in the population dir.
  - RouterEval: no cache, unless `RTE_EMBED_CACHE_DIR` is set. It is recomputed on CPU per unit, which is cheap.
- **Liars:** live text only. RouterEval rendered numbers, so manipulable as in 3.4.1.

### 3.4.5 `dense`: Qwen3-Embedding-8B with its stock instruction (condensed label "dense (Qwen3-8B)", appendix "dense (Qwen3-8B), no instr."; "Qwen3-8B dense" in `figures/shortlist/`)
- **Params:** `{retrieval: embed, dedup: true, embed_model: Qwen/Qwen3-Embedding-8B}` (`&fw10_dense`, `configs/grids/`).
- **Grids:** `fw_live_n1000_sota` only (the 10^3 ablation). Not in Fig E's `MAIN`.
- **Query side:**
  - With no `embed_instruct`, the family texts are encoded with `prompt_name="query"`. That is the model's shipped prompt `"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"` (`$RTE_DATA/models/Qwen3-Embedding-8B/config_sentence_transformers.json`).
  - Documents get no prompt (`_common.py`).
- **Hardware:** bf16 on GPU (`_learned.py`). A routing unit with no cached block and no GPU raises instead of computing (`_common.py`).

### 3.4.6 `dense_icomp` / `dense_idemo`: Qwen3 dense with a task instruction (appendix labels "dense, competence instr." / "dense, demonstration instr."; in the body figures the better of the variants is drawn as "dense (Qwen3-8B)"; "Qwen3-8B dense, I-competent" / "I-demonstrated" in `figures/shortlist/`)
- **Exact instruction strings** (`configs/grids/`; each appears on 30 / 20 lines, plus 18× on each of the three flow-style `re_sl_embed_*` lines):
  - **I-comp:** `"Given a task family, retrieve agents that are competent at solving tasks of that family"` (cache tag `_i0258be9d`)
  - **I-demo:** `"Given a task family, retrieve agents whose demonstrated skill at that family is highest"` (cache tag `_i06c7d226`)
- **Query prompt:** `"Instruct: {embed_instruct}\nQuery:"` (`_common.py`). Only the K = 16 family vectors change (`families_<slug>_i<hash>.npy`); the n document vectors are reused.
- **Params:** `{retrieval: embed, dedup: true, embed_model: Qwen/Qwen3-Embedding-8B, embed_instruct: <string>}` (`&fw10_dense_ic`, `&fw10_dense_id`, from `configs/grids/`).
- **Grids:**
  - `fw_live_n100k_dense_instruct`
  - the backfill grids `fw_live_n{100,1000}[_lowskill]_backfill`, `fw_live_n10k_backfill`, `fw_live_n10k_cartel_backfill` (`configs/grids/`)
  - `re_sl_embed_{small,1k,5k}` (`configs/grids/`)
- **Liars:** live text only. On RouterEval, rendered numbers (manipulable).

### 3.4.7 `hybrid`: RRF(BM25, Qwen3 dense), not drawn
- **Params:** `{retrieval: hybrid, dedup: true, embed_model: Qwen/Qwen3-Embedding-8B}`, plus an I-comp variant in `fw_live_n100k_sota_instruct`.
- **Algorithm:** `_rrf` gives each list the score `1/(60 + 1 + rank)` and sums the two lists, ranking over the pool only (`_common.py`).
- **Grids:** every `*_sota` grid and `fw_live_n100k_sota_instruct`.
- **Figures:** excluded by `source()` ("fusion without the reranker").

### 3.4.8 `sota`, `sota_icomp`, `sota_idemo`: fusion + cross-encoder (body label "fusion + reranker" for the best variant; appendix "fusion + reranker, no instr." / ", competence instr." / ", demonstration instr."; "fusion + reranker[, I-competent / I-demonstrated]" in `figures/shortlist/`)
- **Algorithm** (`sota_shortlist`, `_common.py`), per family f:
  1. `fused = RRF(BM25[f][pool], (Xa @ Xf[f])[pool])`.
  2. Take the top `rerank_pool = 50` of the pool by `fused`.
  3. Rerank them with `CrossEncoder(Qwen/Qwen3-Reranker-4B).predict([(family description, agent description)])`. The model's own template is used (`_rerank`).
  4. Keep the top k = 10, padding with -1 if fewer.
- **Setup:** `pool` is `_pool` because every grid sets `dedup: true`. The table is (K, k) and depends only on the population.
- **Params:**
  - Stock: `{retrieval: sota, dedup: true, embed_model: Qwen/Qwen3-Embedding-8B, rerank_model: Qwen/Qwen3-Reranker-4B}` (`&fw10_sota`, `configs/grids/`).
  - Instruct variants add `embed_instruct`. The instruction affects only the dense half's **query** vectors, and therefore the fused pool. The reranker never sees it.
- **Grids:**
  - Stock: `fw_live_n1000_sota`, `fw_live_n100_sota`, `fw_live_n1000_lowskill_sota`, `fw_live_n100_lowskill_sota`, `fw_live_n10k_sota`, `fw_live_n100k_sota`, `fw_live_n10k_cartel_sota` (9 frameworks), and `re_sl_embed_*`.
  - I-comp and I-demo: `fw_live_n100k_sota_instruct` (sota I-comp, sota I-demo, hybrid I-comp), the backfill grids, and `re_sl_embed_*`.
- **Liars:** live text only. On RouterEval, rendered numbers (manipulable).

### 3.4.9 `declared`: declared top-k (condensed label "declared top-k"; "declared-claim top-k" in `figures/shortlist/`)
- **Params:** `{retrieval: declared, dedup: true}` (`&fw10_declared`, `configs/grids/`).
- **Grids:**
  - `fw_live_n100k_declared`
  - the backfill grids
  - `lie_max_fw_n1000`, `_n10k`, `_n100k` (`configs/grids/`, with `lie_mode: max`)
  - `re_sl_declared_{small,1k,5k}` (`configs/grids/`)
- **Algorithm:** `view.declared[pool, f]`, stable-sorted descending, top 10 (`_common.py`). There is no text retrieval. The framework still receives the **text** descriptions of these 10.
- **Setup:**
  - `_index` still builds a hashed-TF-IDF matrix, which is not used.
  - `dedup` restricts the ranking to one representative per distinct text, so liars who are not the lowest id of their text group are invisible to it.
- **Interpretation.** This idealises agent-card or registry discovery: an A2A-style registry where each agent publishes a structured skill claim and the orchestrator shortlists by that claim (`configs/grids/` calls it "an idealized A2A Agent-Card registry").
- **Liars: fully manipulable.**
  - `inflate` (+0.4) moves liars up. The grid comment at `configs/grids/` reports 74 % liars in the top 10 at beta = 0.5 random, against a 50 % population rate.
  - `max` sets `D[liars] = 1` (`world.py`). Ties are broken by lowest id, so the shortlist is typically ten liars.
  - RouterEval check (m = 1,000, beta 0.5 low_skill_first, `inflate`, seed 1, family 0): 10 of 10 liars in the declared top 10 on the calibrated world H reads (and on the old programmatic one).
  - Honest agents' declared values: on live, `D_self_described` is the model's own self-rating (`llm.py`), so the live declared top-k (E–G) ranks by real, weakly informative claims. H reads the `re_sl_declared_*_norep_cal` reruns, whose claims are `calibrated` (drawn from the live self-rating's distribution given S, 11 values; §1.3.3), so the honest declared top-k is no longer close to an oracle ranking (it was on the old `programmatic` rows, S + N(0, 0.05)), and the rendered description text the frameworks read comes from the same calibrated claims. With 11 values many agents tie; `shuffle: true` makes the lowest-index tie-break random instead of weakest-first.
- **Hardware:** needs no GPU or cache.

### 3.4.10 `va_cohort` (retrieval `midian`, the MIDIAN cohort) and `midian_wo_audit` (MIDIAN w/o audits cohort, not drawn)
- **Params** (stored rows were migrated to the new values; grid names keep `va`):
  - `{retrieval: midian, r: 10}` (before the rename `midian_va`). Grids: `fw_live_n1000_verified_va`, `fw_live_n1000_verified_va_lowskill`, `fw_live_n100_verified_va[_lowskill]`, `fw_live_n10k_verified_va`, `fw_live_n10k_cartel_verified_va` (9 frameworks), `fw_live_n100k_verified_va`, and `fw_routereval_{small,1k,5k}_va` (9).
  - `{retrieval: midian_wo_audit, r: 10 | 5}` (before the rename `midian`) in `fw_live_n1000_verified` and `fw_live_n100_verified`. `source()` returns None (the MIDIAN w/o audits cohort is not drawn).
- **Mechanism:**
  - `build` constructs `Midian(r=10)` (both defenses; `Midian(audit=False, r=10)` for `midian_wo_audit`) and calls its full `build(view, budget)` (`_common.py`).
  - `retrieve` returns `[MIDIAN's cached root pick for the family] + the other members of that pick's leaf cohort`, up to 10 agents, pick first (`_common.py`). The cohort is random (plain `cohort="random"`).
  - `fetch` also calls `self.mid.fetch`, which charges 1 comparison and 2 messages (`midian.py`), and then charges the framework ledger on top.
  - `observe` forwards the **framework's chosen agent's** outcome to MIDIAN (`_common.py`). MIDIAN keeps learning online, including its 5 % report audits, from the framework's routing. That is why prefetch is disabled.
- **Yes, it spends probes: the same budget as a MIDIAN arm.**
  - At the grid's `b = 3`, the level-0 estimate uses `b0 = b − 1 = 2` probes per (agent, family). The remaining `(b − b0)·n` per family is spent on verification re-probes at promotion (`midian.py`).
  - The total is n·K·b = 48n probes at K = 16: 48,000 at n = 10^3 and 4.8 M at 10^5.
  - MIDIAN's build audits add up to 5 % (`AUDIT_RATE`, `midian.py`; §2.3.4).
  - Probes and peer reports are charged to the row's `build_*` ledger columns.
  - The fallback, still declared argmax within the cohort, needs `declared`, which is kept in `needs`.
- **Liars:** they can distort the cohort only through the peer reports, which the audits and exclusion check, and through the fallback. Descriptions play no part in which agents are shortlisted.

### 3.4.11 `shuffle`: position control, not a shortlist source
- **Params:** `{retrieval: midian, r: 10, shuffle: true}` (`midian_va` before the rename). Grid: `fw_live_n100k_verified_va_shuffled` (`configs/grids/`).
- **Mechanism:** the same members in a deterministic permutation seeded by (pick, "fw_shuffle", family) (`_common.py`). It separates "better material" from "the pick is first".
- **Figures:** excluded by `source()`.

### 3.4.12 `lie_text` (erratum 27): not drawn
- **Params:** `{retrieval: tfidf|bm25, dedup: true, lie_text: true}` in `fw_live_n100k_lietext`, which also holds the honest-text twins `{retrieval: tfidf|bm25, dedup: true}` (`configs/grids/`). `fw_live_n100k_lietext_th` adds `claim_threshold: 0.7` (`configs/grids/`).
- **Mechanism** (`_relabel`, `_common.py`), applied to **every** agent's description:
  1. Strip the trailing `Declared areas: ...` clause.
  2. Append `Declared areas: <families>.`, built from `view.declared`: the top 3 families, or with `claim_threshold > 0` every family with D > threshold (at least the argmax).
  3. The LLM prose is untouched, so the prose still describes the true specialty. It is a partial text lie.
  4. Cache names get `_lt<hash of the texts>` (`_common.py`).
- **Liars:** this is the only variant in which the live lie reaches the text. With top 3, it barely changes anything, because `inflate` preserves rank: "only 14 % of cartel liars' top-3 changes" (`_common.py`). With threshold 0.7, a cartel liar claims about 15 of 16 families.
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
| `va_cohort`, "MIDIAN cohort" ("MIDIAN leaf cohort" in `figures/shortlist/`) | `retrieval: midian, r: 10` | MIDIAN's probed and audited pick + its random leaf cohort | only via peer reports (audited) and the fallback | no GPU; **spends 48n probes (+≤5 % audits)** + reports |
| (not drawn) MIDIAN w/o audits cohort | `retrieval: midian_wo_audit, r: 10/5` | MIDIAN w/o audits pick + cohort | reports + fallback | spends probes |
| (not drawn) shuffle | `retrieval: midian, r: 10, shuffle: true` | MIDIAN cohort, permuted | as `va_cohort` | as `va_cohort` |
| (not drawn) lie_text | `lie_text: true` [+ `claim_threshold: 0.7`], tfidf/bm25 | text with the Declared-areas clause taken from D | **yes (partially)**, by construction | no |

---

## 3.5 Which (framework × shortlist × n × regime) combinations exist

I resolved these with `rte.run.blocks` (mirror chains followed) on `configs/grids/`. The field values are:
- live `dist` ⊂ {specialist, heavy_tail, bimodal}
- `declared_source = self_described` on every live framework grid
- `b = 3` on all of them except `budget_b10_*`
- RouterEval `declared_source = programmatic` on the old grids, `calibrated` on the `*_norep_cal` mirrors (the only ones H reads)
- "10" = the ten frameworks and "9" = without Magentic-One
- "cartel" = beta 0.5 with `low_skill_first` liars

| grid(s) | backend, n | dist | beta × liar_select | seeds, Q | frameworks × shortlist | figure label |
|---|---|---|---|---|---|---|
| `fw_live_n1000`, `fw_live_n100` | llm 10^3 / 10^2 | 3 shapes | {0, .1, .25, .5} × random | 1-10, 1000 | 10 × tfidf, + Magentic-One 14B | tfidf |
| `fw_live_n1000_lowskill`, `fw_live_n100_lowskill` | llm 10^3 / 10^2 | 3 | .5 × low_skill_first | 1-10, 1000 | 10 × tfidf, + M1-14B | tfidf |
| `live_n10k_v2` (fw arms) | llm 10^4 | specialist | {0, .25} × random | 1-3, 300 | 10 × tfidf | tfidf |
| `fw_live_n10k_cartel` | llm 10^4 | specialist | cartel | 1-3, 300 | 9 × tfidf | tfidf |
| `fw_live_n10k_cartel_magentic` | llm 10^4 | specialist | cartel | 1-3, 300 | Magentic-One × {tfidf, MiniLM, sota, MIDIAN cohort} | tfidf, embed, sota, va_cohort |
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
| `fw_live_n1000_verified_va` / `fw_live_n100_verified_va` | llm 10^3 / 10^2 | 3 | {0, .1, .25, .5} × random | 1-10, 1000 | 10 × MIDIAN cohort | va_cohort |
| `fw_live_n{1000,100}_verified_va_lowskill` | llm | 3 | cartel | 1-10, 1000 | 10 × MIDIAN cohort | va_cohort |
| `fw_live_n10k_verified_va` / `fw_live_n10k_cartel_verified_va` / `fw_live_n100k_verified_va` | llm 10^4 / 10^4 / 10^5 | spec | {0, .25} random / cartel / 6 regimes | 1-3, 300 | 10 / 9 / 10 × MIDIAN cohort | va_cohort |
| `fw_live_n100k_verified_va_shuffled` | llm 10^5 | spec | 6 | 1-3, 300 | 10 × MIDIAN cohort shuffled | not drawn |
| `fw_live_n{1000,100}_verified` | llm | 3 | 4 × random | 1-10, 1000 | 10 × MIDIAN w/o audits cohort, r ∈ {10, 5} | not drawn |
| `fw_live_n100k_lietext` / `_lietext_th` | llm 10^5 | spec | 6 | 1-3, 300 | 10 × {tfidf, tfidf-lie, bm25, bm25-lie} / {tfidf-lie-th, bm25-lie-th} | not drawn |
| `lie_max_fw_n1000` / `_n10k` / `_n100k` | llm 10^3 / 10^4 / 10^5, `lie_mode: max` | spec | cartel | 1-3; Q 1000 / 300 / 300 | 10 × declared | not drawn (grid-name filter; see 3.9) |
| `fw_routereval_small` (n ∈ {10, 100}) / `_1k` | routereval mmlu | 3 pools | {0, .5} × {random, low_skill_first} | 1-5, 1000 | 9 × tfidf | tfidf (Fig H) |
| `fw_routereval_5k` | routereval leaderboard_mmlu 5,000 | all | same | 1-3, 300 | 9 × tfidf | tfidf (H) |
| `fw_routereval_{small,1k,5k}_em` / `_va` | as above | | | | 9 × MiniLM / MIDIAN cohort | embed / va_cohort (H) |
| `re_sl_declared_{small,1k}` / `_5k` | routereval | strong_to_weak / all | {0, .5} × low_skill_first | 1-5, 1000 / 1-3, 300 | 9 × declared | declared (H) |
| `re_sl_embed_{small,1k}` / `_5k` | same | same | same | same | 9 × {dense-Ic, dense-Id, sota, sota-Ic, sota-Id} | dense_icomp, dense_idemo, sota, sota_icomp, sota_idemo (H) |
| every RouterEval row above, as `<grid>_norep_cal` (`configs/grids/`) | routereval mmlu (`no_repeat`, `shuffle`) / leaderboard_mmlu 5,000 | strong_to_weak / all | {0, .5} × low_skill_first | 1-5, 300 / 1-3, 300 | same frameworks × shortlists, calibrated claims | **these are H's bars** (the rows above are no longer drawn) |
| `fw_k_sensitivity[_dd]` | llm 10^3 | spec | {0, .25} random | 1-3, 300 | LangGraph and AutoGen × k ∈ {5, 10, 20} | not drawn |
| `fw_appendix[_dd]` | llm 10^2 | spec | {0, .25} random | 1-3, 300 | AgentScope | not drawn |
| `budget_b10_shapes` / `_fw_dd`, `churn_n1000` / `_fw_dd` | llm 10^3 | | | | LangGraph, AutoGen | not drawn |

**Results on disk.**
- Every grid above has rows (`lie_max_fw_n1000` / `_n10k` / `_n100k`: 31 / 31 / 29 rows.d files). The 15 RouterEval
  `*_norep_cal` framework mirrors are complete (`shortlist_figs.h30()` is True), with every one of the 9 frameworks at
  full seeds in every H slot.
- The backfill grids and `fw_live_n10k_cartel_magentic` have rows with 3 seeds per framework. At 10^2 / 10^3 the backfill
  shortlists now enter E: the full-seed rule is per (n, shortlist), so their 3 seeds are the full count for those slots
  (§5.1.3).
- The RouterEval embedding cache root is `$RTE_DATA/cache/embed_routereval/` (content-keyed directories).

---

## 3.6 Embedding and shortlist caches

- **`_popdir(view)`** (`_common.py`) decides where the cache files for a population live.
  - **Live:** the backend's population dir, `$RTE_DATA/populations/<dist>_n<n>_K16_seed<s>/`.
  - **Other backends:** a directory under `$RTE_EMBED_CACHE_DIR`, named by `blake2b(digest 12)` of the exact agent texts, a `\x00` separator, and the family texts. It exists **only if that variable is set**; unset means no cache.
  - Keying the directory on the texts means that any change to the declared vector, and therefore to the rendered text, lands in a different directory. For RouterEval, every (n, beta, liar_select, seed) world gets its own directory. The root in use is `$RTE_DATA/cache/embed_routereval/` (102 dirs on 2026-09-24).
- **`_embeddings(view)`** (`_common.py`) holds two blocks.
  - Agent block: `descriptions_<slug><ltag>.npy`. The slug is `minilm` or `qwen_qwen3_embedding_8b`.
  - Family block: `families_<slug><ltag><itag>.npy`, where `itag = _i<blake2b-4 of the instruction>`.
  - A file is reused only if its row count matches. Otherwise it is recomputed. A non-MiniLM model without a GPU **raises** with a pointer to `scripts/embed_populations.py`.
  - Writes are atomic: tmp file, then `os.replace`.
- **`_sota_table(view)`** (`_common.py`) is the (K, k) table `shortlist_sota_<embed>_<rerank>_k<k>p<pool>[_dd][_i<hash>].npy`, named by `sota_cache_name` (where `instruct` is `embed_instruct + ltag`). It is reused if its row count equals K. Otherwise it is computed, and the reranker then runs on whatever device is present (on CPU it is slow, but it does not raise).
- **`scripts/embed_populations.py` (live).**
  - Walks every population dir that has `descriptions.json`  and writes `descriptions_<slug>.npy`.
  - Writes `families_<slug>[_i<hash>].npy`, with the query prompt `"Instruct: ...\nQuery:"` or `prompt_name="query"`.
  - With `--sota`, it also writes the reranked table through the **same** `sota_shortlist` and dedup pool  and `sota_cache_name(..., dedup=True, instruct)`.
  - The family texts are `families.describe(f)` for `FAMILIES_16`. I checked that `families.names(16) == FAMILIES_16`, so they match the backend's texts.
  - An example 10^5 population dir holds `descriptions_minilm.npy`, the Qwen description block, three family blocks (stock, `_i0258be9d`, `_i06c7d226`) and three sota tables.
- **`scripts/embed_routereval.py` (RouterEval).**
  - Asserts that `RTE_EMBED_CACHE_DIR` is set.
  - For each grid block, cell and seed, it builds the `World` and calls `FrameworkMethod._index` once per distinct dense setting, i.e. the exact code a routing unit runs.
  - `--check` replaces `_rerank` with a function that raises, so it passes only if every file hits. The routing units must be launched with the same `RTE_EMBED_CACHE_DIR`. No launcher script sets it (grep finds it only in `_common.py`, the test and `embed_routereval.py`).
- **What never needs a cache:** `tfidf`, `bm25`, `declared`, the MIDIAN modes and `lie_text` tfidf/bm25 are computed inside the job.

---

## 3.7 Ledger accounting of a framework arm
- **Build:** `message(n)`, 0 probes and 0 reports. MIDIAN modes add their own build costs.
- **Per task:** `compare(len(cand))`, `hop(1)`, `message(len(cand) + 2)` (`_common.py`). MIDIAN modes add MIDIAN's `compare(1)`, `message(2)` and its observe-time costs.

---

## 3.8 Facts a paper author should state beside the framework numbers
1. **Live text arms ignore the lie.**
   - On live, every text shortlist is bit-identical across liar regimes (erratum 27).
   - Honest-vs-cartel differences for text arms come only from the declared-argmax fallback.
   - The exception is `lie_text`, which is not drawn.
2. **On RouterEval the text arms read the lied declared vector** (calibrated claims in every drawn H row), and the supervisor sees only the subject name, never the question.
3. **The pre-registered TF-IDF bar is non-dedup**, so it is clone-filled at scale. Every other drawn text bar is dedup. Fig G's lift therefore mixes the retriever gain and the dedup gain.
4. **Declared top-k and the MIDIAN cohort are the only drawn shortlists that are not text retrieval.** The MIDIAN cohort spends a full MIDIAN probe budget. Declared spends none and is directly attackable.
5. **Every shortlist is ordered best-first.** The shuffle control exists because frameworks largely consume position 1 (`docs/methods.md` §5b).
6. **AutoGen's own selector falls back to the first participant** after 3 failed attempts, and the adapter cannot distinguish that from a real pick.

---

## 3.9 Discrepancies / open questions

1. **Dedup is not a no-op on RouterEval.**
   - `configs/grids/` says "dedup is a no-op there: every rendered description is unique". Erratum 25 says 1,000/1,000 and 5,000/5,000 unique.
   - That held only for the old programmatic claims at beta = 0. Rendered texts coincide when two agents' top-5 families and two-decimal values agree: inflated liars whose top 5 all clip to `1.00` on the old rows, and, on the calibrated claims H now reads (11 values), honest agents too. Distinct texts from `_texts` at seed 1, low_skill_first:

     | pool | calibrated (H), beta 0 | calibrated (H), beta 0.5 cartel | old programmatic, beta 0.5 cartel | old programmatic, beta 0.5 random |
     |---|---|---|---|---|
     | n = 1,000 | 931/1,000 | 681/1,000 | 844/1,000 | 694/1,000 |
     | n = 100 | 100/100 | 88/100 | 96/100 | 78/100 |
     | n = 5,000 | 4,134/5,000 | 2,551/5,000 | 4,052/5,000 | 3,310/5,000 |

   - So every `dedup: true` arm in H (MiniLM, dense, rerank, declared top-k; not the MIDIAN cohort, which ignores dedup, and not TF-IDF) retrieves from a pool 7–17 % smaller than m honest and 12–49 % smaller under the cartel. The pre-registered TF-IDF rows (no dedup) are unaffected.
2. **"TF-IDF is degenerate on rendered numbers" (`configs/grids/` fw_routereval_5k comment; `docs/archive/DEVIATIONS.md` §Frameworks on RouterEval (a)) is imprecise.**
   - The query `"Tasks of family X"` matches the family-name tokens in each agent's declared top 5.
   - The retriever therefore shortlists agents that *claim* X among their top 5, and liars flood it: 100 % liars for one family in my 1k check.
   - It carries declared-channel signal, and adversarial signal. It is not a no-signal ranking.
3. **The RouterEval supervisor never sees the prompt.** The task text is `"A task of family <subject> (instance i)."` (`_common.py`). DEVIATIONS does not state this.
4. **Which grids the shortlist figures read is decided by name.**
   - `shortlist_figs.GRIDS`  collects live grids starting with `fw_live_n` (without `lietext`), plus `live_n10k_v2`
     and `live_n100k`, and RouterEval grids starting with `fw_routereval_` or `re_sl_`. H draws `MAIN` on the RouterEval
     rows (`shortlist_condensed.py`); the RouterEval set is all old grids or all `*_norep_cal` grids (`h30()`),
     and today it is the `*_norep_cal` set.
   - `lie_max_fw_*` is excluded by name (no `fw_live_n` prefix). That exclusion is necessary: `regime()` keys on beta
     and liar_select only, so `lie_mode: max` rows would otherwise be pooled with `inflate` "cartel" rows.
5. **Magentic-One coverage differs across 10^4 cartel grids.** It is excluded "for time" from `fw_live_n10k_cartel`
   and its `_dd`, `_em`, `_sota`, `_shapes` and `_verified_va` mirrors (9 frameworks), but included in
   `fw_live_n10k_cartel_backfill` and `lie_max_fw_n10k` (10), and in every live 10^5 grid. `fw_live_n10k_cartel_magentic`
   adds it for four shortlists at the 10^4 cartel (TF-IDF, MiniLM, rerank, MIDIAN cohort). Where its cartel rows have fewer
   than 3 seeds, E's full-seed rule drops it; today it has all 3 in every 10^4 cartel slot, so every 10^4 E bar averages
   all ten frameworks (§5.2).
6. **`docs/methods.md` §5 (line ~119) says `retrieval="declared"` is "grid written, NOT yet run".** `$RTE_DATA/results/fw_live_n100k_declared` has 162 csv rows. The doc is stale.
7. **`docs/methods.md` §5 table:** "`fw_langgraph` | `create_react_agent` supervisor". In the code the supervisor is `langgraph_supervisor.create_supervisor`; `create_react_agent` builds the candidate agents (`langgraph_worker.py`).
8. **Reproducibility or determinism.**
   - `docs/operations.md` T1 says prefetch gives "identical picks; tested". The test uses a mock server.
   - On the real fleet:
     - The replica is chosen per call with an unseeded RNG (`_common.py`).
     - No worker sends a `seed`.
     - vLLM greedy decoding under varying batches is not guaranteed bit-stable.
   - `docs/archive/DEVIATIONS.md` ("Requests set temperature 0 and seed 0") describes the agents' `llm_client` calls, not the framework workers.
   - I did not measure the pick-level variance of a framework row between two sequential runs.
9. **Some worker exceptions still count as infrastructure errors.** `INVALID_ACTION` (`_common.py`) covers four error classes: a tool named after the agent (ADK, OpenAI Agents), `ModelBehaviorError`, MAF's missing next speaker, and a `KeyError: 'agent_NNNNNN'` name lookup (MAF's orchestrator naming a non-candidate; erratum 29, CHANGES §8e). LlamaIndex selector parse failures, AgentScope JSON errors and any other worker exception still become `error`, count as `infra_errors` and can fail a unit past 2 %. Arguably they are framework behaviour (non-picks). This matters only if some framework's parse-failure rate is near 2 %.
10. **`success_strict` after an error is scored as a non-pick.** Both error paths set `_picked = False` before routing by declared argmax (`_common.py`), so `observe` scores such a task 0 under the strict metric.
11. **Erratum 28 wording and counts differ.** CHANGES_AND_ERRATA says "7 framework **conda** envs ... 240 dangling library symlinks each". `docs/operations.md` D2 says "98 dangling library symlinks each". The code-side comment (`_common.py`) says ~4,500 rows; the erratum says 4,269 rows were quarantined. The envs are venvs (`_bridge.py`).
12. **Stale grid comment:** `fw_live_n1000`'s inline comment says "v2 0.4: 5 seeds, Q=1000", but the grid runs `seeds: 1-10` (`configs/grids/`).
13. **The unused TF-IDF matrix** is built for `declared` and `bm25` (`_common.py`), which is harmless. Unknown `retrieval` strings silently run hashed TF-IDF. No current grid uses a misspelled value: every value I resolved is one of tfidf, bm25, embed, hybrid, sota, declared, midian or midian_wo_audit (the MIDIAN keys after the 2026-09-24 rename).
14. **MiniLM family vectors are not on disk at 10^5.** `specialist_n100000_K16_seed1/` has `descriptions_minilm.npy` but no `families_minilm.npy`, even though `_embeddings` would write it. The `_em` rows there were probably produced before the family block was cached, or with a different family-cache name. It is harmless, since 16 MiniLM vectors cost nothing, but it is unexplained.
15. **Open: the "Declared areas" clause on live gives the true specialty list.** It is structured metadata that no real agent card would guarantee honest. Every live text arm (and the supervisor) benefits from it; `lie_text` is the only condition that corrupts it.

---

## 4. Figures A and B: where every bar, line, colour and number comes from

Scope: `figures/paper/A_live_allb.{png,pdf,csv}`, `A_live_stacked.*`, `B_families_allb.*` and `B_families_stacked.*`, written by `python scripts/figures/condensed_figs.py`. Figures C and D (efficiency) are written by `scripts/figures/efficiency_figs.py` and covered in Part 6.
Code, repo-relative to the repository root: `scripts/figures/condensed_figs.py`, `scripts/figures/lib/figspec.py` (canvas, fonts, colours, names, legend order, drawing helpers) and `scripts/figures/lib/seed_tables.py`. They import `scripts/figures/lib/exclusions.py` (`excluded`) and `scripts/figures/lib/stats.py` (`se`), `scripts/analysis/fw_variant_numbers.py` (`load`, `regime`) and `rte/analyze.py` (`ALIAS`). `condensed_figs.py` also reads the CSVs that `scripts/figures/bar_figs.py` wrote.
Data root: `RTE_DATA=$RTE_DATA`, with results under `$RTE_DATA/results/<grid>/`.

**How this section was checked (read-only).** Every value, whisker, `chosen` string and `b_invariant` flag in the tables below is read from `A_live_allb.csv` / `B_families_allb.csv` as written at 2026-09-24 11:35 (144 / 170 rows; the `_stacked` CSVs hold 80 / 90: every arm's b = 3 bar plus MIDIAN's b = 1 and 5). Seed counts and source grids come from running `seed_tables.switched()` and `seed_tables.tables()` read-only on the same rows (`switched()` returns all four non-live families). The live b = 3 values of the single arms are the `results/aggregates/bars/live.csv` numbers (`LIVE_GRIDS`, self_described, specialist, per-seed means): MIDIAN w/o defenses (`midian_wo_defenses`) n = 100 / 1,000 / 10^4 / 10^5: 0.7770 / 0.7890 / 0.7850 / 0.7522 honest and 0.7405 / 0.7379 / 0.7522 / 0.7056 cartel; `oracle` 0.8449 / 0.8612 / 0.8589 / 0.8622. The per-seed tables give the same oracle means, and `narrow()` asserts that every drawn single-arm mean agrees with its table within 0.01.

---

### 4.0 Machinery shared by A and B (read this first)

#### 4.0.1 The pipeline (`condensed_figs.py`)
```
d = load(); C = cells(d); SW |= switched(); T = tables(); add_budgets(C); from_tables(C, T); narrow(C, T)
for key, bb in T.items():                        # per-seed tables for the cross-fitted pooled arms
    if key in C: C[key]["seeds"] = bb
add_best_framework(C)
fig_A(C); fig_B(C)                    # C / D: scripts/figures/efficiency_figs.py
```
A per-seed table whose cell has no b = 3 bar row is dropped (`if key in C`). This script draws only A and B.
1. **`load()`**  concatenates the five per-family bar CSVs `results/aggregates/bars/{live,bernoulli,replay,routereval,llmrouterbench}.csv` and **drops every label starting with `fw_`**, so framework arms are never drawn. These CSVs are *not* regenerated by condensed_figs; `scripts/figures/bar_figs.py` writes them, and they hold **b = 3 only** (4.0.2).
2. **`cells(d)`**  groups the rows by `(family, group, n, regime)`. A cell is kept only if it has an `oracle` row. Each cell stores `oracle = (mean, ci_lo, ci_hi)` and `raw[3] = {label: (mean, lo, hi)}` for every non-oracle label that is not on the do-not-add list (`excluded(l)`). **So every b = 3 number of the five single arms in A/B is a bar_figs number, unchanged**, except in a family that has switched to its erratum-30 rows (step 5).
3. **`switched()`** (`seed_tables.py`) returns the non-live families whose erratum-30 grid is complete; `condensed_figs.SW` holds them. See 4.0.7.
4. **`seed_tables.tables()`** (`seed_tables.py`) builds seed × arm tables per (cell, regime, b) from the raw rows. They feed the two pooled arms and, for a switched family, every bar (4.0.4, 4.0.7).
5. **`add_budgets(C)`**  adds `raw[1]` and `raw[5]` for the families not in `SW` (only live today; 4.0.3); **`from_tables(C, T)`**  replaces the whole cell of a switched family (every arm, every b and the oracle) with the per-seed-table values.
6. **`narrow(C, T)`**  replaces every single-arm and oracle whisker with ±1 s.e. over seeds from the per-seed tables (4.0.5).
7. **`add_best_framework(C)`**  attaches, per live n and regime, the seed × (framework | text shortlist) table that A's framework arm is cross-fitted from (4.0.4, 4.1).
8. **`arms_at(cell)`**  turns `raw[b]` and the tables into the plotted arms (4.0.4).
9. **`budget_bars(...)`**  draws the figure and writes the CSV (4.0.5).

#### 4.0.2 Where the b = 3 numbers come from (`scripts/figures/bar_figs.py`)
Today only live's b = 3 single-arm bars come from here. `load()` still reads all five bar CSVs, and `cells()` needs each family's b = 3 oracle row to create the cell, but `from_tables` then replaces every bar and the oracle of the four switched families (4.0.7). The non-live bullets below describe what those CSVs hold, not what B draws.
- **live** (`family_live`, `bar_figs.py`):
  - Rows are the concatenation of `LIVE_GRIDS[n]` (`bar_figs.py`), loaded through `rte.analyze.load`, which first runs `consolidate` to merge `rows.d` into `rows.csv` (`analyze.py`, `run.py`).
  - Filtered to `declared_source == "self_described"`  and `dist == dist`.
  - Per-regime statistics come from `stats_from_rows` :
    - Filter `beta` ≈ β and `liar_select == liar`.
    - **Honest (β=0) uses `liar_select == "random"`**. It falls back to all β=0 rows only if that set is empty.
    - Then per label: `per = groupby("seed").success.mean()`; plotted mean = `per.mean()`; CI = `stats.ci(per)`, a bootstrap kept in the bar CSV but not drawn: A and B draw ±1 s.e. (4.0.5).
  - Grids per n: 100 → `fw_live_n100, learned_n100, fw_live_n100_lowskill` (`live_core_n100` dropped 2026-09-24: pre-09-02 probe instances); 1000 → `fw_live_n1000, live_f1_n1000, variants_f1, learned_f1, fw_live_n1000_lowskill`; 10000 → `learned_n10k, live_n10k_v2, fw_live_n10k_cartel, live_n10k_cartel_random`; 100000 → `live_n100k`.
  - Verified cell axes of those rows after the filter: all `b = 3`, `K = 16`, `lie_mode = inflate`, `demand = uniform`, `collude = True`; `Q = 1000` at n ≤ 10^3 and `Q = 300` at n ≥ 10^4.
  - **There is no deduplication across grids.** If a (label, seed) appears in several grids, all of its rows are averaged into that seed's value (details in 4.5).
- **bernoulli / replay** (`family_matrix`, `bar_figs.py`):
  - Read `results/<grid>/matrix_success.csv` for `bernoulli_scale_v5` (group label `"specialist"`) and `replay_scale_v5` (group `"all shapes pooled"`), filtered to `b == 3`.
  - The matrix was written by `scripts/analysis/scale_matrix.py`  on 2026-09-17:
    - mean = mean over all (dist, seed) units;
    - CI = `stats.ci` on the (dist, seed)-indexed series, which is a seed bootstrap;
    - honest = β=0 with `liar_select = random` (`scale_matrix.py`).
  - Replay at n=1e6 has 300 units = 3 shapes × 100 seeds, so it is balanced.
- **routereval** (`bar_figs.py`):
  - n=5000 comes from `big = routereval_mmlu5k + fw_routereval_5k`. `fw_routereval_5k` contains only `oracle` rows once frameworks are dropped.
  - Cell axes: `dist = all`, K=16, Q=300, b=3, `declared_source = programmatic`.
  - **`big` is not filtered by pool**, so the n=5000 bars are identical in the `strong_to_weak`, `all_strong` and `all_weak` groups. B picks the `strong_to_weak` copy (`PRIMARY`, `condensed_figs.py`), but the data is the 5000-LLM leaderboard pool, `dist = all`.
- **llmrouterbench** (`bar_figs.py`): grid `llmrouterbench_pool`, n=20, K=15, Q=1000, b=3, programmatic, 5 seeds.


#### 4.0.3 Where the b = 1 and b = 5 numbers of the single arms come from (`add_budgets`, `condensed_figs.py`)
- **Grid list** : for each tag, both `va_b_<tag>` (MIDIAN only) and `rivals_b_<tag>` (budget-matched rivals) are read. A family in `SW` is skipped here : all its bars come from its erratum-30 tables. Every non-live family is in `SW` today, so only the four live tags are used. The tags are:
  - `n100, n1000, n10k, n100k` → key `("live","specialist",n)`
  - `routereval5k` → `("routereval","strong_to_weak",5000)`
  - `llmrouterbench` → `("llmrouterbench","20 models",20)`
  - `bernoulli_1e7` → `("bernoulli","specialist",1e7)`
  - `replay_1e6` → `("replay","all shapes pooled",1e6)`
- **Row loading.** Rows come from `fw_variant_numbers.load` (`fw_variant_numbers.py`): it reads `rows.d/*.json` plus `rows.csv` directly, with **no consolidate** and no write. Each `rows.d` row gets `rid` = its file name, so unmerged rows keep distinct ids; then `drop_duplicates("rid")`, then `drop_duplicates` on `(n, b, dist, beta, liar_select, seed, method, params)`.
- **Labels.** `label(method, params)` (`seed_tables.py`, imported at `condensed_figs.py`) rebuilds the same label string as `rte.analyze.prepare` (`analyze.py`), then applies `ALIAS` (`analyze.py`). For example, `flat_probe_argmax{"online":true}` becomes `flat_probe_argmax_online`, and `linucb_honest{"bonus":"own"}` becomes `linucb_honest[bonus=own]`.
- **Grouping** : `groupby(["n","b","beta","liar_select","label"])`, mapped to a regime by `fw_variant_numbers.regime(beta, ls)` (`fw_variant_numbers.py`): β = 0 → `beta0`, whatever `liar_select` is; β = 0.5 with `low_skill_first` → `cartel`.
- **Rows kept** : only if the key already exists in `C` (so the b = 3 bar CSV must have that cell), b ∈ {1, 5}, the label is not `oracle`, and `excluded(l)` is false.
- **Aggregation**:
  - **Non-replay** : `per = q.groupby("seed").success.mean()`; value = `per.mean()`; interval = `stats.se(per.values)` (mean ∓ 1 s.e.; `narrow` later recomputes it from the per-seed tables).
  - **Replay, "all shapes pooled"** : `by = groupby(["seed","dist"]).success.mean().unstack()`; if all 3 shapes exist, keep only **seeds that have every shape**; per-seed value = mean over the 3 shapes; value = mean over those seeds; interval ±1 s.e. over those seeds. Their b = 1 / 3 numbers come from the matrix, which is balanced (100 × 3), so the two rules agree there.
- **Matrices for b = 1** : for an unswitched bernoulli or replay family, b = 1 would come from `bernoulli_scale_v5` / `replay_scale_v5` `matrix_success.csv` (100 seeds), filtered to `b == 1`, the `success` metric, and regimes `"beta=0 (no liars)"` / `"beta=0.5 CARTEL (low-skill-first)"`. Both families are switched, so this step is skipped today; their b = 1 bars come from the per-seed tables (4.0.7).
- **Nothing is ever pooled across b.** Each bar is one budget.

#### 4.0.4 The arms and how "best learned/declared router" / "best bandit" are chosen (`arms_at`)
`ARMS`  is `figspec.ORDER` without the oracle and the framework arm, so the plotting and legend order is fixed
(`figspec.py`), with key, legend label (`figspec.NAME`) and colour (`figspec.COLOR`):

| order | key | legend label | colour |
|---|---|---|---|
| 1 | `midian` | MIDIAN | `#2ecc71` |
| 2 | `midian_wo_defenses` | MIDIAN w/o defenses | `#c0392b` |
| 3 | `flat_probe_argmax_online` | flat probe argmax | `#3498db` |
| 4 | `best_learned` | best learned/declared router | `#ff7f0e` |
| 5 | `best_bandit` | best bandit | `#9467bd` |
| 6 | `declared_argmax` | declared argmax | `#5d6d7e` |
| 7 | `best_framework` (A only) | best framework | `#8e6c3a` |
| 8 | `random` | random | `#bbbbbb` |

- **Single arms** (1–3, 6, 8): `arms_at` copies `raw[b][key]` for every b that has it; the budget-less arms only at b = 3.
- **Pooled arms** (4, 5): for each b with a per-seed table in the cell :
  - `want`  = the pool (`POOLS`) minus `NOT_RUNNABLE(family, n)`, and minus the pre-registered `warm_start_bandit` (n0 = 5) on bernoulli / replay / RouterEval / LLMRouterBench unless that family is in `SW` (`CLAIM_KEY - SW`). It is the same at every b of a cell. The rules are in §2.1.3.
  - `crossfit(T, want)` (`seed_tables.crossfit`) scores each seed with the arm that has the best mean on the other seeds, among arms that ran on that seed. The bar is the mean of the per-seed scores, and the whisker ±1 s.e. of those scores.
  - Fewer than 2 scored seeds: no bar.
  - `chosen`  = the pick counts, most-picked first, plus `| INCOMPLETE POOL, missing or partial …` when a `want` arm has no column at that b **or lacks a value on any scored seed** (`miss`).
- **The framework arm** (8, A only): when `add_best_framework`  attached a seed × pair table to
  the cell (`fw_seeds`), the same `crossfit` runs over every (framework | text shortlist) pair with the full seed count at
  that (n, regime); the bar is stored at b = 3 only and `chosen` holds the pair counts (e.g. `fw_magentic_one | embed x10`).
  The pairs come from `shortlist_figs.collect("live")` (b = 3 rows, specialist; §5.1.2), restricted to `TEXT` (: the
  nine text shortlists; declared top-k and the MIDIAN cohort are not text retrieval). B skips this arm
  (`skip=("best_framework")`).
- **What the pools really contain**, per docs/methods.md:
  - `cluster_head_router` and `disrouter_cascade` are **"Declared channel only"** arms (docs/methods.md §3). They read declared skill D and never probe. They are also in `bar_figs.DECL` (`bar_figs.py`). Hence the legend name "best learned/declared router".
  - `flat_nsw_router` is a "verified outcomes, centralized" arm (docs/methods.md §4).
  - Only `knn_router(_online)` and `mlp_router` are the published RouterBench routers (docs/methods.md §4).
  - On the old non-live rows, whose claims were nearly the answer key, the "best learned/declared router" was a declared-only arm almost everywhere. On the calibrated rows B draws now, the declared-only arms win only at b = 1 (bernoulli 10^7: `cluster_head_router` ×100 honest, `disrouter_cascade` ×100 cartel; RouterEval 5,000 cartel: `disrouter_cascade` ×3), and `flat_nsw_router` takes most non-live bars (§2.1.3).
  - `warm_start_bandit` (both n0) reads declared skill plus probes (docs/methods.md §4).
- **Which pool members are missing or partial** (the `chosen` column): none. Every A / B pool bar, learned and bandit, has
  every allowed member on every scored seed (no `INCOMPLETE` in either CSV; the last live 10^5 kNN seeds landed
  2026-09-24 11:20).

  `linucb_honest[bonus=own]` comes from the `linucb_fix_*` grids (live, RouterEval-5k, LLMRouterBench cells and bernoulli 10^7) and from the erratum-30 grids (replay 10^6 only from `replay_1e6_split_cal`); all have landed. Post-fix TrueSkill comes from `trueskill_fix_n{100,1000,10k}` and the erratum-30 grids (landed); it cannot run at n ≥ 10^5, so it is not in the live 10^5, bernoulli 10^7 or replay 10^6 pools.
- **Budget-less arms** (`BUDGETLESS = {declared_argmax, random}`) and the framework arm: only their b = 3
  entry is used, drawn as **one bar per regime in its base colour** (no shading, `one(k)`), and written to the CSV
  with `b = "-"`.
- **Every other arm** has a bar for each b ∈ {1, 3, 5} for which data exists (in `_allb`).

#### 4.0.5 Drawing (`budget_bars`; style from `scripts/figures/lib/figspec.py`)
- **Canvas and fonts** (`figspec.figure`, `apply`): `_allb` on the appendix canvas 5.5 × 2.4 in, `_stacked` on the body
  canvas 5.5 × 1.9 in; serif (Times New Roman / Times, else STIX), tick labels and legend 8 pt, axis labels 9 pt,
  `pdf.fonttype 42`; horizontal major grid only (lw 0.3, alpha 0.4); no top or right spine; saved as vector PDF plus a
  300 dpi PNG (`figspec.save`).
- **Groups**: one group per x tick. A: `$n = 10^2$`, `$10^3$`, … (`figspec.pow10_ticks`, the first carries "n ="); B:
  the backend name (`figspec.BACKEND`) over `$n = 10^k$` on a second line (`fig_B`). **Regimes**:
  `REG = {"beta0": "honest", "cartel": "β=0.5 cartel"}`.
- **Slots within a group** :
  - `arms` = the `figspec.ORDER` keys present in any cell of the figure (minus `skip`).
  - **_allb** has one slot per (arm, regime, b), ordered arm → regime (honest, then cartel) → b (1, 3, 5). Budget-less
    arms and the framework arm get one slot per regime: A 5·2·3 + 3·2 = **36 slots**, B 5·2·3 + 2·2 = **34**, bar width
    `w = 0.86/slots`.
  - **_stacked** has one slot per (arm, regime): A **16 slots**, B **14**, `w = 0.86/slots`. Every arm is drawn at
    b = 3 only, except **MIDIAN, whose slot holds its b = 5, 3 and 1 bars**, each at its true height, sorted
    tallest first; each later (shorter) bar gets a higher z-order, so the shortest (normally b = 1, light) is in front
    and the tallest (normally b = 5, dark) behind.
  - Slot positions are fixed, so a missing bar leaves an empty gap.
- **Colour and shade** (`figspec.bar`, `shade`, `B_SHADE`, `figspec.py`): b = 1 = the arm colour mixed 50 %
  toward white (light); b = 3 = the arm colour; b = 5 = the colour × 0.6 (dark). Budget-less arms and the framework arm
  are unshaded.
- **Hatch**: the cartel regime has `hatch="////"`; honest bars are solid (`figspec.HATCH`). Every bar has a black edge,
  lw 0.3.
- **No titles and no markers.** `figspec.finish` clears the title. When any (arm, regime, b) slot has no bar or any drawn
  pooled bar's `chosen` contains `INCOMPLETE`, the script prints `[<name>] INCOMPLETE: a bar is missing or its pool is
  incomplete (see the csv)`; `cluster/ops/figure_status.py` lists the missing (group, arm, regime, b) slots and
  the incomplete-pool bars from the `_allb` CSVs. Today it reports none.
- **Oracle line** (`figspec.oracle`): a dotted grey (`#7f8c8d`, lw 1.0) horizontal line across each group (±0.46).
  A: y = the honest b = 3 oracle mean of the cell; B: y = 1.0. The oracle does not depend on liars (verified: live 0.8449 / 0.8612 / 0.8589 / 0.8622 in both regimes).
- **Y axis** (`figspec.finish`): A "task success" on ylim (0.2, 0.95); B "success relative to oracle" on ylim
  (0.2, 1.05); two-decimal ticks. Bars start at 0, so anything below 0.2 is hidden and bar lengths exaggerate ratios
  (figures audit F8). The lowest bars are B's replay random (0.243 of oracle) and replay cartel declared argmax (0.303);
  A's lowest is live 10^4 random (0.417).
- **Normalisation (B only)** : `z` = the **β = 0, b = 3 oracle mean** of the group's cell (from the bar CSV for live, 0.8622; from the erratum-30 table for the four switched families: bernoulli 0.8462, replay 0.7919, RouterEval 0.9200, LLMRouterBench 0.7500). Mean and both whisker ends of **every** bar in the group (both regimes, all b) are divided by that one scalar: a ratio of means, not a per-seed ratio, and the whisker ignores the oracle's own uncertainty. The b = 1 / 5 rows mirror the b = 3 cells' task streams, so their own oracles equal the b = 3 one (checked in every A / B per-seed table).
- **Error bars** (`figspec.whisker`): drawn **on every bar in _allb**, black, lw 0.6, cap 1.5 pt.
  Every whisker is **±1 standard error over seeds**, a ~68 % interval, never a 95 % CI. `narrow(C, T)` (called
  after `from_tables`) replaces the interval of every single-arm bar and of the oracle with `stats.se` of that
  arm's per-seed values from the per-seed tables (for replay, per-seed means of the 3 shape means), asserting that the
  table mean agrees with the plotted mean within 0.01; a bar whose per-seed values are not in the tables keeps its mean and
  gets no whisker. The pooled arms and the framework arm get `se` of their cross-fitted per-seed scores. The bar CSVs' and matrices' own
  bootstrap intervals are therefore not used; in the four switched families `from_tables` sets the same ±1 s.e. directly. With **3 seeds** (live 10^4 / 10^5, RouterEval 5,000) the s.e. rests on 3
  values; LLMRouterBench has 5, replay 30, bernoulli 100. Intervals are marginal, not paired, so overlap is not a test.
- **_stacked has no error bars** (`if not stacked`); the intervals are in `_allb`.
- **Legend** (`figspec.legend`): one frameless legend above the axes, entries in the fixed order
  `["oracle"] + arms` (never ranked by value: `figspec.legend` builds matplotlib's `Legend` directly, so the value-ranking
  wrapper `legend_rank.install()` puts on `Axes.legend` does not apply). `_stacked`: 3 columns when the framework arm is drawn (A),
  else 4 (B). `_allb`: three rows, with the three grey budget swatches "b = 1 / b = 3 / b = 5" (`figspec.budget_handles`)
  in a column of their own at the right.
- **CSV** : one row per drawn bar with columns `group, regime, arm (the legend name), key, b ('-' for budget-less arms and the framework arm), chosen, value, ci_lo, ci_hi, b_invariant`. The values are **absolute bar heights**; in B they are already divided by the oracle. **_allb CSVs** have one row per bar (A 144, B 170); **_stacked CSVs** have one row per bar drawn (A 80, B 90: MIDIAN's three budgets plus one b = 3 row for every other arm). Missing bars have no row.
- **`b_invariant`**  is true for `declared_argmax` and `random`, and for a pooled bar whose every pick is in `B_INVARIANT = {declared_argmax, random, cluster_head_router, disrouter_cascade}`, the arms that never probe. **These are the arms that do not change with b**: a pooled bar that picked `cluster_head_router` or `disrouter_cascade` on every seed would have the same value at b = 1, 3 and 5. In the current CSVs the flag is true for the declared argmax and random bars and for three pooled bars, all in B at b = 1: bernoulli 10^7 best learned/declared router (0.6662 honest, `cluster_head_router` ×100; 0.6451 cartel, `disrouter_cascade` ×100) and RouterEval 5,000 cartel (0.6969, `disrouter_cascade` ×3). At b = 3 / 5 the same cells pick a probing arm, so those bars differ. (The framework arm's flag is false.)

#### 4.0.6 What "honest" and "β=0.5 cartel" mean here
- **honest** = β = 0, no liars.
  - At b=3: `liar_select = random` rows (bar_figs; the matrices' "beta=0 (no liars)" cell).
  - At b=1/5 (live/RouterEval/LLMRouterBench `va_b_*` / `rivals_b_*`): the rows are tagged `liar_select = low_skill_first, collude = True`, with `n_liars = 0`. The tag is inert at β = 0, as the `configs/grids/` comment notes.
- **β=0.5 cartel** = β = 0.5, `liar_select = low_skill_first`, `collude = True`, `lie_mode = inflate`. Half the agents lie, chosen low-skill first, and they collude.
  - live: `declared_source = self_described`.
  - bernoulli/replay/RouterEval/LLMRouterBench: `calibrated` claims in every drawn row that reads claims (the erratum-30 rows, 4.0.7). Bernoulli's non-claim-reading arms keep their `programmatic`-cell rows, which are bit-identical because nothing they use depends on `declared_source`.
- **What changes with b** (the probe budget: probes per agent per family, per docs/methods.md):
  - Every probe-using arm changes with b: MIDIAN, MIDIAN w/o defenses, flat probe argmax, the bandits' warm-up, and the learned routers' training probes (knn: k = b).
  - The four `B_INVARIANT` arms (`declared_argmax`, `random`, `cluster_head_router`, `disrouter_cascade`) do not (4.0.5).
- **Structural identity at b = 1.** In the **honest** regime MIDIAN and MIDIAN w/o defenses are equal at b = 1 in every cell of A and B: live 0.7105 / 0.6913 / 0.6678 / 0.6400 at n = 10^2 … 10^5, bernoulli 0.8015, replay 0.8289, RouterEval 0.7017 and LLMRouterBench 0.8444 (B values ÷ oracle). Under the cartel they differ (e.g. live 10^2: 0.6972 vs 0.6625; replay 0.8279 vs 0.5138).
  
  This is consistent with README.md: "never run b = 1 beside b = 3 in one table (verification is unfunded at b = 1)". The condensed figures put b = 1 beside b = 3 deliberately (see 4.5).
- **Identical honest and cartel b=5 at live n=100.** MIDIAN b=5 is 0.8121 in both regimes. I checked per seed:
  - all 10 seeds have identical success at β=0 and β=0.5 (for example seed 1: 0.801 / 0.801);
  - `n_liars` is 0 against 50, and `misroute_to_liar` is > 0 in the cartel rows (seed 2: 0.091).
  
  So the routing is identical, and "misroutes" land on liar-flagged agents whose executions are real. At b=5 MIDIAN's selection does not depend on declarations. This is a genuine result, not duplicated rows.


#### 4.0.7 Erratum 30: the rows B's non-live families read, and the leaks of the rows they replaced
The old non-live rows had three leaks (erratum 30, [errata.md](errata.md); found by an internal leakage audit), and **none of those rows is drawn any more**:
- **Claims were an answer key.** bernoulli, replay, RouterEval and LLMRouterBench declared D = clip(S + N(0, 0.05)), corr(S, D) ≈ 0.99, while live self-ratings correlate 0.36 with S. Every claim-reading arm was inflated: declared argmax (0.994 of oracle honest on bernoulli, against 0.72 on live 10^5), `cluster_head_router` / `disrouter_cascade` (then the "best learned/declared router" almost everywhere) and both warm-start bandits. That is why the pre-registered `warm_start_bandit` (n0 = 5) was left out of those families' bandit pool until they switched (`CLAIM_KEY`, `condensed_figs.py`).
- **Replay probed and routed on the same prompts**, and S (oracle, liar selection, claims) was the in-sample accuracy over all of them.
- **RouterEval / LLMRouterBench streams repeated test prompts** (repeat share 0.48 on mmlu, 0.20 on LLMRouterBench), which online learners that see the prompt could memorise (leakage audit L5 / L10). The RouterEval pools are also stored weak → strong, so on 11-valued calibrated claims every lowest-index tie-break picked the weakest tied agent; `shuffle: true` fixes that, and the unshuffled first attempt is quarantined (§1.4.4).

The fixes are new cell values, so old rows are untouched on disk: `declared_source: calibrated` (§1.3.3), replay `split: true` (§1.4.3), and RouterEval / LLMRouterBench `no_repeat: true` with `shuffle: true` and Q = 300 (§1.4.4). The rerun grids (`configs/grids/`) and the switch (`seed_tables.py`, `ERRATUM30`, `switched()`):

| family (B cell) | rows B draws now | what the switch replaced | seeds |
|---|---|---|---|
| bernoulli 10^7 | `bernoulli_scale_v5` + `va_b_` / `rivals_b_` / `pool_fill_` / `linucb_fix_bernoulli_1e7` + `bernoulli_1e7_cal`; the claim-reading arms (`CLAIM_READERS`: `cluster_head_router`, `disrouter_cascade`, both warm-start bandits, `declared_argmax`) only from their calibrated rows (`seed_tables.py`) | only the claim readers' programmatic rows; every other arm keeps its rows, which are bit-identical by construction (world, liars, stream and probes do not read `declared_source`) | 100 |
| replay 10^6 | `replay_1e6_split_cal` only (every B arm; calibrated claims + probe/task split) | the whole cell : `replay_scale_v5` + its b-grids | 30 |
| RouterEval 5,000 | `routereval5k_norep_cal` only (every B arm; calibrated, no-repeat, shuffled, Q = 300) | the whole cell : `routereval_mmlu5k` + its b-grids | 3 |
| LLMRouterBench 20 | `llmrouterbench_norep_cal` only (every B arm; calibrated, no-repeat, shuffled, Q = 300, K = 15) | the whole cell : `llmrouterbench_pool` + its b-grids (Q = 1,000) | 5 |

- **Completeness test** (`complete`, `seed_tables.py`): the set of rows on disk (`rows.csv` + `rows.d`), keyed by (n, b, dist, β, liar_select, seed, arm label), must contain every row the grid planner expects (`planned`: cells × seeds × method specs, oracle excluded). A set comparison, so stray or duplicate rows cannot stand in for a missing one. Partial reruns are never mixed in; a switched cell with no b = 3 oracle would be left empty rather than fall back to old rows (`from_tables`, `condensed_figs.py`). **All four grids pass, so `switched()` returns all four families.**
- **In a switched family** `condensed_figs.SW` holds it : `add_budgets` skips it, `from_tables`  replaces the oracle and every arm at every b with the mean and ±1 s.e. of its per-seed table, and the pre-registered `warm_start_bandit` (n0 = 5) joins the bandit pool (it is the bernoulli pick at every b).
- **TrueSkill** (`seed_tables.py`): in every family only `trueskill_per_family` rows from the `trueskill_fix_*` grids or the erratum-30 grids count, because rows written before the probe-duplicate fix (erratum 30, `World._probe`) gave a repeated agent the same instance.
- **What moved.** Declared argmax fell from 0.994 / 0.998 / 0.958 / 0.990 of oracle honest (bernoulli / replay / RouterEval / LLMRouterBench, old rows) to 0.6625 / 0.3822 / 0.6739 / 0.7440. The declared-only "learned" routers stopped winning except at b = 1 (§2.1.3). The replay oracle is now 0.7919 (task rows) and every replay value rests on 30 seeds.
- On live nothing changes: live claims are real self-ratings and live probes, measurement and tasks use disjoint instance seeds.
- H (RouterEval shortlists) has its own all-or-nothing switch, which has also fired; see §5.5.

---

### 4.1 Figure A: live RTE, specialist population, n = 10^2 … 10^5

**Question answered.** On the live LLM backend with the specialist population, how does MIDIAN compare with MIDIAN w/o defenses, the flat probe-argmax control, the best learned/declared router, the best bandit, declared argmax and random, at every scale n, honest versus a β = 0.5 low-skill colluding cartel, and at each probe budget b = 1, 3, 5?

**Groups** (`fig_A`): n = the n values with a `("live","specialist",n,"beta0")` cell in `C`: 100, 1000, 10000 and 100000.

**_allb** (`A_live_allb`, appendix): in each group, for each arm left to right in the legend order: solid bars for honest b = 1, 3, 5 (light, as is, dark), then hatched cartel b = 1, 3, 5; one solid and one hatched bar for declared argmax, random and the framework arm; the dotted oracle at the honest b = 3 oracle (0.8449, 0.8612, 0.8589, 0.8622); y = "task success", 0.2–0.95. 144 bars.

**_stacked** (`A_live_stacked`, Figure 1, `stacked=True`): one slot per (arm, regime), **every arm at b = 3**. Only MIDIAN's slot carries the budget overlay: its b values are sorted **tallest first**  and each is drawn **from 0 to its own full height** (z-order 2, 3, 4: tallest behind), so b = 1 (light) is in front and b = 5 (dark) behind; MIDIAN is monotone in b in every A cell. The other arms are single b = 3 bars in their base colour, so the shade legend and the empty slots of the old all-arm overlay are gone. No error bars; the CSV holds absolute values, not gains (80 rows).

**The framework arm, "best framework, best text shortlist"** (`add_best_framework`; `arms_at`). Per n and regime, the seed × (framework | text shortlist) success table from `shortlist_figs.collect("live")` (b = 3, specialist; every pair with the full seed count of that cell), cross-fitted per seed like the pooled arms: for each seed the pair with the best mean on the other seeds is picked and scored on this seed. One bar per regime (b = "-"), brown `#8e6c3a`, with a ±1 s.e. whisker in `_allb`. Current values (honest / cartel): 10² 0.6111 / 0.5661, 10³ 0.6315 / 0.5598, 10⁴ 0.6200 / 0.6233, 10⁵ 0.6167 / 0.6089; every pick is Magentic-One, with MiniLM (`embed`) at 10² honest and 10³, fusion + reranker (`sota`) at 10² cartel, and a Qwen3 dense instruction variant at 10⁴ / 10⁵ (the CSV's `chosen` column).

**Provenance.**

| n | b = 3 source of the single arms (`live.csv`, self_described + specialist) | b = 1 / b = 5 source |
|---|---|---|
| 100 | `fw_live_n100` / `fw_live_n100_lowskill` + `learned_n100`. **10 seeds.** | `va_b_n100`, `rivals_b_n100` (10 seeds) |
| 1000 | `fw_live_n1000` / `_lowskill` + `variants_f1` + `live_f1_n1000` + `learned_f1`. **10 seeds.** | `va_b_n1000`, `rivals_b_n1000` (10 seeds) |
| 10000 | `learned_n10k`, `live_n10k_v2`, `fw_live_n10k_cartel`, `live_n10k_cartel_random`. **3 seeds.** | `va_b_n10k`, `rivals_b_n10k` (3 seeds) |
| 100000 | `live_n100k`. **3 seeds.** | `va_b_n100k`, `rivals_b_n100k` (both regimes, b = 1 and 5; 3 seeds) |

The pooled arms read `LIVE_GRIDS[n]` + `va_b_` / `rivals_b_` / `pool_fill_` / `linucb_fix_` / `trueskill_fix_` + `tuned_wsb_` + `pool_seeds_` (`seed_tables.py`).

**A value table** (from `A_live_allb.csv`). Format: value, seeds (source); for the pooled arms: value, scored seeds, pick counts; `*` after a pooled entry = pool missing a member or a member missing seeds; `(b-inv.)` = every pick is a never-probing arm; n/a = budget-less. Whiskers are in the Appendix. Every slot has a bar. The live 10^5 best-learned-router bars (kNN still landing; the two cartel ones are marked `*`) may change slightly at the next figure run.

| group | regime | arm | b=1 | b=3 | b=5 |
|---|---|---|---|---|---|
| n = 100 | honest | MIDIAN | 0.7105 10s (va_b_n100) | 0.7816 10s (bars) | 0.8121 10s (va_b_n100) |
| n = 100 | cartel | MIDIAN | 0.6972 10s (va_b_n100) | 0.7680 10s (bars) | 0.8121 10s (va_b_n100) |
| n = 100 | honest | MIDIAN w/o defenses | 0.7105 10s (rivals_b_n100) | 0.7770 10s (bars) | 0.8002 10s (rivals_b_n100) |
| n = 100 | cartel | MIDIAN w/o defenses | 0.6625 10s (rivals_b_n100) | 0.7405 10s (bars) | 0.7552 10s (rivals_b_n100) |
| n = 100 | honest | flat probe argmax | 0.7023 10s (rivals_b_n100) | 0.7805 10s (bars) | 0.8006 10s (rivals_b_n100) |
| n = 100 | cartel | flat probe argmax | 0.7023 10s (rivals_b_n100) | 0.7805 10s (bars) | 0.8006 10s (rivals_b_n100) |
| n = 100 | honest | best learned/declared router | 0.6151 10s, `mlp_router` ×8; `knn_router_online` ×2 | 0.7376 10s, `knn_router_online` ×10 | 0.7788 10s, `knn_router_online` ×10 |
| n = 100 | cartel | best learned/declared router | 0.6151 10s, `mlp_router` ×8; `knn_router_online` ×2 | 0.7376 10s, `knn_router_online` ×10 | 0.7788 10s, `knn_router_online` ×10 |
| n = 100 | honest | best bandit | 0.6881 10s, `warm_start_bandit[n0=0.5]` ×10 | 0.7524 10s, `linucb_honest[bonus=own]` ×10 | 0.7729 10s, `linucb_honest[bonus=own]` ×10 |
| n = 100 | cartel | best bandit | 0.6798 10s, `linucb_honest[bonus=own]` ×10 | 0.7524 10s, `linucb_honest[bonus=own]` ×10 | 0.7729 10s, `linucb_honest[bonus=own]` ×10 |
| n = 100 | honest | declared argmax | n/a | 0.5992 10s (bars) | n/a |
| n = 100 | cartel | declared argmax | n/a | 0.5181 10s (bars) | n/a |
| n = 100 | honest | random | n/a | 0.4258 10s (bars) | n/a |
| n = 100 | cartel | random | n/a | 0.4258 10s (bars) | n/a |
| n = 1,000 | honest | MIDIAN | 0.6913 10s (va_b_n1000) | 0.8134 10s (bars) | 0.8377 10s (va_b_n1000) |
| n = 1,000 | cartel | MIDIAN | 0.6914 10s (va_b_n1000) | 0.8074 10s (bars) | 0.8375 10s (va_b_n1000) |
| n = 1,000 | honest | MIDIAN w/o defenses | 0.6913 10s (rivals_b_n1000) | 0.7890 10s (bars) | 0.8079 10s (rivals_b_n1000) |
| n = 1,000 | cartel | MIDIAN w/o defenses | 0.6505 10s (rivals_b_n1000) | 0.7379 10s (bars) | 0.7658 10s (rivals_b_n1000) |
| n = 1,000 | honest | flat probe argmax | 0.6860 10s (rivals_b_n1000) | 0.7871 10s (bars) | 0.8108 10s (rivals_b_n1000) |
| n = 1,000 | cartel | flat probe argmax | 0.6860 10s (rivals_b_n1000) | 0.7869 10s (bars) | 0.8108 10s (rivals_b_n1000) |
| n = 1,000 | honest | best learned/declared router | 0.6364 10s, `mlp_router` ×9; `knn_router_online` ×1 | 0.7656 10s, `knn_router_online` ×10 | 0.7976 10s, `knn_router_online` ×10 |
| n = 1,000 | cartel | best learned/declared router | 0.6364 10s, `mlp_router` ×9; `knn_router_online` ×1 | 0.7656 10s, `knn_router_online` ×10 | 0.7976 10s, `knn_router_online` ×10 |
| n = 1,000 | honest | best bandit | 0.7038 10s, `warm_start_bandit[n0=0.5]` ×10 | 0.7737 10s, `warm_start_bandit[n0=0.5]` ×10 | 0.8020 10s, `warm_start_bandit[n0=0.5]` ×10 |
| n = 1,000 | cartel | best bandit | 0.6669 10s, `warm_start_bandit` ×8; `trueskill_per_family` ×2 | 0.7636 10s, `warm_start_bandit` ×7; `warm_start_bandit[n0=0.5]` ×3 | 0.8040 10s, `warm_start_bandit[n0=0.5]` ×10 |
| n = 1,000 | honest | declared argmax | n/a | 0.6154 10s (bars) | n/a |
| n = 1,000 | cartel | declared argmax | n/a | 0.5202 10s (bars) | n/a |
| n = 1,000 | honest | random | n/a | 0.4321 10s (bars) | n/a |
| n = 1,000 | cartel | random | n/a | 0.4321 10s (bars) | n/a |
| n = 10,000 | honest | MIDIAN | 0.6678 3s (va_b_n10k) | 0.8111 3s (bars) | 0.8356 3s (va_b_n10k) |
| n = 10,000 | cartel | MIDIAN | 0.6700 3s (va_b_n10k) | 0.8100 3s (bars) | 0.8311 3s (va_b_n10k) |
| n = 10,000 | honest | MIDIAN w/o defenses | 0.6678 3s (rivals_b_n10k) | 0.7850 3s (bars) | 0.8089 3s (rivals_b_n10k) |
| n = 10,000 | cartel | MIDIAN w/o defenses | 0.6389 3s (rivals_b_n10k) | 0.7522 3s (bars) | 0.7933 3s (rivals_b_n10k) |
| n = 10,000 | honest | flat probe argmax | 0.6522 3s (rivals_b_n10k) | 0.7744 3s (bars) | 0.8022 3s (rivals_b_n10k) |
| n = 10,000 | cartel | flat probe argmax | 0.6522 3s (rivals_b_n10k) | 0.7744 3s (bars) | 0.8022 3s (rivals_b_n10k) |
| n = 10,000 | honest | best learned/declared router | 0.5156 3s, `flat_nsw_router` ×2; `knn_router_online` ×1 | 0.7456 3s, `flat_nsw_router` ×3 | 0.7611 3s, `knn_router_online` ×2; `flat_nsw_router` ×1 |
| n = 10,000 | cartel | best learned/declared router | 0.5156 3s, `flat_nsw_router` ×2; `knn_router_online` ×1 | 0.7456 3s, `flat_nsw_router` ×3 | 0.7622 3s, `knn_router_online` ×2; `flat_nsw_router` ×1 |
| n = 10,000 | honest | best bandit | 0.7200 3s, `trueskill_per_family` ×3 | 0.7622 3s, `trueskill_per_family` ×2; `warm_start_bandit[n0=0.5]` ×1 | 0.8100 3s, `warm_start_bandit[n0=0.5]` ×3 |
| n = 10,000 | cartel | best bandit | 0.7033 3s, `trueskill_per_family` ×2; `warm_start_bandit` ×1 | 0.7833 3s, `warm_start_bandit` ×3 | 0.8178 3s, `warm_start_bandit` ×2; `warm_start_bandit[n0=0.5]` ×1 |
| n = 10,000 | honest | declared argmax | n/a | 0.6578 3s (bars) | n/a |
| n = 10,000 | cartel | declared argmax | n/a | 0.5322 3s (bars) | n/a |
| n = 10,000 | honest | random | n/a | 0.4167 3s (bars) | n/a |
| n = 10,000 | cartel | random | n/a | 0.4167 3s (bars) | n/a |
| n = 100,000 | honest | MIDIAN | 0.6400 3s (va_b_n100k) | 0.8356 3s (bars) | 0.8533 3s (va_b_n100k) |
| n = 100,000 | cartel | MIDIAN | 0.6178 3s (va_b_n100k) | 0.8278 3s (bars) | 0.8522 3s (va_b_n100k) |
| n = 100,000 | honest | MIDIAN w/o defenses | 0.6400 3s (rivals_b_n100k) | 0.7522 3s (bars) | 0.7922 3s (rivals_b_n100k) |
| n = 100,000 | cartel | MIDIAN w/o defenses | 0.5600 3s (rivals_b_n100k) | 0.7056 3s (bars) | 0.7567 3s (rivals_b_n100k) |
| n = 100,000 | honest | flat probe argmax | 0.6622 3s (rivals_b_n100k) | 0.7489 3s (bars) | 0.7856 3s (rivals_b_n100k) |
| n = 100,000 | cartel | flat probe argmax | 0.6622 3s (rivals_b_n100k) | 0.7489 3s (bars) | 0.7856 3s (rivals_b_n100k) |
| n = 100,000 | honest | best learned/declared router | 0.6044 3s, `knn_router_online` ×2; `cluster_head_router` ×1 | 0.7411 3s, `knn_router_online` ×3 | 0.7511 3s, `knn_router_online` ×2; `flat_nsw_router` ×1 |
| n = 100,000 | cartel | best learned/declared router | 0.6478 3s, `knn_router_online` ×3 | 0.7411 3s, `knn_router_online` ×3 | 0.7556 3s, `knn_router_online` ×2; `flat_nsw_router` ×1 |
| n = 100,000 | honest | best bandit | 0.6656 3s, `warm_start_bandit[n0=0.5]` ×2; `warm_start_bandit` ×1 | 0.7544 3s, `warm_start_bandit[n0=0.5]` ×3 | 0.8056 3s, `warm_start_bandit[n0=0.5]` ×3 |
| n = 100,000 | cartel | best bandit | 0.7078 3s, `warm_start_bandit` ×3 | 0.7778 3s, `warm_start_bandit` ×3 | 0.8022 3s, `warm_start_bandit` ×2; `warm_start_bandit[n0=0.5]` ×1 |
| n = 100,000 | honest | declared argmax | n/a | 0.6222 3s (bars) | n/a |
| n = 100,000 | cartel | declared argmax | n/a | 0.5778 3s (bars) | n/a |
| n = 100,000 | honest | random | n/a | 0.4422 3s (bars) | n/a |
| n = 100,000 | cartel | random | n/a | 0.4422 3s (bars) | n/a |

Method and params behind each label: `midian` = `midian {}` (formerly `midian_va {}`); `midian_wo_defenses` = `midian {"audit": false, "verify": false}` (formerly `midian {}`); `flat_probe_argmax_online` = `flat_probe_argmax {"online": true}`; `knn_router_online` = `knn_router {"online": true}`; `warm_start_bandit` = `warm_start_bandit {}` (n0 = 5); `warm_start_bandit[n0=0.5]` = `warm_start_bandit {"n0": 0.5}`; `linucb_honest[bonus=own]` = `linucb_honest {"bonus": "own"}`; `declared_argmax {}`, `random {}`.

---

### 4.2 Figure B: every family at its largest population, success / oracle

**Question answered.** Does the ordering in A hold across every experiment family (live LLM, calibrated Bernoulli, RouterBench replay, RouterEval real-LLM pool, LLMRouterBench) at each family's largest population, on a common oracle-normalised scale, honest versus cartel, at b = 1, 3, 5?

**Group selection** (`fig_B`). For each family in `FAMILY` order: n = the **largest** n whose primary group has **both** a cartel and a β = 0 cell in `C`; primary group = `specialist` for live, `strong_to_weak` for routereval, otherwise whatever group exists (`primary`; `PRIMARY`). Result: live n = 100,000; bernoulli n = 10,000,000; RouterBench replay n = 1,000,000; RouterEval n = 5,000 (key `strong_to_weak`, but the data is the 5,000-LLM leaderboard pool, `dist = all`, 4.0.2); LLMRouterBench n = 20.

**Y axis**: "success relative to oracle" (0.2–1.05), every bar ÷ that group's honest b = 3 oracle: live 0.8622, bernoulli 0.8462, replay 0.7919, RouterEval 0.9200, LLMRouterBench 0.7500 (the four non-live ones from their erratum-30 tables, 4.0.7). The dotted line is at 1.0. No bar exceeds 1; whiskers can (live 10^5 MIDIAN b = 5 reaches 1.010).

**Bars, shades, hatches, legend and stacking** work as in A (4.0.5, 4.1), without the framework arm. B_stacked overlays budgets on MIDIAN only; MIDIAN is non-monotone in one B cell, LLMRouterBench cartel (b = 1 0.8364, b = 3 0.8347, b = 5 0.8720), where the b = 1 bar shows only as a 0.002 sliver above b = 3. Other arms that are non-monotone in b, e.g. LLMRouterBench best learned/declared router (b = 1 0.8684, b = 3 0.9262, b = 5 0.8996), LLMRouterBench cartel MIDIAN w/o defenses (0.8240 / 0.8107 / 0.8204) and RouterEval cartel MIDIAN w/o defenses (0.6920 / 0.6643 / 0.6667), are visible only in `_allb`.

**Provenance per family** (every arm; the per-seed tables behind each switched family are listed in 4.0.7).

| family (cell) | b = 1 | b = 3 | b = 5 |
|---|---|---|---|
| live 1e5 | `va_b_n100k`, `rivals_b_n100k`, 3 seeds | `live_n100k` (bars CSV), 3 seeds | `va_b_n100k`, `rivals_b_n100k`, 3 seeds |
| bernoulli 1e7 (specialist, K16 Q1000) | `bernoulli_scale_v5` rows, 100 seeds; claim readers from `bernoulli_1e7_cal` | same, 100 seeds; declared argmax from `bernoulli_1e7_cal` | `va_b_` / `rivals_b_` / `pool_fill_` / `linucb_fix_bernoulli_1e7`, claim readers from `bernoulli_1e7_cal`, 100 seeds |
| replay 1e6 (3 shapes pooled, K64 Q1000, split, calibrated) | `replay_1e6_split_cal`, 30 seeds with all three shapes | same | same |
| RouterEval 5k (calibrated, no-repeat, shuffled, Q300) | `routereval5k_norep_cal`, 3 seeds | same | same |
| LLMRouterBench 20 (K15, calibrated, no-repeat, shuffled, Q300) | `llmrouterbench_norep_cal`, 5 seeds | same | same |

**B value table** (from `B_families_allb.csv`). All values ÷ oracle. Same format as the A table; whiskers are in the Appendix. Every slot has a bar. The live 10^5 best-learned-router bars (kNN still landing; the two cartel ones are marked `*`) may change slightly at the next figure run.

| group | regime | arm | b=1 | b=3 | b=5 |
|---|---|---|---|---|---|
| live n = 100,000 | honest | MIDIAN | 0.7423 3s (va_b_n100k) | 0.9691 3s (bars) | 0.9897 3s (va_b_n100k) |
| live n = 100,000 | cartel | MIDIAN | 0.7165 3s (va_b_n100k) | 0.9601 3s (bars) | 0.9884 3s (va_b_n100k) |
| live n = 100,000 | honest | MIDIAN w/o defenses | 0.7423 3s (rivals_b_n100k) | 0.8724 3s (bars) | 0.9188 3s (rivals_b_n100k) |
| live n = 100,000 | cartel | MIDIAN w/o defenses | 0.6495 3s (rivals_b_n100k) | 0.8183 3s (bars) | 0.8776 3s (rivals_b_n100k) |
| live n = 100,000 | honest | flat probe argmax | 0.7680 3s (rivals_b_n100k) | 0.8686 3s (bars) | 0.9111 3s (rivals_b_n100k) |
| live n = 100,000 | cartel | flat probe argmax | 0.7680 3s (rivals_b_n100k) | 0.8686 3s (bars) | 0.9111 3s (rivals_b_n100k) |
| live n = 100,000 | honest | best learned/declared router | 0.7010 3s, `knn_router_online` ×2; `cluster_head_router` ×1 | 0.8595 3s, `knn_router_online` ×3 | 0.8711 3s, `knn_router_online` ×2; `flat_nsw_router` ×1 |
| live n = 100,000 | cartel | best learned/declared router | 0.7513 3s, `knn_router_online` ×3 | 0.8595 3s, `knn_router_online` ×3 | 0.8763 3s, `knn_router_online` ×2; `flat_nsw_router` ×1 |
| live n = 100,000 | honest | best bandit | 0.7719 3s, `warm_start_bandit[n0=0.5]` ×2; `warm_start_bandit` ×1 | 0.8750 3s, `warm_start_bandit[n0=0.5]` ×3 | 0.9343 3s, `warm_start_bandit[n0=0.5]` ×3 |
| live n = 100,000 | cartel | best bandit | 0.8209 3s, `warm_start_bandit` ×3 | 0.9021 3s, `warm_start_bandit` ×3 | 0.9304 3s, `warm_start_bandit` ×2; `warm_start_bandit[n0=0.5]` ×1 |
| live n = 100,000 | honest | declared argmax | n/a | 0.7216 3s (bars) | n/a |
| live n = 100,000 | cartel | declared argmax | n/a | 0.6701 3s (bars) | n/a |
| live n = 100,000 | honest | random | n/a | 0.5129 3s (bars) | n/a |
| live n = 100,000 | cartel | random | n/a | 0.5129 3s (bars) | n/a |
| Bernoulli n = 10,000,000 | honest | MIDIAN | 0.8015 100s (bernoulli_scale_v5) | 0.9469 100s (bernoulli_scale_v5) | 0.9764 100s (va_b_bernoulli_1e7) |
| Bernoulli n = 10,000,000 | cartel | MIDIAN | 0.7977 100s (bernoulli_scale_v5) | 0.9386 100s (bernoulli_scale_v5) | 0.9738 100s (va_b_bernoulli_1e7) |
| Bernoulli n = 10,000,000 | honest | MIDIAN w/o defenses | 0.8015 100s (bernoulli_scale_v5) | 0.9110 100s (bernoulli_scale_v5) | 0.9400 100s (rivals_b_bernoulli_1e7) |
| Bernoulli n = 10,000,000 | cartel | MIDIAN w/o defenses | 0.7450 100s (bernoulli_scale_v5) | 0.8463 100s (bernoulli_scale_v5) | 0.8988 100s (rivals_b_bernoulli_1e7) |
| Bernoulli n = 10,000,000 | honest | flat probe argmax | 0.8005 100s (bernoulli_scale_v5) | 0.9088 100s (bernoulli_scale_v5) | 0.9409 100s (rivals_b_bernoulli_1e7) |
| Bernoulli n = 10,000,000 | cartel | flat probe argmax | 0.8005 100s (bernoulli_scale_v5) | 0.9088 100s (bernoulli_scale_v5) | 0.9409 100s (rivals_b_bernoulli_1e7) |
| Bernoulli n = 10,000,000 | honest | best learned/declared router | 0.6662 100s, `cluster_head_router` ×100 (b-inv.) | 0.7832 100s, `flat_nsw_router` ×100 | 0.8489 100s, `flat_nsw_router` ×100 |
| Bernoulli n = 10,000,000 | cartel | best learned/declared router | 0.6451 100s, `disrouter_cascade` ×100 (b-inv.) | 0.7832 100s, `flat_nsw_router` ×100 | 0.8489 100s, `flat_nsw_router` ×100 |
| Bernoulli n = 10,000,000 | honest | best bandit | 0.8907 100s, `warm_start_bandit` ×100 | 0.9380 100s, `warm_start_bandit` ×100 | 0.9513 100s, `warm_start_bandit` ×100 |
| Bernoulli n = 10,000,000 | cartel | best bandit | 0.8353 100s, `warm_start_bandit` ×100 | 0.9254 100s, `warm_start_bandit` ×100 | 0.9470 100s, `warm_start_bandit` ×100 |
| Bernoulli n = 10,000,000 | honest | declared argmax | n/a | 0.6625 100s (bernoulli_1e7_cal) | n/a |
| Bernoulli n = 10,000,000 | cartel | declared argmax | n/a | 0.4980 100s (bernoulli_1e7_cal) | n/a |
| Bernoulli n = 10,000,000 | honest | random | n/a | 0.4936 100s (bernoulli_scale_v5) | n/a |
| Bernoulli n = 10,000,000 | cartel | random | n/a | 0.4936 100s (bernoulli_scale_v5) | n/a |
| RouterBench replay n = 1,000,000 | honest | MIDIAN | 0.8289 30s (replay_1e6_split_cal) | 0.9658 30s (replay_1e6_split_cal) | 0.9833 30s (replay_1e6_split_cal) |
| RouterBench replay n = 1,000,000 | cartel | MIDIAN | 0.8279 30s (replay_1e6_split_cal) | 0.9622 30s (replay_1e6_split_cal) | 0.9824 30s (replay_1e6_split_cal) |
| RouterBench replay n = 1,000,000 | honest | MIDIAN w/o defenses | 0.8289 30s (replay_1e6_split_cal) | 0.8939 30s (replay_1e6_split_cal) | 0.9231 30s (replay_1e6_split_cal) |
| RouterBench replay n = 1,000,000 | cartel | MIDIAN w/o defenses | 0.5138 30s (replay_1e6_split_cal) | 0.7197 30s (replay_1e6_split_cal) | 0.8211 30s (replay_1e6_split_cal) |
| RouterBench replay n = 1,000,000 | honest | flat probe argmax | 0.8302 30s (replay_1e6_split_cal) | 0.8976 30s (replay_1e6_split_cal) | 0.9223 30s (replay_1e6_split_cal) |
| RouterBench replay n = 1,000,000 | cartel | flat probe argmax | 0.8302 30s (replay_1e6_split_cal) | 0.8976 30s (replay_1e6_split_cal) | 0.9223 30s (replay_1e6_split_cal) |
| RouterBench replay n = 1,000,000 | honest | best learned/declared router | 0.7604 30s, `flat_nsw_router` ×30 | 0.9029 30s, `flat_nsw_router` ×30 | 0.9324 30s, `flat_nsw_router` ×30 |
| RouterBench replay n = 1,000,000 | cartel | best learned/declared router | 0.7604 30s, `flat_nsw_router` ×30 | 0.9029 30s, `flat_nsw_router` ×30 | 0.9324 30s, `flat_nsw_router` ×30 |
| RouterBench replay n = 1,000,000 | honest | best bandit | 0.8801 30s, `warm_start_bandit` ×30 | 0.9392 30s, `linucb_honest[bonus=own]` ×30 | 0.9582 30s, `linucb_honest[bonus=own]` ×30 |
| RouterBench replay n = 1,000,000 | cartel | best bandit | 0.8734 30s, `linucb_honest[bonus=own]` ×30 | 0.9392 30s, `linucb_honest[bonus=own]` ×30 | 0.9582 30s, `linucb_honest[bonus=own]` ×30 |
| RouterBench replay n = 1,000,000 | honest | declared argmax | n/a | 0.3822 30s (replay_1e6_split_cal) | n/a |
| RouterBench replay n = 1,000,000 | cartel | declared argmax | n/a | 0.3029 30s (replay_1e6_split_cal) | n/a |
| RouterBench replay n = 1,000,000 | honest | random | n/a | 0.2433 30s (replay_1e6_split_cal) | n/a |
| RouterBench replay n = 1,000,000 | cartel | random | n/a | 0.2433 30s (replay_1e6_split_cal) | n/a |
| RouterEval n = 5,000 | honest | MIDIAN | 0.7017 3s (routereval5k_norep_cal) | 0.8031 3s (routereval5k_norep_cal) | 0.8671 3s (routereval5k_norep_cal) |
| RouterEval n = 5,000 | cartel | MIDIAN | 0.6969 3s (routereval5k_norep_cal) | 0.7971 3s (routereval5k_norep_cal) | 0.8671 3s (routereval5k_norep_cal) |
| RouterEval n = 5,000 | honest | MIDIAN w/o defenses | 0.7017 3s (routereval5k_norep_cal) | 0.7210 3s (routereval5k_norep_cal) | 0.7379 3s (routereval5k_norep_cal) |
| RouterEval n = 5,000 | cartel | MIDIAN w/o defenses | 0.6920 3s (routereval5k_norep_cal) | 0.6643 3s (routereval5k_norep_cal) | 0.6667 3s (routereval5k_norep_cal) |
| RouterEval n = 5,000 | honest | flat probe argmax | 0.6304 3s (routereval5k_norep_cal) | 0.6896 3s (routereval5k_norep_cal) | 0.7379 3s (routereval5k_norep_cal) |
| RouterEval n = 5,000 | cartel | flat probe argmax | 0.6304 3s (routereval5k_norep_cal) | 0.6896 3s (routereval5k_norep_cal) | 0.7379 3s (routereval5k_norep_cal) |
| RouterEval n = 5,000 | honest | best learned/declared router | 0.6123 3s, `flat_nsw_router` ×2; `cluster_head_router` ×1 | 0.7729 3s, `flat_nsw_router` ×3 | 0.8188 3s, `flat_nsw_router` ×3 |
| RouterEval n = 5,000 | cartel | best learned/declared router | 0.6969 3s, `disrouter_cascade` ×3 (b-inv.) | 0.7729 3s, `flat_nsw_router` ×3 | 0.8188 3s, `flat_nsw_router` ×3 |
| RouterEval n = 5,000 | honest | best bandit | 0.7802 3s, `linucb_honest[bonus=own]` ×3 | 0.8949 3s, `linucb_honest[bonus=own]` ×3 | 0.9046 3s, `linucb_honest[bonus=own]` ×3 |
| RouterEval n = 5,000 | cartel | best bandit | 0.7802 3s, `linucb_honest[bonus=own]` ×3 | 0.8949 3s, `linucb_honest[bonus=own]` ×3 | 0.9046 3s, `linucb_honest[bonus=own]` ×3 |
| RouterEval n = 5,000 | honest | declared argmax | n/a | 0.6739 3s (routereval5k_norep_cal) | n/a |
| RouterEval n = 5,000 | cartel | declared argmax | n/a | 0.5568 3s (routereval5k_norep_cal) | n/a |
| RouterEval n = 5,000 | honest | random | n/a | 0.5725 3s (routereval5k_norep_cal) | n/a |
| RouterEval n = 5,000 | cartel | random | n/a | 0.5725 3s (routereval5k_norep_cal) | n/a |
| LLMRouterBench n = 20 | honest | MIDIAN | 0.8444 5s (llmrouterbench_norep_cal) | 0.8844 5s (llmrouterbench_norep_cal) | 0.9120 5s (llmrouterbench_norep_cal) |
| LLMRouterBench n = 20 | cartel | MIDIAN | 0.8364 5s (llmrouterbench_norep_cal) | 0.8347 5s (llmrouterbench_norep_cal) | 0.8720 5s (llmrouterbench_norep_cal) |
| LLMRouterBench n = 20 | honest | MIDIAN w/o defenses | 0.8444 5s (llmrouterbench_norep_cal) | 0.8862 5s (llmrouterbench_norep_cal) | 0.8951 5s (llmrouterbench_norep_cal) |
| LLMRouterBench n = 20 | cartel | MIDIAN w/o defenses | 0.8240 5s (llmrouterbench_norep_cal) | 0.8107 5s (llmrouterbench_norep_cal) | 0.8204 5s (llmrouterbench_norep_cal) |
| LLMRouterBench n = 20 | honest | flat probe argmax | 0.8124 5s (llmrouterbench_norep_cal) | 0.8800 5s (llmrouterbench_norep_cal) | 0.8996 5s (llmrouterbench_norep_cal) |
| LLMRouterBench n = 20 | cartel | flat probe argmax | 0.8124 5s (llmrouterbench_norep_cal) | 0.8800 5s (llmrouterbench_norep_cal) | 0.8996 5s (llmrouterbench_norep_cal) |
| LLMRouterBench n = 20 | honest | best learned/declared router | 0.8684 5s, `mlp_router` ×5 | 0.9262 5s, `mlp_router` ×5 | 0.8996 5s, `mlp_router` ×5 |
| LLMRouterBench n = 20 | cartel | best learned/declared router | 0.8684 5s, `mlp_router` ×5 | 0.9262 5s, `mlp_router` ×5 | 0.8996 5s, `mlp_router` ×5 |
| LLMRouterBench n = 20 | honest | best bandit | 0.8311 5s, `linucb_honest[bonus=own]` ×4; `warm_start_bandit` ×1 | 0.8596 5s, `linucb_honest[bonus=own]` ×4; `warm_start_bandit[n0=0.5]` ×1 | 0.8844 5s, `linucb_honest[bonus=own]` ×4; `warm_start_bandit[n0=0.5]` ×1 |
| LLMRouterBench n = 20 | cartel | best bandit | 0.8489 5s, `linucb_honest[bonus=own]` ×5 | 0.8693 5s, `linucb_honest[bonus=own]` ×5 | 0.8996 5s, `linucb_honest[bonus=own]` ×5 |
| LLMRouterBench n = 20 | honest | declared argmax | n/a | 0.7440 5s (llmrouterbench_norep_cal) | n/a |
| LLMRouterBench n = 20 | cartel | declared argmax | n/a | 0.6142 5s (llmrouterbench_norep_cal) | n/a |
| LLMRouterBench n = 20 | honest | random | n/a | 0.6898 5s (llmrouterbench_norep_cal) | n/a |
| LLMRouterBench n = 20 | cartel | random | n/a | 0.6898 5s (llmrouterbench_norep_cal) | n/a |

Reading notes. In the switched families several pooled bars are identical across regimes (replay best learned/declared router at every b and best bandit at b = 3 / 5, RouterEval best bandit at every b and best learned/declared router at b = 3 / 5, LLMRouterBench best learned/declared router at every b): the picked arm (`flat_nsw_router`, `linucb_honest[bonus=own]`, `mlp_router`) reads neither claims nor reports, and β does not change the world, stream or probes. Flat probe argmax and random are regime-invariant for the same reason.

---

### 4.3 Completeness, and how it is reported

**What is missing, and why:**
- **No slot is empty.** Every (arm, regime, b) slot of A (144 bars, including the framework arm's 8) and B (170) has a
  bar; `rivals_b_n100k` holds every rival at live 10^5 in both regimes at b = 1 and 5.
- **Every pool is complete**: the last live 10^5 kNN seeds (`knn_router`, `knn_router_online`) landed 2026-09-24 11:20,
  and `linucb_honest[bonus=own]` and post-fix TrueSkill have landed wherever they can run. No `chosen` entry says
  `INCOMPLETE POOL`.
- **Erratum 30**: all four non-live families have switched (4.0.7).

**How missing data is reported:**
- Never in the figure: no title, no marker (`docs/figures.md` §1). `budget_bars` prints `[<name>] INCOMPLETE: …` when
  any slot has no bar or any pooled bar's pool is missing a member or has a member missing seeds, and
  `cluster/ops/figure_status.py` lists every missing (group, arm, regime, b) slot and every incomplete-pool bar from the
  `_allb` CSVs (§1.8.4). Both report nothing today.
- A missing bar is an empty slot; the CSV omits it. An incomplete pool is flagged in `chosen` ("missing or partial").
- Seed counts are not marked on the figure (3 at live 10^4 / 10^5 and RouterEval, 5 LLMRouterBench, 10 live 10^2 / 10^3, 30 replay, 100 bernoulli; the same at every b of a cell); the Appendix lists them.

---

### 4.4 Figures C and D
They are not drawn by `condensed_figs.py`. See Part 6.

### 4.5 Discrepancies, open questions and possible issues

1. **A pooled bar is not one arm's success.** "Best learned/declared router" / "best bandit" are cross-fitted (§2.1.3): each seed is scored with the arm that is best on the other seeds, so the bar can mix arms (e.g. live 10^3 cartel best bandit b = 3: `warm_start_bandit` ×7, `warm_start_bandit[n0=0.5]` ×3). It is an honest estimate of "pick the best arm from past seeds, then deploy it", and at most the best single arm's mean. With 3 seeds (live 10^4 / 10^5, RouterEval) each pick rests on 2 seeds.
2. **The pools mix channels.** `LEARNED` includes `cluster_head_router` and `disrouter_cascade`, *declared-channel-only* arms (docs/methods.md §3) that never probe, and `flat_nsw_router`, a verified-centralised arm (an ANN index over flat probe means), not a published learned router. On the calibrated rows B now draws, the declared-only pair is the majority pick in 3 of the 54 best-learned-router bars, all at b = 1 (bernoulli 10^7 both regimes, RouterEval 5,000 cartel), and flat-NSW in 19 (every replay bar, bernoulli b = 3 / 5, most RouterEval bars, live 10^4 b = 1 / 3). 32 of the 54 bars are majority-picked by a published RouterBench router (kNN 22, MLP 10). The legend label "best learned/declared router" names the declared-only pair but not the ANN index (rivals audit F5). On the old answer-key rows the declared-only pair won almost every non-live bar; those rows are no longer drawn.
3. **Every pool is complete** (the live 10^5 kNN seeds have landed); nothing is flagged in the CSVs.
4. **b = 1 / 5 and b = 3 come from different runs.** b = 3 of the single arms is the bars CSV, pooled from several grids per seed; b = 1 / 5 come from one grid each. The task streams match (identical oracles), but, e.g., MIDIAN n = 100 b = 3 averages rows from 3 grids per seed while its b = 1 / 5 twin comes from 1.
5. **Cross-grid averaging within a seed at b = 3 (live).** `stats_from_rows` averages every row of a (label, seed) across `LIVE_GRIDS`, with no dedup, and the duplicate rows are **not identical**: MIDIAN w/o defenses (then `midian`) n = 100 β = 0 seed 1: 0.816 (`fw_live_n100`), 0.816 (`learned_n100`), 0.783 (`live_core_n100`). The maximum cross-grid spread is 0.076 at n = 100, 0.020 at n = 1,000 and 0.017 at 10^4. The protocol audit (F5) traces most of it to the old per-method probe instances of `live_core_n100` and early `live_f1_n1000`, averaged into MIDIAN w/o defenses, not into MIDIAN. Since 2026-09-24 `live_core_n100` is out of `LIVE_GRIDS`: n = 100 MIDIAN w/o defenses 0.7770 / 0.7405 (honest / cartel; was 0.7741 / 0.7359), MIDIAN w/o audits 0.7815 / 0.6793, and the β = 0.25-cartel random and r = 5 bars of the per-family plots, which only that grid had, are gone.
6. **Unequal seed counts.** Live 10^4 / 10^5 and RouterEval have 3 seeds per bar, LLMRouterBench 5, live 10^2 / 10^3 10, replay 30, bernoulli 100; their whiskers are not comparable in coverage (4.0.5).
7. **Honest `liar_select` differs by source.** At b = 3 honest uses `liar_select = random` rows (`bar_figs.py`); at b = 1 / 5 the `va_b` / `rivals_b` β = 0 rows are tagged `low_skill_first`. The pooled arms take β = 0 rows under either tag (`seed_tables.py`). Inert at β = 0 (`n_liars = 0`), but a different cell tag.
8. **Normalisation in B uses one scalar**, the β = 0, b = 3 oracle; the ratio CIs ignore oracle uncertainty and are not per-seed ratios.
9. **Replay pooling.** In the switched replay cell every bar comes from `replay_1e6_split_cal` through `_table(pooled_shapes=True)` (`seed_tables.py`): a seed counts only if the arm ran on all three shapes, and its value is the mean of the three shape means. All 30 seeds have all three shapes for every arm. (The unswitched path would take b = 1 / 3 from the scale matrix, which averages all (shape, seed) units, and b = 5 from `add_budgets`, which keeps complete-shape seeds; it is not used today.)
10. **Documentation tension about b = 1.** README.md says "never run b = 1 beside b = 3 in one table (verification is unfunded at b = 1)". A and B put them side by side, and honest MIDIAN b = 1 equals MIDIAN w/o defenses at b = 1 in every cell (4.0.6).
11. **Stacked presentation.** In `_stacked` only MIDIAN overlays its budgets, as overlapping full-height bars, not additive segments; a negative b-gain appears as an out-of-order shade (one B cell, 4.2); the other arms are b = 3 only; the stacked CSV holds absolute values; no CIs, by design.
12. **Stale docstrings.** The module docstring (`condensed_figs.py`) still says an incomplete pool "puts the one * in the figure title", and `add_budgets`'s  that a missing cell is marked by "a *"; the code draws no title and no marker (`figspec.finish` clears the title).
13. **Latent loader hazard (no effect today).** `fw_variant_numbers.load` deduplicates on `(n, b, dist, beta, liar_select, seed, method, params)`, which omits `declared_source`, `K`, `Q`, `collude`, `lie_mode` and `backend_kwargs`. `add_budgets` uses it only for the live `va_b_*` / `rivals_b_*` grids, which have one value of each, so A and B are unaffected. The per-seed tables use `seed_tables.rows`, which dedups on `rid` only, and bernoulli's claim readers are separated by `declared_source` explicitly (`seed_tables.py`).
14. **Hidden arms.** The do-not-add filter (`exclusions.excluded`, which drops every label containing "halving") removes every halving arm; no ARMS entry or pool member is a halving arm in any case. Peer-reported halving, a pre-registered rival that reads the same report channel, beats MIDIAN paired on every seed in every honest b = 3 cell of A and, on the pre-erratum-30 rows the audit read, in 3 of the 4 other B families, and loses under the cartel (protocol audit F1, F6; the B-family part was not re-checked on the reruns B now draws). Trusted-observer halving is withdrawn and never reported (erratum 26).
15. **The "RouterEval strong_to_weak" key is a misnomer at n = 5,000.** The cell key comes from `bar_figs.py`, which does not filter the 5,000-LLM pool by `dist`; `seed_tables` keeps the same key for `routereval5k_norep_cal`, whose pool is the 5,000-LLM leaderboard (`dist = all`).
16. **Live n = 100 cartel MIDIAN b = 5 equals honest (0.8121) seed for seed.** Genuine (4.0.6), but worth a caption sentence because it looks like a copy error.
17. **Caption disclosures the protocol audit requires** (internal protocol audit, findings F3, F6, F7):
    - A and B show only two regimes, honest and the β = 0.5 low-skill-first cartel. That cartel is the one liar regime in which MIDIAN beats peer-reported halving (5 of 24 live regimes); showing it alone is a post-hoc presentation choice, although the regime is pre-registered.
    - Post-hoc choices, none marked on the figures: MIDIAN as the headline arm (its own pre-registered target V2-11 was scored a miss); the tuned warm-start bandit (n0 = 0.5, tuned on unreported seeds 11-15); the pool edits (LinUCB replaced by its fixed-bonus variant, TrueSkill from post-fix rows only); the erratum-30 reruns B's non-live groups now draw (calibrated-claim model, replay split with 30 seeds, no-repeat Q = 300, agent shuffle); hidden halving.
    - MIDIAN overspends its budget: 1.03–1.07 × n·K·b (audit re-probes) while every rival spends ≤ 1.00 ×, so "same budget" is not literally true.


---

## 5. Figures E, F, G, H — the shortlist figures

Scope: `figures/paper/{E_shortlists_by_n, F_shortlists_1e5, G_shortlist_lift_1e5, H_routereval_shortlists}.{png,pdf,csv}`.
All claims below are traced to code (repo root the repository root). The value tables are read from
`E_shortlists_by_n.csv`, `F_shortlists_1e5.csv`, `G_shortlist_lift_1e5.csv` and `H_routereval_shortlists.csv` (written
2026-09-24 11:41); the per-bar framework lists and seed counts from `figures/shortlist/{live,routereval}.csv` (written with them),
the input those CSVs were summarised from. `shortlist_figs.h30()` and `pending_reruns()` were run read-only on the rows
(`RTE_DATA=$RTE_DATA`).

## 5.1 The pipeline E–H share

### 5.1.1 Two stages, two scripts

```
$RTE_DATA/results/<grid>/rows.csv + rows.d/*.json
        │  scripts/figures/shortlist_figs.py  (collect → draw → main)
        ▼
figures/shortlist/{live,routereval}.csv   (one row per framework × shortlist × (n, dist, regime), plus oracle / MIDIAN rows)
        │  scripts/figures/shortlist_condensed.py (load → summarise → body → by_n / fig_F / fig_G)
        ▼
figures/paper/{E,F,G,H}_*.{png,pdf,csv}
```

- The condensed script never reads raw rows. It reads only `figures/shortlist/{family}.csv` (`shortlist_condensed.py`),
  written by `shortlist_figs.main`. **The condensed figures are only as fresh as the last `shortlist_figs.py` run.**
- Both scripts must run with `RTE_DATA` set. Otherwise `R` falls back to `/scratch/rte/results` (`shortlist_figs.py`,
  `fw_variant_numbers.py`), which holds no grids, and the run writes empty CSVs.

### 5.1.2 Stage 1: `shortlist_figs.collect` (raw rows → per-framework seed means)

1. **Grids read** (`GRIDS`, `shortlist_figs.py`):
   - live: every directory whose name starts with `fw_live_n` and does not contain `lietext`, plus `live_n10k_v2` and
     `live_n100k` (Python precedence makes this `(A and B) or C`, the intended reading).
   - routereval: every directory starting with `fw_routereval_` or `re_sl_` **whose name ends in `_norep_cal` exactly
     when `h30()` is true**. `h30()`  is true only once every grid in `H30_FW` (the 15 `*_norep_cal` framework
     mirrors) and `H30_REF` (`routereval_mmlu_norep_cal`, `routereval5k_norep_cal`) is complete by
     `seed_tables.complete`. So H reads either all old grids or all erratum-30 grids, never a mix (§5.5). **Today
     `h30()` is True: only the 15 `*_norep_cal` framework grids are read.**
   Grids are visited in sorted name order.
2. **Row loading** (`rows`): `fw_variant_numbers.load` concatenates `rows.d/*.json`
   (rid = file name) and `rows.csv`, drops duplicate `rid`, then duplicates on (n, b, dist, beta, liar_select, seed, method,
   params); rows stored under the old MIDIAN keys are translated on read (`keys.normalize`). Only **b = 3** is kept.
3. **Framework rows**: `method` starts with `fw_`. Each row gets a shortlist key from its params JSON via `source()`; rows that map to `None` are dropped:

   | params | source key | display: `figures/shortlist/` (`_LONG`) / condensed (`figspec.SHORTLIST_BODY` / `_APPENDIX`) |
   |---|---|---|
   | contains `supervisor` (14B Magentic arm) or `lie_text` | dropped | — |
   | no `retrieval`, no `dedup` | `tfidf` | hashed TF-IDF (pre-registered) / hashed TF-IDF |
   | no `retrieval`, `dedup: true` | dropped (dedup TF-IDF not drawn) | — |
   | `retrieval: midian` (formerly `midian_va`), `r: 10`, not `shuffle` | `va_cohort` | MIDIAN leaf cohort / MIDIAN cohort |
   | `retrieval: midian` with shuffle or r ≠ 10 | dropped | — |
   | `retrieval: embed`, model contains "Qwen" | `dense` / `dense_icomp` / `dense_idemo` (instruction contains "competent" / "demonstrated") | Qwen3-8B dense [, I-competent / I-demonstrated] / dense (Qwen3-8B)[, no instr.] or dense, competence / demonstration instr. |
   | `retrieval: embed`, no Qwen model (MiniLM default) | `embed` | MiniLM / MiniLM |
   | `retrieval: sota` | `sota` / `sota_icomp` / `sota_idemo` | fusion + reranker [, I-…] / fusion + reranker[, no / competence / demonstration instr.] |
   | `retrieval: bm25` / `declared` | `bm25` / `declared` | BM25 / declared-claim top-k; condensed BM25 / declared top-k |
   | `retrieval: hybrid` (fusion without reranker), `retrieval: midian_wo_audit` (formerly `midian`, the cohort of MIDIAN w/o audits) | dropped | — |

   `dedup` is ignored for every retrieval shortlist, and every `embed`, `sota`, `declared`, `bm25` and `dense` row carries
   `"dedup": true`. So only the `tfidf` bar is un-deduplicated: at 10^5 it is the clone-filled pre-registered adapter
   (erratum 25).
4. **Regime** (`tagged` → `fw_variant_numbers.regime`, `fw_variant_numbers.py`): β = 0 → `beta0` whatever the
   liar_select; β = 0.5 with `low_skill_first` → `cartel`; everything else → `beta<x>_<random|cartel>`. When a β = 0 cell has
   rows under both liar_select values, only `random` is kept (the same liar-free cell).
5. **Per-framework seed series**: `q.groupby(["method","seed"]).success.mean().unstack(0)`; one value per seed.
6. **One grid per (framework, shortlist, cell), with no pooling across grids**: the grid with the most seeds wins; on a
   tie the alphabetically first grid wins.
7. **Erratum-28 status** (`pending_reruns`, `fw_variant_numbers.py`): every (grid, framework, dist, regime) of
   `$RTE_DATA/results/quarantine_units.tsv` whose rerun has **not landed** — `landed()` checks that every param variant of
   the method in that grid has a row for that (dist, β, liar_select, seed) — unless `logs/DONE_stage2` exists (then none;
   with only `DONE_stage1`, just the `fw_live_n(100|1000)(_lowskill)?_sota` grids). A framework × shortlist series touched
   by such a unit is written with `rerun_outstanding = True`. Today `DONE_stage1` exists (not `DONE_stage2`), and 51
   keys are outstanding, all in `fw_live_n{100,1000}[_lowskill]_sota`; they flag three rows of `live.csv` (heavy_tail /
   bimodal cartel cells), none of them in E.
8. **Reference lines**: for each (n, dist, regime) cell, rows with `method ∈ {oracle, random}` or MIDIAN (`method == "midian"`
   with params `{}`) from `REF_GRIDS[family][n]` (`H30_REF` instead once `h30()` is true, as it is today), same n, dist and regime, turned into per-seed means; the grid with
   the most seeds wins, the first listed on a tie. No pooling and no seed matching to the framework bars, despite the
   comment above `REF_GRIDS` ("pooled and matched on (dist, regime, seed)"). Written as `"MIDIAN, no framework"` (`figspec.NAME["midian_ref"]`),
   `oracle` and `random` with `shortlist = "-"`; the per-condition figures draw the first two (random is in
   the CSV for the condensed figures). Since 2026-09-24 `routereval_mmlu_norep_cal` also runs `random`, so H has a random
   value at every m.
9. **CSV row** (`draw`, `main`): `family, regime, dist, n, arm (ABBR name), shortlist, mean, ci_lo, ci_hi, seeds,
   rerun_outstanding`; `mean` = mean over seeds, `ci_lo` / `ci_hi` = mean ∓ 1 standard error over seeds (`stats.se`,
   imported as `_ci`, `shortlist_figs.py`), which is also the per-condition figures' whisker. A cell with
   fewer than two shortlists gets no framework rows. The per-condition PNGs draw **no markers**: `draw` collects star
   positions but never plots them, although the module docstring and the `INDEX.md` text  still say "* marks a bar".

### 5.1.3 Stage 2: `shortlist_condensed` (per-framework means → bars)

Style, names, colours and saving come from `scripts/figures/lib/figspec.py` (docs/figures.md).
- `load(family)` (`shortlist_condensed.py`): live keeps `dist == "specialist"`; routereval keeps
  `dist == "strong_to_weak"` **or** `n == 5000`. Both keep only `regime ∈ {beta0, cartel}` (`REG`).
- `summarise(d)`, over framework rows (`shortlist != "-"`):
  1. **Full seeds only** : a framework row counts only if its seed count equals the **maximum seed count of that
     (n, shortlist)**, taken over every framework and both regimes. Live: 10 at n = 10^2 / 10^3 for TF-IDF, MiniLM,
     rerank, MIDIAN cohort, BM25 and plain dense, 3 for the five backfill shortlists there (dense I-comp / I-demo, rerank
     I-comp / I-demo, declared top-k), and 3 for everything at 10^4 / 10^5; RouterEval: 5 at m ≤ 1,000, 3 at 5,000.
  2. **`MIN_FW = 6`** : an (n, regime, shortlist) with fewer than 6 such frameworks is dropped, which leaves
     an empty slot instead of averaging a thinner, different framework set.
  3. `mean` = **unweighted mean over frameworks** of each framework's seed mean; `best` = max over frameworks; `k` =
     number of frameworks; `star` = any framework's `rerun_outstanding`; `name` = the appendix display name.
  4. `ref` = a pivot of the `-` rows to (n, regime) × {oracle, random, MIDIAN, no framework}.
- `body(s)`, for the body figures F and H: per n, dense and fusion + reranker are each replaced by **their best
  instruction variant by honest mean over frameworks** (`VARIANTS`: `dense` / `dense_icomp` / `dense_idemo` and
  `sota` / `sota_icomp` / `sota_idemo`), kept under the family key (`dense`, `sota`) with a `variant` column naming the
  variant drawn; the other body shortlists are kept as they are; names are the body names (`figspec.SHORTLIST_BODY`).
- **Consequence of rule 1 at n = 10^2 / 10^3**: the backfill shortlists (`fw_live_n{100,1000}[_lowskill]_backfill`, seeds
  1–3) are drawn, with every framework at 3 seeds, next to 10-seed bars in the same group. Bars in one group can therefore
  rest on different seed counts; nothing on the figure says so.
- `lines(ax, ref, n, x0, x1)` : draws **only the (n, "beta0") reference values**: the oracle as a grey dotted
  line (`figspec.oracle`, `#7f8c8d`, lw 1.0), random as a light-grey dotted line when the cell has one (`figspec.ref(…,
  "random")`, `#bbbbbb`) and MIDIAN as a solid green line (`figspec.ref`, `#2ecc71`, lw 1.0). Hatched bars are therefore
  read against the honest MIDIAN line (0.7816 vs its own cartel 0.7680 at live 10^2, 0.8356 vs 0.8278 at 10^5, 0.6860 vs
  0.6953 at RouterEval 1,000). The oracle is regime-invariant.
- `pair(ax, x, w, q, key, dots)`, for h ∈ {0: beta0, 1: cartel}, at `x + (h − 0.5)·w`:
  - regime missing for this (n, shortlist) → no bar, nothing drawn, and `INCOMPLETE` is set;
  - otherwise a bar of height `mean` in the shortlist's colour (`figspec.SHORTLIST_COLOR`; cartel: `hatch="////"`), a
    black dot (ms 3) at `best` if `dots`, and `INCOMPLETE |= star`. **No marker on any bar.**
- `finish` : y label "task success", `ylim(0.2, 0.95)`, two-decimal ticks, no title (`figspec.finish`); the
  legend via `figspec.legend` (one frameless legend in fixed order, never ranked by value); if `INCOMPLETE` the script
  prints `[<name>] INCOMPLETE: …` (the figure carries nothing); saves PDF + 300 dpi PNG and the frame to `<name>.csv`.
- **Legend order**: the shortlists in `figspec.SHORTLIST_ORDER` order as drawn (E, H) followed by the reference lines
  oracle, random, MIDIAN, no framework (`LINES`); F lists the reference lines and "best framework" (the dot) only,
  since its tick labels name the shortlists.

### 5.1.4 Who contributes: frameworks, seeds, cell parameters

- **The ten frameworks** (`configs/grids/` `_sets.frameworks`, ABBR in `scripts/figures/lib/figspec.py`): autogen,
  camel (CAMEL workforce), crewai, adk (Google ADK), langgraph, llama (LlamaIndex), maf, magentic (Magentic-One), openai
  (OpenAI Agents), smol (smolagents). Magentic-One is **excluded by design** from `fw_live_n10k_cartel` and its
  mirrors (`configs/grids/`) and from every `fw_routereval_*` and `re_sl_*` grid (and their `_norep_cal` mirrors), so H
  averages at most 9; at the 10^4 cartel it is supplied by `fw_live_n10k_cartel_magentic` (`configs/grids/`) and the
  backfill grid, and it now counts in every 10^4 cartel E bar.
- **Which frameworks enter a bar** is decided by the full-seed rule, not by the grid lists: a framework whose rows were
  quarantined (erratum 28) or refused (ADK's "Tool 'agent_…' not found", MAF's "KeyError: 'agent_…'", both now
  `invalid_action` non-picks under erratum 29, §3.2.6) and not yet rerun on every seed simply drops out of that bar. The
  tables in 5.2 and 5.5 list, per bar, the frameworks that count and those dropped for partial seeds; today four E bars
  drop one framework each and no H bar drops any.
- **Probes.** A framework reads only the declared/self-described channel (`FrameworkMethod.needs = {"declared"}`,
  `_common.py`). **The one exception is the MIDIAN-cohort shortlist**, which adds `{"probe", "reports"}`  and
  builds an internal `Midian(r=10)` with the cell's budget. So the MIDIAN-cohort bars spend the same b = 3 probe
  budget as MIDIAN (1.03 × n·K·b); all other bars spend none, and E–H are not a budget-matched comparison.
- **Cell parameters** (`configs/grids/`; mirrors match their base grid cell for cell):

  | family / n | base grid (line) | backend | dist | K | b | Q | seeds | declared channel |
  |---|---|---|---|---|---|---|---|---|
  | live 10^2 | `fw_live_n100`  and `fw_live_n100_lowskill`  | llm | specialist | 16 | 3 | 1000 | 1–10 | self_described |
  | live 10^3 | `fw_live_n1000`, `fw_live_n1000_lowskill`  | llm | specialist | 16 | 3 | 1000 | 1–10 | self_described |
  | live 10^4 | `live_n10k_v2`, `fw_live_n10k_cartel`  | llm | specialist | 16 | 3 | 300 | 1–3 | self_described |
  | live 10^5 | `live_n100k`  | llm | specialist | 16 | 3 | 300 | 1–3 | self_described |
  | RouterEval m = 10, 100 | `fw_routereval_small_norep_cal` (mirror of `fw_routereval_small`) | routereval (mmlu, `no_repeat`, `shuffle`) | strong_to_weak | 16 | 3 | 300 | 1–5 | calibrated, rendered to text |
  | RouterEval m = 1000 | `fw_routereval_1k_norep_cal` (of `fw_routereval_1k`) | routereval (mmlu, `no_repeat`, `shuffle`) | strong_to_weak | 16 | 3 | 300 | 1–5 | calibrated |
  | RouterEval 5000 | `fw_routereval_5k_norep_cal` (of `fw_routereval_5k`) | routereval (leaderboard_mmlu, `no_repeat`, `shuffle`) | all | 16 | 3 | 300 | 1–3 | calibrated |

  The `_em`, `_va`, `re_sl_declared_*` and `re_sl_embed_*` `_norep_cal` mirrors  share these cells; every
  H cell is β ∈ {0, 0.5} with `low_skill_first`.

### 5.1.5 Reference-line provenance

| family, n | oracle (grid, seeds, β = 0 value) | MIDIAN, no framework (grid, seeds, β = 0 value) | random (β = 0 value) |
|---|---|---|---|
| live 10^2 | `fw_live_n100`, 1–10, 0.8449 | `fw_live_n100`, 1–10, 0.7816 | 0.4258 |
| live 10^3 | `fw_live_n1000`, 1–10, 0.8612 | `fw_live_n1000`, 1–10, 0.8134 | 0.4322 |
| live 10^4 | `learned_n10k`, 1–3, 0.8589 | `learned_n10k`, 1–3, 0.8111 | 0.4167 |
| live 10^5 | `live_n100k`, 1–3, 0.8622 | `live_n100k`, 1–3, 0.8356 | 0.4422 |
| RouterEval 10 / 100 / 1000 | `routereval_mmlu_norep_cal`, 1–5: 0.8747 / 0.7873 / 0.8760 | `routereval_mmlu_norep_cal`, 1–5: 0.7080 / 0.6653 / 0.6860 | 0.5440 / 0.5400 / 0.5307 |
| RouterEval 5000 | `routereval5k_norep_cal`, 1–3, 0.9200 | `routereval5k_norep_cal`, 1–3, 0.7389 | 0.5267 |

Random comes from the same grid set as the oracle row of the cell (the grid with the most seeds wins per arm). The lines are not restricted to the seeds each framework actually has, and at 10^4 it comes from `learned_n10k`, not the
frameworks' `live_n10k_v2` (identical values). The RouterEval lines now come from the no-repeat reruns, so MIDIAN can
no longer memorise repeated test prompts (the old `routereval_mmlu` line carried about +0.02 of that, leakage audit L10).
Their cartel values are 0.7080 / 0.6620 / 0.6953 / 0.7333; at m = 1,000 the cartel line would sit above the honest one.

## 5.2 Figure E: `E_shortlists_by_n` (live, specialist, n = 10^2 … 10^5)

**Question.** On the live LLM backend (specialist population), how well do deployed frameworks route under each shortlist
source, and how does that change with n? How far are they from the oracle and from MIDIAN routing the whole population,
honest and under the β = 0.5 low-skill cartel?

**Code.** `by_n(s, ref, "E_shortlists_by_n", MAIN, figspec.SHORTLIST_APPENDIX, "appendix", "n", ncol=3)` (`shortlist_condensed.py`); appendix canvas 5.5 × 2.4 in.
- **x axis**: one group per n (100, 1,000, 10,000, 100,000), ticks $n = 10^2$, $10^3$, … (`figspec.pow10_ticks`).
- **Within a group**: one fixed slot per source in `MAIN` order (`figspec.SHORTLIST_ORDER` without BM25 and plain dense): hashed TF-IDF, MiniLM, dense competence instr., dense demonstration instr., fusion + reranker no instr., fusion + reranker competence instr., fusion + reranker demonstration instr., declared top-k, MIDIAN cohort. BM25 and plain Qwen dense are left out (10^3 ablation only).
- **Slot geometry**: w = 0.86/18 ≈ 0.048; source j at `i + (2j+1−9)·w`, honest at −w/2 and cartel at +w/2.
- **Marks**: bars = mean over frameworks (no dots), so **E has no black dots**; the three reference lines (oracle, random,
  MIDIAN, no framework) span `i ± 0.46` and use β = 0 values; y 0.2–0.95; colours and legend labels from `figspec`
  (appendix names); one legend row block above the axes, 3 columns.
- **Completeness**: reported (not drawn) while any of the 72 slots is empty or any drawn bar has `star`. Today all 72 are
  drawn and no counted framework has an outstanding erratum-28 rerun.
- **Declared top-k on live** ranks by the frameworks' real self-described claims (`declared_source = self_described`, the
  model's own self-rating; §3.4.9), and the lie inflates them for the cartel.

**Provenance per bar.** Winning grids:

| n | TF-IDF | MiniLM | rerank | MIDIAN cohort | declared / dense-I / rerank-I |
|---|---|---|---|---|---|
| 100 | `fw_live_n100` / `fw_live_n100_lowskill` | `fw_live_n100_em` / `…_lowskill_em` | `fw_live_n100_sota` / `…_lowskill_sota` | `fw_live_n100_verified_va` / `…_verified_va_lowskill` | `fw_live_n100_backfill` / `fw_live_n100_lowskill_backfill` (3 seeds; drawn) |
| 1,000 | `fw_live_n1000` / `fw_live_n1000_lowskill` | `fw_live_n1000_em` / `…_lowskill_em` | `fw_live_n1000_sota` / `…_lowskill_sota` | `fw_live_n1000_verified_va` / `…_verified_va_lowskill` | `fw_live_n1000_backfill` / `fw_live_n1000_lowskill_backfill` (3 seeds; drawn) |
| 10,000 | `live_n10k_v2` / `fw_live_n10k_cartel` | `fw_live_n10k_em` / `fw_live_n10k_cartel_em` | `fw_live_n10k_sota` / `fw_live_n10k_cartel_sota` | `fw_live_n10k_verified_va` / `fw_live_n10k_cartel_verified_va` | `fw_live_n10k_backfill` / `fw_live_n10k_cartel_backfill` |
| 100,000 | `live_n100k` (both) | `fw_live_n100k_em` | `fw_live_n100k_sota` | `fw_live_n100k_verified_va` | `fw_live_n100k_declared`, `fw_live_n100k_dense_instruct`, `fw_live_n100k_sota_instruct` |

(Magentic-One at the 10^4 cartel comes from `fw_live_n10k_cartel_magentic` and `fw_live_n10k_cartel_backfill`.)

**E and F table** (`E_shortlists_by_n.csv`). mean = bar height, best = best single framework (F's black dot), k =
frameworks in the bar, seeds/fw = the full seed count of that (n, shortlist). The last column lists the frameworks that
count, then those dropped for having fewer seeds. No bar has an outstanding erratum-28 rerun. BM25 and plain dense (10^3
honest only) are in the CSV but not drawn.

| slot | shortlist | regime | mean | best | k | seeds/fw | frameworks |
|---|---|---|---|---|---|---|---|
| n = 100 | tfidf | honest | 0.4794 | 0.5559 | 10 | 10 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 100 | tfidf | cartel | 0.4670 | 0.4844 | 10 | 10 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 100 | embed | honest | 0.5319 | 0.6111 | 10 | 10 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 100 | embed | cartel | 0.5233 | 0.5532 | 10 | 10 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 100 | dense_icomp | honest | 0.4884 | 0.5870 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 100 | dense_icomp | cartel | 0.4791 | 0.5257 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 100 | dense_idemo | honest | 0.4884 | 0.5850 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 100 | dense_idemo | cartel | 0.4794 | 0.5170 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 100 | sota | honest | 0.5389 | 0.6037 | 10 | 10 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 100 | sota | cartel | 0.5352 | 0.5661 | 10 | 10 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 100 | sota_icomp | honest | 0.5392 | 0.6127 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 100 | sota_icomp | cartel | 0.5320 | 0.5703 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 100 | sota_idemo | honest | 0.5290 | 0.6123 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 100 | sota_idemo | cartel | 0.5235 | 0.5730 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 100 | declared | honest | 0.6307 | 0.6500 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 100 | declared | cartel | 0.4944 | 0.5103 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 100 | va_cohort | honest | 0.5676 | 0.6482 | 10 | 10 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 100 | va_cohort | cartel | 0.5597 | 0.6103 | 10 | 10 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 1,000 | tfidf | honest | 0.3846 | 0.4313 | 10 | 10 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 1,000 | tfidf | cartel | 0.3766 | 0.4109 | 10 | 10 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 1,000 | embed | honest | 0.5088 | 0.6315 | 10 | 10 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 1,000 | embed | cartel | 0.5000 | 0.5598 | 10 | 10 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 1,000 | dense_icomp | honest | 0.5032 | 0.6087 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 1,000 | dense_icomp | cartel | 0.5001 | 0.5730 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 1,000 | dense_idemo | honest | 0.4964 | 0.6163 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 1,000 | dense_idemo | cartel | 0.4897 | 0.5507 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 1,000 | sota | honest | 0.4748 | 0.5131 | 9 | 10 | adk autogen camel crewai langgraph llama maf openai smol; dropped (partial seeds) magentic:9 |
| n = 1,000 | sota | cartel | 0.4813 | 0.5398 | 10 | 10 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 1,000 | sota_icomp | honest | 0.5024 | 0.5930 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 1,000 | sota_icomp | cartel | 0.5013 | 0.5797 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 1,000 | sota_idemo | honest | 0.4976 | 0.5903 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 1,000 | sota_idemo | cartel | 0.4939 | 0.5507 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 1,000 | declared | honest | 0.6132 | 0.6263 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 1,000 | declared | cartel | 0.5259 | 0.5510 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 1,000 | va_cohort | honest | 0.5742 | 0.6593 | 10 | 10 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 1,000 | va_cohort | cartel | 0.5678 | 0.6211 | 10 | 10 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 1,000 | bm25 | honest | 0.4229 | 0.5321 | 9 | 10 | adk autogen camel crewai langgraph maf magentic openai smol; dropped (partial seeds) llama:9 |
| n = 1,000 | dense | honest | 0.4629 | 0.5947 | 10 | 10 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 10,000 | tfidf | honest | 0.2928 | 0.3667 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 10,000 | tfidf | cartel | 0.2916 | 0.3522 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 10,000 | embed | honest | 0.4352 | 0.5711 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 10,000 | embed | cartel | 0.4232 | 0.4856 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 10,000 | dense_icomp | honest | 0.4968 | 0.6078 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 10,000 | dense_icomp | cartel | 0.4996 | 0.6233 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 10,000 | dense_idemo | honest | 0.4823 | 0.6200 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 10,000 | dense_idemo | cartel | 0.4784 | 0.5822 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 10,000 | sota | honest | 0.4107 | 0.5156 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 10,000 | sota | cartel | 0.4113 | 0.4878 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 10,000 | sota_icomp | honest | 0.4528 | 0.5589 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 10,000 | sota_icomp | cartel | 0.4483 | 0.5322 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 10,000 | sota_idemo | honest | 0.4348 | 0.5344 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 10,000 | sota_idemo | cartel | 0.4352 | 0.5256 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 10,000 | declared | honest | 0.6462 | 0.6756 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 10,000 | declared | cartel | 0.5289 | 0.5544 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 10,000 | va_cohort | honest | 0.5947 | 0.6778 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 10,000 | va_cohort | cartel | 0.5599 | 0.6300 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 100,000 | tfidf | honest | 0.3789 | 0.3789 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 100,000 | tfidf | cartel | 0.3789 | 0.3789 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 100,000 | embed | honest | 0.4610 | 0.6011 | 9 | 3 | adk autogen camel langgraph llama maf magentic openai smol; dropped (partial seeds) crewai:2 |
| n = 100,000 | embed | cartel | 0.4424 | 0.4978 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 100,000 | dense_icomp | honest | 0.4833 | 0.6267 | 9 | 3 | autogen camel crewai langgraph llama maf magentic openai smol; dropped (partial seeds) adk:2 |
| n = 100,000 | dense_icomp | cartel | 0.4870 | 0.6089 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 100,000 | dense_idemo | honest | 0.4702 | 0.6189 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 100,000 | dense_idemo | cartel | 0.4633 | 0.5711 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 100,000 | sota | honest | 0.4110 | 0.5122 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 100,000 | sota | cartel | 0.4094 | 0.4856 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 100,000 | sota_icomp | honest | 0.4320 | 0.5533 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 100,000 | sota_icomp | cartel | 0.4287 | 0.5278 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 100,000 | sota_idemo | honest | 0.4231 | 0.5211 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 100,000 | sota_idemo | cartel | 0.4172 | 0.5233 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 100,000 | declared | honest | 0.6244 | 0.6578 | 9 | 3 | autogen camel crewai langgraph llama maf magentic openai smol; dropped (partial seeds) adk:2 |
| n = 100,000 | declared | cartel | 0.5481 | 0.5700 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 100,000 | va_cohort | honest | 0.5711 | 0.6656 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |
| n = 100,000 | va_cohort | cartel | 0.5560 | 0.6667 | 10 | 3 | adk autogen camel crewai langgraph llama maf magentic openai smol |

**Reading notes.**
- Every slot is drawn. At 10^2 / 10^3 the five backfill shortlists average 3 seeds per framework and the others 10 (the
  full-seed rule is per (n, shortlist), 5.1.3).
- Four bars drop one framework for partial seeds: 10^3 rerank honest (Magentic-One, 9 of 10 seeds), 10^5 MiniLM honest
  (CrewAI, 2 of 3), 10^5 dense I-comp honest and 10^5 declared honest (ADK, 2 of 3). Their cartel twins count all ten, so
  those honest / cartel pairs average different framework sets. Every 10^4 bar, including the cartel ones, has all ten.
- The CSV also carries BM25 and plain dense at 10^3 honest, which E does not draw.

## 5.3 Figure F: `F_shortlists_1e5` (live, specialist, n = 100,000)

**Question.** At the one scale where all nine shortlist sources ran (live 10^5), which shortlist makes deployed
frameworks route best? What does the best single framework reach? How far are both from the oracle and from MIDIAN?

**Code.** `fig_F(body(s), ref, "F_shortlists_1e5", figspec.SHORTLIST_BODY, "body")` and `fig_F(s, ref, "F_shortlists_1e5_appendix", figspec.SHORTLIST_APPENDIX, "appendix")` (`shortlist_condensed.py`).
- **Two versions.** `F_shortlists_1e5` (Figure 3, body canvas 5.5 × 1.9 in) draws the body shortlists: declared top-k,
  MIDIAN cohort, MiniLM, hashed TF-IDF, and dense (Qwen3-8B) and fusion + reranker each as its best instruction variant at
  10^5 by honest mean (`body`; today both the competence instruction, `dense_icomp` 0.4833 and `sota_icomp` 0.4320; the
  CSV's `variant` column names it). BM25 has no 10^5 bar, so the body figure has six shortlists (12 bars).
  `F_shortlists_1e5_appendix` (appendix canvas) draws all nine 10^5 shortlists with the appendix names (18 bars).
- **x axis**: shortlists at n = 10^5, **sorted by the honest mean, descending**; tick labels are the figspec names,
  rotated 30°.
- **Bars**: at each x a solid honest bar at x − 0.19 and a hatched cartel bar at x + 0.19 (w = 0.38), height = mean over
  the full-seed frameworks (at least 6).
- **Black dot**: on each bar, the **best single framework's seed mean** (`best`).
- **Reference lines**: dotted oracle 0.8622, dotted random 0.4422 and solid MIDIAN 0.8356 ("MIDIAN, no framework")
  across the full width, the β = 0 values from `live_n100k` seeds 1–3. MIDIAN's own cartel value (0.8278) is not drawn.
- **Completeness** is reported, not drawn; today every bar of both versions is drawn and none is flagged. The legend is
  one row above the axes: oracle, random, MIDIAN, no framework, "best framework" (the dot).
- Every cell is a `mirror_of: live_n100k` (K = 16, b = 3, Q = 300, specialist, self-described, seeds 1–3). The TF-IDF dot
  equals its bar: every framework scores exactly 0.378889 = mean(0.3467, 0.3667, 0.4233) in every regime (the clone
  shortlist, §5.4).

| x slot | shortlist | regime | mean (bar) | best (black dot) | k |
|---|---|---|---|---|---|
| 0 | declared | honest | 0.6244 | 0.6578 | 9 |
| 0 | declared | cartel | 0.5481 | 0.5700 | 10 |
| 1 | va_cohort | honest | 0.5711 | 0.6656 | 10 |
| 1 | va_cohort | cartel | 0.5560 | 0.6667 | 10 |
| 2 | dense_icomp | honest | 0.4833 | 0.6267 | 9 |
| 2 | dense_icomp | cartel | 0.4870 | 0.6089 | 10 |
| 3 | dense_idemo | honest | 0.4702 | 0.6189 | 10 |
| 3 | dense_idemo | cartel | 0.4633 | 0.5711 | 10 |
| 4 | embed | honest | 0.4610 | 0.6011 | 9 |
| 4 | embed | cartel | 0.4424 | 0.4978 | 10 |
| 5 | sota_icomp | honest | 0.4320 | 0.5533 | 10 |
| 5 | sota_icomp | cartel | 0.4287 | 0.5278 | 10 |
| 6 | sota_idemo | honest | 0.4231 | 0.5211 | 10 |
| 6 | sota_idemo | cartel | 0.4172 | 0.5233 | 10 |
| 7 | sota | honest | 0.4110 | 0.5122 | 10 |
| 7 | sota | cartel | 0.4094 | 0.4856 | 10 |
| 8 | tfidf | honest | 0.3789 | 0.3789 | 10 |
| 8 | tfidf | cartel | 0.3789 | 0.3789 | 10 |

## 5.4 Figure G: `G_shortlist_lift_1e5` (gain over TF-IDF at n = 100,000, paired within framework)

**Question.** How much does each shortlist source improve a framework over the pre-registered hashed TF-IDF shortlist,
framework by framework, honest and under the cartel?

**Code.** `fig_G(d, n=100000)` (`shortlist_condensed.py`), the only figure computed from the per-framework rows; appendix canvas.
1. `fw` = the framework rows at n = 10^5 **with the maximum seed count of their shortlist** (3 for every shortlist); `base` = TF-IDF means by (regime,
   framework).
2. For each (regime, shortlist ≠ tfidf): `diff = mean[shortlist] − base` aligned on (regime, framework), `.dropna()`.
   **Paired within framework**: each framework is compared with its own TF-IDF number in the same regime.
3. `k` = number of paired frameworks; if `k < MIN_FW` (6) the bar is left out and the script reports it.
4. `lift = diff.mean()`; the whisker is **±1 standard error across frameworks**,
   `half = sd(diff, ddof=1) / √k`, a ~68 % interval, not a 95 % CI. It measures spread across frameworks; seed
   noise enters only through each framework's 3-seed mean.
5. `star` = any `rerun_outstanding` among the shortlist's rows; the script reports it (TF-IDF's own flag is not read).
6. **Order** by honest lift, descending; bars like F without dots, plus a ±1 s.e. whisker (`figspec.whisker`) and a zero
   line; tick labels are the appendix names, rotated 30°; y label "gain in task success over hashed TF-IDF" on two lines,
   no legend. The y range is automatic (no 0.2 floor).

**Erratum 25: G is a gain over a floor.** At 10^5 the pre-registered adapter's TF-IDF top 10 is **exactly one prompt
signature** (ten clones; `docs/archive/CHANGES_AND_ERRATA.md`), so every framework scores the same 0.378889 in every seed and
regime (`success.nunique() == 1` per (β, liar_select, seed) in `live_n100k`: 0.3467 / 0.3667 / 0.4233). `base` is a
constant, and `diff` is the shortlist's per-framework mean minus 0.378889: "how far above the clone floor", not a gain over
a working TF-IDF retriever. The de-duplicated TF-IDF (`*_dd` grids, never drawn) is 0.313 honest / 0.302 cartel (10 frameworks × 3 seeds, `fw_live_n100k_dd`), below the
floor. Every other drawn shortlist is deduplicated, so G mixes the retriever gain with the dedup gain.

| x slot | shortlist | regime | lift (bar) | ± half (1 s.e.) | k paired |
|---|---|---|---|---|---|
| 0 | declared | honest | +0.2456 | 0.0060 | 9 |
| 0 | declared | cartel | +0.1692 | 0.0050 | 10 |
| 1 | va_cohort | honest | +0.1922 | 0.0203 | 10 |
| 1 | va_cohort | cartel | +0.1771 | 0.0198 | 10 |
| 2 | dense_icomp | honest | +0.1044 | 0.0214 | 9 |
| 2 | dense_icomp | cartel | +0.1081 | 0.0160 | 10 |
| 3 | dense_idemo | honest | +0.0913 | 0.0186 | 10 |
| 3 | dense_idemo | cartel | +0.0844 | 0.0142 | 10 |
| 4 | embed | honest | +0.0821 | 0.0210 | 9 |
| 4 | embed | cartel | +0.0636 | 0.0121 | 10 |
| 5 | sota_icomp | honest | +0.0531 | 0.0168 | 10 |
| 5 | sota_icomp | cartel | +0.0498 | 0.0149 | 10 |
| 6 | sota_idemo | honest | +0.0442 | 0.0158 | 10 |
| 6 | sota_idemo | cartel | +0.0383 | 0.0169 | 10 |
| 7 | sota | honest | +0.0321 | 0.0194 | 10 |
| 7 | sota | cartel | +0.0306 | 0.0156 | 10 |

No G bar is flagged (`star` False throughout) and none is left out.

## 5.5 Figure H: `H_routereval_shortlists` (RouterEval, strong-to-weak pools m = 10/100/1000 + leaderboard 5000)

**Question.** Is the live shortlist ranking reproduced on real LLM pools (RouterEval, MMLU), where "agents" are real LLMs with
real per-question correctness and self-descriptions are rendered from the declaration vector?

**Code.** The same `by_n` as E, called as `by_n(body(s), ref, "H_routereval_shortlists", BODY_H, figspec.SHORTLIST_BODY, "body", "m", ncol=5)`
(`shortlist_condensed.py`) on `summarise(load("routereval"))`: Figure 4, body canvas 5.5 × 1.9 in, **six body
shortlists** per m (`BODY_H`: hashed TF-IDF, MiniLM, dense (Qwen3-8B), fusion + reranker, declared top-k, MIDIAN
cohort), dense and fusion + reranker each at its best instruction variant at that m by honest mean (`body`; today dense
competence instr. at every m, fusion + reranker stock at m = 10 and 1,000, demonstration instr. at 100, competence instr.
at 5,000; the CSV's `variant` column). The same full-seed and `MIN_FW` rules, no dots, y 0.4–0.95 (random is 0.527–0.544 here), the three β = 0
reference lines (oracle, random, MIDIAN, no framework; random at every m since `routereval_mmlu_norep_cal` runs it),
ticks $m = 10$, $10^2$, $10^3$, $5{,}000$, legend one block above in 5 columns. **Pool m** is the backend's `n`. The
instruction variants are in the table below (from `summarise`) but not in the figure.

**Which rows H reads: the erratum-30 reruns, switched all at once.** `shortlist_figs.GRIDS["routereval"]` reads the
`*_norep_cal` grids only when `h30()` is true, i.e. when **all** 15 framework mirrors in `H30_FW` and both reference grids
in `H30_REF` are complete; otherwise only the old grids, never a mix. `h30()` is True today, so H is drawn entirely from
the reruns. On those rows:
- **Claims are calibrated**: D is drawn from the live self-rating's distribution given S (11 values, corr ≈ 0.36; §1.3.3),
  S being the agent's train-prompt accuracy. The **declared top-k** shortlist ranks by that vector (plus the +0.4 lie for
  cartel liars), so it is no longer an oracle-like ranking: honest 0.618 / 0.573 / 0.602 / 0.565 at m = 10 / 100 / 1,000 /
  5,000, the best H bar only at m = 10.
- **The framework description text is rendered from the same claims**: every RouterEval agent's description is
  "Self-rated competence: <its top-5 families by `view.declared`, two decimals>" (`_common.py`, §3.2.4), so every
  text shortlist (TF-IDF, MiniLM, dense, rerank) and every supervisor read the calibrated claims, and the supervisor sees
  only the subject name, never the prompt. Calibrated claims tie often, so rendered texts coincide and the dedup
  shortlists retrieve from a smaller pool (§3.9 #1).
- **No test prompt repeats** within a stream (`no_repeat`, Q = 300), so neither the frameworks nor the MIDIAN reference
  line can gain from repeats.
- **The agent order is shuffled per seed** (`shuffle`), so lowest-index tie-breaks (declared top-k, TF-IDF on tied scores)
  are random rather than weakest-first.
- The pre-registered hashed **TF-IDF is the best honest H bar at m = 100, 1,000 and 5,000** (0.628 / 0.643 / 0.660).
- The old rows (programmatic claims, S + N(0, 0.05); repeated prompts; stored weak → strong order; Q = 1,000 at m ≤ 1,000),
  on which declared top-k was the best honest bar (0.668 / 0.750 / 0.819 / 0.856) and collapsed under the cartel, are not
  drawn.

**Why 5,000 is dist "all".** The m ≤ 1,000 pools are RouterEval's `mmlu` hard-setting pools, whose `dist` axis is a pool
configuration; the `_norep_cal` mirrors keep only `strong_to_weak`, and `load` keeps `strong_to_weak`. The 5,000 point is
the leaderboard backend (`dataset: leaderboard_mmlu`), a single pool with `dist: [all]`, admitted by `n == 5000`. Its model
names are synthetic.

**Provenance.** `fw_routereval_small_norep_cal` (m = 10, 100), `fw_routereval_1k_norep_cal`, `fw_routereval_5k_norep_cal`
(TF-IDF), each with `_em` (MiniLM) and `_va` (MIDIAN cohort) mirrors, `re_sl_declared_{small,1k,5k}_norep_cal` (declared
top-k) and `re_sl_embed_{small,1k,5k}_norep_cal` (Qwen3 dense I-comp / I-demo, rerank stock / I-comp / I-demo)
(`configs/grids/`): cells `dist: [strong_to_weak]` (the 5k grids `all`), β ∈ {0, 0.5}, `low_skill_first`,
calibrated claims, `no_repeat`, `shuffle`, Q = 300. Magentic-One is excluded by design, so at most 9 frameworks; all 9 have
full seeds in every slot. Seeds 1–5 at m ≤ 1,000 and 1–3 at 5,000. Reference lines from `routereval_mmlu_norep_cal` /
`routereval5k_norep_cal` (§5.1.5); the oracle at m = 100 (0.787) is lower than at m = 10 (0.875) or 1,000 (0.876). The
rows show one `oracle_success` per unit (no family-order split, §1.10 #1).

**At m = 10 the shortlist is the whole pool.** With k = 10 (`_common.py`) TF-IDF, MiniLM and declared top-k return all
10 agents (fewer after dedup when rendered texts coincide, §3.9 #1); only the order differs. Differences at m = 10
reflect ordering and the supervisor, not retrieval quality.

**H table** (every shortlist `summarise` computes on RouterEval; same columns as the E table). `H_routereval_shortlists.csv` holds the 48 drawn rows (six body shortlists × 4 m × 2 regimes, with `variant`).

| slot | shortlist | regime | mean | best | k | seeds/fw | frameworks |
|---|---|---|---|---|---|---|---|
| pool m = 10 | tfidf | honest | 0.5984 | 0.6180 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 10 | tfidf | cartel | 0.5702 | 0.6000 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 10 | embed | honest | 0.5923 | 0.6147 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 10 | embed | cartel | 0.5536 | 0.5693 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 10 | dense_icomp | honest | 0.5955 | 0.6287 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 10 | dense_icomp | cartel | 0.5453 | 0.5713 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 10 | dense_idemo | honest | 0.5811 | 0.6007 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 10 | dense_idemo | cartel | 0.5379 | 0.5660 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 10 | sota | honest | 0.5979 | 0.6327 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 10 | sota | cartel | 0.5446 | 0.5773 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 10 | sota_icomp | honest | 0.5969 | 0.6207 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 10 | sota_icomp | cartel | 0.5498 | 0.5840 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 10 | sota_idemo | honest | 0.5958 | 0.6147 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 10 | sota_idemo | cartel | 0.5471 | 0.5787 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 10 | declared | honest | 0.6183 | 0.6393 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 10 | declared | cartel | 0.5910 | 0.6107 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 10 | va_cohort | honest | 0.6022 | 0.6333 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 10 | va_cohort | cartel | 0.5574 | 0.5920 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 100 | tfidf | honest | 0.6276 | 0.6400 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 100 | tfidf | cartel | 0.5886 | 0.6047 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 100 | embed | honest | 0.5387 | 0.5620 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 100 | embed | cartel | 0.5701 | 0.6013 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 100 | dense_icomp | honest | 0.5384 | 0.5693 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 100 | dense_icomp | cartel | 0.5576 | 0.5733 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 100 | dense_idemo | honest | 0.5307 | 0.5587 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 100 | dense_idemo | cartel | 0.5610 | 0.5767 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 100 | sota | honest | 0.5191 | 0.5607 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 100 | sota | cartel | 0.5290 | 0.5493 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 100 | sota_icomp | honest | 0.5215 | 0.5620 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 100 | sota_icomp | cartel | 0.5276 | 0.5513 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 100 | sota_idemo | honest | 0.5223 | 0.5680 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 100 | sota_idemo | cartel | 0.5239 | 0.5493 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 100 | declared | honest | 0.5727 | 0.5967 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 100 | declared | cartel | 0.5495 | 0.5700 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 100 | va_cohort | honest | 0.5721 | 0.5860 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 100 | va_cohort | cartel | 0.5945 | 0.6127 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 1,000 | tfidf | honest | 0.6430 | 0.6513 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 1,000 | tfidf | cartel | 0.5790 | 0.5927 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 1,000 | embed | honest | 0.5010 | 0.5220 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 1,000 | embed | cartel | 0.5797 | 0.5980 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 1,000 | dense_icomp | honest | 0.5328 | 0.5653 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 1,000 | dense_icomp | cartel | 0.5793 | 0.5980 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 1,000 | dense_idemo | honest | 0.5232 | 0.5633 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 1,000 | dense_idemo | cartel | 0.5691 | 0.5847 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 1,000 | sota | honest | 0.5136 | 0.5560 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 1,000 | sota | cartel | 0.5576 | 0.5713 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 1,000 | sota_icomp | honest | 0.5027 | 0.5433 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 1,000 | sota_icomp | cartel | 0.5647 | 0.5753 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 1,000 | sota_idemo | honest | 0.5035 | 0.5480 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 1,000 | sota_idemo | cartel | 0.5755 | 0.5893 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 1,000 | declared | honest | 0.6021 | 0.6273 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 1,000 | declared | cartel | 0.5742 | 0.5980 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 1,000 | va_cohort | honest | 0.5781 | 0.6193 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 1,000 | va_cohort | cartel | 0.5700 | 0.5993 | 9 | 5 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 5,000 | tfidf | honest | 0.6596 | 0.6667 | 9 | 3 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 5,000 | tfidf | cartel | 0.6122 | 0.6267 | 9 | 3 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 5,000 | embed | honest | 0.4921 | 0.5311 | 9 | 3 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 5,000 | embed | cartel | 0.5870 | 0.6033 | 9 | 3 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 5,000 | dense_icomp | honest | 0.5657 | 0.5900 | 9 | 3 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 5,000 | dense_icomp | cartel | 0.5590 | 0.5800 | 9 | 3 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 5,000 | dense_idemo | honest | 0.5515 | 0.5822 | 9 | 3 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 5,000 | dense_idemo | cartel | 0.5711 | 0.5856 | 9 | 3 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 5,000 | sota | honest | 0.5284 | 0.5544 | 9 | 3 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 5,000 | sota | cartel | 0.5637 | 0.5867 | 9 | 3 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 5,000 | sota_icomp | honest | 0.5332 | 0.5922 | 9 | 3 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 5,000 | sota_icomp | cartel | 0.5460 | 0.5678 | 9 | 3 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 5,000 | sota_idemo | honest | 0.5184 | 0.5700 | 9 | 3 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 5,000 | sota_idemo | cartel | 0.5470 | 0.5722 | 9 | 3 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 5,000 | declared | honest | 0.5649 | 0.6056 | 9 | 3 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 5,000 | declared | cartel | 0.5335 | 0.5600 | 9 | 3 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 5,000 | va_cohort | honest | 0.6109 | 0.6333 | 9 | 3 | adk autogen camel crewai langgraph llama maf openai smol |
| pool m = 5,000 | va_cohort | cartel | 0.5512 | 0.5756 | 9 | 3 | adk autogen camel crewai langgraph llama maf openai smol |

**Completeness (H).** All 48 slots are drawn, each averaging all 9 frameworks at full seeds, and no bar has an outstanding
erratum-28 rerun.

## 5.6 Discrepancies, open questions, possible issues

Severity: **[H]** can change a reader's conclusion; **[M]** misleading labelling or a comparability caveat; **[L]** cosmetic.

- **D1 [M]: Frameworks can drop out of bars silently.** A framework enters a bar only with the full seed count of that
  (n, shortlist) (5.1.3); nothing on the figure says which frameworks a bar averages. Today four E bars drop one framework
  each (5.2 reading notes) and no H bar drops any. The completeness report (script output, `cluster/ops/figure_status.py`) covers empty slots, outstanding erratum-28
  reruns and bars under `MIN_FW`, not partial-seed exclusions of a drawn bar.
- **D2 [M]: Means over different framework sets or seed counts sit side by side.** The four E honest bars that drop a
  framework sit next to cartel twins with all ten; at 10^2 / 10^3 the backfill bars rest on 3 seeds per framework and the
  others on 10. `MIN_FW = 6` bounds how thin a bar can be, not how different two bars' sets are. G is paired within
  framework and so immune, apart from reducing to the common set.
- **D3 [M]: H is not live.** On RouterEval the agent descriptions are rendered claim vectors (calibrated claims now), the
  supervisor sees only the subject name, and the pool is fixed across seeds; only live (E–G) has real self-descriptions
  and real prompts (5.5). The calibrated claims are a post-hoc model of live self-ratings (§1.3.3).
- **D4 [M]: G's whisker is a between-framework spread.** It is ±1 standard error across frameworks, not seed
  uncertainty.
- **D5 [M]: Unequal seeds within one framework series are never averaged** (full-seed rule), but bars in one group can have
  different seed counts (the 10^2 / 10^3 backfill shortlists, 3 vs 10).
- **D6 [M]: The figures are only as fresh as the last `shortlist_figs.py` run.** The condensed script never checks freshness.
- **D7 [M]: The reference lines are honest-only and not seed-matched** (5.1.3, 5.1.5); the comment above `REF_GRIDS`
  claims pooling and seed matching the code does not do.
- **D8 [M]: E–H are not budget-matched.** The frameworks spend no probes; the MIDIAN-cohort bars and the MIDIAN line spend
  about 1.03 × n·K·b (protocol audit F3).
- **D9 [M]: G is a lift over a degenerate floor** (5.4). The figure's y label only says "gain in task success over hashed
  TF-IDF"; the caption must say that at 10^5 TF-IDF is the clone floor.
- **D10 [M]: Only the TF-IDF bar is un-deduplicated**, so at 10^5 "better retriever" and "no clones" are conflated.
- **D11 [L]: At RouterEval m = 10, k = 10 = m** (5.5).
- **D12 [L]: (resolved)** Legends are no longer ranked by value; E and H list the shortlists in the fixed `figspec.SHORTLIST_ORDER`.
- **D13 [L]: The bars start at y = 0.2 in E, F and H** (`finish`), which exaggerates ratios; G has an automatic
  range with a zero line.
- **D14 [L]: Documentation drift.** `shortlist_figs.py`'s docstring  and `INDEX.md` text  still say
  an asterisk marks a bar with an outstanding rerun and that the lines carry a 95 % seed-bootstrap band; no per-condition
  figure draws a marker, and every interval is ±1 s.e. (`stats.se`). The comment on `summarise`  is
  accurate about the per-(n, shortlist) seed rule.
- **D15 [L]: The `RTE_DATA` default is a trap** (5.1.1).
- **Open question: is the per-framework "best" dot comparable across bars?** It is the max over frameworks of a 3-seed
  mean at 10^5, upward-biased by selection, more so for bars with more frameworks (TF-IDF: all tie).

---

## 6. Figures C and D: routing work and energy per query

Scope: `figures/paper/C_routing_work_vs_n.{png,pdf,csv}` and `D_energy_per_query.{png,pdf,csv}`, written
by `python scripts/figures/efficiency_figs.py`. The script reuses the ledger cache `cost_by_n.csv` when it exists
(§6.1.1). It imports `scripts/figures/lib/figspec.py` for canvas, fonts, colours, names and legends, `scripts/analysis/energy.py`
for the energy model and the framework numbers (in `draw_D`), `label` from `scripts/figures/lib/rows.py`,
`rte.methods.keys` (every row is read through `keys.normalize`, so rows stored under the old MIDIAN keys load under
the new ones), and `BANDIT`, `LEARNED`, `NOT_RUNNABLE`, `OUT`, `POOLS` from `scripts/figures/condensed_figs.py`. Line
citations in this part are to `efficiency_figs.py` unless another file is named.

**How this part was checked (read-only).**
- The C CSV was re-derived from `cost_by_n.csv`, including the flat-NSW exclusion and both edges' slopes. The cache itself was re-derived from `bernoulli_scale_v5/rows.csv`
  (medians, row counts and the across-seed ranges of every pool member): all 83 rows agree.
- Every D row was re-derived from the cache and the energy constants (`build_J`, `marginal_J` to 1e-6 J).
- The `va_build` inputs were re-read from the five grids, with row counts per source, and `va_build` itself was called on
  the current cache. It returns the nine values of D's CSV, all tagged `ledger`.
- `energy.table()` was re-run with its row loader swapped for a read-only one (rows.csv plus rows.d, deduplicated on
  `rid`). The stock loader, `rte.analyze.load`, runs `consolidate`, which writes rows.csv (`rte/analyze.py`). The band
  edges came out the same: 20.600235 and 219.754123 J.

---

### 6.1 Shared inputs

#### 6.1.1 The ledger cache `cost_by_n.csv` (`costs`)
- **Source.** `$RTE_DATA/results/bernoulli_scale_v5/rows.csv`, read in chunks of 500,000 rows. Only the columns in
  `COLS` are kept, and every chunk goes through `keys.normalize`. The grid has no `rows.d` files, so rows.csv is complete.
- **Filter**. `beta == 0`, and either `b == 3` or MIDIAN (`method == "midian"` with params `{}`). So every arm is kept at b = 3, and
  MIDIAN at every b the grid ran: b = 3 at every n, plus b = 1 at 10^6 and 10^7 (the grid's only b = 1 cells, §1.6.3).
  The b = 1 rows are there for D's build ledger. There is no filter on `dist`, `liar_select` or `declared_source`. It
  needs none: every kept row is `dist = specialist`, `liar_select = random`, `declared_source = programmatic` (checked).
- **Labels.** `seed_tables.label(method, params)`  applies `ALIAS`, so `flat_probe_argmax{"online":true}` becomes
  `flat_probe_argmax_online`. Only labels in `SHOW` are kept. `SHOW`  is the four fixed arms in `FIXED`
  (: `midian`, `midian_wo_defenses`, `flat_probe_argmax_online`, `declared_argmax`) plus every member of `LEARNED`
  and `BANDIT` (`condensed_figs.py`). `warm_start_bandit[n0=0.5]` and `linucb_honest[bonus=own]` are in the bandit pool (`POOLS`,
  `condensed_figs.py`) but not in `BANDIT`, so they are not in `SHOW`, and `bernoulli_scale_v5` has no rows for them anyway.
  Conversely the cache holds the old context-bonus `linucb_honest` (it is in `BANDIT`), which `POOLS` no longer contains, so
  `draw_C` leaves it out of "any flat scan"'s member check (its work is n like every other bandit).
- **Aggregation.** The median over rows per (label, n, b) of `messages_per_task`, `comparisons_per_task`, `build_probes`
  and `build_messages`. There is one row per seed. The file has **83 rows** and a `b` column:
  - 81 at b = 3: 11 labels × 7 n (n = 10 … 10^7), plus `trueskill_per_family` at 4 n (10 … 10^4). The grid has no
    TrueSkill rows at n ≥ 10^5; that arm is also `NOT_RUNNABLE` there (`condensed_figs.py`).
  - 2 at b = 1: MIDIAN at 10^6 (16,799,988.5 build probes) and 10^7 (167,999,523.5).
  - `knn_router`, `knn_router_online` and `mlp_router` have no rows: they need prompt text, so they are `NOT_RUNNABLE` on
    bernoulli (`condensed_figs.py`) and were never run there.

  | n | 10, 100, 1,000, 10^4 | 10^5 | 10^6 | 10^7 |
  |---|---|---|---|---|
  | seeds per (arm, n, b) | 1,000 | 500 | 200 | 100 |

  The seed counts are the same for every arm, and for MIDIAN b = 1. Per-query counts are identical across seeds for
  every arm except `cluster_head_router` (n ≥ 100) and `disrouter_cascade` (all n), whose medians are plotted. For
  example, at n = 10^7 the per-seed range is 1,000,011–1,000,018 and 3,471,301–4,023,174. Among build counts, only
  MIDIAN's `build_probes` varies (e.g. 49,325–49,586 at n = 10^3).
- **Caching.** When the file exists it is read back and the rows are not touched. The docstring says to delete it
  to re-read. It covers n = 10 … 10^7; C
  draws n ≥ 100 at b = 3. The current file was written 2026-09-24 11:03 with the new labels.
- **Arms, labels and colours.** C draws MIDIAN, MIDIAN w/o defenses, "any flat scan" and the learned band, with the names
  and colours of `figspec` (`NAME`, `COLOR`): MIDIAN `#2ecc71`, MIDIAN w/o defenses `#c0392b`, any flat scan `#3498db`
  (flat probe argmax's blue), best learned/declared router `#ff7f0e`. The oracle, random and the frameworks are not in C.
  D uses only MIDIAN's rows (`label == "midian"`).

#### 6.1.2 What "routing work" counts
C's y value is `messages_per_task + comparisons_per_task` at b = 3. These are the ledger's per-task counts (§1.1.7). Hops
and reports are not counted, and neither is the build. Per arm (§2.11), with depth = ⌈log₁₀ n⌉ at r = 10:
- MIDIAN: 2 + depth messages and 1 + 10·depth comparisons, so work = 3 + 11·depth.
- MIDIAN w/o defenses: 3·depth messages and 20·depth comparisons, so work = 23·depth.
- Flat probe argmax, declared argmax and every bandit (UCB, Thompson, warm-start, LinUCB, TrueSkill): 0 messages and n
  comparisons (C's "any flat scan").
- Flat-NSW router: 0 messages and ef = 50 comparisons. Its ⌈log₂ n⌉ hops are not in C.
- Cluster-head router: 4 messages plus about n/10 + 10 comparisons (#heads + cluster size, §2.11). The cache has 24.6 at
  n = 10², 114.7 at 10³ and 1,000,014 at 10⁷.
- DisRouter cascade: messages only, 0 comparisons. They grow almost linearly: 54.4 at 10², 3,742,796 at 10⁷.

The cache confirms each formula at every n. MIDIAN w/o defenses' and MIDIAN's work is therefore linear in tree depth, which is
log₁₀ n rounded up: it grows like log n, not like a power of n. MIDIAN's b = 1 rows at 10^6 / 10^7 have the same
per-query counts as its b = 3 rows. Messages and comparisons are added one for one, although they cost very different
amounts (§6.3.2: 1e-3 J against 1e-8 J).

---

### 6.2 Figure C: `C_routing_work_vs_n`

**Question answered.** For the arms of A and B, how does the communication and computation a router spends on each query
grow with the population n? In particular, how do MIDIAN and MIDIAN w/o defenses compare with flat arms that scan every
agent?

**What is drawn** (`draw_C`; appendix canvas 5.5 × 2.4 in, `fig(draw_C, …, "appendix")`):
- **Three lines** : MIDIAN, MIDIAN w/o defenses and **"any flat scan"**, circle markers (ms 3), lw 1.4, in
  `figspec.COLOR` (green, red, blue), at n = 10², 10³ … 10⁷ (b = 3 rows only). There are no error bars.
  - "Any flat scan" stands for flat probe argmax, declared argmax and every best-bandit pool member (`FLAT_SCAN`), which all compare n agents per query and send no messages. The line is drawn from flat probe argmax's rows; the script first prints how far the runnable members differ at any n (0.0 % on the current cache:
    every member is exactly n). The bandits therefore get no band of their own, and declared argmax no line of its own.
- **The learned band** (best learned/declared router). A pooled bar in A / B is a different member per cell
  and per seed (§2.1.3), so C draws the pool's range instead of one line.
  - At each n ≥ 10², the pool is `POOLS["best_learned"]` minus `NOT_RUNNABLE("bernoulli", n)` (`runnable`),
    limited to the members that have cache rows.
  - **Constant-cost members are left out**. A member whose own work over n ≥ 10² has `growth(slope) ==
    "constant"` is dropped from the band at every n. Only `flat_nsw_router` qualifies: its work is ef = 50 at every n,
    slope about 0. The cross-fit picks it often (the majority pick in 19 of the 54 A / B "best learned/declared router"
    bars, table below).
  - The band spans the smallest to the largest remaining member's work at that n. It is filled in orange at α 0.3, with a lw 1.0 line along each edge. There are no markers.
  - The band's CSV rows hold the lower edge in `work` and the upper edge in `work_max`, with a slope for each (`slope`,
    `slope_max`).
- **Random** routes blindly and does no routing work; it is not drawn and has no CSV rows.
- Both axes are log. x = "population size n", y = "messages + comparisons per query" (`figspec.AXIS`).
  Horizontal grid only; no title.
- **Legend** : inside the axes, top left, one column, in drawing order (never ranked by value).
  - Each entry is `name (growth)`. `growth(s)`  turns a fitted slope s into words:
    "constant" if |s| < 0.02, "∝ log n" if s < 0.3, otherwise "∝ n^s" with two decimals.
  - The band's entry gives `growth` of both edges' slopes: if the two phrases agree it is that phrase; else, if both
    slopes are ≥ 0.3, a range "∝ n^a to n^z" with the smaller slope first.
  - The current entries are: MIDIAN (∝ log n), MIDIAN w/o defenses (∝ log n), any flat scan (∝ n^1.00), best
    learned/declared router (∝ n^0.97 to n^0.99). The learned range runs from the upper edge's slope (0.971) to the lower
    edge's (0.987).

**The slope** (`slope`). An ordinary least-squares line through (log₁₀ n, log₁₀ work) for n ≥ 10³ and work > 0,
via `np.polyfit(..., 1)`. That is five points, n = 10³ … 10⁷; n = 10² is drawn but not fitted. For the band it is fitted to
each edge separately. The slope goes into the CSV but only picks the legend's wording.
- MIDIAN w/o defenses' and MIDIAN's work is 23·depth and 3 + 11·depth (§6.1.2), linear in log n. A power law fitted to
  such a curve has a small slope (0.09) that falls as n grows, so `growth` labels it "∝ log n".
- The learned band's lower edge (cluster head) has slope 0.987 and its upper edge (DisRouter) 0.971. The flat-NSW
  router's slope is about 0, which is what removes it from the band.
- The thresholds 0.02 and 0.3 are fixed in the code. The fitted values here are about 0 (flat-NSW), 0.086–0.091,
  0.971, 0.987 and 1.000, so the thresholds sit far from all of them.

**Values** (`C_routing_work_vs_n.csv`: columns `arm, key, n, work, slope, work_max, slope_max`; one row per drawn n per
line or band, 24 rows):

| n | MIDIAN | MIDIAN w/o defenses | any flat scan (flat probe argmax, declared argmax, every bandit) | best learned/declared router, lower edge | best learned/declared router, upper edge |
|---|---|---|---|---|---|
| 10² | 25 | 46 | 100 | 24.58 (cluster head) | 54.35 (DisRouter) |
| 10³ | 36 | 69 | 1,000 | 114.74 (cluster head) | 488.06 (DisRouter) |
| 10⁴ | 47 | 92 | 10,000 | 1,014.0 (cluster head) | 4,533.7 (DisRouter) |
| 10⁵ | 58 | 115 | 100,000 | 10,013.8 (cluster head) | 42,499.9 (DisRouter) |
| 10⁶ | 69 | 138 | 1,000,000 | 100,013.8 (cluster head) | 394,053.4 (DisRouter) |
| 10⁷ | 80 | 161 | 10,000,000 | 1,000,013.7 (cluster head) | 3,742,795.6 (DisRouter) |
| fitted slope (n ≥ 10³) | 0.0860 | 0.0912 | 1.0000 | 0.9875 | 0.9709 |

**Pool members' work per query** (`cost_by_n.csv`, b = 3, medians over seeds; messages + comparisons). "—" means no row.
The pool at a given n is the members with a value there. The last column says where C draws the member.

| member | pool | drawn as | 10² | 10³ | 10⁴ | 10⁵ | 10⁶ | 10⁷ |
|---|---|---|---|---|---|---|---|---|
| `cluster_head_router` | learned | band (lower edge) | 24.58 | 114.74 | 1,014.0 | 10,013.8 | 100,013.8 | 1,000,013.7 |
| `flat_nsw_router` | learned | not drawn (constant) | 50 | 50 | 50 | 50 | 50 | 50 |
| `disrouter_cascade` | learned | band (upper edge) | 54.35 | 488.06 | 4,533.7 | 42,499.9 | 394,053.4 | 3,742,795.6 |
| `knn_router`, `knn_router_online`, `mlp_router` | learned | no rows | — | — | — | — | — | — |
| `ucb_per_family`, `thompson_per_family`, `warm_start_bandit` | bandit | any flat scan | 100 | 1,000 | 10,000 | 100,000 | 1,000,000 | 10,000,000 |
| `linucb_honest` (old context bonus) | not in `POOLS` | not drawn | 100 | 1,000 | 10,000 | 100,000 | 1,000,000 | 10,000,000 |
| `linucb_honest[bonus=own]` | bandit | no rows | — | — | — | — | — | — |
| `trueskill_per_family` | bandit | any flat scan | 100 | 1,000 | 10,000 | — | — | — |
| `warm_start_bandit[n0=0.5]` | bandit | no rows | — | — | — | — | — | — |

`knn_router`, `knn_router_online` and `mlp_router` cannot run on bernoulli, so C has no values for them. Where they did
run, their work is exactly n per query: 0 messages plus n comparisons. This was checked on every rows.csv that holds
them (live `learned_n100`, `learned_f1`, `learned_n10k*`, `live_n100k*`, plus RouterEval and LLMRouterBench): the
minimum and maximum of `messages_per_task + comparisons_per_task` equal n in every row, at n = 10 … 10⁵ for kNN and
n = 10 … 10³ for MLP (MLP is `NOT_RUNNABLE` from n = 5,000).

**Reading.**
- At n = 10² the learned band's floor (cluster head, 24.58) is just under MIDIAN (25), and MIDIAN w/o defenses (46) is
  under any flat scan (100).
- From 10³ on, both edges of the learned band are above MIDIAN w/o defenses and MIDIAN and grow almost like n: the floor
  is about n/10 + 14 (cluster head), the ceiling about 0.37·n at 10⁷ (DisRouter).
- The flat-NSW router's constant 50, if it were drawn, would sit under MIDIAN up to 10⁴ (47) and over it from 10⁵ (58).

**A band is not the cost of the arm B reports.** B's "best learned/declared router" bars are the cross-fitted pick of one
member per seed (§2.1.3). C's band brackets what the pick could cost only among the members it draws.
- **The band is not the whole pool.** `flat_nsw_router` is left out of C's band, but it is still a candidate in A and
  B's "best learned/declared router": `POOLS` (`condensed_figs.py`) feeds `arms_at` (`condensed_figs.py`),
  which calls `seed_tables.crossfit` with the whole runnable pool. A seed that picks it pays 50 comparisons per query,
  below the band.
- **Who the cross-fit picks.** From the "best learned/declared router" rows of `A_live_allb.csv` and `B_families_allb.csv`
  (`chosen` column of the 11:35 CSVs; 54 bars, the live 10⁵ bars counted in both files; none is incomplete). "Majority" is
  the number of bars whose most-picked member it is (the first listed in `chosen`); "seed-level" sums the per-seed picks
  over all 54 bars (1,002 seeds). Growth is the member's work per query over n: C's cache for the three declared /
  probe-index members, the live and other ledgers for kNN and MLP (above).

  | member | growth | majority (of 54 bars) | seed-level picks |
  |---|---|---|---|
  | `flat_nsw_router` | constant (50) | 19 (live 10⁴ b = 1, 3 both regimes; bernoulli 10⁷ b = 3, 5 both regimes; all 6 replay 10⁶ bars; RouterEval 5,000 honest b = 1, 3, 5 and cartel b = 3, 5) | 610 |
  | `knn_router_online` | n | 22 | 122 |
  | `mlp_router` | n | 10 | 64 |
  | `disrouter_cascade` | ~n^0.97 | 2 | 103 |
  | `cluster_head_router` | ~n^0.99 | 1 | 103 |

  So a constant-cost member is the majority pick in 19 of 54 bars and a minority pick in 6 more (live 10⁴ b = 5 both
  regimes, live 10⁵ b = 5 both regimes in A and B); for those bars the arm A / B reports does 50 comparisons per query,
  below MIDIAN w/o defenses at every n and below MIDIAN from 10⁵ up (figures audit F4).
- At bernoulli n = 10⁷, b = 3 (C's budget): best learned/declared router `flat_nsw_router` ×100 in both regimes (50
  comparisons per query, below the drawn band and below MIDIAN's 80); best bandit `warm_start_bandit` (n0 = 5, on
  calibrated claims) ×100 in both regimes, n per query (on the "any flat scan" line). At b = 1 the learned pick is
  `cluster_head_router` ×100 honest (about n/10 = 1,000,014 per query) and `disrouter_cascade` ×100 cartel (the band's
  ceiling, 3.7M).

So at C's budget the arm B reports as "best learned/declared router" at 10⁷ does constant work, outside C's band; its
"best bandit" does work of order n.

### 6.3 Figure D: `D_energy_per_query`

**Question answered.** Counting LLM calls, messages and comparisons in joules, after how many routed queries does
MIDIAN's one-off probing build cost less than a framework that calls a supervisor LLM on every query? How does that
point move with n and b?

**What is drawn** (`draw_D`; canvas "d", 5.5 × 1.8 in):
- **x** = queries served T, 200 log-spaced points from 10² to 10⁹. **y** = energy per query in J, with the build
  amortised. Both axes are log; the y axis starts at 1e-6 to leave room for the legend. Axis
  labels "queries served T" and "energy per query (J)" (`figspec.AXIS`); no title (the word "estimated" is for the
  caption).
- **MIDIAN lines.** Nine lines, one per (n, b) with n ∈ {10³, 10⁵, 10⁷} and b ∈ {1, 3, 5} :
  `y(T) = build_J / T + marginal_J`.
  - The colour is MIDIAN's green shaded by n (`N_SHADE`, through `figspec.shade`): n = 10³ light (mixed 50 %
    toward white), 10⁵ the base green, 10⁷ dark (× 0.6).
  - The line style gives b (`figspec.B_LINESTYLE`): dotted `:` for b = 1, solid for b = 3, dashed `--` for b = 5. lw 1.2.
  - As T grows each line falls with slope −1 and flattens at `marginal_J`. At T = 10⁹ only the n = 10³ lines have
    reached their floor, about 0.005 J.
- **The framework band**  spans all T, from the cheapest to the costliest framework's per-query energy
  (`energy.table().per_task_J` over the `fw_*` rows without a `supervisor` override): 20.60 J (AutoGen) to
  219.75 J (Magentic-One, 7B). Light grey `#e5e5e5` fill; both edges are drawn as grey lines (lw 0.6) and named at the
  right margin, outside the axes: "AutoGen, 20.6 J" and "Magentic-One, 220 J" (`doc_tables.NAMES`). The band
  has no legend entry.
- **The ×s** : black, ms 3, drawn above the lines. Each is at (break-even T, framework J), where a MIDIAN line
  crosses the lower edge (vs the cheapest framework) or the upper edge (vs the costliest). There are 18: 9 lines × 2 edges.
- **Legend** : inside the axes, bottom left, two columns: three colour entries $n = 10^3$ / $10^5$ / $10^7$
  (lw 2.5 in the three greens) and three line-style entries b = 1 / b = 3 / b = 5 (black), instead of nine separate
  entries.

#### 6.3.1 MIDIAN's build probes (`va_build`)
- **Grids read** (`VA_GRIDS`): `va_b_bernoulli_1e7`, `va_b_n1000`, `va_b_n100k`, `fw_live_n1000`, `live_n100k`. For
  each, every `rows.d/*.json` and the `rows.csv` are read, each through `keys.normalize`. rows.csv and rows.d hold the same rows in four of
  the five grids. Rows are deduplicated on `rid`; rows without a `rid` are all kept. Every rows.csv here has a
  `rid` column.
- **Aggregation** : `method == "midian"` with params `{}` (the full method), then the median of `build_probes` per (n, b). There is no filter on
  β, `dist`, `liar_select` or channel.
- **Bernoulli fill-in** : for every (n, b) of MIDIAN in the cache, the grid value is kept if there is one, else the
  cache's value (the `bernoulli_scale_v5` median) is used. So the live and `va_b` ledgers come first, then the bernoulli
  sweep. D uses the sweep for (10⁷, 1) and (10⁷, 3). The cache values at 10³ (49,440) and 10⁵ (4,959,820) are overridden
  by the live grids.
- **Ratio fallback** : for each b, `ratio[b]` = the median over the n available at that b of
  `build_probes / (n · 16 · b)`. Any (n, b) with no value would become `ratio[b] · n · 16 · b` and be tagged
  `ratio r x nKb` in the CSV's `build_source` column. K = 16 is hard-coded. No D cell uses it now; the docstring says so.

| n | b | build probes | ÷ n·K·b | CSV `build_source` | where the number comes from |
|---|---|---|---|---|---|
| 10³ | 1 | 16,805 | 1.0503 | ledger | `va_b_n1000`: 10 seeds × β {0, 0.5}, live specialist (20 rows) |
| 10³ | 3 | 49,427.5 | 1.0297 | ledger | `fw_live_n1000`: 10 seeds × β {0, 0.1, 0.25, 0.5} × 3 shapes (120 rows) |
| 10³ | 5 | 83,007.5 | 1.0376 | ledger | `va_b_n1000` (20 rows) |
| 10⁵ | 1 | 1,680,005 | 1.0500 | ledger | `va_b_n100k`: 3 seeds × β {0, 0.5} (6 rows) |
| 10⁵ | 3 | 4,959,662 | 1.0333 | ledger | `live_n100k`: 3 seeds, β {0, 0.25, 0.5} (18 rows) |
| 10⁵ | 5 | 8,319,642 | 1.0400 | ledger | `va_b_n100k`: 3 seeds × β {0, 0.5} (6 rows) |
| 10⁷ | 1 | 167,999,523.5 | 1.0500 | ledger | the cache: `bernoulli_scale_v5`, 100 seeds, β = 0 |
| 10⁷ | 3 | 495,999,294.5 | 1.0333 | ledger | the cache: `bernoulli_scale_v5`, 100 seeds, β = 0 |
| 10⁷ | 5 | 831,998,807 | 1.0400 | ledger | `va_b_bernoulli_1e7`: 100 seeds × β {0, 0.5} (200 rows) |

Every build overspends n·K·b by 3–5 % (§2.3.4). "ledger" means a median of measured `build_probes`, from a live or a
bernoulli run: the column does not say which backend.

#### 6.3.2 The energy model (`scripts/analysis/energy.py`)
Everything is estimated from call counts, not measured on a power meter. The module docstring gives the model
(`energy.py`).
- **GPU-seconds per LLM call** (`call`, `energy.py`): `params_b · (A · prompt_tokens + B · gen_tokens)`. `params_b` comes
  from `configs/models.yaml` (0.5, 1.5, 2, 3, 7, 9 and 14 B; `energy.py`).
- **B = 5A** (`energy.py`). A decode token is taken as 5× a prefill token on H100 / vLLM (docstring). This ratio is
  assumed, not measured.
- **A is calibrated to one measurement** (`energy.py`). A 7B supervisor call has 1,900 prompt and 65 generated tokens
  (`SUP_TOK`). It is set to cost 1/34 GPU-s: 34 req/s is the throughput of a saturated 1-GPU 7B replica, measured on
  2026-09-03 from 4 samples of 32–36 req/s (docstring). So A = (1/34) / (7 · (1,900 + 5·65)) = 1.888e-6 and B = 9.442e-6.
- **A supervisor call** (`SUP`, `energy.py`) = 1/34 = 0.029412 GPU-s. At 700 W that is **20.588 J**.
- **A probe** is one call by the probed agent's own model, using that model's measured token means (`PROBE_TOK`,
  `energy.py`, "measured per-server lifetime means"). 7B and 14B use the 3B means as a proxy (294 prompt / 86
  generated tokens, the tool-enabled prompt). The expected cost per probe is the mix-weighted mean over the population's
  models (`MIX`, `energy.py`; `probe_cost`, `energy.py`). The mixes follow `draw_profiles` (`rte/backends/population.py`):

  | shape | model mix | GPU-s per probe | J at 700 W |
  |---|---|---|---|
  | specialist (used in D) | all 7 models, 1/7 each | 0.0057697 | **4.0388** |
  | heavy_tail | 90 % split over 0.5B / 1.5B, 10 % over 7B / 9B / 14B | 0.0017557 | 1.2290 |
  | bimodal | 80 % 0.5B, 20 % 7B | 0.0020319 | 1.4223 |

  Per model at 700 W: 0.5B 0.10 J, 1.5B 0.86 J, 2B 0.50 J, 3B 2.87 J, 7B 6.70 J, 9B 3.84 J, 14B 13.40 J.
- **Watts.** 700 W, the H100 TDP (`WATTS`, `energy.py`; `table(watts=700)`, `energy.py`). `draw_D` multiplies by 700 itself. The 400 W "typical draw" entry is used only in the docs/archive/results/RESULTS_energy.md table (`energy.py`).
- **Message = 1e-3 J** (`J_MSG`, `energy.py`): one RPC handled in about 100 µs on a 10 W core (`energy.py`).
- **Comparison = 1e-8 J** (`J_CMP`, `energy.py`): one float compare.
- **Pessimistic messages.** `table()` also computes `*_pess` columns with 10 × J_MSG = 1e-2 J (`energy.py`). D
  does not use them. With them, every break-even in §6.3.4 moves by less than 0.31 %.
- **Excluded.** The routed task's own execution. `exec_gpu_s_per_task` is computed (`energy.py`) but left out of
  `per_task_gpu_s` (`energy.py`). It is the same for every method: 0.0058 GPU-s per task on specialist (`energy.py`).
  CPU-side routing work (tree descent, TF-IDF) is left out of the LLM term (`energy.py`). It enters only through the
  ledger's message and comparison counts.

**MIDIAN in D** :
- `build_J = build_probes · 4.0388 + build_messages · 1e-3`. The build messages are the b = 3 cache value at every b
  (1,010 / 101,110 / 10,111,110 at n = 10³ / 10⁵ / 10⁷), a rounding term. Build comparisons are not charged.
- `marginal_J = messages_per_task · 1e-3 + comparisons_per_task · 1e-8` from the cache (b = 3). That is 5 msgs + 31 cmp =
  0.00500031 J at 10³, 7 + 51 = 0.00700051 J at 10⁵ and 9 + 71 = 0.00900071 J at 10⁷. It is the same at every b, as the
  ledger's per-query counts are.
- No LLM term per query: MIDIAN routes without an LLM call and without run-time probes (§2.3.6).

#### 6.3.3 The framework band (`energy.table()`, `energy.py`)
- **Rows** (`rows`, `energy.py`): grids `live_f1_n1000`, `variants_f1` and `fw_live_n1000` at n = 1,000, filtered to
  `declared_source == "self_described"`. The per-method **mean** of each ledger column is taken over every such row. β,
  `dist`, seed and quarantine are not filtered. Every `fw_` row comes from `fw_live_n1000`: 10 seeds × 3 shapes ×
  β {0, 0.1, 0.25, 0.5}, `liar_select = random`. That is 120 rows per framework (CAMEL 114, LlamaIndex 119, Magentic-One
  204 over its 7B and 14B arms). No erratum-28 quarantine is applied, so CrewAI and ADK, which E–H drop, are in the band.
- **Supervisor call-equivalents** (`energy.py`) = the framework's mean `wall_clock_per_task` divided by
  AutoGen's (1.8987 s). AutoGen makes one supervisor call per task, so it is 1.0 by construction. The code comments call
  the latency a "measured median" (`energy.py`), but it is the mean over rows. This assumes latency is proportional
  to GPU work. It was measured on a shared fleet.
- **The 14B Magentic-One arm** would be scaled by 14/7 (`energy.py`). D excludes it: `~fw.index.str.contains("supervisor")`
   drops the `fw_magentic_one{"supervisor":"Qwen/Qwen2.5-14B-Instruct"}` row (8.93 call-equivalents, 367.87 J).
  The band is also restricted to `fw_*` rows, so `llm_supervisor` is not in it.
- **`per_task_J`** (`energy.py`) = `(probes_per_task · probe_cost + call-equivalents · SUP) · 700 +
  messages_per_task · 1e-3 + comparisons_per_task · 1e-8`. For every framework, probes_per_task = 0, messages = 12
  (k + 2) and comparisons = 10 (k) per task (`_common.py`, k = 10).

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
A MIDIAN line meets a framework level F where build/T + marginal = F, so 

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
2. **`build_source` does not name the backend.** All nine D rows say "ledger". Seven are live or `va_b` ledgers (the six
   cells at 10³ / 10⁵, and 10⁷ b = 5 from `va_b_bernoulli_1e7`), and two are the `bernoulli_scale_v5` sweep at 10⁷ b = 1
   and 3. The table
   in §6.3.1 gives each one's source. The ratio fallback  is still in the code but unused.
3. **The learned band leaves out the one constant-cost member, the flat-NSW router.** `draw_C` drops any pool member whose
   own growth over n ≥ 10² is "constant", and only `flat_nsw_router` is (50 comparisons at every n). It is an ANN
   index over flat probe means held in one place (§2.8.3; docs/methods.md calls it a verified-centralised arm). It shows that
   sublinear routing is possible centrally; what MIDIAN adds is routing that is sublinear, decentralised and verified.
   - It is left out because its cost is constant (`draw_C`'s docstring: "constant-cost members, flat_nsw, left out").
     Yet it is the majority pick in 19 of the 54 "best learned/declared router" bars (live 10⁴ b = 1 / 3,
     bernoulli 10⁷ b = 3 / 5, every replay 10⁶ bar, most RouterEval 5,000 bars), including bernoulli 10⁷ at C's own
     budget b = 3 (§6.2).
   - It is still a candidate in A and B's cross-fitted "best learned/declared router" (`POOLS`, `condensed_figs.py`;
     `seed_tables.crossfit`). In those bars, and in 610 of the 1,002 seed-level picks, the arm A / B
     reports pays 50 per query: below the drawn band, below MIDIAN w/o defenses at every n and below MIDIAN from 10⁵ up. The band is
     not the whole pool, and a caption should say so (figures audit F4 suggests drawing it as its own dashed line).
   - The exclusion rule itself does not depend on the pick counts: it reads only C's cache.
   - C also leaves out the router's ⌈log₂ n⌉ hops.
4. **D's advantage is over the frameworks, not over flat probing.** Flat probe argmax builds with exactly n·K·b probes,
   slightly fewer than MIDIAN's 1.03–1.05 × n·K·b. Its per-query energy is n comparisons: 1e-5 J at 10³, 1e-3 J at 10⁵,
   0.1 J at 10⁷. Its D curve is not drawn. It would lie within a few percent of MIDIAN's: its build is 3–5 % smaller,
   and its per-query floor is lower at 10³ and higher at 10⁷. The asymptotic advantage over flat methods is C's routing work, which D prices at near zero.
5. **The framework band is a live n = 1,000 measurement, applied at every n.**
   - Retrieval over the n descriptions (TF-IDF or embedding scoring) is not charged. The ledger charges each framework
     query `compare(k)` and `message(k + 2)` (`_common.py`), which is 10 comparisons and 12 messages whatever n
     is. `energy.py` says CPU-side routing is "microseconds per task and omitted" (`energy.py`).
   - The framework build is not drawn: `message(n)` for registration (`_common.py`), 1 J at n = 10³ and 1e4 J at 10⁷.
     Neither is any description-embedding cost.
   - The supervisor prompt holds k = 10 descriptions, so its token count, and hence the call-equivalents, is taken as
     constant in n. It was measured only at n = 1,000.
   
   Every omission favours the frameworks, so the band is a lower bound on their cost at large n under this model.
6. **The band pools regimes and shapes.** It averages over β ∈ {0, 0.1, 0.25, 0.5} and all three population shapes. Per-β
   means of AutoGen's latency range from 0.92 s to 3.13 s. It includes CrewAI and ADK rows that E–H drop (no quarantine
   filter). Only the edges matter for D: AutoGen and Magentic-One.
7. **Call-equivalents are a latency ratio, not a count of calls.** The band's edges are named "AutoGen, 20.6 J" and
   "Magentic-One, 220 J"; the 1 to 10.7 call-equivalents behind them (§6.3.3) are for the caption. The floor
   is AutoGen, one call by construction. The 10.7 at the ceiling is Magentic-One's mean latency
   divided by AutoGen's (§6.3.3). No call count was recorded.
8. **Energy is an estimate** (the caption must say "estimated"; the figure has no title and its y label is "energy per query (J)"). Its uncertain parts: B = 5A is assumed; A rests on one 34 req/s throughput figure;
   7B / 14B probe tokens are 3B proxies; latency stands in for GPU work; 700 W is the TDP. A 400 W draw would scale every
   LLM term by 4/7. That lowers every y value but leaves T\* almost unchanged, because the build and the band scale
   together. Messages at 1e-3 J are a guess. The 1e-2 J variant moves T\* by under 0.31 %.
9. **MIDIAN's build is pooled over regimes and shapes.** `va_build` does not filter β or `dist`. The 10³ b = 3 value
   mixes 3 live shapes and 4 β values. The spread is small (49,368–49,468 at 10³ b = 3), so the effect is under 0.2 %.
10. **The 10⁵ builds rest on 3 seeds.** `va_b_n100k` has 6 rows per b (3 seeds × 2 regimes, whose build counts match per
    seed); `live_n100k` 18 at b = 3.
11. **The cache does not refresh itself.** `cost_by_n.csv` is reused until it is deleted. The `va_build` grids and
    the framework rows are re-read on every run. A rerun can therefore mix a new build with an old cache. For the counts in
    C this does not matter, because they are deterministic. A cache written before the `b` column was added would fail in
    `draw_C` (`d.b`), so an old file has to be deleted. A cache written before the rename would carry old labels
    (`midian_va`, plain `midian`): the current one (11:03) carries `midian` and `midian_wo_defenses`.
12. **Running `efficiency_figs.py` writes to result directories.** `energy.table()` loads through `rte.analyze.load`,
    which consolidates rows.d into rows.csv for `live_f1_n1000`, `variants_f1` and `fw_live_n1000` (`rte/analyze.py`),
    unless a grid has a `.merge_owner` file (`rte/run.py`). Without `RTE_DATA`, the default
    `/scratch/rte/results`  holds no grids. The script then stops with an error (`rte/analyze.py`) and draws
    nothing.
13. **C adds unlike units.** A message and a comparison each count 1 in C, although the energy model prices them 1e5 apart.
    In messages alone, the flat arms send 0 per query and MIDIAN sends 2 + depth.

---

## 7. Figure I: `I_max_lie` (appendix; the strongest declared lie)

**What it shows.** Live RTE, specialist population, β = 0.5 low-skill-first cartel, b = 3, seeds 1-3, at n = 10^3,
10^4 and 10^5. Each bar is an arm's mean task success under the **max lie** (`lie_mode: max`, `World.apply_lying`:
every cartel member declares skill 1.0 in every family), hatched as a cartel cell, with ±1 s.e. over seeds. A short
black tick on each bar marks the same arm under the **standard lie** of Figure A (the liar's true skill + 0.4, clipped;
`lie_mode: inflate`). The oracle is the grey dotted line. The legend order is `figspec.ORDER` without the oracle entry,
plus "standard lie".

**Grids.** `lie_max_n1000` mirrors `fw_live_n1000_lowskill` restricted to the specialist population and seeds 1-3;
`lie_max_n10k` mirrors `learned_n10k` and `lie_max_n100k` mirrors `live_n100k`, both restricted to β = 0.5 low-skill
first. All three run one arm list: declared argmax, MIDIAN, MIDIAN w/o defenses, flat probe argmax (online),
warm-start bandit (n0 = 5 and 0.5), online kNN router, random. `lie_max_fw_{n1000,n10k,n100k}` mirror them with the ten
frameworks on the declared top-k shortlist (`retrieval: declared`). Only b = 3 rows are read.

**Pooled arms.** As in A, but chosen by the best **mean over seeds**, not cross-fitted per seed, which favours the
rival: "best learned/declared router" = online kNN (the only learned router run here); "best bandit" = the better of the
two warm-start bandits; "best framework" = the best of the ten frameworks among those with the full seed count. MIDIAN,
MIDIAN w/o defenses, flat probe argmax, declared argmax and random are single arms.

**The standard-lie ticks** are read from `A_live_allb.csv` (the cartel regime, b = 3, or the budget-less row), and for
"best framework" from `results/aggregates/shortlist/live.csv` (declared top-k shortlist, cartel, specialist, the best framework's
mean).

**Values** (`I_max_lie.csv`; s.e. over 3 seeds):

| n | arm | max lie | s.e. | standard lie |
|---|---|---|---|---|
| 10^3 | MIDIAN | 0.805 | 0.005 | 0.807 |
| 10^3 | MIDIAN w/o defenses | 0.719 | 0.012 | 0.738 |
| 10^3 | flat probe argmax | 0.785 | 0.010 | 0.787 |
| 10^3 | best learned/declared router | 0.768 | 0.015 | 0.766 |
| 10^3 | best bandit | 0.769 | 0.009 | 0.764 |
| 10^3 | declared argmax | 0.218 | 0.047 | 0.520 |
| 10^3 | best framework | 0.432 | 0.068 | 0.560 |
| 10^3 | random | 0.428 | 0.015 | 0.432 |
| 10^4 | MIDIAN | 0.810 | 0.022 | 0.810 |
| 10^4 | MIDIAN w/o defenses | 0.752 | 0.028 | 0.752 |
| 10^4 | flat probe argmax | 0.774 | 0.033 | 0.774 |
| 10^4 | best learned/declared router | 0.731 | 0.030 | 0.746 |
| 10^4 | best bandit | 0.771 | 0.041 | 0.783 |
| 10^4 | declared argmax | 0.407 | 0.088 | 0.532 |
| 10^4 | best framework | 0.479 | 0.054 | 0.623 |
| 10^4 | random | 0.417 | 0.025 | 0.417 |
| 10^5 | MIDIAN | 0.832 | 0.008 | 0.828 |
| 10^5 | MIDIAN w/o defenses | 0.706 | 0.021 | 0.706 |
| 10^5 | flat probe argmax | 0.749 | 0.030 | 0.749 |
| 10^5 | best learned/declared router | 0.741 | 0.019 | 0.741 |
| 10^5 | best bandit | 0.749 | 0.015 | 0.778 |
| 10^5 | declared argmax | 0.491 | 0.057 | 0.578 |
| 10^5 | best framework | 0.493 | 0.020 | 0.609 |
| 10^5 | random | 0.442 | 0.016 | 0.442 |

**Reading it.** The max lie moves the arms that read declarations: declared argmax falls by 0.09-0.30 and the best
framework by 0.12-0.14 (the declared top-k shortlist fills with liars claiming 1.0), and the warm-start bandit, whose
prior reads the claims, by up to 0.03. The arms that route on probes alone (MIDIAN, MIDIAN w/o defenses, flat probe
argmax, the kNN router) move by at most 0.02. At 10^4 and 10^5 several of them coincide exactly with the tick: an arm
that never reads D sees the same world, stream and probe outcomes under either lie. At 10^3 the tick averages the 10
seeds of Figure A and the bar the 3 seeds of `lie_max_n1000`, so small differences there are seed differences.

**Caveats.** Three seeds per cell. The 10^4 and 10^5 framework bars were provisional when the CSV was written (the last
`lie_max_fw` units were still running); regenerate `I_max_lie` after they land.

---

## Appendix: figures A and B, per row (seed counts, source grid)

Values as in `A_live_allb.csv` and `B_families_allb.csv` (written 2026-09-24 11:35); B is ÷ oracle. Format: value [mean − 1 s.e., mean + 1 s.e.], seeds, source. For the pooled arms: value [whisker], scored seeds, pick counts, then the pool members missing at that b or missing some of the scored seeds ("missing or partial"; reported by the script and `cluster/ops/figure_status.py`, never drawn; none today) and "(b-invariant)" when every pick is a never-probing arm. Every slot has a bar. Sources for the four switched families are the erratum-30 per-seed tables (4.0.7); the framework arm of A ("best framework, best text shortlist") is listed in §4.1, not here.

### App. A_live_allb

| group | regime | arm | b=1 | b=3 | b=5 |
|---|---|---|---|---|---|
| n = 100 | honest | MIDIAN | 0.7105 [0.699,0.722] 10s (va_b_n100) | 0.7816 [0.769,0.794] 10s (bars) | 0.8121 [0.804,0.820] 10s (va_b_n100) |
| n = 100 | β=0.5 cartel | MIDIAN | 0.6972 [0.686,0.709] 10s (va_b_n100) | 0.7680 [0.756,0.780] 10s (bars) | 0.8121 [0.804,0.820] 10s (va_b_n100) |
| n = 100 | honest | MIDIAN w/o defenses | 0.7105 [0.699,0.722] 10s (rivals_b_n100) | 0.7770 [0.767,0.787] 10s (bars) | 0.8002 [0.792,0.808] 10s (rivals_b_n100) |
| n = 100 | β=0.5 cartel | MIDIAN w/o defenses | 0.6625 [0.653,0.672] 10s (rivals_b_n100) | 0.7405 [0.732,0.749] 10s (bars) | 0.7552 [0.747,0.764] 10s (rivals_b_n100) |
| n = 100 | honest | flat probe argmax | 0.7023 [0.692,0.712] 10s (rivals_b_n100) | 0.7805 [0.773,0.788] 10s (bars) | 0.8006 [0.794,0.807] 10s (rivals_b_n100) |
| n = 100 | β=0.5 cartel | flat probe argmax | 0.7023 [0.692,0.712] 10s (rivals_b_n100) | 0.7805 [0.773,0.788] 10s (bars) | 0.8006 [0.794,0.807] 10s (rivals_b_n100) |
| n = 100 | honest | best learned/declared router | 0.6151 [0.600,0.630] 10s, picks `mlp_router` ×8; `knn_router_online` ×2 | 0.7376 [0.726,0.749] 10s, picks `knn_router_online` ×10 | 0.7788 [0.769,0.789] 10s, picks `knn_router_online` ×10 |
| n = 100 | β=0.5 cartel | best learned/declared router | 0.6151 [0.600,0.630] 10s, picks `mlp_router` ×8; `knn_router_online` ×2 | 0.7376 [0.726,0.749] 10s, picks `knn_router_online` ×10 | 0.7788 [0.769,0.789] 10s, picks `knn_router_online` ×10 |
| n = 100 | honest | best bandit | 0.6881 [0.681,0.695] 10s, picks `warm_start_bandit[n0=0.5]` ×10 | 0.7524 [0.744,0.761] 10s, picks `linucb_honest[bonus=own]` ×10 | 0.7729 [0.764,0.782] 10s, picks `linucb_honest[bonus=own]` ×10 |
| n = 100 | β=0.5 cartel | best bandit | 0.6798 [0.671,0.689] 10s, picks `linucb_honest[bonus=own]` ×10 | 0.7524 [0.744,0.761] 10s, picks `linucb_honest[bonus=own]` ×10 | 0.7729 [0.764,0.782] 10s, picks `linucb_honest[bonus=own]` ×10 |
| n = 100 | honest | declared argmax | n/a | 0.5992 [0.589,0.609] 10s (bars) | n/a |
| n = 100 | β=0.5 cartel | declared argmax | n/a | 0.5181 [0.500,0.536] 10s (bars) | n/a |
| n = 100 | honest | random | n/a | 0.4258 [0.417,0.435] 10s (bars) | n/a |
| n = 100 | β=0.5 cartel | random | n/a | 0.4258 [0.417,0.435] 10s (bars) | n/a |
| n = 1,000 | honest | MIDIAN | 0.6913 [0.686,0.697] 10s (va_b_n1000) | 0.8134 [0.809,0.818] 10s (bars) | 0.8377 [0.834,0.841] 10s (va_b_n1000) |
| n = 1,000 | β=0.5 cartel | MIDIAN | 0.6914 [0.684,0.698] 10s (va_b_n1000) | 0.8074 [0.802,0.813] 10s (bars) | 0.8375 [0.834,0.841] 10s (va_b_n1000) |
| n = 1,000 | honest | MIDIAN w/o defenses | 0.6913 [0.686,0.697] 10s (rivals_b_n1000) | 0.7890 [0.788,0.791] 10s (bars) | 0.8079 [0.805,0.811] 10s (rivals_b_n1000) |
| n = 1,000 | β=0.5 cartel | MIDIAN w/o defenses | 0.6505 [0.638,0.663] 10s (rivals_b_n1000) | 0.7379 [0.731,0.745] 10s (bars) | 0.7658 [0.761,0.771] 10s (rivals_b_n1000) |
| n = 1,000 | honest | flat probe argmax | 0.6860 [0.680,0.692] 10s (rivals_b_n1000) | 0.7871 [0.784,0.790] 10s (bars) | 0.8108 [0.808,0.813] 10s (rivals_b_n1000) |
| n = 1,000 | β=0.5 cartel | flat probe argmax | 0.6860 [0.680,0.692] 10s (rivals_b_n1000) | 0.7869 [0.784,0.790] 10s (bars) | 0.8108 [0.808,0.813] 10s (rivals_b_n1000) |
| n = 1,000 | honest | best learned/declared router | 0.6364 [0.625,0.648] 10s, picks `mlp_router` ×9; `knn_router_online` ×1 | 0.7656 [0.760,0.772] 10s, picks `knn_router_online` ×10 | 0.7976 [0.794,0.801] 10s, picks `knn_router_online` ×10 |
| n = 1,000 | β=0.5 cartel | best learned/declared router | 0.6364 [0.625,0.648] 10s, picks `mlp_router` ×9; `knn_router_online` ×1 | 0.7656 [0.760,0.772] 10s, picks `knn_router_online` ×10 | 0.7976 [0.794,0.801] 10s, picks `knn_router_online` ×10 |
| n = 1,000 | honest | best bandit | 0.7038 [0.698,0.710] 10s, picks `warm_start_bandit[n0=0.5]` ×10 | 0.7737 [0.770,0.777] 10s, picks `warm_start_bandit[n0=0.5]` ×10 | 0.8020 [0.799,0.805] 10s, picks `warm_start_bandit[n0=0.5]` ×10 |
| n = 1,000 | β=0.5 cartel | best bandit | 0.6669 [0.653,0.680] 10s, picks `warm_start_bandit` ×8; `trueskill_per_family` ×2 | 0.7636 [0.760,0.767] 10s, picks `warm_start_bandit` ×7; `warm_start_bandit[n0=0.5]` ×3 | 0.8040 [0.801,0.807] 10s, picks `warm_start_bandit[n0=0.5]` ×10 |
| n = 1,000 | honest | declared argmax | n/a | 0.6154 [0.604,0.627] 10s (bars) | n/a |
| n = 1,000 | β=0.5 cartel | declared argmax | n/a | 0.5202 [0.506,0.535] 10s (bars) | n/a |
| n = 1,000 | honest | random | n/a | 0.4321 [0.427,0.437] 10s (bars) | n/a |
| n = 1,000 | β=0.5 cartel | random | n/a | 0.4321 [0.427,0.437] 10s (bars) | n/a |
| n = 10,000 | honest | MIDIAN | 0.6678 [0.656,0.680] 3s (va_b_n10k) | 0.8111 [0.788,0.834] 3s (bars) | 0.8356 [0.814,0.857] 3s (va_b_n10k) |
| n = 10,000 | β=0.5 cartel | MIDIAN | 0.6700 [0.657,0.683] 3s (va_b_n10k) | 0.8100 [0.788,0.832] 3s (bars) | 0.8311 [0.816,0.847] 3s (va_b_n10k) |
| n = 10,000 | honest | MIDIAN w/o defenses | 0.6678 [0.656,0.680] 3s (rivals_b_n10k) | 0.7850 [0.764,0.806] 3s (bars) | 0.8089 [0.786,0.831] 3s (rivals_b_n10k) |
| n = 10,000 | β=0.5 cartel | MIDIAN w/o defenses | 0.6389 [0.623,0.655] 3s (rivals_b_n10k) | 0.7522 [0.724,0.780] 3s (bars) | 0.7933 [0.768,0.818] 3s (rivals_b_n10k) |
| n = 10,000 | honest | flat probe argmax | 0.6522 [0.626,0.679] 3s (rivals_b_n10k) | 0.7744 [0.741,0.808] 3s (bars) | 0.8022 [0.769,0.835] 3s (rivals_b_n10k) |
| n = 10,000 | β=0.5 cartel | flat probe argmax | 0.6522 [0.626,0.679] 3s (rivals_b_n10k) | 0.7744 [0.742,0.807] 3s (bars) | 0.8022 [0.769,0.835] 3s (rivals_b_n10k) |
| n = 10,000 | honest | best learned/declared router | 0.5156 [0.475,0.556] 3s, picks `flat_nsw_router` ×2; `knn_router_online` ×1 | 0.7456 [0.726,0.765] 3s, picks `flat_nsw_router` ×3 | 0.7611 [0.735,0.787] 3s, picks `knn_router_online` ×2; `flat_nsw_router` ×1 |
| n = 10,000 | β=0.5 cartel | best learned/declared router | 0.5156 [0.475,0.556] 3s, picks `flat_nsw_router` ×2; `knn_router_online` ×1 | 0.7456 [0.726,0.765] 3s, picks `flat_nsw_router` ×3 | 0.7622 [0.737,0.788] 3s, picks `knn_router_online` ×2; `flat_nsw_router` ×1 |
| n = 10,000 | honest | best bandit | 0.7200 [0.707,0.733] 3s, picks `trueskill_per_family` ×3 | 0.7622 [0.743,0.782] 3s, picks `trueskill_per_family` ×2; `warm_start_bandit[n0=0.5]` ×1 | 0.8100 [0.790,0.830] 3s, picks `warm_start_bandit[n0=0.5]` ×3 |
| n = 10,000 | β=0.5 cartel | best bandit | 0.7033 [0.676,0.731] 3s, picks `trueskill_per_family` ×2; `warm_start_bandit` ×1 | 0.7833 [0.750,0.817] 3s, picks `warm_start_bandit` ×3 | 0.8178 [0.792,0.844] 3s, picks `warm_start_bandit` ×2; `warm_start_bandit[n0=0.5]` ×1 |
| n = 10,000 | honest | declared argmax | n/a | 0.6578 [0.621,0.694] 3s (bars) | n/a |
| n = 10,000 | β=0.5 cartel | declared argmax | n/a | 0.5322 [0.492,0.572] 3s (bars) | n/a |
| n = 10,000 | honest | random | n/a | 0.4167 [0.391,0.442] 3s (bars) | n/a |
| n = 10,000 | β=0.5 cartel | random | n/a | 0.4167 [0.391,0.442] 3s (bars) | n/a |
| n = 100,000 | honest | MIDIAN | 0.6400 [0.611,0.669] 3s (va_b_n100k) | 0.8356 [0.822,0.849] 3s (bars) | 0.8533 [0.836,0.871] 3s (va_b_n100k) |
| n = 100,000 | β=0.5 cartel | MIDIAN | 0.6178 [0.595,0.641] 3s (va_b_n100k) | 0.8278 [0.820,0.835] 3s (bars) | 0.8522 [0.835,0.869] 3s (va_b_n100k) |
| n = 100,000 | honest | MIDIAN w/o defenses | 0.6400 [0.611,0.669] 3s (rivals_b_n100k) | 0.7522 [0.730,0.774] 3s (bars) | 0.7922 [0.770,0.814] 3s (rivals_b_n100k) |
| n = 100,000 | β=0.5 cartel | MIDIAN w/o defenses | 0.5600 [0.487,0.633] 3s (rivals_b_n100k) | 0.7056 [0.685,0.726] 3s (bars) | 0.7567 [0.744,0.769] 3s (rivals_b_n100k) |
| n = 100,000 | honest | flat probe argmax | 0.6622 [0.620,0.705] 3s (rivals_b_n100k) | 0.7489 [0.718,0.779] 3s (bars) | 0.7856 [0.759,0.813] 3s (rivals_b_n100k) |
| n = 100,000 | β=0.5 cartel | flat probe argmax | 0.6622 [0.620,0.705] 3s (rivals_b_n100k) | 0.7489 [0.718,0.779] 3s (bars) | 0.7856 [0.759,0.813] 3s (rivals_b_n100k) |
| n = 100,000 | honest | best learned/declared router | 0.6044 [0.586,0.623] 3s, picks `knn_router_online` ×2; `cluster_head_router` ×1 | 0.7411 [0.722,0.760] 3s, picks `knn_router_online` ×3 | 0.7511 [0.741,0.762] 3s, picks `knn_router_online` ×2; `flat_nsw_router` ×1 |
| n = 100,000 | β=0.5 cartel | best learned/declared router | 0.6478 [0.598,0.697] 3s, picks `knn_router_online` ×3 | 0.7411 [0.722,0.760] 3s, picks `knn_router_online` ×3 | 0.7556 [0.744,0.767] 3s, picks `knn_router_online` ×2; `flat_nsw_router` ×1 |
| n = 100,000 | honest | best bandit | 0.6656 [0.643,0.688] 3s, picks `warm_start_bandit[n0=0.5]` ×2; `warm_start_bandit` ×1 | 0.7544 [0.732,0.777] 3s, picks `warm_start_bandit[n0=0.5]` ×3 | 0.8056 [0.778,0.833] 3s, picks `warm_start_bandit[n0=0.5]` ×3 |
| n = 100,000 | β=0.5 cartel | best bandit | 0.7078 [0.675,0.740] 3s, picks `warm_start_bandit` ×3 | 0.7778 [0.756,0.800] 3s, picks `warm_start_bandit` ×3 | 0.8022 [0.772,0.833] 3s, picks `warm_start_bandit` ×2; `warm_start_bandit[n0=0.5]` ×1 |
| n = 100,000 | honest | declared argmax | n/a | 0.6222 [0.597,0.648] 3s (bars) | n/a |
| n = 100,000 | β=0.5 cartel | declared argmax | n/a | 0.5778 [0.558,0.598] 3s (bars) | n/a |
| n = 100,000 | honest | random | n/a | 0.4422 [0.426,0.458] 3s (bars) | n/a |
| n = 100,000 | β=0.5 cartel | random | n/a | 0.4422 [0.426,0.458] 3s (bars) | n/a |

### App. B_families_allb

| group | regime | arm | b=1 | b=3 | b=5 |
|---|---|---|---|---|---|
| live n = 100,000 | honest | MIDIAN | 0.7423 [0.709,0.776] 3s (va_b_n100k) | 0.9691 [0.954,0.984] 3s (bars) | 0.9897 [0.969,1.010] 3s (va_b_n100k) |
| live n = 100,000 | β=0.5 cartel | MIDIAN | 0.7165 [0.690,0.743] 3s (va_b_n100k) | 0.9601 [0.952,0.969] 3s (bars) | 0.9884 [0.969,1.008] 3s (va_b_n100k) |
| live n = 100,000 | honest | MIDIAN w/o defenses | 0.7423 [0.709,0.776] 3s (rivals_b_n100k) | 0.8724 [0.847,0.898] 3s (bars) | 0.9188 [0.893,0.944] 3s (rivals_b_n100k) |
| live n = 100,000 | β=0.5 cartel | MIDIAN w/o defenses | 0.6495 [0.564,0.735] 3s (rivals_b_n100k) | 0.8183 [0.794,0.842] 3s (bars) | 0.8776 [0.863,0.892] 3s (rivals_b_n100k) |
| live n = 100,000 | honest | flat probe argmax | 0.7680 [0.719,0.818] 3s (rivals_b_n100k) | 0.8686 [0.833,0.904] 3s (bars) | 0.9111 [0.880,0.942] 3s (rivals_b_n100k) |
| live n = 100,000 | β=0.5 cartel | flat probe argmax | 0.7680 [0.719,0.818] 3s (rivals_b_n100k) | 0.8686 [0.833,0.904] 3s (bars) | 0.9111 [0.880,0.942] 3s (rivals_b_n100k) |
| live n = 100,000 | honest | best learned/declared router | 0.7010 [0.680,0.722] 3s, picks `knn_router_online` ×2; `cluster_head_router` ×1 | 0.8595 [0.837,0.882] 3s, picks `knn_router_online` ×3 | 0.8711 [0.859,0.883] 3s, picks `knn_router_online` ×2; `flat_nsw_router` ×1 |
| live n = 100,000 | β=0.5 cartel | best learned/declared router | 0.7513 [0.694,0.809] 3s, picks `knn_router_online` ×3 | 0.8595 [0.837,0.882] 3s, picks `knn_router_online` ×3 | 0.8763 [0.863,0.889] 3s, picks `knn_router_online` ×2; `flat_nsw_router` ×1 |
| live n = 100,000 | honest | best bandit | 0.7719 [0.746,0.798] 3s, picks `warm_start_bandit[n0=0.5]` ×2; `warm_start_bandit` ×1 | 0.8750 [0.849,0.901] 3s, picks `warm_start_bandit[n0=0.5]` ×3 | 0.9343 [0.902,0.967] 3s, picks `warm_start_bandit[n0=0.5]` ×3 |
| live n = 100,000 | β=0.5 cartel | best bandit | 0.8209 [0.783,0.858] 3s, picks `warm_start_bandit` ×3 | 0.9021 [0.876,0.928] 3s, picks `warm_start_bandit` ×3 | 0.9304 [0.895,0.966] 3s, picks `warm_start_bandit` ×2; `warm_start_bandit[n0=0.5]` ×1 |
| live n = 100,000 | honest | declared argmax | n/a | 0.7216 [0.692,0.751] 3s (bars) | n/a |
| live n = 100,000 | β=0.5 cartel | declared argmax | n/a | 0.6701 [0.647,0.694] 3s (bars) | n/a |
| live n = 100,000 | honest | random | n/a | 0.5129 [0.494,0.531] 3s (bars) | n/a |
| live n = 100,000 | β=0.5 cartel | random | n/a | 0.5129 [0.494,0.531] 3s (bars) | n/a |
| Bernoulli n = 10,000,000 | honest | MIDIAN | 0.8015 [0.799,0.804] 100s (bernoulli_scale_v5) | 0.9469 [0.945,0.949] 100s (bernoulli_scale_v5) | 0.9764 [0.975,0.978] 100s (va_b_bernoulli_1e7) |
| Bernoulli n = 10,000,000 | β=0.5 cartel | MIDIAN | 0.7977 [0.795,0.801] 100s (bernoulli_scale_v5) | 0.9386 [0.937,0.941] 100s (bernoulli_scale_v5) | 0.9738 [0.972,0.976] 100s (va_b_bernoulli_1e7) |
| Bernoulli n = 10,000,000 | honest | MIDIAN w/o defenses | 0.8015 [0.799,0.804] 100s (bernoulli_scale_v5) | 0.9110 [0.909,0.913] 100s (bernoulli_scale_v5) | 0.9400 [0.938,0.942] 100s (rivals_b_bernoulli_1e7) |
| Bernoulli n = 10,000,000 | β=0.5 cartel | MIDIAN w/o defenses | 0.7450 [0.740,0.750] 100s (bernoulli_scale_v5) | 0.8463 [0.843,0.850] 100s (bernoulli_scale_v5) | 0.8988 [0.896,0.901] 100s (rivals_b_bernoulli_1e7) |
| Bernoulli n = 10,000,000 | honest | flat probe argmax | 0.8005 [0.798,0.803] 100s (bernoulli_scale_v5) | 0.9088 [0.907,0.911] 100s (bernoulli_scale_v5) | 0.9409 [0.939,0.942] 100s (rivals_b_bernoulli_1e7) |
| Bernoulli n = 10,000,000 | β=0.5 cartel | flat probe argmax | 0.8005 [0.798,0.803] 100s (bernoulli_scale_v5) | 0.9088 [0.907,0.911] 100s (bernoulli_scale_v5) | 0.9409 [0.939,0.942] 100s (rivals_b_bernoulli_1e7) |
| Bernoulli n = 10,000,000 | honest | best learned/declared router | 0.6662 [0.659,0.674] 100s, picks `cluster_head_router` ×100 (b-invariant) | 0.7832 [0.778,0.788] 100s, picks `flat_nsw_router` ×100 | 0.8489 [0.844,0.853] 100s, picks `flat_nsw_router` ×100 |
| Bernoulli n = 10,000,000 | β=0.5 cartel | best learned/declared router | 0.6451 [0.640,0.650] 100s, picks `disrouter_cascade` ×100 (b-invariant) | 0.7832 [0.778,0.788] 100s, picks `flat_nsw_router` ×100 | 0.8489 [0.844,0.853] 100s, picks `flat_nsw_router` ×100 |
| Bernoulli n = 10,000,000 | honest | best bandit | 0.8907 [0.889,0.893] 100s, picks `warm_start_bandit` ×100 | 0.9380 [0.936,0.940] 100s, picks `warm_start_bandit` ×100 | 0.9513 [0.950,0.953] 100s, picks `warm_start_bandit` ×100 |
| Bernoulli n = 10,000,000 | β=0.5 cartel | best bandit | 0.8353 [0.833,0.838] 100s, picks `warm_start_bandit` ×100 | 0.9254 [0.924,0.927] 100s, picks `warm_start_bandit` ×100 | 0.9470 [0.945,0.949] 100s, picks `warm_start_bandit` ×100 |
| Bernoulli n = 10,000,000 | honest | declared argmax | n/a | 0.6625 [0.655,0.670] 100s (bernoulli_1e7_cal) | n/a |
| Bernoulli n = 10,000,000 | β=0.5 cartel | declared argmax | n/a | 0.4980 [0.487,0.509] 100s (bernoulli_1e7_cal) | n/a |
| Bernoulli n = 10,000,000 | honest | random | n/a | 0.4936 [0.492,0.496] 100s (bernoulli_scale_v5) | n/a |
| Bernoulli n = 10,000,000 | β=0.5 cartel | random | n/a | 0.4936 [0.492,0.496] 100s (bernoulli_scale_v5) | n/a |
| RouterBench replay n = 1,000,000 | honest | MIDIAN | 0.8289 [0.825,0.833] 30s (replay_1e6_split_cal) | 0.9658 [0.963,0.969] 30s (replay_1e6_split_cal) | 0.9833 [0.981,0.986] 30s (replay_1e6_split_cal) |
| RouterBench replay n = 1,000,000 | β=0.5 cartel | MIDIAN | 0.8279 [0.824,0.832] 30s (replay_1e6_split_cal) | 0.9622 [0.959,0.965] 30s (replay_1e6_split_cal) | 0.9824 [0.980,0.985] 30s (replay_1e6_split_cal) |
| RouterBench replay n = 1,000,000 | honest | MIDIAN w/o defenses | 0.8289 [0.825,0.833] 30s (replay_1e6_split_cal) | 0.8939 [0.890,0.897] 30s (replay_1e6_split_cal) | 0.9231 [0.920,0.927] 30s (replay_1e6_split_cal) |
| RouterBench replay n = 1,000,000 | β=0.5 cartel | MIDIAN w/o defenses | 0.5138 [0.495,0.533] 30s (replay_1e6_split_cal) | 0.7197 [0.707,0.732] 30s (replay_1e6_split_cal) | 0.8211 [0.816,0.826] 30s (replay_1e6_split_cal) |
| RouterBench replay n = 1,000,000 | honest | flat probe argmax | 0.8302 [0.827,0.834] 30s (replay_1e6_split_cal) | 0.8976 [0.895,0.900] 30s (replay_1e6_split_cal) | 0.9223 [0.920,0.925] 30s (replay_1e6_split_cal) |
| RouterBench replay n = 1,000,000 | β=0.5 cartel | flat probe argmax | 0.8302 [0.827,0.834] 30s (replay_1e6_split_cal) | 0.8976 [0.895,0.900] 30s (replay_1e6_split_cal) | 0.9223 [0.920,0.925] 30s (replay_1e6_split_cal) |
| RouterBench replay n = 1,000,000 | honest | best learned/declared router | 0.7604 [0.755,0.766] 30s, picks `flat_nsw_router` ×30 | 0.9029 [0.899,0.906] 30s, picks `flat_nsw_router` ×30 | 0.9324 [0.929,0.936] 30s, picks `flat_nsw_router` ×30 |
| RouterBench replay n = 1,000,000 | β=0.5 cartel | best learned/declared router | 0.7604 [0.755,0.766] 30s, picks `flat_nsw_router` ×30 | 0.9029 [0.899,0.906] 30s, picks `flat_nsw_router` ×30 | 0.9324 [0.929,0.936] 30s, picks `flat_nsw_router` ×30 |
| RouterBench replay n = 1,000,000 | honest | best bandit | 0.8801 [0.877,0.883] 30s, picks `warm_start_bandit` ×30 | 0.9392 [0.937,0.942] 30s, picks `linucb_honest[bonus=own]` ×30 | 0.9582 [0.956,0.960] 30s, picks `linucb_honest[bonus=own]` ×30 |
| RouterBench replay n = 1,000,000 | β=0.5 cartel | best bandit | 0.8734 [0.871,0.876] 30s, picks `linucb_honest[bonus=own]` ×30 | 0.9392 [0.937,0.942] 30s, picks `linucb_honest[bonus=own]` ×30 | 0.9582 [0.956,0.960] 30s, picks `linucb_honest[bonus=own]` ×30 |
| RouterBench replay n = 1,000,000 | honest | declared argmax | n/a | 0.3822 [0.368,0.397] 30s (replay_1e6_split_cal) | n/a |
| RouterBench replay n = 1,000,000 | β=0.5 cartel | declared argmax | n/a | 0.3029 [0.288,0.317] 30s (replay_1e6_split_cal) | n/a |
| RouterBench replay n = 1,000,000 | honest | random | n/a | 0.2433 [0.242,0.245] 30s (replay_1e6_split_cal) | n/a |
| RouterBench replay n = 1,000,000 | β=0.5 cartel | random | n/a | 0.2433 [0.242,0.245] 30s (replay_1e6_split_cal) | n/a |
| RouterEval n = 5,000 | honest | MIDIAN | 0.7017 [0.670,0.733] 3s (routereval5k_norep_cal) | 0.8031 [0.787,0.819] 3s (routereval5k_norep_cal) | 0.8671 [0.846,0.888] 3s (routereval5k_norep_cal) |
| RouterEval n = 5,000 | β=0.5 cartel | MIDIAN | 0.6969 [0.665,0.728] 3s (routereval5k_norep_cal) | 0.7971 [0.783,0.811] 3s (routereval5k_norep_cal) | 0.8671 [0.846,0.888] 3s (routereval5k_norep_cal) |
| RouterEval n = 5,000 | honest | MIDIAN w/o defenses | 0.7017 [0.670,0.733] 3s (routereval5k_norep_cal) | 0.7210 [0.687,0.755] 3s (routereval5k_norep_cal) | 0.7379 [0.715,0.761] 3s (routereval5k_norep_cal) |
| RouterEval n = 5,000 | β=0.5 cartel | MIDIAN w/o defenses | 0.6920 [0.658,0.726] 3s (routereval5k_norep_cal) | 0.6643 [0.647,0.681] 3s (routereval5k_norep_cal) | 0.6667 [0.648,0.685] 3s (routereval5k_norep_cal) |
| RouterEval n = 5,000 | honest | flat probe argmax | 0.6304 [0.608,0.653] 3s (routereval5k_norep_cal) | 0.6896 [0.670,0.710] 3s (routereval5k_norep_cal) | 0.7379 [0.723,0.753] 3s (routereval5k_norep_cal) |
| RouterEval n = 5,000 | β=0.5 cartel | flat probe argmax | 0.6304 [0.608,0.653] 3s (routereval5k_norep_cal) | 0.6896 [0.670,0.710] 3s (routereval5k_norep_cal) | 0.7379 [0.723,0.753] 3s (routereval5k_norep_cal) |
| RouterEval n = 5,000 | honest | best learned/declared router | 0.6123 [0.580,0.645] 3s, picks `flat_nsw_router` ×2; `cluster_head_router` ×1 | 0.7729 [0.748,0.798] 3s, picks `flat_nsw_router` ×3 | 0.8188 [0.788,0.849] 3s, picks `flat_nsw_router` ×3 |
| RouterEval n = 5,000 | β=0.5 cartel | best learned/declared router | 0.6969 [0.693,0.701] 3s, picks `disrouter_cascade` ×3 (b-invariant) | 0.7729 [0.748,0.798] 3s, picks `flat_nsw_router` ×3 | 0.8188 [0.788,0.849] 3s, picks `flat_nsw_router` ×3 |
| RouterEval n = 5,000 | honest | best bandit | 0.7802 [0.772,0.788] 3s, picks `linucb_honest[bonus=own]` ×3 | 0.8949 [0.878,0.912] 3s, picks `linucb_honest[bonus=own]` ×3 | 0.9046 [0.885,0.924] 3s, picks `linucb_honest[bonus=own]` ×3 |
| RouterEval n = 5,000 | β=0.5 cartel | best bandit | 0.7802 [0.772,0.788] 3s, picks `linucb_honest[bonus=own]` ×3 | 0.8949 [0.878,0.912] 3s, picks `linucb_honest[bonus=own]` ×3 | 0.9046 [0.885,0.924] 3s, picks `linucb_honest[bonus=own]` ×3 |
| RouterEval n = 5,000 | honest | declared argmax | n/a | 0.6739 [0.635,0.713] 3s (routereval5k_norep_cal) | n/a |
| RouterEval n = 5,000 | β=0.5 cartel | declared argmax | n/a | 0.5568 [0.512,0.601] 3s (routereval5k_norep_cal) | n/a |
| RouterEval n = 5,000 | honest | random | n/a | 0.5725 [0.550,0.595] 3s (routereval5k_norep_cal) | n/a |
| RouterEval n = 5,000 | β=0.5 cartel | random | n/a | 0.5725 [0.550,0.595] 3s (routereval5k_norep_cal) | n/a |
| LLMRouterBench n = 20 | honest | MIDIAN | 0.8444 [0.835,0.854] 5s (llmrouterbench_norep_cal) | 0.8844 [0.874,0.895] 5s (llmrouterbench_norep_cal) | 0.9120 [0.906,0.918] 5s (llmrouterbench_norep_cal) |
| LLMRouterBench n = 20 | β=0.5 cartel | MIDIAN | 0.8364 [0.831,0.842] 5s (llmrouterbench_norep_cal) | 0.8347 [0.824,0.845] 5s (llmrouterbench_norep_cal) | 0.8720 [0.851,0.893] 5s (llmrouterbench_norep_cal) |
| LLMRouterBench n = 20 | honest | MIDIAN w/o defenses | 0.8444 [0.835,0.854] 5s (llmrouterbench_norep_cal) | 0.8862 [0.877,0.896] 5s (llmrouterbench_norep_cal) | 0.8951 [0.882,0.908] 5s (llmrouterbench_norep_cal) |
| LLMRouterBench n = 20 | β=0.5 cartel | MIDIAN w/o defenses | 0.8240 [0.804,0.844] 5s (llmrouterbench_norep_cal) | 0.8107 [0.793,0.829] 5s (llmrouterbench_norep_cal) | 0.8204 [0.799,0.842] 5s (llmrouterbench_norep_cal) |
| LLMRouterBench n = 20 | honest | flat probe argmax | 0.8124 [0.799,0.826] 5s (llmrouterbench_norep_cal) | 0.8800 [0.871,0.889] 5s (llmrouterbench_norep_cal) | 0.8996 [0.895,0.905] 5s (llmrouterbench_norep_cal) |
| LLMRouterBench n = 20 | β=0.5 cartel | flat probe argmax | 0.8124 [0.799,0.826] 5s (llmrouterbench_norep_cal) | 0.8800 [0.871,0.889] 5s (llmrouterbench_norep_cal) | 0.8996 [0.895,0.905] 5s (llmrouterbench_norep_cal) |
| LLMRouterBench n = 20 | honest | best learned/declared router | 0.8684 [0.842,0.895] 5s, picks `mlp_router` ×5 | 0.9262 [0.909,0.944] 5s, picks `mlp_router` ×5 | 0.8996 [0.882,0.917] 5s, picks `mlp_router` ×5 |
| LLMRouterBench n = 20 | β=0.5 cartel | best learned/declared router | 0.8684 [0.842,0.895] 5s, picks `mlp_router` ×5 | 0.9262 [0.909,0.944] 5s, picks `mlp_router` ×5 | 0.8996 [0.882,0.917] 5s, picks `mlp_router` ×5 |
| LLMRouterBench n = 20 | honest | best bandit | 0.8311 [0.817,0.845] 5s, picks `linucb_honest[bonus=own]` ×4; `warm_start_bandit` ×1 | 0.8596 [0.853,0.866] 5s, picks `linucb_honest[bonus=own]` ×4; `warm_start_bandit[n0=0.5]` ×1 | 0.8844 [0.873,0.896] 5s, picks `linucb_honest[bonus=own]` ×4; `warm_start_bandit[n0=0.5]` ×1 |
| LLMRouterBench n = 20 | β=0.5 cartel | best bandit | 0.8489 [0.835,0.863] 5s, picks `linucb_honest[bonus=own]` ×5 | 0.8693 [0.856,0.883] 5s, picks `linucb_honest[bonus=own]` ×5 | 0.8996 [0.892,0.907] 5s, picks `linucb_honest[bonus=own]` ×5 |
| LLMRouterBench n = 20 | honest | declared argmax | n/a | 0.7440 [0.701,0.787] 5s (llmrouterbench_norep_cal) | n/a |
| LLMRouterBench n = 20 | β=0.5 cartel | declared argmax | n/a | 0.6142 [0.552,0.676] 5s (llmrouterbench_norep_cal) | n/a |
| LLMRouterBench n = 20 | honest | random | n/a | 0.6898 [0.665,0.714] 5s (llmrouterbench_norep_cal) | n/a |
| LLMRouterBench n = 20 | β=0.5 cartel | random | n/a | 0.6898 [0.665,0.714] 5s (llmrouterbench_norep_cal) | n/a |
