# Experimental design

How an experiment is specified (a grid), what one result is (a row), how results are paired and aggregated, and which
grids exist. The world, channels and accounting are in [architecture.md](architecture.md); the methods in
[methods.md](methods.md).

## Cells, units and rows

A **cell** fixes every axis of the world:

| axis | values used | meaning |
|---|---|---|
| `backend` | `llm`, `replay`, `routereval`, `bernoulli` | where agents and outcomes come from ([architecture.md](architecture.md#core-api)) |
| `n` | 10 to 10<sup>7</sup> | population size (a pool size on `routereval`) |
| `K` | 16 (live, bernoulli, RouterEval), 64 (replay), 15 (LLMRouterBench) | task families |
| `dist` | `specialist`, `heavy_tail`, `bimodal`, `correlated`, `iid_uniform`; pool configs on `routereval` | population shape |
| `beta` | 0, 0.1, 0.25, 0.5 | liar fraction |
| `liar_select` | `random`, `low_skill_first` | who lies |
| `collude` | true | liars' peer reports favour fellow liars and smear the top-20% honest |
| `declared_source` | `programmatic`, `self_described`, `calibrated` | how honest declarations are produced (below) |
| `lie_mode` | `inflate` (+0.4, clipped), `max` (1.0 in every family) | the declared lie |
| `demand` | `uniform` | family distribution of the task stream |
| `b` | 1, 3, 5 (up to 30 in the budget sweeps) | probes per (agent, family) in the build budget |
| `Q` | 300 to 2,000 | tasks in the stream |
| `backend_kwargs` | e.g. `calibrate_from`, `split`, `no_repeat`, `shuffle` | backend options |

A **unit** is one (cell, seed) and a **row** is one (cell, seed, method, params). Every grid also gets one `oracle` row
per unit.

The **β = 0.5 cartel** of the paper's figures is `beta = 0.5, liar_select = low_skill_first, collude = true`: half the
population, the least skilled half, claims inflated skill and reports for one another. **Honest** is `beta = 0`.

**Declarations.** `self_described` (live only) is the rating each agent gives itself after writing its own
self-description; it is over-confident and weakly informative (mean 0.69 against a mean true skill of 0.43,
correlation 0.36). `programmatic` is `clip(S + N(0, 0.05))`, an almost exact answer key. `calibrated` (non-live backends)
draws the declaration from the live self-rating's empirical distribution given the decile of S, so the non-live
backends see live-like claims ([errata.md](errata.md), erratum 30).

## Pairing

Within a cell and seed every method sees the same agents, the same liars and the same task stream, and the k-th probe of
any (agent, family) is the same instance for every method (index-seeded). Differences between methods within a unit are
therefore paired: the analysis differences methods per seed before averaging, and the live backend's memoised answers
are shared across methods. Seeds enter every random choice through `stable_seed_32(seed, ...)`, so a unit is
reproducible in any process. A method's own randomness is seeded by the seed and its `needs`, never by its name, so two
methods with the same needs draw the same stream.

## Seeds

A seed indexes an independent population, liar set and task stream. Seed counts are per grid and are chosen by cost:
10 on the live grids at n <= 10<sup>3</sup>, 3 at 10<sup>4</sup> and 10<sup>5</sup> (Q = 300), 3 to 10 on framework
grids, and 30 to 1,000 on the non-live scale grids. The live true skill S is measured once per prompt signature (200
probes) and shared across seeds, so the seed intervals cover population and stream variation, not the binomial error of
S itself.

## Metrics (one row)

| column | meaning |
|---|---|
| cell columns, `method`, `params`, `seed`, `grid` | the unit and the arm |
| `success` | fraction of the Q routed tasks solved |
| `success_late`, `n_late` | success on the last quarter of the stream (the online steady state) |
| `success_by_block` | success per block of the stream (learning curves, churn) |
| `oracle_success`, `regret` | the oracle's success on the same unit, and the difference |
| `misroute_to_liar` | fraction of tasks routed to a liar |
| `build_{probes,reports,messages,hops,comparisons,total_comm}` | ledger counters spent in `build` |
| `{probes,reports,messages,hops,comparisons,tasks,total_comm}_per_task` | ledger counters per routed task |
| `wall_clock_build`, `wall_clock_per_task`, `wall_clock_per_task_total` | wall time (mixes memo hits and misses; never used for cost claims) |
| `n_agents`, `n_liars`, `skill_*` | the realised population: mean skill, spread, excess of the best agent over the mean |
| `method_stats` | the method's own counters as JSON, e.g. a framework's `fallback_rate`, `success_strict`, `infra_errors`, `invalid_action` |
| `rid` | the row id |

### Rows and row ids

The row id is a 128-bit BLAKE2b hash of the canonical JSON of the cell, its `backend_kwargs` (and churn, if any), the
method, its params and the seed (`row_id` / `rid_of_row` in `rte/run.py`). The runner writes each row atomically to `rows.d/<rid>.json` and skips any
unit whose rid is already on disk, so a grid can be sharded over many jobs (`--only k=v` filters cells, `--seeds`
selects seeds) and any job can be killed and rerun. Rows are later folded into `rows.csv`. `backend_kwargs` values that
name files under the data root are hashed in their unexpanded form (`$RTE_DATA/...`), so a row id does not depend on
where the data root is mounted ([errata.md](errata.md#row-ids-of-grids-that-name-data-files)).

## Aggregation

`python -m rte.analyze --grid G` (package `rte/analysis/`) writes, per grid:

- per-arm tables by cell and by method class, with **95% percentile-bootstrap intervals over seeds**;
- **paired deltas** of every arm against the reference arm, MIDIAN w/o defenses (the pre-registered tree), with sign
  tests and a `WITHIN_FLOOR` flag when a delta is inside the reference's own seed envelope (`paired_vs_midian.csv`);
- **cost exponents**: log-log fits of each ledger counter against n, one fit per b (`cost_exponents.csv`);
- the pre-registered target checks, and a `summary.md` with all of the above.

The paper figures use ±1 standard error over seeds instead of intervals, and two further rules:

- **Budgets are never pooled.** Every bar is one b; b = 1 and b = 3 never share a table (verification is unfunded at
  b = 1).
- **Pooled arms are cross-fitted.** "Best learned/declared router" and "best bandit" are chosen per seed on the *other*
  seeds and scored on this one, so no bar is the maximum of noisy means over the seeds it reports.

[figure_provenance.md](figure_provenance.md) documents the aggregation behind each figure in full.

## Grids

An experiment is a named **grid**: a set of axis lists whose cartesian product is the cells, a seed range and a list of
methods. `python -m rte.run --grid <name>` runs one; `--dry-run` lists the units still to run.

**Where grids live.** `configs/grids/*.yaml`, one file per family: `00_base.yaml` (the axis `defaults` and the named
`_sets`, and nothing else), `smoke.yaml`, `live.yaml`, `synthetic.yaml`, `routereval.yaml`, `frameworks.yaml`,
`erratum30.yaml` and `archive.yaml` (grids no figure reads any more, kept because a grid's name is its results directory
and part of every row id). `rte.run.load_config()` reads them in sorted file order into one mapping and raises on a grid
name defined twice or a duplicate key anywhere. Every grid carries a one-line header, `# [TAGS] purpose; read by:
scripts`, with tags `FIG:<letters>` (drawn in figures A-I), `NUM` (feeds a quoted number or table, no figure), `HIST`
(historical, read by no current script), `SUPERSEDED-BY:<grid>`, `PENDING` (rows outstanding) and `SMOKE` (quick check);
`00_base.yaml`'s header defines the vocabulary.

**Grammar.**

- `defaults:` gives every axis a default; a grid overrides any of them.
- Lists are axes: `beta: [0, 0.25, 0.5]` makes three cells.
- `seeds: 1-10` is a range.
- `methods:` is a list whose items are `name`, `{name, params}` or `{set: NAME}`; a set of methods is flattened into
  the list. The former `methods: all` is frozen as the explicit set `all_arms`, so adding a method file never changes an
  existing grid.
- **Named sets** live once in `_sets` of `00_base.yaml`. A method set is referenced with `{set: NAME}`; a grid-level
  `use: NAME` merges the mapping `_sets[NAME]` (for example the live axis block) under the grid's own keys. YAML
  anchors are not used, because they cannot cross files.
- `mirror_of: <grid>` copies another grid (from any file), then applies this grid's overrides. It is how every variant
  is paired cell for cell with its source: a bernoulli twin of a live grid, a framework grid with another shortlist, a
  budget variant.
- `blocks:` puts several axis blocks into one grid (for example different seed counts per n).

**Separate grids, never edits.** A new variant of an existing experiment is a new grid (usually a `mirror_of`), never an
edit to the old one. A grid's results directory and row ids are then its own, and a number quoted from a grid cannot
move because rows were added to it. The golden file `tests/golden/grid_fingerprints.tsv` pins every grid's units, row
count and row-id hash; `tests/test_golden_fingerprints.py` fails if a configuration change alters any of them.

### Catalogue

197 grids, 183,793 units and 4,520,534 method rows in the golden fingerprint file, plus `reviewer_bernoulli`. Each grid is counted once, in the
first family it matches:

| family | grids | what they are |
|---|---|---|
| `smoke`, `bernoulli_cost_smoke` (+ `reviewer_bernoulli`) | 2 (+1) | CPU checks in seconds; `reviewer_bernoulli` (MIDIAN, its ablations and the main rivals on a laptop) was added after the golden file |
| `*_cal`, `*_norep_cal`, `rivals5_*` | 26 | erratum 30: calibrated claims, split probe / task prompts, no repeated prompts, shuffled pools (non-live backends and their framework mirrors) |
| `fw_live_*` | 51 | the ten frameworks on the live population, one grid per (n, regime, shortlist source): `_dd` deduplicated TF-IDF, `_em` MiniLM, `_sota` BM25 / dense / fusion / reranker, `_verified*` MIDIAN cohorts, `_declared`, `_lietext`, `_backfill` |
| `live_*` | 12 | live LLM population, every self-contained rival, n = 10<sup>2</sup> to 10<sup>5</sup> |
| `learned_*` | 9 | RouterBench's kNN and MLP routers as methods on our terms |
| `va_b_*`, `rivals_b_*`, `pool_fill_*`, `pool_seeds_n1000`, `tune_wsb_*`, `tuned_wsb_*`, `linucb_fix_*`, `trueskill_fix_*` | 39 | budgets b = 1 / 5 for MIDIAN and its rivals, and every member of the cross-fitted pools, per figure cell |
| `lie_max_*` | 6 | the max lie (Figure I) |
| `routereval_*`, `llmrouterbench_pool`, `fw_routereval_*`, `re_sl_*` | 18 | RouterEval pools of 10 to 5,000 real LLMs and LLMRouterBench, with liars and with frameworks |
| `bernoulli_*`, `replay_*`, `scale_100k` | 15 | calibrated synthetic and RouterBench replay populations to n = 10<sup>7</sup>; bernoulli and replay twins of the live grids |
| ablations: `budget_*`, `midian_*`, `internals_v2`, `stratify`, `variants_f1`, `churn_*`, `cohort_*`, `fw_k_sensitivity*`, `fw_appendix*` | 19 | v2 / v4: budget, r and δ sweeps, verification and audits, churn, cohort formation, shortlist size k |

[figures.md](figures.md#figures-and-their-grids) lists the grids behind each paper figure.
