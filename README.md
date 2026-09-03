# RTE — Routing To Experts: MIDIAN vs self-contained rivals and ten real agent frameworks

RTE is a clean-room benchmark for one question: **given n agents of unknown and possibly misreported skill, how do you
route each incoming task to the right one, cheaply, when some agents lie?** It compares MIDIAN, a hierarchical
peer-verified routing tree, against 18 self-contained rival mechanisms and the selection primitives of 10 popular
agent frameworks (LangGraph, CrewAI, AutoGen, Magentic-One, Microsoft Agent Framework, OpenAI Agents SDK, Google ADK,
LlamaIndex, smolagents, CAMEL), along three axes: population size n (10² to 10⁷), liar fraction β, and population
skill structure. Every method is charged for every probe, report, message, hop and comparison it makes, so the
comparison is success *and* cost.

The headline result (live LLM population, n = 1000, paired cells, 95% CI): MIDIAN routes correctly 0.64 of the
time and MIDIAN-V 0.66, versus 0.51–0.54 for every one of the ten frameworks, 0.61 for a flat probe scan, 0.56 for a
one-call LLM supervisor and 0.72 for the oracle. The average hides a split: on populations where skill is legible from
a self-description (a minority of big tool-using models) the frameworks sit on the oracle, and on populations where
skill is family-specific they collapse to 0.39 against MIDIAN's 0.75. MIDIAN-V does it with 2 messages and 1 comparison
per task; a framework spends 12 messages, 10 comparisons and at least one supervisor LLM call. Full results, the
honest control that beats MIDIAN (adaptive sequential halving) and the six pre-registered targets (five misses and one split
verdict) are in [`RESULTS_rte.md`](RESULTS_rte.md).

---

## Contents

1. [The problem and the world model](#1-the-problem-and-the-world-model)
2. [Methods](#2-methods)
3. [Accounting: what every method pays for](#3-accounting-what-every-method-pays-for)
4. [Experimental design](#4-experimental-design)
5. [Repository layout](#5-repository-layout)
6. [Running it](#6-running-it)
7. [Results and figures](#7-results-and-figures)
8. [Design principles and how to extend](#8-design-principles-and-how-to-extend)
9. [Documents](#9-documents)
10. [Caveats](#10-caveats)

---

## 1. The problem and the world model

A **World** holds n agents and K task families. Each agent a has a true skill S[a, f] ∈ [0, 1] per family: the
probability it solves a task of that family. S is never shown to any method; only the runner uses it, for the oracle
line and for regret. Methods learn about agents through four **channels**, each of which a method must *declare* it
needs (`needs = {"declared", "probe", "reports", "bus"}`), and a `View` raises `AccessError` on anything undeclared:

| channel | what it gives | who can corrupt it |
|---|---|---|
| `declared` | D[n, K], what agents *say* they can do | liars inflate their own row by +0.4 |
| `probe(a, f)` | run agent a on a fresh instance of family f, get 0/1 | nobody: outcomes are real |
| `reports` | ask peer j what it observed about agent a | a lying j reports 1 for fellow liars, 0 for the top-20% honest (collusion) |
| `bus` | send a message from one agent to another | nobody, but every message is counted |

A fraction β of agents are **liars**, chosen at random or low-skill-first. Lying changes what agents *say*, never what
they *do*, which is why probe-only methods are flat in β and declaration-based methods are not.

Three **backends** implement the same six-member protocol (`true_skill`, `declared`, `execute`, `execute_many`,
`stats`, plus n / K / families), so any method runs unchanged on all of them:

- **llm** (the headline): agents are real LLM calls against a vLLM fleet. Each agent is a (model, per-family handicap,
  tool) *signature* drawn from a 7-model ladder (Qwen2.5 0.5B/1.5B/3B/7B/14B, Gemma-2 2B/9B, `configs/models.yaml`);
  tasks are K = 16 Reasoning Gym families (`rte/backends/families.py`). S is *measured* per signature (200 probes) and
  agents write their own self-descriptions, which the `self_described` declaration channel exposes.
- **replay**: RouterBench's recorded outcomes of 11 real models on 64 real evaluation categories. No GPU; executing
  an agent is a table lookup. Used for the 10⁴–10⁶ scale axis and as a twin of the live sweeps.
- **bernoulli**: synthetic S with five population shapes, calibrated to the measured live S. Used for the 10²–10⁷
  cost curves and as a CPU mirror of every live grid.

Population shapes (`dist`): `specialist` (three strong families per agent), `heavy_tail` (one in ten is a big model),
`bimodal` (20% big-with-tools, 80% small), `correlated` (group-level skill over 4 family groups), `iid_uniform`.

## 2. Methods

Every method is one file in `rte/methods/`, ≤150 lines, discovered by file name, with the interface

```python
class M(Method):
    name = "..."; needs = frozenset({...})
    def __init__(self, **params): ...
    def build(self, view, budget): ...          # one-time setup; probes are capped at budget.total_probes(n, K) = n·K·b
    def fetch(self, task) -> int: ...           # route one task (or a list of agents for route-to-many)
    def observe(self, task, agent, outcome): ... # online update after the outcome is known
```

**MIDIAN** (`midian.py`, pre-registered, byte-identical since the first run). Agents are grouped into random cohorts
of r. At level 0 each member is probed b times per family and its cohort peers report what they saw; the cohort's
estimate of each member is a trimmed mean over reporters (drop ⌊δ(r−1)⌋ from each end), so up to that many liars per
cohort are absorbed. Each cohort's best member per family is its summary; cohorts are grouped again into cohorts of
r, up a tree of depth ⌈log_r n⌉. Routing descends the tree from the root following the best child for the task's
family: 2 messages per level, r comparisons per level. After each outcome the running estimate of the chosen agent is
updated and its path recomputed. Build costs O(n) probes and reports and O(n) messages; a route costs O(log n).

**MIDIAN-V** (`midian(verify=True, cached=True, r=…)`, labeled post-hoc variant). Identical tree, but a promotion is
*verified*: candidates promoted to a parent node are re-probed, budget-exactly (level 0 spends b−1 per cell and the
saved probes go to the promoted candidates), by reporters drawn from sibling subtrees, and the verified value is
written back. The root's pick per family is cached, so a route costs 1 comparison and 2 messages. `r=5` halves the
reports again. Other knobs (`observers`, `b0`, `top`) are documented in the file; `stratify=True` (v2) draws cohorts
stratified by declared family instead of at random.

**MIDIAN-A** (`midian_a.py`, v2). Plain MIDIAN plus audits: 5% of instances are re-probed by the auditor, a reporter
whose report disagrees with the audit twice is excluded from every later estimate, and audits continue online. Costs
1.05× build probes, nothing per task. **MIDIAN-VA** (`midian_va.py`, v2) is MIDIAN-A with V's verified promotion and
cached root pick: the excluded reporters are also removed from the verification. The recommended order of adding
mechanisms is A then V — see RESULTS_rte_v2.md §4. `midian_sh.py` (successive halving inside cohorts), `midian_sha.py`,
`stratify` and `linucb_honest.py` are the v2 negative controls.

**Self-contained rivals** (SPEC §6), grouped by what they read:

| class | methods |
|---|---|
| floor / ceiling | `random`, oracle (runner) |
| declared channel only | `declared_argmax`, `declared_softmax`, `cnp_self_bid` (contract-net bidding), `route_to_k_majority`, `cluster_head_router` (k-means heads), `disrouter_cascade` (cost-ordered cascade), `llm_supervisor` (one LLM call over the top-20 declarations) |
| verified outcomes, centralized | `flat_probe_argmax` (frozen or `online=True`), `flat_nsw_router` (HNSW over probe vectors), `ucb_per_family`, `thompson_per_family`, `trueskill_per_family`, `warm_start_bandit` (declarations as prior, probes as updates), `verify_on_claim`, `sequential_halving` (adaptive: halve the candidate set per family each round) and `sequential_halving(peer_reported=True)`, the fair control that spends the same budget adaptively but learns only through MIDIAN's trimmed report channel |
| verified outcomes, decentralized | `referral_network` (d-regular referral walk), `gossip_reputation_greedy` (EigenTrust-style gossip) |
| MIDIAN family | `midian`, variants above, `midian_llm_descent` (an LLM chooses the child at each level) |

**Frameworks** (SPEC §6A, `rte/methods/frameworks/`). Each `fw_*.py` is 6–11 lines: it hands a shortlist and the task
to a worker running inside the framework's own virtual environment (`workers/*_worker.py`, JSON lines over
`_bridge.py`) and returns the agent the framework's *own* selection primitive chose. The shared adapter
(`_common.FrameworkMethod`) does what a practitioner would: hashed TF-IDF over the agents' self-descriptions picks the
top k = 10, the framework's supervisor (Qwen2.5-7B) picks one, with declared-argmax over the shortlist as a counted
fallback when the framework returns no valid name. `retrieval="midian"` is the labeled variant that gives the framework
MIDIAN-V's verified cohort as its shortlist instead. Interception notes per framework are in `NOTES_*.md`; MetaGPT has
no selection primitive to intercept and AgentScope is an appendix.

## 3. Accounting: what every method pays for

The `Ledger` has one increment site per counter and the world charges the channels itself, so a method cannot
under-report: **probes** (one per agent execution at build), **reports** (one per peer report), **messages** (a query
and its reply are 2; declared methods pay n at build for reading n declarations), **hops** (one per tree level or
graph step), **comparisons** (a flat scan over n candidates is n; a max over r children is r; a cached argmax is 1),
**tasks** (executions at route time). Formulas per method are in `CONTRACT.md`; `scripts/check_methods.py` asserts
them against the counters. Wall-clock is recorded too but mixes cache hits and misses, so cost claims use counts.

Measured at n = 1000 (per task / at build): MIDIAN 6 messages, 30 comparisons / 48k probes, 432k reports, 1k
messages; MIDIAN-V r=5 2 / 1 / 48k, 80k, 1k; flat probe argmax 0 / 1000 / 48k, 0, 0; a framework 12 / 10 + an LLM
call / 0, 0, 1k. Across n = 10² … 10⁷, MIDIAN's per-task comparisons, messages and hops scale as n^0.14 (2⌈log_r n⌉),
MIDIAN-V's as n^0, flat and declared scans as n^1.0.

## 4. Experimental design

- **Cells and pairing.** A cell fixes (backend, n, K, shape, β, liar selection, collusion, declaration channel, lie
  mode, demand, b, Q). Within a cell and seed, every method sees the same agents, the same liars and the same task
  stream, and the k-th probe of any (agent, family) is the same instance for every method (index-seeded), so the
  memoised LLM answers are shared and deltas are paired.
- **Metrics per row.** success, success on the last quarter of the stream, regret vs oracle, misroute-to-liar rate,
  the six counters at build and per task, wall-clock, and the method's own stats (e.g. framework fallbacks).
- **Analysis** (`rte/analyze.py`): per-class tables with 95% percentile-bootstrap CIs over seeds, MIDIAN-vs-rival
  paired deltas with sign tests and a `WITHIN_FLOOR` flag (delta inside MIDIAN's own seed envelope), log-log cost
  exponent fits across n, the six pre-registered target checks, and figures F1–F7.
- **Grids** (`configs/grid.yaml`): `live_core_n100`, `live_f1_n1000` (every rival), `live_extra_n1000` (two more
  shapes), `live_n10k`, `budget_sweep` (b = 1, 3, 10), `midian_internals` (r × δ × verification), `fw_live_n100`,
  `fw_live_n1000`, their `_verified` variants, `fw_k_sensitivity`, `fw_appendix`, `replay_scale`, `bernoulli_scale`,
  and `replay_mirror_*` / `bernoulli_mirror_*` twins of every live grid. Seeds: 5 (algorithmic), 3 (frameworks).
- **Pre-registration.** `TARGETS_rte.md` was committed before the first run. Plain MIDIAN's parameters were never
  changed; every improvement is a labeled variant, and misses are reported as misses.

## 5. Repository layout

```
rte/
  world.py            World, View, channels, lying, paired task streams, index-seeded probes
  ledger.py           the six counters (one increment site each)
  budget.py           probes per (agent, family) -> total build budget
  run.py              grid runner: cells x seeds x methods -> one JSON row per result, resumable, --only sharding
  analyze.py          tables, CIs, paired deltas, exponents, targets, figures
  llm_client.py       vLLM client: endpoints, replica choice, content-hash memo (sharded SQLite, live refresh)
  measure.py          measure S per prompt signature on the fleet
  stable_hash.py      the only seeding primitive
  backends/           bernoulli.py, replay.py, llm.py (+ families.py, population.py, prompts.py, tools.py)
  methods/            one file per method; _est.py (probe->estimate helpers, trimmed peer reports), _decl.py
    frameworks/       _common.py (adapter), _bridge.py, fw_*.py, workers/, NOTES_*.md
configs/              grid.yaml (every experiment), models.yaml (the model ladder)
scripts/              env build (00-03), fleet + replica sbatch, launch_live.sh, run_grid.sbatch, measure.sbatch,
                      check_methods.py, extra_figs.py, mock_openai_server.py, fw_envs/ (per-framework venvs)
tests/                152 tests: every method (needs enforcement, budget, ledger), MIDIAN tree invariants,
                      backends, framework adapters against a mock OpenAI server
```

Data, environments, model weights, the answer memo, logs and results all live under `$RTE_DATA` (not in the repo).

## 6. Running it

**Environment.** `scripts/00_build_env.sh` builds `$RTE_DATA/env/rte` (Python 3.12 venv, vLLM, reasoning-gym,
numpy/pandas/scipy, hnswlib, trueskill); `01_download_weights.py` and `02_download_routerbench.py` fetch the model
snapshots and RouterBench; `scripts/fw_envs/<framework>.sh` builds each framework's isolated venv from
`requirements-frameworks/<framework>.txt`. Everything runs with `PYTHONPATH=~/rte` (or `pip install -e .`).

**CPU-only (no GPU needed).**
```bash
python -m pytest tests -q                                  # 152 tests
python scripts/check_methods.py                            # ledger formulas vs counters
python -m rte.run --grid bernoulli_mirror_live_f1_n1000    # any bernoulli_* or replay_* grid, minutes to an hour
python -m rte.analyze --grid bernoulli_mirror_live_f1_n1000
```

**Live LLM grids** (SLURM, FAS RC): start the fleet, then launch one job per (grid, method, shard):
```bash
sbatch scripts/serve_fleet.sbatch                          # 4 GPUs, all 7 models, registers endpoints in $RTE_DATA/endpoints.d
RTE_REPLICA_MODEL=Qwen/Qwen2.5-7B-Instruct sbatch scripts/serve_replica.sbatch   # extra capacity for a hot model
sbatch scripts/measure.sbatch                              # measure S per signature (once per model ladder)
RTE_GRIDS="live_f1_n1000" RTE_SHARD=dist,beta RTE_SEED_SHARD=1 scripts/launch_live.sh   # ~60 shard jobs per method
python -m rte.analyze --grid live_f1_n1000
python -m rte.analyze --grid live_f1_n1000 --grids live_core_n100,live_f1_n1000,live_n10k,replay_scale,bernoulli_scale --out .../combined
python scripts/extra_figs.py                               # the seven synthesis figures
python -m rte.llm_client compact                           # merge memo shards between stages
```
`launch_live.sh` takes `RTE_GRIDS`, `RTE_ONLY` (methods), `RTE_SHARD` (cell axes to split on), `RTE_SEED_SHARD`,
`RTE_RUN_ARGS` (e.g. `--seeds 4-5`). Rows are per-file and atomic, so any job can be killed and relaunched; the memo
makes re-runs of finished units nearly free.

## 7. Results and figures

`RESULTS_rte.md` is the write-up: the framework headline with CIs and fallback rates, the frameworks given MIDIAN's
shortlist, every rival by β and by population shape, the MIDIAN-vs-halving control, the internals ablation, budget
and scale, cost exponents and break-even, the learning curve, target verdicts, replay, caveats and deviations.
Per-grid machine summaries are `$RTE_DATA/results/<grid>/summary.md`. The figures are in [`figures/`](figures/) (regenerate with `scripts/extra_figs.py`; seed-bootstrap error bars within cells, 300 dpi):

- **H1** headline by population shape, frameworks as a min–max band with fallback rates
- **H2** legibility: Spearman(self-description, true skill) vs framework − MIDIAN
- **H3** consistency vs robustness: success at β=0 vs β=0.5 with colluding low-skill liars
- **H4** cost–quality Pareto with break-even Q
- **H5** cost scaling 10² to 10⁷, plus supervisor latency
- **H6** MIDIAN, MIDIAN-A, MIDIAN-VA (and V, SH, SH+A) vs sequential halving by β and liar selection; replay twin below
- **H7** frameworks given MIDIAN's verified shortlist
- **H8** budget sweep by declaration channel
- **H9** churn: success and cumulative probes across churn events
- **H10** runtime and energy estimate (GPU-seconds and Wh per 1,000 tasks)
- appendix: internals, learning curve, k-sensitivity, replay mirror, fallback table, UCB/Thompson; the 2026-09-02 figures A–G are kept under `figures/v1/`

In one paragraph: the frameworks' only signal is self-description, which overclaims by +0.27 and correlates 0.36 with
true skill, so they sit at 0.5 regardless of β and fall to 0.4 on specialist populations; handing them MIDIAN's
verified cohort lifts every one by 0.04–0.12. Among mechanisms that verify, adaptive sequential halving with a trusted
observer sits on the oracle, and its peer-reported version still beats MIDIAN-V by 0.04 at β ≤ 0.25 in every cell,
but collapses at β = 0.5 with low-skill liars (0.41) where MIDIAN holds 0.60: the tree's per-cohort trimming survives
poisoned reports that early elimination does not. Adding audits (MIDIAN-A) makes MIDIAN flat in β at 5% more probes
(+0.07 at β = 0.5, +0.10 with low-skill liars, nothing lost at β ≤ 0.25); adding verification on top (MIDIAN-VA) loses
nothing at any β and halves the per-task cost (31.6 comparisons and 5 messages instead of 60 and 9). Verification alone
(MIDIAN-V) is +0.02 at β ≤ 0.25 but the most exposed variant at β = 0.5 (−0.03 vs plain), which is why the order is A
then V. (RESULTS_rte_v2.md §4.)

## 8. Design principles and how to extend

The code follows a standing directive: minimum lines, zero duplication, everything swappable by configuration.

- **Add a routing method:** one file in `rte/methods/` with the four-method interface and a `needs` set. It is
  discovered by name, tested by `tests/test_each_method.py` (needs enforcement, budget, ledger) and runnable on all
  three backends with no other change. Shared helpers: `_est.py` (probe-then-estimate, trimmed peer reports, bandits),
  `_decl.py` (declared-channel argmax).
- **Add a framework:** a `fw_<name>.py` of ~10 lines subclassing `FrameworkMethod`, a worker in `workers/` that
  imports the framework and exposes its selection primitive, a `requirements-frameworks/<name>.txt`, and a
  `scripts/fw_envs/<name>.sh`. The adapter, bridge, retrieval and accounting are shared.
- **Swap a model:** edit `configs/models.yaml` (id, parameter count, GPU share, tool gating); nothing else names a
  model except the supervisor constant in `frameworks/_common.py`.
- **Add a task source:** one adapter with `generate(seed) -> entry`, `question(entry) -> str`, `score(answer, entry)
  -> float` in `rte/backends/families.py`; Reasoning Gym is one such adapter.
- **Add a backend:** the six-member protocol in `rte/backends/__init__.py`; `bernoulli.py` is the reference implementation.
- **Add an experiment:** a block in `configs/grid.yaml` (`mirror_of` clones a grid onto another backend).

Every departure from `SPEC.md` is a dated bullet in `DEVIATIONS.md`; the correctness and ledger formulas are in
`CONTRACT.md`; the operational lessons from running ~1000 SLURM jobs against a shared vLLM fleet (locks that hang on
NFS, replica herding, memo sharding, tool-run memoisation) are in `STATUS.md` and `DEVIATIONS.md`.

## 9. Documents

| file | what it is |
|---|---|
| `SPEC.md` | the study specification: world, MIDIAN, rivals, frameworks (§6A), figures, compute plan |
| `TARGETS_rte.md` | the six pre-registered expectations, committed before any run |
| `CONTRACT.md` | frozen interfaces, accounting formulas, correctness checks, simplicity directive |
| `DEVIATIONS.md` | every departure from the spec, dated, with the reason |
| `STATUS.md` | run log and handoff: what exists, what ran, findings, how to finish |
| `RESULTS_rte.md` | the results |
| `rte/methods/frameworks/NOTES_*.md` | how each framework's selection primitive is intercepted, and what broke |

## 10. Caveats

- b = 3 probes give a 4-valued estimate, so at n = 1000 about 116 agents per family tie at 1.0 and any argmax over
  raw probes picks among them blindly; this is what verification and adaptive halving fix.
- True skill is measured once per prompt signature and shared across seeds; the seed CIs cover population and stream
  variation, not the ±0.035 / ±0.065 binomial error of S itself.
- Framework grids use 300 tasks and 3 seeds; CrewAI's rows mostly measure its fallback (it delegates on ~20% of tasks).
- Plain MIDIAN charges one report per probe per peer as the spec reads; MIDIAN-V and peer-reported halving charge one
  per peer (its mean), so their report counts are ~3× lower for the same information.
- Wall-clock columns mix cache hits and misses; use the counters for cost claims.
