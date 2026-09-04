# RESULTS_rte_v2.md — RTE after the v2 work order (FINAL, 2026-09-04)

Status (FINAL, 2026-09-04 15:30): every grid in this document is complete — the MIDIAN-side grids (variants_f1 incl.
MIDIAN-VA, internals_v2, midian_r20, stratify, budget_b10_shapes, live_f1_core_s6_10, midian_v_replication, live_n10k_v2,
churn_n1000), fw_live_n100 / fw_live_n1000 (10 seeds, Q = 1000, fixed adapters; 2,640 rows each), fw_live_n{100,1000}_verified
(2,520 rows each), fw_live_n1000_verified_va (1,320 rows) and the two low-skill-first framework grids. No number in this
document is provisional; every one comes from `rte.analyze` over a completed grid (per-channel tables, one name per arm,
reports per probe, no wall-clock; `python -m rte.analyze --grid <g> --grids ...` prints the v1 and v2 target verdicts).
Pre-registrations: v1 `TARGETS_rte.md` (frozen), v2 `TARGETS_rte_v2.md` (committed before any v2 launch), v3
`TARGETS_rte_v3.md`. Plain MIDIAN's parameters never changed; every mechanism added after the first run is a labeled
variant.

## Figure guide (`figures/`; every error bar is a 95% bootstrap over seeds within a cell, stated in each caption; per-task message and comparison axes include MIDIAN's observe-time path update, commit 3415f03)

| figure | what it shows | read it with |
|---|---|---|
| H1 `H1_headline_by_shape.png` | The headline: success by population shape at n=1000 on the live self-described channel — oracle, MIDIAN-V, MIDIAN, flat online as bars, the ten frameworks as a min–max band with each one's fallback rate annotated. Specialist is where frameworks collapse; bimodal is where they sit on the oracle. | §1 |
| H2 `H2_legibility.png` | Why: x = Spearman(self-description, true skill) per shape (and declared-argmax success as a second x), y = framework − MIDIAN. Not monotone: the specialist failure is retrieval, not ranking. | §2 |
| H3 `H3_consistency_robustness.png` | Every method as a point: success at β=0 (x) vs at β=0.5 with colluding low-skill liars (y). Top-right is good; the diagonal is "immune to lying". | §3, §4 |
| H4 `H4_cost_quality_breakeven.png` | Cost–quality Pareto: total cost (build + Q·per-task) at Q ∈ {10², 10³, 10⁴, 10⁵} vs success, per currency (messages, comparisons, LLM calls); break-even Q marked. | §6 |
| H5 `H5_cost_scaling.png` | Cost vs n from 10² to 10⁷ on the calibrated bernoulli world (comparisons, messages, build probes), plus the frameworks' measured supervisor latency under shared-fleet load. | §6 |
| H6 `H6_midian_vs_halving.png` | MIDIAN, MIDIAN-V, MIDIAN-SH, MIDIAN-A, MIDIAN-SH+A, MIDIAN-VA vs sequential halving (trusted and peer-reported) by β and liar selection; second row = RouterBench replay twin. Read as MIDIAN → +A (flat in β) → +VA (same, half the per-task cost); V alone is the most exposed line at β=0.5; SH and peer halving collapse at β=0.5 low-skill. | §4 |
| H7 `H7_shortlist_lift.png` | Each framework with its own TF-IDF shortlist vs with MIDIAN-V's verified cohort (r=10, r=5); reference line = MIDIAN-V alone. (Provisional until the verified grids close.) | §1b |
| H8 `H8_budget_by_channel.png` | Success vs probe budget b ∈ {1, 3, 10}, split by declaration channel (programmatic = upper bound). | §7 |
| H9 `H9_churn.png` | Success per 100-task block and cumulative probes across churn events (10% / 30% of agents replaced every 200 tasks): MIDIAN's local repair vs halving-rebuild vs halving-stale vs flat online. | §5 |
| H11 `H11_joules_latency.png` | Combined currencies: cumulative joules (LLM call 20.6 J, message 1e-3 J, comparison 1e-8 J) with MIDIAN's break-even crossings, and per-task critical-path latency (1 ms per sequential hop + the supervisor call's measured median). | §6b |
| H10 `H10_runtime_energy.png` | Cumulative GPU-seconds, Wh, messages and comparisons vs tasks routed: probe-based methods start at their build cost and stay flat, frameworks start at zero and grow per task; the crossings are the energy break-even points. Estimate: counts × per-call cost measured on the fleet. | §6b |
| appendix `A_*.png` | internals_v2 (r × δ × collude × liar, MIDIAN vs MIDIAN-A), learning curve (success − oracle per 100 tasks), framework k-sensitivity, replay mirror, fallback table, UCB/Thompson. | Appendices |

The 2026-09-02 figures A–G (5 seeds, pooled error bars) are kept under `figures/v1/` for the v1 write-up only.

## Methods, as run (from DEVIATIONS.md, not SPEC.md)

- **World.** n agents, K = 16 Reasoning Gym families, true skill S[a, f] never visible to a method. Live backend:
  each agent is a (model, per-family handicap, tool) signature from the 7-model ladder (Qwen2.5 0.5B/1.5B/3B/7B/14B,
  Gemma-2 2B/9B). **S is measured once per prompt signature (200 probes; 60 on the ≥9B rungs) and shared by every
  seed**: the seed CIs below cover population draw and task-stream variation only; the binomial error of S itself,
  ±0.035 / ±0.065 per cell at p = 0.5, is a common offset that is not inside any interval.
- **Channels.** `declared` D[n, K]; `probe(a, f)` → a fresh index-seeded instance (the k-th probe of (a, f) is the same
  instance for every method); `reports` (peer j's observation of a); `bus`. A method declares its `needs`; the View
  raises on anything else. Declaration channels: **self_described** (the live one: each agent writes its own
  description and self-ratings) and **programmatic** = clip(S + N(0, 0.05)), an honest declaration no live agent
  produces, reported only as an **upper bound**.
- **Lying, verbatim from `rte/world.py`.** A fraction β of agents are liars (random, or low-skill-first). Declared
  channel, `inflate`: `D[liars] = clip(D[liars] + 0.4, 0, 1)`. Report channel with `collude=True`: a liar j reports 1
  about every fellow liar and 0 about the honest agents in the top 20% of j's observed batch means; otherwise j reports
  what it saw. Liars execute at true skill.
- **Budget.** b = 3 probes per (agent, family) at build (48,000 at n = 1000). With 3 probes an estimate takes 4 values,
  so ≈ 116 agents per family tie at 1.0 at n = 1000; any argmax over raw probes picks among them blindly. This tie
  structure is what verification (MIDIAN-V), adaptive allocation (halving, MIDIAN-SH) and audits (MIDIAN-A) act on.
- **Frameworks.** Ten libraries run through their own selection primitives behind one adapter: hashed TF-IDF over
  self-descriptions picks a top-10 shortlist, the framework's supervisor picks one. Supervisor = **Qwen2.5-7B-Instruct**
  for every framework; the labeled `fw_magentic_one[supervisor=Qwen2.5-14B]` arm is **asymmetric** (14B orchestrator).
  Two accountings per task: *lenient* (a task the framework did not delegate is routed to declared-argmax over the
  shortlist and scored) and *strict* (`success_strict`: such a task scores 0); `fallback_rate` = 1 − picks/tasks.
- **Accounting.** One increment site per counter; the world charges probes/reports/tasks itself. Reports are charged
  one per (peer, member, family, probe) in every arm (v2 0.3; MIDIAN-V's rows before 2026-09-03 carried one report
  per peer and are recomputed by formula in §6). Wall-clock appears only in the supervisor-latency table.
- **Pairing and statistics.** A cell fixes (shape, β, liar selection, channel, …); within a cell and seed every method
  sees the same agents, liars and task stream. Deltas are paired by seed; intervals are 95% percentile bootstrap;
  `WITHIN_FLOOR` = a delta inside plain MIDIAN's own seed envelope in that cell (0.044 on average at n = 1000).

Per-task costs include the observe-time path update (commit 3415f03: after every routed task MIDIAN recomputes the
chosen agent's path, r comparisons and one child→parent message per level, for plain and cached variants alike; the
analyzer adds it to rows written before the fix). MIDIAN-V's saving over MIDIAN is the descent (1 vs 30 comparisons,
2 vs 6 messages), not the update (30 comparisons, 3 messages, common to both): at n = 1000, r = 10 (depth 3) plain
MIDIAN is 60 comparisons / 9 messages per task, MIDIAN-V and MIDIAN-VA 31 / 5, MIDIAN-A 60 / 9, MIDIAN r = 5 (depth 5)
50 / 15, MIDIAN-V r = 5 26 / 7; at n = 10,000 (depth 4) 80 / 12 and 41 / 6; r = 20 120 / 9 and 61 / 5.

Arms, one name each: `midian` (plain, r = 10, δ = 1/3, online), `midian_v` / `midian_v_r5` (verified promotion,
cached root pick; post-hoc 2026-09-02, definition in `rte/methods/midian_v.py`), `midian_sh` (successive halving
inside cohorts), `midian_a` (5% audits, exclusion at two strikes), `midian_sha` (both), `midian_stratified`,
`flat_probe_argmax_frozen` / `flat_probe_argmax_online`, `sequential_halving` (trusted observer) /
`sequential_halving_peer` (MIDIAN's report channel) / `..._rebuild` / `..._stale` (churn modes), `linucb_honest`.

---

## 1. Headline: the ten frameworks vs MIDIAN (H1)

Self-described channel, 10 seeds, Q = 1000 tasks, the fixed adapters (private CrewAI store and delegating manager,
Magentic-One robust ledger with a labeled 14B-orchestrator arm, ADK transfer capture). **FINAL: fw_live_n100 and
fw_live_n1000 are complete (2,640 rows each) — every arm on all 120 cell × seed pairs.** Paired deltas are over the 120
pairs (95% CI = bootstrap over pairs); the `n` column is the pair count.

| method | n=100 [95% CI] | n=1000 [95% CI] | n | Δ vs MIDIAN, n=1000 (cells won) | strict success | fallback rate (failure / fallback) |
|---|---|---|---|---|---|---|
| oracle | 0.716 [0.696, 0.736] | 0.723 [0.702, 0.744] | 120 | +0.069 (120/120) | — | — |
| midian_va | 0.653 [0.637, 0.670] | 0.675 [0.655, 0.695] | 120 | +0.021 [+0.015, +0.026] (86/120) | — | — |
| midian_a | 0.656 [0.638, 0.673] | 0.668 [0.651, 0.686] | 120 | +0.014 (83/120) | — | — |
| midian_v_r5 | 0.651 [0.632, 0.669] | 0.665 [0.643, 0.685] | 120 | +0.011 (84/120) | — | — |
| midian_v | 0.642 [0.623, 0.662] | 0.663 [0.642, 0.683] | 120 | +0.009 (79/120) | — | — |
| **midian** | 0.639 [0.620, 0.658] | 0.654 [0.636, 0.672] | 120 | 0 | — | — |
| midian r=5 | 0.636 [0.618, 0.654] | 0.642 [0.625, 0.660] | 120 | -0.012 (40/120) | — | — |
| flat_probe_argmax_frozen | 0.588 [0.572, 0.603] | 0.612 [0.595, 0.629] | 120 | -0.042 (35/120) | — | — |
| declared_argmax | 0.570 [0.561, 0.578] | 0.575 [0.567, 0.584] | 120 | -0.079 (27/120) | — | — |
| llm_supervisor | 0.568 [0.561, 0.574] | 0.568 [0.556, 0.578] | 120 | -0.086 (53/120) | 0.000 | 0% |
| Magentic-One (7B orchestrator) | 0.553 [0.546, 0.559] | 0.546 [0.530, 0.561] | 120 | -0.108 (53/120) | 0.172 | 60.4% (40.3 / 20.1) |
| Google ADK | 0.540 [0.532, 0.549] | 0.542 [0.527, 0.558] | 120 | -0.112 (53/120) | 0.380 | 28.5% (10.6 / 17.9) |
| Magentic-One, **14B orchestrator** (asymmetric arm) | 0.540 [0.532, 0.549] | 0.533 [0.514, 0.551] | 120 | -0.121 (53/120) | 0.287 | 43.2% (2.3 / 40.9) |
| LangGraph | 0.530 [0.521, 0.539] | 0.531 [0.512, 0.550] | 120 | -0.123 (53/120) | 0.427 | 18.7% (0.0 / 18.7) |
| LlamaIndex | 0.522 [0.512, 0.531] | 0.531 [0.511, 0.550] | 120 | -0.123 (53/120) | 0.443 | 16.1% (0.0 / 16.1) |
| OpenAI Agents SDK | 0.527 [0.517, 0.536] | 0.528 [0.509, 0.547] | 120 | -0.125 (53/120) | 0.444 | 15.3% (0.0 / 15.3) |
| AutoGen | 0.528 [0.518, 0.538] | 0.528 [0.508, 0.548] | 120 | -0.126 (53/120) | 0.497 | 6.0% (0.0 / 6.0) |
| CrewAI (fixed) | 0.529 [0.519, 0.539] | 0.528 [0.508, 0.547] | 120 | -0.126 (53/120) | 0.463 | 11.8% (0.0 / 11.7) |
| CAMEL Workforce | 0.534 [0.524, 0.543] | 0.527 [0.506, 0.547] | 120 | -0.127 (53/120) | 0.429 | 17.7% (0.0 / 17.7) |
| smolagents | 0.529 [0.519, 0.539] | 0.526 [0.506, 0.546] | 120 | -0.128 (53/120) | 0.462 | 6.9% (0 / 0.7; rest bad names) |
| MAF | 0.530 [0.519, 0.541] | 0.524 [0.503, 0.545] | 120 | -0.130 (53/120) | 0.501 | 2.8% (0.0 / 2.8) |
| random | 0.315 [0.300, 0.330] | 0.314 [0.300, 0.329] | 120 | -0.340 (0/120) | — | — |

All rows final: fw_live_n100 and fw_live_n1000 complete (2,640 rows each) on 2026-09-04.
Strict success scores every task the framework did not delegate (or named nobody) as 0; "failure" = the framework
answered the task itself, "fallback" = it returned no usable name and the adapter routed declared-argmax over its
shortlist. With Q = 1000 the CrewAI rows are a measurement of CrewAI (12% fallback, none of it failures): the 2026-09-02
84% was the corrupted shared store. Magentic-One's 40% failure rate is real (its orchestrator solves the task in the
planning stage and declares it satisfied with no speaker); the 14B orchestrator converts almost all of those into
fallbacks (2.3% failures, 41% fallbacks) and is 0.013 below the 7B arm on lenient success (paired: strictly below in
the 4 specialist cells, identical in the other 8 — on bimodal and heavy_tail both orchestrators pick the same agent from
the same shortlist, so the whole difference is the specialist population) — so under strict accounting Magentic-One is the worst framework (0.17–0.29) and under lenient accounting the
7B arm is the best (0.546).

The average hides the split that decides the paper (figure H1; n = 1000, all 120 cell × seed pairs per arm):

| n=1000, by shape | oracle | midian_v | midian | midian_va | flat (frozen) | frameworks mean (min–max) | pairs framework > MIDIAN |
|---|---|---|---|---|---|---|---|
| specialist (3 strong families per agent) | 0.861 | 0.795 | 0.778 | 0.802 | 0.728 | **0.390** (0.367–0.434) | 0 / 400 |
| heavy_tail (1 in 10 is a big model) | 0.734 | 0.666 | 0.643 | 0.679 | 0.586 | 0.630 (all ten identical) | 130 / 400 |
| bimodal (20% big-with-tools, 80% small) | 0.573 | 0.528 | 0.540 | 0.544 | 0.521 | **0.573** (all ten identical) | 400 / 400 |

(n = 100: specialist 0.845 / 0.765 / 0.762 / 0.769 / 0.694 / frameworks 0.479, 0 / 400; heavy_tail 0.729 / 0.628 / 0.627 /
0.645 / 0.543 / 0.557, 29 / 400; bimodal 0.573 / 0.534 / 0.528 / 0.546 / 0.526 / 0.561, 320 / 400.) Where skill is legible from a
description (bimodal: "is this one of the 20% big models with a tool?") every framework sits on the oracle and beats
MIDIAN in every cell; where skill is family-specific (specialist) the frameworks collapse to 0.39 against 0.78 and lose
every pair by 0.34–0.41; on heavy_tail they trail by 0.01. Frameworks are flat in β (0.530–0.532 from β = 0 to 0.5, table columns by β in
`results/fw_live_n1000/summary.md`): self-descriptions overclaim and carry little family-level signal, so lying about
that channel changes little. At Q = 1000 and 10 seeds the picture is the 2026-09-02 one within 0.02 on every row.


**Low-skill-first collusion at β = 0.5, frameworks included (fw_live_n{100,1000}_lowskill, 3 shapes × 10 seeds, Q = 1000;
complete 2026-09-04 05:10).** The liars are the weakest half of the population and they collude on their reports:

| arm | n = 1000 | n = 100 | strict success (n = 1000) | fallback rate |
|---|---|---|---|---|
| oracle | 0.723 | 0.716 | | |
| **MIDIAN-VA** | **0.679** | **0.657** | | |
| MIDIAN-A | 0.665 | 0.656 | | |
| flat probe argmax (frozen) | 0.612 | 0.588 | | |
| MIDIAN | 0.569 | 0.537 | | |
| LLM supervisor (7B, whole roster) | 0.552 | 0.552 | | 0.000 |
| Magentic-One (7B) | 0.538 | 0.537 | 0.217 | 0.507 |
| Google ADK | 0.536 | 0.534 | 0.413 | 0.220 |
| MIDIAN-V | 0.531 | 0.528 | | |
| CrewAI / smolagents / LangGraph / AutoGen / OpenAI Agents / CAMEL / MAF / LlamaIndex | 0.520–0.528 | 0.519–0.532 | 0.41–0.53 | 0.00–0.23 |
| declared argmax | 0.515 | 0.520 | | |
| random | 0.315 | 0.315 | | |

Paired on identical cells (30 each): MIDIAN-VA − AutoGen **+0.154** [+0.082, +0.225], − Magentic-One +0.141, − Google
ADK +0.144 at n = 1000 (+0.134 / +0.120 / +0.123 at n = 100); even a frozen flat probe scan beats AutoGen by +0.086
here. The frameworks are unmoved by collusion (they never read the report channel), so this is the same 0.52–0.54 as
§1; what collapses is anything that trusts reports without audits — plain MIDIAN to 0.57, MIDIAN-V to 0.53 — and what
holds is A / VA. The 14B-supervisor Magentic-One arm is in the grid (`fw_magentic_one[supervisor=...14B]`) and sits with
the 7B arm.

## 1b. Frameworks given MIDIAN's verified shortlist (H7) — and MIDIAN-VA's audited one (T3-19)

Frameworks given MIDIAN-V's verified leaf cohort as their shortlist (`retrieval: midian`, r = 10 or r = 5 candidates)
instead of the TF-IDF top-10 of self-descriptions; everything else unchanged (same supervisor, prompts, accounting).
Grids fw_live_n{100,1000}_verified (3 shapes × 4 β × 10 seeds, Q = 1000; 2,520 rows each, complete) and
fw_live_n1000_verified_va (the same ten frameworks given MIDIAN-VA's cohort, r = 10; 1,320 rows, complete), paired
against the plain arm of fw_live_n{100,1000} on identical cell × seed pairs (120 per framework; 95% CI = bootstrap over
pairs).

| framework | plain | MIDIAN-V shortlist r=10 | r=5 | lift r=10 [95% CI] | lift r=5 | MIDIAN-VA shortlist r=10 | lift VA [95% CI] | VA − V (r=10) [95% CI] |
|---|---|---|---|---|---|---|---|---|
| Magentic-One (7B orchestrator) | 0.546 | 0.633 | 0.623 | +0.087 [+0.069, +0.104] | +0.077 | 0.622 | +0.076 [+0.059, +0.092] | -0.011 [-0.017, -0.005] |
| AutoGen | 0.528 | 0.609 | 0.594 | +0.081 [+0.062, +0.101] | +0.065 | 0.612 | +0.084 [+0.066, +0.102] | +0.003 [-0.005, +0.012] |
| smolagents | 0.526 | 0.595 | 0.608 | +0.069 [+0.053, +0.087] | +0.081 | 0.600 | +0.073 [+0.058, +0.090] | +0.004 [-0.001, +0.010] |
| Google ADK | 0.542 | 0.606 | 0.608 | +0.064 [+0.049, +0.080] | +0.066 | 0.605 | +0.063 [+0.048, +0.079] | -0.001 [-0.008, +0.006] |
| CAMEL Workforce | 0.527 | 0.577 | 0.589 | +0.050 [+0.033, +0.067] | +0.063 | 0.570 | +0.044 [+0.029, +0.058] | -0.006 [-0.015, +0.002] |
| LangGraph | 0.531 | 0.581 | 0.585 | +0.049 [+0.032, +0.067] | +0.054 | 0.586 | +0.054 [+0.038, +0.072] | +0.005 [-0.005, +0.015] |
| CrewAI (fixed) | 0.528 | 0.577 | 0.587 | +0.049 [+0.030, +0.067] | +0.059 | 0.573 | +0.045 [+0.028, +0.062] | -0.004 [-0.011, +0.003] |
| OpenAI Agents SDK | 0.528 | 0.566 | 0.568 | +0.038 [+0.026, +0.051] | +0.039 | 0.569 | +0.040 [+0.027, +0.054] | +0.003 [-0.004, +0.009] |
| MAF | 0.524 | 0.553 | 0.580 | +0.030 [+0.013, +0.046] | +0.056 | 0.551 | +0.028 [+0.011, +0.045] | -0.002 [-0.008, +0.004] |
| LlamaIndex | 0.531 | 0.541 | 0.528 | +0.010 [-0.014, +0.034] | -0.003 | 0.539 | +0.008 [-0.016, +0.031] | -0.002 [-0.011, +0.007] |
| MIDIAN-V / MIDIAN / MIDIAN-VA / oracle | 0.663 / 0.654 / 0.675 / 0.723 | | | | | | | |

n = 100 (V shortlist only):

| framework | plain | MIDIAN-V shortlist r=10 | r=5 | lift r=10 [95% CI] | lift r=5 |
|---|---|---|---|---|---|
| AutoGen | 0.528 | 0.589 | 0.589 | +0.060 [+0.046, +0.075] | +0.061 |
| Magentic-One (7B orchestrator) | 0.553 | 0.611 | 0.609 | +0.058 [+0.045, +0.072] | +0.056 |
| smolagents | 0.529 | 0.583 | 0.582 | +0.053 [+0.042, +0.066] | +0.053 |
| Google ADK | 0.540 | 0.588 | 0.590 | +0.048 [+0.036, +0.062] | +0.050 |
| CAMEL Workforce | 0.534 | 0.579 | 0.582 | +0.045 [+0.032, +0.058] | +0.048 |
| OpenAI Agents SDK | 0.527 | 0.565 | 0.566 | +0.038 [+0.025, +0.051] | +0.040 |
| LangGraph | 0.530 | 0.567 | 0.566 | +0.037 [+0.023, +0.050] | +0.037 |
| CrewAI (fixed) | 0.529 | 0.566 | 0.581 | +0.037 [+0.025, +0.048] | +0.052 |
| MAF | 0.530 | 0.555 | 0.575 | +0.025 [+0.011, +0.039] | +0.045 |
| LlamaIndex | 0.522 | 0.531 | 0.521 | +0.009 [-0.006, +0.024] | -0.001 |

Nine of ten frameworks gain +0.03 to +0.09 from a verified shortlist at n = 1000 and +0.03 to +0.06 at n = 100 (the
phase-1 range +0.04 … +0.12 at Q = 300 / 3 seeds is reproduced at 10 seeds), and none reaches MIDIAN-V itself (0.663 /
0.642): given a cohort of ten verified candidates, the supervisor's own choice among them still costs 0.03–0.12.
LlamaIndex's selector ignores the shortlist order and gains nothing (CI covers 0 at both n). r = 5 vs r = 10: within
±0.03 of each other for every framework.

**MIDIAN-VA's shortlist (T3-19).** Averaged over the ten frameworks the VA-shortlisted arm is 0.593 / 0.575 / 0.579 /
0.583 at β = 0 / 0.1 / 0.25 / 0.5 against the V-shortlisted 0.596 / 0.592 / 0.585 / 0.562 (plain 0.531 at every β):
VA − V is −0.003 / −0.017 / −0.006 at β ≤ 0.25 and **+0.021** at β = 0.5, where V's unaudited cohort starts to admit
liars (MIDIAN-V itself drops 0.685 → 0.607 across β while MIDIAN-VA holds 0.684 → 0.680). Per framework at β = 0.5:
LangGraph +0.035, AutoGen +0.033, smolagents +0.030, OpenAI Agents +0.027, LlamaIndex +0.022, ADK +0.020, CrewAI +0.015,
CAMEL +0.011, MAF +0.010, Magentic-One +0.007. **T3-19 SPLIT**: the β ≤ 0.25 clause holds (every framework mean within
±0.02; Magentic-One at β = 0.1 is −0.025), the β = 0.5 clause (≥ +0.03) holds for three of ten frameworks and misses on
the mean (+0.021). The audited cohort protects the supervisor less than it protects MIDIAN-VA's own pick: the
frameworks re-rank the ten candidates by their self-descriptions, which liars inflate, so part of what the audits
remove from the cohort the supervisor puts back.
## 2. Legibility (H2)

Self-descriptions are *most* correlated with true skill exactly where the frameworks fail, so legibility of the
ranking is not what limits them; the retrieval step is. Spearman(D_self_described, S) over all (agent, family) cells,
and the mean within-family rank correlation, from `scripts/legibility.py` (10 seeds per shape, n = 1000; n = 100 within
±0.02):

| shape | Spearman(D, S), all cells | within-family Spearman | declared_argmax (self-described, n=1000) | frameworks − MIDIAN (v1, 60 cells) |
|---|---|---|---|---|
| specialist | 0.443 ± 0.005 | 0.573 ± 0.006 | see §3 | −0.361 (0 / 200 cells) |
| heavy_tail | 0.194 ± 0.021 | 0.464 ± 0.018 | | −0.017 (83 / 200) |
| bimodal | 0.295 ± 0.020 | 1.000 ± 0.000 | | +0.039 (200 / 200) |

Bimodal has only two skill levels, so the within-family ranking is perfect and a description-driven shortlist cannot miss;
specialist has the highest overall correlation but the frameworks collapse there, because the TF-IDF shortlist has to
match a task's text to the right *specialty* paragraph before the supervisor ever ranks anyone. The framework-delta axis (figure H2, final): frameworks − MIDIAN at n = 1000 is −0.388 on specialist, −0.013 on
heavy_tail, +0.033 on bimodal (n = 100: −0.283 / −0.070 / +0.034) — monotone in *within-family* Spearman (0.573 → 0.464 → 1.000
is not monotone, but bimodal's two-level skill makes any description-based shortlist exact), not in the overall correlation.

## 3. Every rival, by class, on the self-described channel (live_f1_n1000; programmatic in Appendix A)

n = 1000, 48 cells × 5 seeds; the 24 self-described cells for declared-channel readers. Paired Δ = rival − MIDIAN.

| class | method | self-described success | Δ vs MIDIAN | rival wins | programmatic (upper bound) | Δ |
|---|---|---|---|---|---|---|
| ceiling | oracle | 0.723 | +0.081 | — | 0.723 | +0.081 |
| verified, central | sequential_halving (trusted) | 0.722 | +0.080 | 240/240 | 0.722 | +0.080 |
| verified, central | sequential_halving_peer | 0.677 | +0.035 | — | 0.677 | +0.035 |
| verified, central | warm_start_bandit | 0.678 | +0.036 | 93/120 | 0.685 | +0.043 |
| verified, central | verify_on_claim | 0.647 | +0.005 | 76/119 | 0.672 | +0.030 |
| verified, central | flat_probe_argmax_online | 0.667 | +0.025 | — | 0.667 | +0.025 |
| midian | midian_v | 0.654 | +0.012 | — | 0.654 | +0.012 |
| midian | **midian** | 0.642 | 0 | — | 0.642 | 0 |
| midian | midian_llm_descent | 0.628 | −0.014 | 28/117 | 0.627 | −0.015 |
| verified, central | flat_probe_argmax_frozen | 0.619 | −0.023 | — | 0.619 | −0.023 |
| declared | cluster_head_router | 0.567 | −0.075 | 36/120 | 0.654 | +0.012 |
| declared | llm_supervisor | 0.567 | −0.076 | 53/119 | 0.584 | −0.058 |
| declared | cnp_self_bid | 0.560 | −0.082 | 26/118 | 0.652 | +0.010 |
| declared | route_to_k_majority | 0.557 | −0.085 | 29/120 | 0.659 | +0.017 |
| declared | declared_argmax | 0.549 | −0.094 | 23/120 | 0.656 | +0.014 |
| verified, central | trueskill / thompson / ucb | 0.623 / 0.604 / 0.597 | −0.019 / −0.039 / −0.045 | — | same | same |
| declared | declared_softmax | 0.473 | −0.169 | 0/120 | 0.595 | −0.047 |
| verified, decentral | gossip_reputation_greedy | 0.508 | −0.134 | — | 0.508 | −0.134 |
| declared | disrouter_cascade | 0.369 | −0.273 | 0/120 | 0.619 | −0.023 |
| verified, decentral | referral_network | 0.409 | −0.233 | — | 0.409 | −0.233 |
| floor | random | 0.311 | −0.331 | 0/240 | 0.311 | −0.331 |

The pooled 2026-09-02 tables had flattered every pure-declaration rival by ≈ 0.05: on the live channel they all lose
to MIDIAN by 0.08–0.27; on the programmatic upper bound they tie or edge it. By β on the self-described channel the
declared class drops 0.05–0.22 from β = 0 to 0.5 (declared_argmax 0.594 → 0.531, disrouter 0.524 → 0.340) while every
probe-only method moves ≤ 0.01. **Headline arms at 10 seeds (live_f1_n1000 seeds 1–5 + live_f1_core_s6_10 seeds 6–10; 471 paired units, both channels;
231 on the self-described channel).** Doubling the seeds moves no mean by more than 0.006 and halves nothing important:
the ordering and every sign are the 5-seed ones.

| arm | β=0 | β=0.1 | β=0.25 | β=0.5 | mean [95% CI] | sd (units) | Δ vs MIDIAN, self-described (cells won) |
|---|---|---|---|---|---|---|---|
| oracle | 0.723 | 0.719 | 0.719 | 0.719 | 0.720 [0.710, 0.731] | 0.118 | +0.078 (231/231) |
| sequential_halving (trusted) | 0.722 | 0.719 | 0.718 | 0.718 | 0.720 [0.709, 0.730] | 0.117 | +0.078 (231/231) |
| warm_start_bandit | 0.694 | 0.681 | 0.673 | 0.669 | 0.679 [0.671, 0.688] | 0.099 | +0.035 (181/231) |
| sequential_halving_peer | 0.722 | 0.718 | 0.714 | 0.536 | 0.673 [0.658, 0.688] | 0.166 | +0.031 (202/231; −0.057 at β=0.5) |
| flat_probe_argmax_online | 0.667 | 0.664 | 0.664 | 0.664 | 0.665 [0.656, 0.674] | 0.097 | +0.023 (167/227) |
| verify_on_claim | 0.689 | 0.657 | 0.645 | 0.646 | 0.660 [0.651, 0.669] | 0.100 | +0.010 (150/229) |
| midian_v_r5 | 0.691 | 0.687 | 0.677 | 0.564 | 0.655 [0.644, 0.667] | 0.127 | +0.013 (171/231) |
| midian_v | 0.682 | 0.681 | 0.672 | 0.567 | 0.651 [0.639, 0.662] | 0.128 | +0.009 (161/227) |
| **midian** | 0.667 | 0.661 | 0.644 | 0.594 | 0.642 [0.632, 0.652] | 0.108 | 0 |
| midian r=5 | 0.663 | 0.655 | 0.636 | 0.560 | 0.629 [0.619, 0.639] | 0.112 | −0.013 (69/225) |
| flat_probe_argmax_frozen | 0.615 | 0.612 | 0.610 | 0.608 | 0.611 [0.603, 0.621] | 0.099 | −0.030 (77/228) |
| declared_argmax | 0.659 | 0.598 | 0.587 | 0.583 | 0.607 [0.598, 0.617] | 0.106 | −0.080 (54/231; 0.602 → 0.539 self-described) |
| llm_supervisor | 0.581 | 0.574 | 0.573 | 0.569 | 0.574 [0.569, 0.580] | 0.056 | −0.071 (111/230) |
| random | 0.315 | 0.312 | 0.312 | 0.312 | 0.312 [0.305, 0.320] | 0.084 | −0.329 (0/231) |

The unit-level sd (0.10–0.17) is population and stream variation across cells; within a cell the seed envelope of
MIDIAN is 0.070 (n = 1000), which is the WITHIN_FLOOR threshold used in the verdicts.

The v2 variants on the same cells (variants_f1, self-described, 10 seeds, 240 units each), success at β = 0 / 0.1 / 0.25 / 0.5:
`midian_a` 0.668 / 0.668 / 0.667 / 0.667 (flat in β), `midian_sha` 0.670 / 0.670 / 0.670 / 0.663, `midian_sh` 0.670 / 0.670 /
0.648 / 0.474, `linucb_honest` 0.642 / 0.641 / 0.642 / 0.642 (below flat_probe_argmax_online 0.667 at every β, −0.025
paired, 19/239 cells); UCB/Thompson are in Appendix C ("16k arms, Q = 1,000, under-explored by construction").

## 4. MIDIAN → MIDIAN-A → MIDIAN-VA, against sequential halving (H6)

Two mechanisms were added to plain MIDIAN as labeled variants: **A** (5% instance audits of reports, two strikes exclude a
reporter) and **V** (verified promotion + cached root pick). The order in which they are added is the story: audits
first is monotone at every β, verification first is not. All deltas below are paired on identical (cell, seed) units of
variants_f1 (3 shapes × 4 β × 2 liar selections × 10 seeds = 240 units, n = 1000, self-described channel, 60 pairs per β;
mean [95% bootstrap CI over pairs]).

**Step 0 — plain MIDIAN against halving.** `sequential_halving_peer` spends MIDIAN's budget adaptively but learns only through MIDIAN's trimmed report channel.
240 paired units (live_f1_n1000, both channels; the halving arms do not read declarations).

| midian_v − sequential_halving_peer | β=0 | β=0.1 | β=0.25 | β=0.5 |
|---|---|---|---|---|
| random liars | −0.040 | −0.038 | −0.046 | −0.064 |
| low-skill-first liars | −0.040 | −0.037 | −0.037 | **+0.123** |
| wins / non-tied cells | 0/60 | 0/60 | 0/60 | 26/58 |

Plain MIDIAN: −0.056 / −0.060 / −0.072 at β ≤ 0.25 (0/60, p ≈ 2e-18) and +0.049 at β = 0.5 (32/60). Absolute at
β = 0.5 with low-skill liars: MIDIAN 0.57, midian_v 0.53, halving_peer 0.41. Trusted-observer halving sits on the
oracle at every β (−0.001 … 0.000), so its edge is adaptive allocation, not information. On RouterBench replay the
ordering is the same: peer halving +0.04 at β ≤ 0.25, plain MIDIAN +0.10 at β = 0.5.

**Step 1 — audits (MIDIAN-A).** midian_a − midian by β = 0 / 0.1 / 0.25 / 0.5: **+0.000 / +0.004 / +0.019 / +0.069**
(+0.023 [+0.018, +0.028] overall, 175/214 non-tied cells); at β = 0.5 with colluding low-skill-first liars +0.097, 30/30
cells, and +0.127 over peer halving. The audits exclude the colluding reporters (at n = 1000 ≈ 99% of liars and no
honest reporter, bernoulli check) and the estimate stops moving with β: MIDIAN-A is 0.668 / 0.668 / 0.667 / 0.667. Cost:
1.050× build probes (50,382; audits are probes; online audits are reports), per-task cost unchanged (60 comparisons,
9 messages). Nothing is given up anywhere. **V2-2 HIT.**

**Step 2 — verification on top (MIDIAN-VA).** midian_va − midian_a by β: **+0.016 / +0.003 / −0.000 / +0.013** (+0.008
[+0.005, +0.011] overall; +0.014 [+0.005, +0.023] at β = 0.5 low-skill-first, 30 pairs). Again nothing is given up, and
the per-task cost halves: 31.6 comparisons and 5.06 messages per task instead of 60 and 9 (the descent is one
comparison at a verified, cached root; the observe-time path update is the same 30 / 3), build probes 0.98× A's
(49,420), reports 430,560, hops 0. VA is the best MIDIAN variant on average (0.675 vs A 0.667, V 0.653, plain 0.645),
above V and A on every shape (bimodal 0.545 / heavy_tail 0.679 / specialist 0.802 vs V's 0.521 / 0.648 / 0.791 and A's
0.552 / 0.660 / 0.790), and flat under collusion (0.680 at β = 0.5, 0.679 with low-skill-first liars). Energy and
latency sit on V's curve (§6b: 285 GPU-s build, 20.0 J/task). vs plain MIDIAN: +0.031 [+0.025, +0.036] over all 240
pairs (+0.016 / +0.007 / +0.019 / +0.082); vs peer halving −0.039 / −0.051 / −0.051 / +0.140 (+0.278 at β = 0.5
low-skill-first).

**Why not verification first.** midian_v − midian: +0.016 / +0.020 / +0.028 at β ≤ 0.25 and **−0.029** at β = 0.5
(−0.038 with low-skill-first liars: V is 0.531 where plain MIDIAN is 0.569 and peer halving 0.402). V is cheaper and
better while liars are few and worse than plain MIDIAN once they coordinate, because the verified promotion trusts the
same corrupted reports. Adding audits then repairs it — midian_va − midian_v **+0.000 / −0.013** [−0.017, −0.010] /
**−0.009** [−0.014, −0.004] / **+0.110** [+0.089, +0.134] (+0.148 at β = 0.5 low-skill-first) — but gives back 0.009–0.013
of V's low-β edge. Told in this order the second step is a regression fix. The honest reading of both orders together:
once audits are in place, V's accuracy edge is mostly absorbed (VA − A is +0.003 / −0.000 at β = 0.1 / 0.25); what
verification reliably buys on top of a robust estimator is the halved per-task cost, plus +0.016 at β = 0 and +0.013 at
β = 0.5. At β = 0 VA equals V exactly (no reporter is ever struck). Where VA is below V (β = 0.1, 0.25) the code-path
hypothesis — not a measured decomposition — is that a struck reporter's probes are also removed from V's verified
promotion (`_verify(..., exclude)`), which shrinks the verification set. **V2-11 WITHIN_FLOOR**: (i) VA ≥ max(V, A) − 0.01
fails at β = 0.1 by 0.003 beyond the tolerance (−0.013 on variants_f1; −0.010 / −0.013 / −0.017 / +0.001 by β in the
final merged evaluation with the replication grid), inside MIDIAN's 0.074 seed envelope; (ii) HIT (+0.014 vs A at β = 0.5
low-skill-first); (iii) HIT (1.033× V's build probes, per-task cost V's + 2%).

**All variants on the same 240 units, with the negative results (SH, SH+A, LinUCB).**

| success | β=0 | β=0.1 | β=0.25 | β=0.5 all | β=0.5 random liars | β=0.5 low-skill liars | sd (units) |
|---|---|---|---|---|---|---|---|
| oracle | 0.723 | 0.723 | 0.723 | 0.723 | 0.723 | 0.723 | 0.118 |
| midian_va | 0.684 | 0.671 | 0.667 | 0.680 | 0.680 | 0.679 | 0.107 |
| sequential_halving_peer | 0.722 | 0.722 | 0.718 | 0.540 | 0.678 | 0.402 | 0.165 |
| midian_sha (SH+A) | 0.670 | 0.670 | 0.670 | 0.663 | 0.663 | 0.663 | 0.098 |
| midian_a | 0.668 | 0.668 | 0.667 | 0.667 | 0.668 | 0.666 | 0.099 |
| flat_probe_argmax_online | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 | 0.097 |
| midian_v | 0.684 | 0.684 | 0.676 | 0.569 | 0.608 | 0.531 | 0.130 |
| midian | 0.668 | 0.664 | 0.648 | 0.598 | 0.627 | 0.569 | 0.109 |
| midian_sh | 0.670 | 0.670 | 0.648 | 0.474 | 0.522 | 0.426 | 0.137 |
| linucb_honest | 0.642 | 0.641 | 0.642 | 0.642 | 0.642 | 0.642 | 0.078 |

In-cohort successive halving (SH) buys nothing at β ≤ 0.25 and loses badly under collusion. midian_sh − halving_peer
**−0.060** [−0.068, −0.052], 18/240 (−0.052 / −0.052 / −0.070 at β ≤ 0.25, every shape: bimodal −0.035, heavy_tail −0.066,
specialist −0.076): halving inside a cohort of 10 cannot reproduce halving over 1,000 — the cohort's best member is
found, but the cohort was random. midian_sh − midian: +0.002 / +0.006 / −0.000 at β ≤ 0.25 and **−0.124** at β = 0.5
(0/30 at low-skill collusion, −0.143): early elimination on poisoned reports. midian_sha − midian_a: +0.003 / +0.002 /
+0.003 / −0.004 — the audits repair SH's collapse (+0.189 over SH at β = 0.5) but SH adds nothing on top of A. SH's
costs are plain MIDIAN's (48,000 probes, 432,000 reports, 9 messages and 60 comparisons per task — 6 / 30 for the descent
plus 3 / 30 for the observe-time path update — exponent n^0.14 by construction). Pre-registered: **V2-1 MISS** (both
halves), **V2-3 MISS** by −0.015 at β = 0 only (−0.000 / −0.008 / −0.004 elsewhere; inside MIDIAN's 0.070 seed
envelope), **V2-5 MISS** (LinUCB is 0.025 below flat_online, not between it and warm-start; it is flat in β, +0.000).

**MIDIAN-V replication on ten fresh seeds (midian_v_replication: seeds 11–20, n ∈ {100, 1000}, 3 shapes, β ∈ {0, 0.25,
0.5}, both channels; 360 units per arm).** This is the confirmatory run for the post-hoc variant, under its definition of
record in `midian_v.py`: reports are charged and aggregated per probe (the 2026-09-02 rows aggregated per-peer means, a
different corrupted value under collusion — DEVIATIONS 2026-09-03). midian_v − midian, paired: n = 1000 **+0.021** at
β = 0 and **+0.029** at β = 0.25 (sd across units 0.020 / 0.027, 60 pairs each), **−0.017** at β = 0.5 collude; n = 100
+0.007 / +0.026 / 0.000. By shape at n = 1000: heavy_tail +0.037 / +0.048, specialist +0.025 / +0.036, bimodal +0.001 /
+0.003 at β = 0 / 0.25. Against the controls on the same fresh seeds (n = 1000): midian_v − flat_online +0.021 / +0.017 /
−0.060; midian_v − halving_peer −0.043 / −0.044 / −0.083 (3/145 cells); halving_peer − midian −0.067 / −0.077 / −0.069.
Absolute (n = 1000, β = 0 / 0.25 / 0.5): oracle 0.714 / 0.715 / 0.714, halving_peer 0.744 / 0.742 / 0.703, midian_v
0.675 / 0.670 / 0.594, flat_online 0.654 / 0.653 / 0.654, midian 0.653 / 0.641 / 0.611. **V2-8 HIT** (+0.02 ± 0.02 at
β ≤ 0.25); the β = 0.5 exposure is real and reported as measured. Note that on these seeds peer-reported halving does
not collapse at β = 0.5 (0.703): the replication grid uses random liars only, and the collapse is a low-skill-first
effect (§4 table). **MIDIAN-VA on the same fresh seeds** (360 pairs): VA − V **+0.020** [+0.016, +0.025] overall;
n = 1000: +0.001 / −0.004 / **+0.074** [+0.066, +0.082] at β = 0 / 0.25 / 0.5; n = 100: +0.000 / −0.008 / +0.058.
VA − MIDIAN +0.031 [+0.028, +0.035] (n = 1000: +0.022 / +0.025 / +0.057). VA − halving_peer at n = 1000: −0.038 /
−0.045 / −0.006. The variants_f1 picture (§4 steps 1–2) replicates: VA equals V at β = 0, gives back ≤ 0.008 at
β = 0.25, and is +0.06–0.07 above V under collusion — on random liars, without the low-skill-first selection.

## 5. Churn (H9)

churn_n1000: 10% or 30% of agents replaced in place every 200 tasks (fresh profiles, liars redrawn at rate β, probe
indices reset; the first task routed to a replaced agent the method has not re-probed scores 0), n = 1000, specialist +
heavy_tail, β ∈ {0, 0.25}, self-described, 5 seeds, Q = 1000; 20 units per arm and fraction (complete 2026-09-03 21:30).

| arm | success, 10% churn | Δ vs same cells no churn | success, 30% churn | Δ vs no churn | repair probes / event (% of build) |
|---|---|---|---|---|---|
| oracle | 0.797 | — | 0.797 | — | — |
| sequential_halving_peer, rebuild | 0.794 | +0.07* | 0.794 | +0.07* | 44,928 (100%) |
| warm_start_bandit | 0.732 | −0.001 | 0.729 | −0.005 | 4,800 / 14,400 (10 / 30%) |
| flat_probe_argmax_online | 0.722 | −0.002 | 0.710 | −0.015 | 4,800 / 14,400 |
| sequential_halving_peer, stale | 0.721 | +0.00 | 0.589 | −0.13 | 0 |
| midian_va (added 2026-09-04) | 0.687 | −0.02 vs MIDIAN | 0.675 | −0.03 vs MIDIAN | 6,400 (13%) |
| midian_a | 0.713 | −0.012 | 0.697 | −0.028 | 4,800 / 14,400 (9.5 / 28.6%) |
| midian_sh | 0.712 | −0.009 | 0.703 | −0.018 | 4,800 / 14,400 |
| **midian** | 0.709 | **−0.008** | 0.702 | **−0.014** | 4,800 / 14,400 (10 / 30%) |
| midian_v | 0.689 | −0.059 | 0.683 | −0.064 | 3,200 / 9,600 (6.7 / 20%) |
| linucb_honest | 0.624 | −0.062 | 0.511 | −0.175 | 0 |
| fw_langgraph / fw_autogen | 0.517 / 0.503 | — | 0.431 / 0.435 | — | 0 |

(*rebuild re-spends the whole n·K·b budget at every event, so after four events it has probed 5× more than anyone
else; its "no-churn" reference is a single-budget run.) The no-churn baseline is the same (shape, β, seed) cell from
live_f1_n1000 / variants_f1 on the self-described channel. Per 100-task block (events at 200, 400, 600, 800), MIDIAN
dips ≤ 0.03 after an event and is back within the next block; midian_v, whose cached root pick is not re-verified on
churn, loses 0.06 and does not recover; linucb and the frameworks degrade monotonically at 30% (0.60 → 0.37 and 0.52 →
0.39 over the stream) because their statistics/descriptions refer to agents that no longer exist. MIDIAN's repair is the
work-order formula exactly (K·b probes, K·b·(r−1) reports, (r−1)+depth messages per arrival: 4,800 / 43,200 / 1,200 per
event at 10%; runs after commit 3415f03 also charge the path recompute per arrival, K·r·depth = 480 comparisons and K·depth = 48
messages, which these rows predate). **V2-6**: quality half HIT (−0.008 at 10%, within 0.03; halving-stale loses 0.13 at 30%, ≥ 0.05);
cost half MISS as written — repair is 10% of build per event at 10% churn, not ≤ 3% (the 3% was mis-derived: re-probing
10% of agents at full b is 10% of the build by definition), and halving-rebuild's repair is 9.4× MIDIAN's, not ≥ 10×.
Figure H9 shows success and cumulative probes vs task index.

**MIDIAN-VA under churn, budget and r = 20 (rows added 2026-09-04; pre-registered T3-19–21 in TARGETS_rte_v3.md).**
Churn: VA 0.687 / 0.675 at 10% / 30% vs V 0.689 / 0.683, MIDIAN 0.709 / 0.702, A 0.713 / 0.697 — VA − V −0.002 / −0.008,
VA − MIDIAN −0.022 / −0.028 (20 pairs each): the cached root pick goes stale exactly like V's, and the audits do not
repair staleness; **T3-20 MISS** (VA − V ≥ +0.03 expected). Budget (β = 0.25, random liars): VA 0.588 / 0.662 / 0.715 at
b = 1 / 3 / 10 vs V 0.593 / 0.675 / 0.698 — VA − V −0.006 [−0.013, +0.002] / −0.013 [−0.025, −0.001] / +0.017
[+0.008, +0.028]; **T3-21 MISS at b ≤ 3** (with random liars V's exposure is small and the 5% audit budget costs more
than it returns), HIT at b = 10. b = 10 shapes: VA 0.568 (bimodal) / 0.723 (heavy_tail), the best MIDIAN variant, 0.015
under peer halving. n = 10k (random liars): VA 0.811 / 0.803 vs V 0.813 / 0.806 (−0.002 [−0.010, +0.006]), +0.019 over
MIDIAN. r = 20: VA 0.661 / 0.665 at β = 0.25 / 0.5 vs V(r = 20) 0.653 / 0.636 (+0.018 [+0.011, +0.026]) and MIDIAN(r = 20)
0.641 / 0.644. Net: VA's advantage is the collusion regime (§4, part E of v3) and b = 10; under churn and small budgets
with random liars it trails V and plain MIDIAN by 0.01–0.03.

## 6. Cost scaling and break-even (H4, H5)

Per task and at build, n = 1000, reports per probe in every arm:

| method | build probes | build reports | build msgs | msgs/task | comps/task | LLM calls/task |
|---|---|---|---|---|---|---|
| midian | 48,000 | 432,000 | 1,010 | 9 (6 descent + 3 update) | 60 (30 + 30) | 0 |
| midian_v | 47,840 | 430,560 | 1,010 | 5 (2 + 3) | 31 (1 + 30) | 0 |
| midian_v_r5 | 48,000 | 192,000 | 1,050 | 7 (2 + 5) | 26 (1 + 25) | 0 |
| midian_a | 50,382 (1.050×) | 432,000 (+ online audits as reports) | 1,010 | 9 | 60 | 0 |
| flat_probe_argmax (either) | 48,000 | 0 | 0 | 0 | 1,000 | 0 |
| sequential_halving_peer | 44,928 | 404,352 | 0 | 0 | 1 | 0 |
| declared_argmax | 0 | 0 | 1,000 | 0 | 1,000 | 0 |
| llm_supervisor | 0 | 0 | 1,000 | 22 | 20 | 1 |
| any framework (own selection) | 0 | 0 | 1,000 | 12 | 10 | ≥ 1 |
| framework + midian_v shortlist | 47,840 | 430,560 | 2,010 | 17 | 41 | ≥ 1 |

Exponents over n = 10² … 10⁷ (bernoulli calibrated to the measured S; replay 10⁴ … 10⁶; live 10² … 10⁴): MIDIAN
per-task comparisons = r·⌈log_r n⌉ for the descent plus the same for the path update, messages 3·⌈log_r n⌉ (fitted n^0.14 [0.13, 0.15] on the mixed set, n^0.11 on bernoulli
alone); midian_v per-task n^0; flat / declared / CNP comparisons n^1.00; build probes n^1.03 for every probe-based
method (n·K·b), n^0.94 for midian_v; MIDIAN build reports n^1.03.

Break-even against a framework = build cost / per-task saving. Messages: after the first task (1,010 vs 1,000 at build, then
5 vs 12 per task for midian_v, 9 vs 12 for midian). Comparisons: **never** — with the path update charged, midian_v spends 31
and midian 60 comparisons per task against a framework's 10 (its shortlist scan); the frameworks' cost is the LLM call, not
the comparisons. Messages + reports: (432,000 + 1,010 − 1,000) / (12 − 5) ≈ **61,700 tasks** for midian_v with per-probe
reports (43,000 before the update was charged; 15,900 under per-peer reports), ≈ 38,400 for midian_v_r5. LLM calls: ≈ 48,000 tasks at b = 3 against a one-call
framework (16,000 at b = 1; 5,000–8,000 against Magentic-One / CAMEL, which make several calls per task).
Figure H4 plots build + Q·per-task cost against success at Q ∈ {10², 10³, 10⁴, 10⁵} with the break-even Q marked (the framework points refresh when fw_live_* closes).

Supervisor latency (the one wall-clock table; frameworks' calls are never memoised; under shared-fleet load), medians
at n = 1000: AutoGen 0.7 s, CrewAI 0.6, Google ADK 1.0, OpenAI Agents 1.0, LangGraph 1.1, MAF 1.3, smolagents 1.6,
LlamaIndex 3.1, CAMEL 4.5, Magentic-One 6.8.

## 6b. Runtime, energy and combined cost (estimate*)

From `scripts/energy.py` (counts × per-call GPU-seconds measured on the live fleet; per-task counts include the observe-time path update); cumulative cost vs tasks routed in GPU-s, Wh, messages, comparisons, plus joules and critical-path latency with stated per-event weights. Figures H10, H11.
Runtime/energy ESTIMATE (*) per method from LLM-call counts x measured per-call GPU cost.  python scripts/energy.py
Wall-clock in the rows is not used for probe methods (memo hits). Model: GPU-seconds per call = params_b * (A*prompt_tok + B*gen_tok),
B = 5A (decode is ~5x prefill per token on H100/vLLM), A calibrated so a 7B supervisor call (1,900 prompt + 65 gen tokens) costs
1/34 GPU-s = the throughput measured on the saturated 1-GPU 7B replicas (2026-09-03, 4 samples, 32-36 req/s). Energy = GPU-s * W.

Per-call GPU-seconds: 7B supervisor call 0.0294; expected probe/execution call by population shape: specialist 0.00577, heavy_tail 0.00176, bimodal 0.00203 (specialist mixes all 7 models uniformly; heavy_tail 90% 0.5-1.5B; bimodal 80% 0.5B / 20% 7B).
Framework supervisor cost = measured latency ratio to AutoGen (one call) x the 7B call cost; the 14B Magentic-One arm is scaled by 14/7. CPU-side routing (tree descent, TF-IDF) is microseconds per task and omitted. The routed task's own execution (0.0058 GPU-s per task on specialist) is common to every method and excluded.

CUMULATIVE LLM compute after t routed tasks = build + t x per-task (n=1000 specialist). Probe-based methods pay their build up front and then ~0 per task; frameworks pay 0 up front and a supervisor call per task, so the crossing of the two lines is the break-even.

| method | build GPU-s | per-task GPU-s | cumulative GPU-s @ t=1k | @ 10k | @ 100k | cumulative Wh @ 10k (700 W) | (400 W) | messages @ 10k (build + 10k/task) | comparisons @ 10k |
|---|---|---|---|---|---|---|---|---|---|
| declared_argmax | 0 | 0.0000 | 0 | 0 | 0 | 0.0 | 0.0 | 1,000 (1,000 + 0/task) | 10,000,000 (0 + 1000/task) |
| llm_supervisor | 0 | 0.0080 | 8 | 80 | 796 | 15.5 | 8.8 | 221,000 (1,000 + 22/task) | 200,000 (0 + 20/task) |
| verify_on_claim | 0 | 0.0101 | 10 | 101 | 1,013 | 19.7 | 11.3 | 1,000 (1,000 + 0/task) | 169,840 (0 + 17/task) |
| sequential_halving | 259 | 0.0000 | 259 | 259 | 259 | 50.4 | 28.8 | 0 (0 + 0/task) | 10,000 (0 + 1/task) |
| sequential_halving{"peer_reported":true} | 259 | 0.0000 | 259 | 259 | 259 | 50.4 | 28.8 | 0 (0 + 0/task) | 10,000 (0 + 1/task) |
| midian_v | 276 | 0.0000 | 276 | 276 | 276 | 53.7 | 30.7 | 51,010 (1,010 + 5/task) | 310,000 (0 + 31/task) |
| flat_probe_argmax{"online":true} | 277 | 0.0000 | 277 | 277 | 277 | 53.9 | 30.8 | 0 (0 + 0/task) | 10,000,000 (0 + 1000/task) |
| flat_probe_argmax | 277 | 0.0000 | 277 | 277 | 277 | 53.9 | 30.8 | 0 (0 + 0/task) | 10,000,000 (0 + 1000/task) |
| linucb_honest | 277 | 0.0000 | 277 | 277 | 277 | 53.9 | 30.8 | 0 (0 + 0/task) | 10,000,000 (0 + 1000/task) |
| warm_start_bandit | 277 | 0.0000 | 277 | 277 | 277 | 53.9 | 30.8 | 1,000 (1,000 + 0/task) | 10,000,000 (0 + 1000/task) |
| midian | 277 | 0.0000 | 277 | 277 | 277 | 53.9 | 30.8 | 91,010 (1,010 + 9/task) | 600,000 (0 + 60/task) |
| midian_sh | 277 | 0.0000 | 277 | 277 | 277 | 53.9 | 30.8 | 91,010 (1,010 + 9/task) | 600,000 (0 + 60/task) |
| midian_va | 285 | 0.0000 | 285 | 285 | 285 | 55.4 | 31.7 | 51,772 (1,010 + 5/task) | 317,620 (0 + 32/task) |
| midian_a | 291 | 0.0000 | 291 | 291 | 291 | 56.5 | 32.3 | 91,194 (1,010 + 9/task) | 601,840 (0 + 60/task) |
| midian_sha | 291 | 0.0000 | 291 | 291 | 291 | 56.5 | 32.3 | 91,010 (1,010 + 9/task) | 600,000 (0 + 60/task) |
| fw_autogen | 0 | 0.0294 | 29 | 294 | 2,941 | 57.2 | 32.7 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_maf | 0 | 0.0353 | 35 | 353 | 3,534 | 68.7 | 39.3 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_smolagents | 0 | 0.0440 | 44 | 440 | 4,397 | 85.5 | 48.9 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_google_adk | 0 | 0.0473 | 47 | 473 | 4,732 | 92.0 | 52.6 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_langgraph | 0 | 0.0568 | 57 | 568 | 5,681 | 110.5 | 63.1 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_openai_agents | 0 | 0.0617 | 62 | 617 | 6,169 | 119.9 | 68.5 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_crewai | 0 | 0.1188 | 119 | 1,188 | 11,879 | 231.0 | 132.0 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_llamaindex | 0 | 0.1453 | 145 | 1,453 | 14,533 | 282.6 | 161.5 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_camel_workforce | 0 | 0.1484 | 148 | 1,484 | 14,835 | 288.5 | 164.8 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_magentic_one | 0 | 0.2806 | 281 | 2,806 | 28,058 | 545.6 | 311.8 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_magentic_one{"supervisor":"Qwen/Qwen2.5-14B-Instruct"} | 0 | 0.4721 | 472 | 4,721 | 47,212 | 918.0 | 524.6 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |

**Crossing points (tasks routed before the probe-based method's cumulative cost drops below the framework's):**

| | autogen | maf | smolagents | google_adk | langgraph | openai_agents | crewai | llamaindex | camel_workforce | magentic_one | magentic_one{"supervisor":"Qwen/Qwen2.5-14B-Instruct"} |
|---|---|---|---|---|---|---|---|---|---|---|---|
| midian | 9,416 | 7,836 | 6,298 | 5,852 | 4,875 | 4,489 | 2,331 | 1,906 | 1,867 | 987 | 587 |
| midian_a | 9,882 | 8,224 | 6,610 | 6,142 | 5,116 | 4,712 | 2,447 | 2,000 | 1,959 | 1,036 | 616 |
| midian_v | 9,385 | 7,810 | 6,277 | 5,833 | 4,858 | 4,474 | 2,324 | 1,899 | 1,861 | 984 | 585 |

In MESSAGES (fetch 2 per level + observe-time update 1 per level, commit 3415f03) the crossing is immediate: MIDIAN 1,010 + 9t vs a framework 1,000 + 12t crosses at t = 3; MIDIAN-V (1,010 + 5t) at t = 1. In COMPARISONS MIDIAN pays 60 per task (30 descent + 30 observe-time update) vs a framework's 10 (MIDIAN-V 31 = 1 cached pick + 30 update; halving 1; flat 1,000), so no MIDIAN variant undercuts a framework on comparisons; MIDIAN-V's saving over MIDIAN is the descent, not the update. Per task, MIDIAN's cost is communication (messages and comparisons), not LLM compute: it makes no LLM call at route time.

MIDIAN's 48,000-probe build by population shape (GPU-s): specialist 277, heavy_tail 84, bimodal 98; the crossings scale with it (heavy_tail and bimodal cross ~3x sooner).
Reading: against a one-call framework (AutoGen) MIDIAN breaks even after ~9,400 tasks on specialist (~2,900 on heavy_tail), MIDIAN-A after ~9,900; against the multi-call frameworks (CrewAI, LlamaIndex, CAMEL) after 1,900-2,300 tasks; against Magentic-One after ~990 (7B) / ~590 (14B arm). Before the crossing the framework is cheaper; after it, the probe-based methods' cost is flat while every framework's keeps growing linearly.

## Combined currencies (*)

(a) ENERGY, joules per event: LLM call = GPU-s x 700 W (a 7B supervisor call = 20.6 J; a specialist probe = 4.04 J); message = one RPC handled in ~100 us on a ~10 W core = 0.001 J (pessimistic column: 0.01 J); comparison = one float compare = 1e-08 J.
(b) LATENCY on the critical path per task: each sequential message hop = 1 ms RTT (MIDIAN: the 2·depth = 6 fetch hops; the observe-time update propagation (1 message per level) is off the critical path and excluded; MIDIAN-V 2 ms; frameworks and llm_supervisor: 2 hops + the supervisor call at its measured median latency under shared-fleet load); comparisons 10 ns each (flat's 1,000 = 10 us); a route-time probe call (verify_on_claim) 0.3 s.

| method | J/task at t=10k (build amortised) | J/task, pessimistic messages | of which LLM J/task | messages J/task | comparisons J/task | latency s/task |
|---|---|---|---|---|---|---|
| declared_argmax | 0.000 | 0.001 | 0.000 | 0.0001 | 1.00e-05 | 0.0000 |
| llm_supervisor | 5.591 | 5.790 | 5.569 | 0.0221 | 2.00e-07 | 0.5218 |
| verify_on_claim | 7.093 | 7.094 | 7.093 | 0.0001 | 1.70e-07 | 0.5269 |
| sequential_halving | 18.146 | 18.146 | 18.146 | 0.0000 | 1.00e-08 | 0.0000 |
| sequential_halving{"peer_reported":true} | 18.146 | 18.146 | 18.146 | 0.0000 | 1.00e-08 | 0.0000 |
| midian_v | 19.327 | 19.373 | 19.322 | 0.0051 | 3.10e-07 | 0.0020 |
| flat_probe_argmax{"online":true} | 19.386 | 19.386 | 19.386 | 0.0000 | 1.00e-05 | 0.0000 |
| flat_probe_argmax | 19.386 | 19.386 | 19.386 | 0.0000 | 1.00e-05 | 0.0000 |
| linucb_honest | 19.386 | 19.386 | 19.386 | 0.0000 | 1.00e-05 | 0.0000 |
| warm_start_bandit | 19.386 | 19.387 | 19.386 | 0.0001 | 1.00e-05 | 0.0000 |
| midian | 19.395 | 19.477 | 19.386 | 0.0091 | 6.00e-07 | 0.0060 |
| midian_sh | 19.395 | 19.477 | 19.386 | 0.0091 | 6.00e-07 | 0.0060 |
| midian_va | 19.963 | 20.009 | 19.958 | 0.0052 | 3.18e-07 | 0.0021 |
| midian_a | 20.355 | 20.437 | 20.346 | 0.0091 | 6.02e-07 | 0.0060 |
| midian_sha | 20.357 | 20.439 | 20.348 | 0.0091 | 6.00e-07 | 0.0060 |
| fw_autogen | 20.600 | 20.709 | 20.588 | 0.0121 | 1.00e-07 | 1.9236 |
| fw_maf | 24.752 | 24.860 | 24.739 | 0.0121 | 1.00e-07 | 2.3111 |
| fw_smolagents | 30.794 | 30.903 | 30.782 | 0.0121 | 1.00e-07 | 2.8751 |
| fw_google_adk | 33.138 | 33.247 | 33.126 | 0.0121 | 1.00e-07 | 3.0939 |
| fw_langgraph | 39.782 | 39.891 | 39.770 | 0.0121 | 1.00e-07 | 3.7140 |
| fw_openai_agents | 43.194 | 43.303 | 43.182 | 0.0121 | 1.00e-07 | 4.0325 |
| fw_crewai | 83.163 | 83.272 | 83.151 | 0.0121 | 1.00e-07 | 7.7630 |
| fw_llamaindex | 101.742 | 101.851 | 101.730 | 0.0121 | 1.00e-07 | 9.4971 |
| fw_camel_workforce | 103.858 | 103.967 | 103.846 | 0.0121 | 1.00e-07 | 9.6947 |
| fw_magentic_one | 196.420 | 196.529 | 196.408 | 0.0121 | 1.00e-07 | 18.3341 |
| fw_magentic_one{"supervisor":"Qwen/Qwen2.5-14B-Instruct"} | 330.494 | 330.603 | 330.482 | 0.0121 | 1.00e-07 | 15.4251 |

Crossings in joules (tasks after which the probe-based method's cumulative energy falls below the framework's): midian: autogen 9,415, maf 7,835, smolagents 6,297, google_adk 5,852, langgraph 4,874, openai_agents 4,489, crewai 2,331, llamaindex 1,906, camel_workforce 1,867, magentic_one 987, magentic_one{"supervisor":"Qwen/Qwen2.5-14B-Instruct"} 587; midian_a: autogen 9,881, maf 8,223, smolagents 6,609, google_adk 6,141, langgraph 5,115, openai_agents 4,711, crewai 2,447, llamaindex 2,000, camel_workforce 1,959, magentic_one 1,036, magentic_one{"supervisor":"Qwen/Qwen2.5-14B-Instruct"} 616; midian_v: autogen 9,382, maf 7,808, smolagents 6,276, google_adk 5,831, langgraph 4,858, openai_agents 4,474, crewai 2,323, llamaindex 1,899, camel_workforce 1,860, magentic_one 984, magentic_one{"supervisor":"Qwen/Qwen2.5-14B-Instruct"} 585.
Under any sane weighting the LLM call dominates energy by 3-5 orders of magnitude (20.6 J per supervisor call vs 6e-3 J for MIDIAN's six messages and 3e-7 J for its thirty comparisons per task), so the joule crossings equal the GPU-second crossings to the task; messages dominate MIDIAN's latency (6 ms of fetch hops vs 0.6 us of comparisons) while the supervisor call dominates every framework's (0.5-19 s).


## 7. Budget, by channel (H8)

budget_sweep (n = 1000, β = 0.25, 3 shapes × 5 seeds, all 45 cells paired; both channels pooled for probe-only arms):

| method | b=1 | b=3 | b=10 |
|---|---|---|---|
| oracle | 0.723 | 0.723 | 0.723 |
| sequential_halving (trusted) | 0.479 | 0.722 | 0.723 |
| midian_v | 0.593 | 0.675 | 0.698 |
| midian_va (added 2026-09-04) | 0.588 | 0.662 | 0.715 |
| midian_a (added 2026-09-04) | 0.588 | 0.667 | 0.710 |
| midian | 0.585 | 0.650 | 0.702 |
| flat_probe_argmax_frozen | 0.481 | 0.620 | 0.695 |
| warm_start_bandit | 0.676 | 0.684 | 0.704 |
| declared_argmax | 0.673 | 0.673 | 0.673 |
| ucb / thompson (late quarter) | 0.449 / 0.477 | 0.572 / 0.606 | 0.671 / 0.679 |

At b = 1 the tree is the best probe-only arm (+0.10 over flat and halving); any arm that also reads the *programmatic*
declaration beats every probe-only arm at b = 1 — a property of the synthetic channel (declared_argmax is 0.55 on the
live one). Figure H8 splits by channel (budget_sweep is programmatic-only for the declared arms; the self-described panel comes
from budget_b10_shapes).

**b = 10 on the two non-specialist shapes (budget_b10_shapes: β = 0.25, both channels, 3 seeds, Q = 300).** At b = 10
the bimodal gap disappears and heavy_tail opens up:

| shape | oracle | sequential_halving (trusted) | sequential_halving_peer | midian_v | midian | flat_probe_argmax_online | LangGraph / AutoGen |
|---|---|---|---|---|---|---|---|
| bimodal | 0.568 | 0.568 | 0.569 | 0.563 | 0.566 | 0.567 | 0.568 / 0.568 |
| bimodal, midian_va / midian_a | | | | 0.568 / 0.568 | | | |
| heavy_tail | 0.739 | 0.741 | 0.738 | 0.722 | 0.701 | 0.712 | 0.604 / 0.604 |
| heavy_tail, midian_va / midian_a | | | | 0.723 / 0.718 | | | |

Paired, frameworks − MIDIAN: bimodal **+0.002**, heavy_tail **−0.097** [−0.112, −0.081] (6 pairs each); trusted halving − MIDIAN
+0.040 and peer halving − MIDIAN +0.037 on heavy_tail (all 108 rows in).
With ten probes per cell every probe method sits on the oracle on bimodal, i.e. the v1 −0.04 was the 4-valued estimate,
not the tree. **V2-9 HIT.** (LangGraph and AutoGen give identical numbers here because with the same shortlist both
supervisors pick the same agent on these tasks.)

**n = 10,000 at b = 3 on the live channel (live_n10k_v2: specialist, β ∈ {0, 0.25}, 3 seeds, Q = 300).** Replaces the
2026-09-02 b = 1 / programmatic row (where declared_argmax ≈ oracle):

| arm | β=0 | β=0.25 | build probes | reports | msgs / comps per task |
|---|---|---|---|---|---|
| sequential_halving_peer | 0.863 | 0.863 | 460,016 | 4,140,144 | 0 / 1 |
| oracle | 0.859 | 0.859 | | | |
| midian_v | 0.813 | 0.806 | 479,840 | 4,318,560 | 6 / 41 |
| midian_va (added 2026-09-04) | 0.811 | 0.803 | 495,651 | 4,318,560 | 6 / 42 |
| midian_sh | 0.794 | 0.772 | 480,000 | 4,320,000 | 12 / 80 |
| midian | 0.786 | 0.790 | 480,000 | 4,320,000 | 12 / 80 |
| midian_a | 0.784 | 0.786 | 503,806 | 4,320,000 | 12 / 80 |
| flat_probe_argmax_online | 0.774 | 0.774 | 480,000 | 0 | 0 / 10,000 |
| warm_start_bandit | 0.749 | 0.803 | 480,000 | 0 | 0 / 10,000 |
| verify_on_claim | 0.724 | 0.709 | 0 | 0 | 0 / 534 |
| linucb_honest | 0.487 | 0.493 | 480,000 | 0 | 0 / 10,000 |
| random | 0.417 | 0.417 | | | |
| ten frameworks | 0.257–0.367 | 0.257–0.362 | 0 | 0 | 12 / 10 |

Paired: midian − flat_online +0.013 [−0.004, +0.031] (3/6), midian_v − midian +0.022 [+0.010, +0.033] (6/6); the ten
frameworks average 0.29, **below random** (0.417): with 10,000 self-descriptions the TF-IDF top-10 is dominated by
descriptions that match the task's words rather than the task's family, and the supervisor never sees a competent
agent. frameworks − flat_online = −0.48. LinUCB also collapses at this scale (16k × 10 arms, 300 tasks). **V2-7 HIT.**
Peer-reported halving at n = 10k (the six units landed 2026-09-03 18:45): 0.863 at both β, **+0.054** [+0.042, +0.067]
over midian_v (6/6) and +0.076 over plain MIDIAN, +0.004 over the oracle (2/6; the oracle is the argmax of S measured
with ±0.035 binomial error, so a method can sit on it). With random liars at β = 0.25 its picks are unchanged from
β = 0: at n = 10k the trimmed report of 16 reporters per cohort absorbs a quarter of random liars entirely. The
low-skill-first collusion that breaks it at n = 1000 (§4) is not in this grid; the learned_n10k grid (RESULTS_rte_v3
part C) runs both liar selections at n = 10k.

## 8. Pre-registered verdicts

**v1 (TARGETS_rte.md), six, with what each miss taught:**

| # | expectation | verdict | number | what it taught |
|---|---|---|---|---|
| 1 | declared lose ≥ 0.25 from β 0→0.5; probe-only move ≤ 0.03 | MISS | declared lose 0.07–0.12 (0.05–0.22 on self-described); MIDIAN loses 0.071 under collusion | the +0.4 lie is capped by clip and the top-20% collusion rule is what hurts; the tree is not probe-only in the report channel |
| 2 | MIDIAN = flat within 0.02 at β=0; comparisons r·log_r n vs n | MISS / PASS | online MIDIAN +0.044 over frozen flat (max-tree alone −0.009); exponents 0.14 vs 1.00 | the online path recompute, not the tree, is the difference |
| 3 | trimming separates only where β·r exceeds the trim | MISS | δ=1/3 − δ=0 at r=10, collude: +0.004 / −0.004 / −0.036 at β 0.1 / 0.25 / 0.5 | in the predicted regime, wrong direction: with half the reporters colluding, trimming removes honest extremes (→ MIDIAN-A) |
| 4 | verify_on_claim within 0.03 of oracle at β ≤ 0.1, loses ≥ 0.10 by β=0.5 | MISS | gap 0.050; loss 0.037 | verification budget is spent on claims, and honest claims dominate at low β |
| 5 | halving ≈ flat at equal budget; bandits' late success ≥ MIDIAN's at b=1 | MISS | halving − flat +0.103; ucb/thompson late −0.138 / −0.110 vs MIDIAN at b=1 | adaptive allocation is worth 0.10 with 4-valued estimates; 16k arms cannot be explored in 1,000 tasks |
| 6 | argmax-vs-floor gap largest under heavy_tail, smallest under iid_uniform | MISS | heavy_tail 0.175 ≈ specialist 0.173 > bimodal 0.140; iid_uniform 0.166 ≈ correlated 0.163 | the gap tracks how much the population's skill is family-specific, not its tail |

**v2 (TARGETS_rte_v2.md), eleven** (`rte.analyze … --out results/v2_targets`; paired on `label`, plain MIDIAN only;
WITHIN_FLOOR = decisive delta inside MIDIAN's mean seed envelope, 0.070 at n = 1000):

| # | expectation | verdict | numbers |
|---|---|---|---|
| V2-1 | MIDIAN-SH within 0.02 of halving_peer at β ≤ 0.25; ≥ MIDIAN − 0.02 at β=0.5 low-skill; per-task cost unchanged | **MISS** | −0.035 / −0.066 / −0.076 vs halving_peer by shape; −0.143 vs MIDIAN at β=0.5 low-skill (0/30); cost unchanged |
| V2-2 | MIDIAN-A: β=0.5-collude loss ≤ 0.02 on specialist; unchanged within 0.01 at β ≤ 0.25; build ≤ 1.05× | **HIT** | loss −0.004; +0.007 at β ≤ 0.25 (186 pairs); 1.050× probes |
| V2-3 | MIDIAN-SH+A ≥ max(SH, A) − 0.01 at every β | **MISS** (within floor) | −0.015 / −0.000 / −0.008 / −0.004 at β = 0 / 0.1 / 0.25 / 0.5 |
| V2-4 | stratified cohorts vs random, reported as measured | REPORTED | +0.007 / −0.012 (heavy_tail β 0 / 0.5), −0.003 / +0.004 (specialist); −0.004 overall, 8/20 |
| V2-5 | LinUCB-honest between flat_online and warm_start at β=0, flat in β | **MISS** | −0.050 below flat_online at β=0 (grid merge; −0.025 on variants_f1 alone), warm_start +0.084 above it; flat in β (+0.024 over 0→0.5 in the merge, 0.000 on variants_f1) |
| V2-6 | churn: MIDIAN within 0.03 at 10% with repair ≤ 3%; halving-stale ≥ 0.05 worse at 30%; rebuild ≥ 10× repair | quality **HIT** / cost **MISS** | −0.008 at 10%; repair 10% of build; stale −0.13 at 30%; rebuild repair 9.4× (halving-rebuild rows 10–11/20 at write time) |
| V2-7 | n=10k b=3: MIDIAN ≥ flat_online − 0.02; frameworks ≥ 0.10 below flat_online on specialist | **HIT** | +0.013 (6 pairs); frameworks −0.48 |
| V2-8 | replication, fresh seeds 11–20: midian_v − midian = +0.02 ± 0.02 at β ≤ 0.25 | **HIT** | +0.021 (240 pairs; n=1000: +0.021 / +0.029 at β = 0 / 0.25, n=100: +0.007 / +0.026); at β=0.5 collude −0.008 (n=1000 −0.017, n=100 0.000) |
| V2-9 | b=10 closes the bimodal framework gap to ±0.02; heavy_tail MIDIAN ≥ fw + 0.03 | **HIT** | bimodal +0.002; heavy_tail −0.097 |
| V2-10 | trimming hurts plain MIDIAN ≥ 0.02 under collusion, not MIDIAN-A | **HIT** | r=10: MIDIAN −0.038 (δ=1/3 − δ=0, collude), MIDIAN-A −0.000 |
| V2-11 | MIDIAN-VA ≥ max(V, A) − 0.01 at every β; within 0.02 of A at β=0.5 low-skill collude; build probes ≤ 1.05× V, per-task cost V's | **MISS** (within floor) | VA − max(V, A): +0.000 / −0.013 / −0.009 / +0.001 at β = 0 / 0.1 / 0.25 / 0.5 on variants_f1 (−0.010 / −0.013 / −0.017 / +0.001 merged with the complete replication grid); VA − A at β=0.5 low-skill +0.014 (30 pairs); 1.033× probes; 31.6 / 5.06 per task vs V's 31 / 5 |

Six hits, four misses (SH does not help; SH+A adds nothing over A; a history-only LinUCB is below a flat scan; VA gives
back ~0.01 of V's edge at β = 0.1–0.25 while repairing its collusion collapse — the two within-floor misses, V2-3 and
V2-11, are 0.003–0.007 beyond a 0.01 tolerance and inside MIDIAN's own seed envelope), one split (churn: quality as
predicted, repair-cost arithmetic wrong in the pre-registration), one reported.

## 9. Replay (RouterBench real outcomes, K = 64, n = 1000, 10 seeds; programmatic channel only — replay has no
self-description)

At β = 0 / 0.25 / 0.5: oracle 0.783; sequential_halving 0.779 / 0.779 / 0.779; sequential_halving_peer 0.779 / 0.773
/ 0.519; declared_argmax 0.779 / 0.730 / 0.690; midian_v 0.738 / 0.731 / 0.524; midian 0.705 / 0.688 / 0.623;
flat_probe_argmax_online 0.707; flat_probe_argmax_frozen 0.682; random 0.189. With ≈ 11 distinct real models many
agents are copies a 3-probe estimate cannot separate, so MIDIAN ≈ flat ≈ midian_v at β ≤ 0.25 while the honest
declaration is near-oracle; the β = 0.5 ordering (midian > midian_v > halving_peer) matches the live one.

## 10. Learning over the stream (Appendix figure)

9 live cells (3 shapes × 3 seeds, n = 1000, β = 0.25), success minus oracle per block of 100 tasks: MIDIAN −0.088 →
−0.05 over the stream (the online update is worth +0.06 in total, half of it inside the first 100 tasks); midian_v
starts at −0.05 and stays; MIDIAN with updates off never moves (−0.12); frameworks gain 0.00–0.015 from early to late.

## 11. Caveats

- **CrewAI on 2026-09-02 was not measured.** Its shared SQLite task-output store was corrupted by hundreds of
  concurrent workers on NFS at ~11:05; every later kickoff failed before the manager ran and was counted as a
  fallback (79–84%, 100% with the MIDIAN shortlist). Fixed (private store, delegating manager); rows re-run.
- **Magentic-One's failures are real**: its orchestrator solves the task itself during planning and returns
  `is_request_satisfied` with no speaker (5/6 live tasks on specialist self-descriptions with the 7B; 3/6 with the
  14B). Reported under both accountings and both supervisors.
- **S measured once and shared** (Methods): ±0.035 / ±0.065 per cell is a common offset outside the CIs.
- **Framework grids** were Q = 300 / 3 seeds on 2026-09-02 (5 seeds for the paired non-framework arms); the v2 rerun
  is Q = 1000 / 5 seeds for all arms.
- **Report accounting**: MIDIAN-V rows before 2026-09-03 carry per-peer report counts (159,840 at n = 1000); the
  per-probe figure is 430,560 and is what §6 uses.
- **Wall-clock**: only the supervisor-latency table; everything else mixes memo hits and misses.
- **Bernoulli sanity numbers** in §4 are 2-seed smoke tests on the synthetic backend, not evidence for any target.

- **Per-task cost accounting was corrected on 2026-09-03 (commit 3415f03).** MIDIAN's observe-time path recompute (r
  comparisons and one message per level per routed task) had never been charged; every per-task message/comparison
  figure in this document includes it (plain 60 / 9, MIDIAN-V 31 / 5 at n = 1000; the analyzer adds it to rows written
  before the fix). Churn rows predate the matching repair charge (K·r·depth comparisons, K·depth messages per arrival).
  §6b's energy/latency tables carry their own accounting and are not restated here.

- **Why the probe-based arms look this good, and where that stops (added 2026-09-03 after review).** The benchmark
  supplies the one thing probing needs and most deployments lack: a cheap probe whose outcome is checkable on the same
  distribution as the tasks. Four consequences. (a) Every probe here is a free, exact bit of skill (programmatic ground
  truth, memoised); with a judge model in the loop the judge's error is the estimate's floor and each probe is a paid
  call. (b) True skill is defined per family signature, so family is the whole story and a family-level scan is a
  sufficient statistic; real difficulty varies more within a family than across agents, which is why learned routers
  condition on the prompt, not on the agent. (c) n = 1000 anonymous, stationary strangers is a marketplace that does not
  exist yet; a typical deployment has 5–20 agents the developer wrote, whose self-descriptions are honest because the
  developer wrote them — there declared_argmax *is* the oracle and every probe is wasted budget. (d) The frameworks are
  orchestration plumbing whose supervisor picks a worker from a self-description by default; beating them on selection
  says more about that default than about the frameworks, and the gap is zero on bimodal populations (§1). Flat probe
  argmax is an offline eval with a lookup table, which every serious team already runs; RouterBench, RouteLLM, Martian
  and Not Diamond are learned routers over exactly that kind of labelled-outcome data. What is not standard practice is
  what MIDIAN adds on top — O(log n) route cost, no trusted central observer, robustness to reporters who lie — and it
  pays off only at scale with untrusted parties. Untested here: non-verifiable outcomes with a judge, continuous drift
  (churn is block-wise), within-family task heterogeneity, and n = 10–50 where the build is not amortised.
  RESULTS_rte_v3.md (in progress) meets the two obvious follow-ups: our arms on RouterBench's own protocol and metric
  against its own routers, and against real released routers (RouteLLM) on their model pair and their metrics.

## 12. Deviations as the as-run protocol (summary of DEVIATIONS.md; every bullet is dated there)

v1: NVIDIA/CUDA fleet, Gemma-2 for the non-Qwen half; six families swapped for separable ones (final K = 16); python
tool gated at ≥ 3B, calculator dropped; S per prompt signature shared across seeds; index-seeded shared probes; content-
hash memo, sharded per process, tool runs memoised; declared lie clip(D_honest + 0.4); collusion rule in scalar and
vectorised forms; MIDIAN pads per level, leaders not materialised, one report_many per cohort chunk; MIDIAN-V, r = 5,
cached, peer-reported halving, framework `retrieval: midian` are labeled post-hoc; bernoulli_scale K = 16, b = 1 above
10⁶; replay twins programmatic-only. v2 (2026-09-03): churn in place with the epoch rule; reports per probe in every
arm (MIDIAN-V changed, halving/referral/gossip already compliant); `midian_v.py` as definition of record; stratify by
measured probe mean (not declared); per-channel analysis, one name per arm, no wall-clock; framework strict accounting
and the CrewAI / Magentic-One / ADK fixes; MIDIAN-SH trims over peers' means (no single trim depth with unequal pulls);
MIDIAN-A audits 5% of probe *instances* (each audit checks all s−1 claims) via `View.probe_at`, online audits charge
s−1 reports, and a new exclusion re-aggregates the cohort from stored per-peer means; audited builds are 1.05× n·K·b by
design.

## Appendix A — declared-channel readers on the programmatic upper bound

See `RESULTS_channel_tables.md` (generated) for the full β × shape tables per channel; §3 carries the roll-up.

## Appendix B — internals (midian_internals, specialist, collude on/off, 5 seeds) and internals_v2

| variant | β=0.1 | β=0.25 | β=0.5 collude | β=0.5 no collude | build reports | msgs / comps per task |
|---|---|---|---|---|---|---|
| flat_probe_argmax_frozen | 0.745 | 0.745 | 0.745 | 0.745 | 0 | 0 / 1000 |
| midian r=10 δ=1/3 | 0.790 | 0.789 | 0.743 | 0.788 | 432k | 9 / 60 |
| midian r=10 δ=0 | 0.789 | 0.789 | 0.778 | 0.788 | 432k | 9 / 60 |
| midian r=5 δ=1/3 | 0.792 | 0.791 | 0.728 | 0.789 | 192k | 15 / 50 |
| midian r=20 δ=1/3 | 0.782 | 0.785 | 0.765 | 0.784 | 912k | 9 / 120 |
| midian_v r=5 | 0.822 | 0.821 | 0.730 | 0.820 | 192k (per probe) | 7 / 26 |
| midian_v r=10 | 0.816 | 0.816 | 0.749 | 0.816 | 431k (per probe) | 5 / 31 |
| midian_v r=20 | 0.785 | 0.786 | 0.764 | 0.785 | 863k (per probe) | 5 / 61 |
| oracle | 0.861 | | | | | |

**internals_v2 (β = 0.5, specialist, self-described, 5 seeds; r × δ × collude × liar selection; plain MIDIAN vs MIDIAN-A):**

| r, δ | MIDIAN, no collusion | MIDIAN, collude random | MIDIAN, collude low-skill | MIDIAN-A, no collusion | MIDIAN-A, collude random | MIDIAN-A, collude low-skill |
|---|---|---|---|---|---|---|
| 5, 0 | 0.792 | 0.736 | 0.713 | 0.792 | 0.743 | 0.724 |
| 5, 1/3 | 0.791 | 0.729 | 0.701 | 0.791 | 0.744 | 0.724 |
| 10, 0 | 0.789 | 0.778 | 0.769 | 0.789 | 0.790 | 0.790 |
| 10, 1/3 | 0.789 | 0.742 | 0.730 | 0.789 | 0.789 | 0.790 |
| 20, 0 | 0.784 | 0.776 | 0.773 | 0.784 | 0.784 | 0.784 |
| 20, 1/3 | 0.784 | 0.766 | 0.762 | 0.784 | 0.784 | 0.784 |
| flat_probe_argmax_frozen | 0.726 | | | | | |
| oracle | 0.861 | | | | | |

Without collusion nothing matters (every cell 0.784–0.792). Under collusion trimming *hurts* plain MIDIAN (δ=1/3 − δ=0
paired: r=10 −0.019 [−0.030, −0.010], 0/10 wins; r=5 −0.005; r=20 −0.005) because with half the reporters lying the
trimmed extremes are the honest ones; MIDIAN-A is unaffected by δ (−0.000 at every r) and at r ≥ 10 sits at its
no-collusion value — the audits remove the colluders before aggregation. At r = 5 audits only partly help (0.724 vs
0.713): a cohort of five with two or three liars has too few honest reporters left. **V2-10 HIT.**

**midian_r20 (all shapes, both channels, β ∈ {0.25, 0.5}, 5 seeds):** midian r=20 vs r=10: −0.013 at β = 0.25, +0.023 at
β = 0.5 (+0.005 overall, 26/58); midian_v r=20 vs midian r=10: −0.000 / +0.015 (+0.008, 34/60). Cost: r=20 doubles
build reports (912,000) and per-task comparisons (120) for plain, 898,016 reports and 61 comparisons for midian_v. Larger
cohorts buy collusion robustness at β = 0.5 and nothing below it.

## Appendix C — UCB / Thompson (16k arms, Q = 1,000, under-explored by construction)

live_f1_n1000: ucb_per_family 0.596 / 0.597 / 0.598 / 0.598 and thompson_per_family 0.603 / 0.603 / 0.604 / 0.604 at
β = 0 / 0.1 / 0.25 / 0.5; both below flat_probe_argmax_frozen (0.62) at every β. `linucb_honest` (variants_f1, 10 seeds) replaces them in §3: 0.642 at every β, −0.025 below
flat_probe_argmax_online (19/239 cells) and −0.003 vs MIDIAN overall; at n = 10,000 it collapses to 0.49 (Appendix, §7).

## Appendix D — framework k-sensitivity and AgentScope

fw_k_sensitivity (specialist, self-described, n = 1000): LangGraph 0.35 / 0.39 / 0.38 and AutoGen 0.35 / 0.38 / 0.37 at
k = 5 / 10 / 20 (oracle 0.855): the shortlist size is not what limits them. fw_appendix (n = 100): AgentScope 0.415
vs random 0.405 vs MIDIAN 0.73; MetaGPT not run (no selection primitive to intercept).

## Appendix E — fallback table

Final (fw_live_n100 and fw_live_n1000 complete, 2,640 rows each; means over the 120 cell × seed pairs; `fallback_rate` = 1 −
picks / (picks + fallbacks + failures + bad_name), as the adapter reports it):

| framework | n=100: fallback rate / failures / fallbacks | lenient → strict | n=1000: fallback rate / failures / fallbacks | lenient → strict |
|---|---|---|---|---|
| Magentic-One (7B orchestrator) | 61.2% / 45.0% / 16.1% | 0.553 → 0.180 | 60.4% / 40.3% / 20.1% | 0.546 → 0.172 |
| Magentic-One, **14B orchestrator** (asymmetric arm) | 45.3% / 2.7% / 42.6% | 0.540 → 0.286 | 43.2% / 2.3% / 40.9% | 0.533 → 0.287 |
| Google ADK | 31.3% / 10.7% / 20.6% | 0.540 → 0.360 | 28.5% / 10.6% / 17.9% | 0.542 → 0.380 |
| LangGraph | 13.3% / 0.0% / 13.3% | 0.530 → 0.454 | 18.7% / 0.0% / 18.7% | 0.531 → 0.427 |
| CAMEL Workforce | 19.6% / 0.0% / 19.6% | 0.534 → 0.423 | 17.7% / 0.0% / 17.7% | 0.527 → 0.429 |
| LlamaIndex | 14.2% / 0.0% / 14.2% | 0.522 → 0.441 | 16.1% / 0.0% / 16.1% | 0.531 → 0.443 |
| OpenAI Agents SDK | 15.9% / 0.0% / 15.9% | 0.527 → 0.440 | 15.3% / 0.0% / 15.3% | 0.528 → 0.444 |
| CrewAI (fixed) | 11.2% / 2.5% / 8.7% | 0.529 → 0.470 | 11.8% / 0.0% / 11.7% | 0.528 → 0.463 |
| smolagents | 7.4% / 0.0% / 1.2% (6% bad names) | 0.529 → 0.462 | 6.9% / 0.0% / 0.7% (6% bad names) | 0.526 → 0.462 |
| AutoGen | 8.1% / 0.0% / 8.1% | 0.528 → 0.483 | 6.0% / 0.0% / 6.0% | 0.528 → 0.497 |
| MAF | 6.2% / 0.0% / 6.2% | 0.530 → 0.485 | 2.8% / 0.0% / 2.8% | 0.524 → 0.501 |

The 2026-09-02 CrewAI rows (79–84%, all "fallbacks") were the corrupted shared task store, not CrewAI.
