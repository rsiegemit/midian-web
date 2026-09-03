# RESULTS_rte_v2.md — RTE after the v2 work order (DRAFT, 2026-09-03)

Status: v2 grids complete except fw_live_n100 / fw_live_n1000 (10 seeds, Q=1000, fixed adapters) and the two
_verified grids, which were still running at 13:30 on 2026-09-03; their sections carry `TODO(fw_live_*)` / `TODO(verified)`. Every number without a `TODO(grid)` mark is final and comes from completed grids (the 2026-09-02
programme, re-analysed with the v2 analyzer: per-channel tables, one name per arm, reports per probe, no wall-clock).
`TODO(grid)` marks a v2 grid that was launched on 2026-09-03 and has not finished; `rte.analyze` fills its section
(`python -m rte.analyze --grid <g> --grids ...` prints the v1 and v2 target verdicts). Pre-registrations: v1
`TARGETS_rte.md` (frozen), v2 `TARGETS_rte_v2.md` (committed before any v2 launch). Plain MIDIAN's parameters never
changed; every mechanism added after the first run is a labeled variant.

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

Arms, one name each: `midian` (plain, r = 10, δ = 1/3, online), `midian_v` / `midian_v_r5` (verified promotion,
cached root pick; post-hoc 2026-09-02, definition in `rte/methods/midian_v.py`), `midian_sh` (successive halving
inside cohorts), `midian_a` (5% audits, exclusion at two strikes), `midian_sha` (both), `midian_stratified`,
`flat_probe_argmax_frozen` / `flat_probe_argmax_online`, `sequential_halving` (trusted observer) /
`sequential_halving_peer` (MIDIAN's report channel) / `..._rebuild` / `..._stale` (churn modes), `linucb_honest`.

---

## 1. Headline: the ten frameworks vs MIDIAN (H1)

Self-described channel, n = 1000, all 60 cells (3 shapes × 4 β × 5 seeds), Q = 1000 tasks.
**TODO(fw_live_n1000, fw_live_n100)** — the 2026-09-03 rerun on the fixed adapters (private CrewAI store, delegating
manager, Magentic-One robust ledger + 14B arm, ADK transfer capture) at 5 seeds / Q = 1000. Until it closes, the
2026-09-02 numbers (Q = 300; CrewAI's rows invalid, see §11) are:

| method | n=100 | n=1000 [95% CI] | Δ vs MIDIAN (paired) | cells rival > MIDIAN | fallback rate |
|---|---|---|---|---|---|
| oracle | 0.711 | 0.719 [0.689, 0.748] | +0.081 | 60/60 | — |
| midian_v | 0.623 | 0.659 [0.630, 0.689] | +0.022 | 37/56 | — |
| midian_v_r5 | 0.630 | 0.656 [0.623, 0.688] | +0.019 | 45/60 | — |
| **midian** | 0.626 | 0.637 [0.613, 0.663] | 0 | — | — |
| flat_probe_argmax_frozen | 0.572 | 0.612 [0.583, 0.640] | −0.025 | 24/59 | — |
| declared_argmax | 0.560 | 0.562 [0.549, 0.574] | −0.076 | 13/60 | — |
| llm_supervisor | 0.555 | 0.562 [0.546, 0.577] | −0.076 | 28/60 | — |
| Magentic-One | 0.546 | 0.541 [0.521, 0.562] | −0.096 | 28/59 | 52% (2026-09-02 adapter) |
| Google ADK | 0.518 | 0.536 [0.514, 0.557] | −0.102 | 29/60 | 21% |
| CrewAI | 0.536 | 0.535 [0.512, 0.558] | −0.102 | 28/60 | 84% (invalid rows) |
| smolagents | 0.522 | 0.523 [0.496, 0.549] | −0.114 | 28/60 | 0% |
| LangGraph | 0.506 | 0.521 [0.494, 0.548] | −0.116 | 28/59 | 6% |
| AutoGen | 0.511 | 0.520 [0.493, 0.546] | −0.118 | 28/59 | 0% |
| LlamaIndex | 0.512 | 0.519 [0.491, 0.547] | −0.118 | 28/59 | 0% |
| OpenAI Agents SDK | 0.506 | 0.519 [0.491, 0.546] | −0.119 | 29/60 | 5% |
| MAF | 0.517 | 0.518 [0.489, 0.546] | −0.120 | 29/60 | 2% |
| CAMEL Workforce | 0.514 | 0.510 [0.479, 0.540] | −0.127 | 28/59 | 0% |
| random | 0.307 | 0.313 [0.291, 0.336] | −0.325 | 0/60 | — |

The average hides the split that decides the paper (figure H1):

| n=1000, by shape | oracle | midian_v | midian | flat (frozen) | frameworks (mean) | cells framework > MIDIAN |
|---|---|---|---|---|---|---|
| specialist (3 strong families per agent) | 0.857 | 0.789 | 0.751 | 0.751 | **0.390** | 0 / 200 |
| heavy_tail (1 in 10 is a big model) | 0.733 | 0.664 | 0.634 | 0.592 | 0.617 | 83 / 200 |
| bimodal (20% big-with-tools, 80% small) | 0.566 | 0.525 | 0.527 | 0.493 | **0.566** | 200 / 200 |

Where skill is legible from a description (bimodal: "is this one of the 20% big models with a tool?") every
framework sits on the oracle and beats MIDIAN in all 200 cells; where skill is family-specific (specialist) the
frameworks collapse to 0.39 against 0.75 and lose all 200. Frameworks are flat in β (±0.02): self-descriptions
overclaim by +0.27 and correlate 0.36 with S, so lying about that channel changes little.
**TODO(fw_live_*)**: strict-accounting column (`success_strict`), `fallback_rate` for all ten on the fixed adapters,
and the 14B Magentic-One arm.

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
match a task's text to the right *specialty* paragraph before the supervisor ever ranks anyone. The framework-delta axis
is refilled from the 10-seed rerun when it closes (**TODO(fw_live_*)**, figure H2).

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

## 4. MIDIAN vs sequential halving, including SH and A (H6)

`sequential_halving_peer` spends MIDIAN's budget adaptively but learns only through MIDIAN's trimmed report channel.
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
**SH, A and SH+A on the same cells (variants_f1: 3 shapes × 4 β × 2 liar selections × 10 seeds = 240 paired units,
self-described).** Audits are what buy robustness; in-cohort halving buys nothing at β ≤ 0.25 and loses badly under
collusion.

| success | β=0 | β=0.1 | β=0.25 | β=0.5 all | β=0.5 random liars | β=0.5 low-skill liars | sd (units) |
|---|---|---|---|---|---|---|---|
| oracle | 0.723 | 0.723 | 0.723 | 0.723 | 0.723 | 0.723 | 0.118 |
| sequential_halving_peer | 0.722 | 0.722 | 0.718 | 0.540 | 0.678 | 0.402 | 0.165 |
| midian_sha (SH+A) | 0.670 | 0.670 | 0.670 | 0.663 | 0.663 | 0.663 | 0.098 |
| midian_a | 0.668 | 0.668 | 0.667 | 0.667 | 0.668 | 0.666 | 0.099 |
| flat_probe_argmax_online | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 | 0.097 |
| midian_v | 0.684 | 0.684 | 0.676 | 0.569 | 0.608 | 0.531 | 0.130 |
| midian | 0.668 | 0.664 | 0.648 | 0.598 | 0.627 | 0.569 | 0.109 |
| midian_sh | 0.670 | 0.670 | 0.648 | 0.474 | 0.522 | 0.426 | 0.137 |
| linucb_honest | 0.642 | 0.641 | 0.642 | 0.642 | 0.642 | 0.642 | 0.078 |

Paired deltas (mean [95% CI], cells won / non-tied): midian_a − midian **+0.023** [+0.018, +0.028], 175/214, entirely
from β = 0.25 (+0.019) and β = 0.5 (+0.069); at β = 0.5 with colluding low-skill liars midian_a beats plain MIDIAN in
30/30 cells (+0.097) and peer halving by +0.127 — the audits exclude the colluding reporters (at n = 1000 ≈ 99% of liars
and no honest reporter, bernoulli check) and the estimate stops moving with β. midian_sh − halving_peer **−0.060**
[−0.068, −0.052], 18/240 (−0.052 / −0.052 / −0.070 at β ≤ 0.25, every shape: bimodal −0.035, heavy_tail −0.066,
specialist −0.076): halving inside a cohort of 10 cannot reproduce halving over 1,000 — the cohort's best member is
found, but the cohort was random. midian_sh − midian: +0.002 / +0.006 / −0.000 at β ≤ 0.25 and **−0.124** at β = 0.5
(0/30 at low-skill collusion, −0.143): early elimination on poisoned reports. midian_sha − midian_a: +0.003 / +0.002 /
+0.003 / −0.004 — the audits repair SH's collapse (+0.189 over SH at β = 0.5) but SH adds nothing on top of A.
midian_v − midian +0.009 [+0.003, +0.015] here (+0.016 / +0.020 / +0.028 at β ≤ 0.25, −0.029 at β = 0.5). Costs are
unchanged for SH (48,000 probes, 432,000 reports, 6 messages and 30 comparisons per task, exponent n^0.14 by
construction) and 1.050× build probes for A / SH+A (50,382; audits are probes; online audits are reports).
Pre-registered: **V2-1 MISS** (both halves), **V2-2 HIT**, **V2-3 MISS** by −0.015 at β = 0 only (−0.000 / −0.008 / −0.004
elsewhere; inside MIDIAN's 0.070 seed envelope), **V2-5 MISS** (LinUCB is 0.025 below flat_online, not between it and
warm-start; it is flat in β, +0.000).

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
effect (§4 table).

## 5. Churn (H9)

churn_n1000: 10% or 30% of agents replaced in place every 200 tasks (fresh profiles, liars redrawn at rate β, probe
indices reset; the first task routed to a replaced agent the method has not re-probed scores 0), n = 1000, specialist +
heavy_tail, β ∈ {0, 0.25}, self-described, 5 seeds, Q = 1000; 20 units per arm and fraction (halving-rebuild 10–11: its
full re-probe per event is slow and its last units were still running at write time).

| arm | success, 10% churn | Δ vs same cells no churn | success, 30% churn | Δ vs no churn | repair probes / event (% of build) |
|---|---|---|---|---|---|
| oracle | 0.797 | — | 0.797 | — | — |
| sequential_halving_peer, rebuild | 0.759 (n=10) | +0.04* | 0.833 (n=11) | +0.12* | 44,928 (100%) |
| warm_start_bandit | 0.732 | −0.001 | 0.729 | −0.005 | 4,800 / 14,400 (10 / 30%) |
| flat_probe_argmax_online | 0.722 | −0.002 | 0.710 | −0.015 | 4,800 / 14,400 |
| sequential_halving_peer, stale | 0.721 | +0.00 | 0.589 | −0.13 | 0 |
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
event at 10%). **V2-6**: quality half HIT (−0.008 at 10%, within 0.03; halving-stale loses 0.13 at 30%, ≥ 0.05);
cost half MISS as written — repair is 10% of build per event at 10% churn, not ≤ 3% (the 3% was mis-derived: re-probing
10% of agents at full b is 10% of the build by definition), and halving-rebuild's repair is 9.4× MIDIAN's, not ≥ 10×.
Figure H9 shows success and cumulative probes vs task index.

## 6. Cost scaling and break-even (H4, H5)

Per task and at build, n = 1000, reports per probe in every arm:

| method | build probes | build reports | build msgs | msgs/task | comps/task | LLM calls/task |
|---|---|---|---|---|---|---|
| midian | 48,000 | 432,000 | 1,010 | 6 | 30 | 0 |
| midian_v | 47,840 | 430,560 | 1,010 | 2 | 1 | 0 |
| midian_v_r5 | 48,000 | 192,000 | 1,050 | 2 | 1 | 0 |
| midian_a | 50,382 (1.050×) | 432,000 (+ online audits as reports) | 1,010 | 6 | 30 | 0 |
| flat_probe_argmax (either) | 48,000 | 0 | 0 | 0 | 1,000 | 0 |
| sequential_halving_peer | 44,928 | 404,352 | 0 | 0 | 1 | 0 |
| declared_argmax | 0 | 0 | 1,000 | 0 | 1,000 | 0 |
| llm_supervisor | 0 | 0 | 1,000 | 22 | 20 | 1 |
| any framework (own selection) | 0 | 0 | 1,000 | 12 | 10 | ≥ 1 |
| framework + midian_v shortlist | 47,840 | 430,560 | 2,010 | 14 | 11 | ≥ 1 |

Exponents over n = 10² … 10⁷ (bernoulli calibrated to the measured S; replay 10⁴ … 10⁶; live 10² … 10⁴): MIDIAN
per-task comparisons/messages/hops = r·⌈log_r n⌉ (fitted n^0.14 [0.13, 0.15] on the mixed set, n^0.11 on bernoulli
alone); midian_v per-task n^0; flat / declared / CNP comparisons n^1.00; build probes n^1.03 for every probe-based
method (n·K·b), n^0.94 for midian_v; MIDIAN build reports n^1.03.

Break-even against a framework = build cost / per-task saving. Messages and comparisons: after the first task.
Messages + reports: (432,000 + 1,010 − 1,000) / (12 − 2 − 0) ≈ **43,000 tasks** for midian_v with per-probe reports
(was 15,900 under per-peer reports), ≈ 19,000 for midian_v_r5. LLM calls: ≈ 48,000 tasks at b = 3 against a one-call
framework (16,000 at b = 1; 5,000–8,000 against Magentic-One / CAMEL, which make several calls per task).
Figure H4 plots build + Q·per-task cost against success at Q ∈ {10², 10³, 10⁴, 10⁵} with the break-even Q marked (the framework points refresh when fw_live_* closes).

Supervisor latency (the one wall-clock table; frameworks' calls are never memoised; under shared-fleet load), medians
at n = 1000: AutoGen 0.7 s, CrewAI 0.6, Google ADK 1.0, OpenAI Agents 1.0, LangGraph 1.1, MAF 1.3, smolagents 1.6,
LlamaIndex 3.1, CAMEL 4.5, Magentic-One 6.8.

## 7. Budget, by channel (H8)

budget_sweep (n = 1000, β = 0.25, 3 shapes × 5 seeds, all 45 cells paired; both channels pooled for probe-only arms):

| method | b=1 | b=3 | b=10 |
|---|---|---|---|
| oracle | 0.723 | 0.723 | 0.723 |
| sequential_halving (trusted) | 0.479 | 0.722 | 0.723 |
| midian_v | 0.593 | 0.675 | 0.698 |
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
| bimodal | 0.568 | 0.568 | 0.562 (n=4) | 0.563 | 0.566 | 0.567 | 0.568 / 0.568 |
| heavy_tail | 0.739 | 0.741 | 0.758 | 0.722 | 0.701 | 0.712 | 0.604 / 0.604 |

Paired, frameworks − MIDIAN: bimodal **+0.002**, heavy_tail **−0.097** (2/8 cells); halving − MIDIAN +0.040 on heavy_tail.
With ten probes per cell every probe method sits on the oracle on bimodal, i.e. the v1 −0.04 was the 4-valued estimate,
not the tree. **V2-9 HIT.** (LangGraph and AutoGen give identical numbers here because with the same shortlist both
supervisors pick the same agent on these tasks.)

**n = 10,000 at b = 3 on the live channel (live_n10k_v2: specialist, β ∈ {0, 0.25}, 3 seeds, Q = 300).** Replaces the
2026-09-02 b = 1 / programmatic row (where declared_argmax ≈ oracle):

| arm | β=0 | β=0.25 | build probes | reports | msgs / comps per task |
|---|---|---|---|---|---|
| oracle | 0.859 | 0.859 | | | |
| midian_v | 0.813 | 0.806 | 479,840 | 4,318,560 | 2 / 1 |
| midian_sh | 0.794 | 0.772 | 480,000 | 4,320,000 | 8 / 40 |
| midian | 0.786 | 0.790 | 480,000 | 4,320,000 | 8 / 40 |
| midian_a | 0.784 | 0.786 | 503,806 | 4,320,000 | 8 / 40 |
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
The two peer-reported-halving units at n = 10k were still running at write time (**TODO(live_n10k_v2, halving row)**).

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

**v2 (TARGETS_rte_v2.md), ten** (`rte.analyze … --out results/v2_targets`; paired on `label`, plain MIDIAN only;
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

Six hits, three misses (SH does not help; SH+A adds nothing over A; a history-only LinUCB is below a flat scan), one
split (churn: quality as predicted, repair-cost arithmetic wrong in the pre-registration), one reported.

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
| midian r=10 δ=1/3 | 0.790 | 0.789 | 0.743 | 0.788 | 432k | 6 / 30 |
| midian r=10 δ=0 | 0.789 | 0.789 | 0.778 | 0.788 | 432k | 6 / 30 |
| midian r=5 δ=1/3 | 0.792 | 0.791 | 0.728 | 0.789 | 192k | 10 / 25 |
| midian r=20 δ=1/3 | 0.782 | 0.785 | 0.765 | 0.784 | 912k | 6 / 60 |
| midian_v r=5 | 0.822 | 0.821 | 0.730 | 0.820 | 192k (per probe) | 2 / 1 |
| midian_v r=10 | 0.816 | 0.816 | 0.749 | 0.816 | 431k (per probe) | 2 / 1 |
| midian_v r=20 | 0.785 | 0.786 | 0.764 | 0.785 | 863k (per probe) | 2 / 1 |
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
build reports (912,000) and per-task comparisons (60) for plain, 898,016 reports and 1 comparison for midian_v. Larger
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

**TODO(fw_live_*)**: per framework and n, `picks / fallbacks / failures / bad_name`, `fallback_rate`, lenient vs
strict success (from `method_stats`).
