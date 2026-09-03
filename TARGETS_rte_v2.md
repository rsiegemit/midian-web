# TARGETS_rte_v2.md — pre-registered expectations for the v2 work order (v1 in TARGETS_rte.md stays frozen)

Committed 2026-09-03 before any v2 grid was launched. Every new mechanism is a labeled variant; plain MIDIAN's
parameters do not change. Verdicts are HIT / MISS / WITHIN_FLOOR (delta inside MIDIAN's own seed envelope), reported
as measured.

## Phase 1 variants
- **V2-1 MIDIAN-SH** (`midian_sh`: successive halving inside each cohort on the trimmed peer-report channel, same
  r·b probes per (cohort, family)): (i) within 0.02 of `sequential_halving(peer_reported)` at β ≤ 0.25, all shapes,
  self-described channel; (ii) ≥ plain MIDIAN − 0.02 at β = 0.5, collude, low-skill-first; (iii) per-task cost
  unchanged (comparisons/messages exponent ≈ n^0.14).
- **V2-2 MIDIAN-A** (`midian_a`: 5% of reports audited by re-running the same index-seeded probe instance; a reporter
  with 2 mismatches is excluded; audits charged as probes; 5% of routed outcomes audited online): (i) β = 0.5 collude
  loss vs β = 0 ≤ 0.02 on specialist (plain: 0.05–0.09); (ii) β ≤ 0.25 unchanged within 0.01; (iii) build probes
  ≤ 1.05× plain.
- **V2-3 MIDIAN-SH+A** (`midian_sha`): ≥ max(MIDIAN-SH, MIDIAN-A) − 0.01 at every β.
- **V2-4 stratified cohorts** (`midian(stratify=True)`, strata by measured-probe mean): reported as measured vs random
  cohorts on specialist + heavy_tail at β ∈ {0, 0.5}; no directional expectation.
- **V2-5 LinUCB-honest** (`linucb_honest`, features = own observed per-family history only, warm-up = n·K·b):
  replaces UCB/Thompson in the headline; expected between flat_probe_argmax_online and warm_start_bandit at β = 0,
  flat in β.

- **V2-11 MIDIAN-VA** (`midian_va` = MIDIAN-V's verified promotion + cached root pick with MIDIAN-A's report audits; added
  2026-09-03 15:00 before any run): (i) ≥ max(MIDIAN-V, MIDIAN-A) − 0.01 at every β on the self-described channel;
  (ii) at β = 0.5 collude low-skill-first within 0.02 of MIDIAN-A (i.e. audits repair V's exposure); (iii) build probes ≤ 1.05× V,
  per-task cost = V's (1 comparison, 2 messages).

## Phase 2 grids
- **V2-6 churn_n1000** (10% / 30% of agents replaced every 200 tasks): MIDIAN within 0.03 of its no-churn success at
  10% churn with repair probes ≤ 3% of build per event; halving-stale loses ≥ 0.05 at 30%; halving-rebuild matches
  quality at ≥ 10× MIDIAN's repair cost.
- **V2-7 live_n10k_v2** (n = 10,000, b = 3, self-described): MIDIAN ≥ flat_probe_argmax_online − 0.02; frameworks
  below flat_probe_argmax_online on specialist by ≥ 0.10.
- **V2-8 midian_v_replication** (10 fresh seeds): MIDIAN-V − MIDIAN = +0.02 ± 0.02 at β ≤ 0.25; exposure at β = 0.5
  collude reported as measured.
- **V2-9 budget_b10_shapes**: on bimodal at b = 10 the MIDIAN-vs-framework gap (−0.04 at b = 3) closes to within
  ±0.02; on heavy_tail MIDIAN ≥ frameworks + 0.03.
- **V2-10 internals_v2** (r × δ × collude × liar selection at β = 0.5, plain MIDIAN and MIDIAN-A): trimming (δ = 1/3
  vs 0) hurts plain MIDIAN under collusion by ≥ 0.02 and does not hurt MIDIAN-A (|Δ| ≤ 0.02).

## Phase 0 corrections (no expectations; recorded so verdict text can be checked)
- Per-channel tables: programmatic is the upper bound (S + N(0, 0.05)); the self-described channel is the live one.
- Framework fallbacks re-accounted as failures (success 0) when the framework answers instead of delegating; both
  accountings reported; `fallback_rate` column in the headline.
- Reports charged per (peer, member, family, probe) in every arm.
