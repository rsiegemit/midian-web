# Evaluation-protocol audit (audit-protocol2, 2026-09-23)

Scope: selection on test data, budget fairness, pairing/replication, data hygiene, statistics, hidden arms.
Read-only audit; findings appended as confirmed.
Names follow the 2026-09-24 rename (CHANGES_AND_ERRATA §8g): MIDIAN = both defenses (formerly VA); MIDIAN w/o defenses = the
plain tree; MIDIAN w/o audits / w/o verification = the single-defense ablations. Code references are to the code audited.

## Findings

### F1. HIGH: peer-reported sequential halving beats MIDIAN in every honest b = 3 cell of A and B, and no figure draws it

**Evidence.**
- METHODS.md lists `sequential_halving(peer_reported=True)` as a pre-registered SPEC §6 rival with status "reported (peer halving)". It reads the same peer-report channel as MIDIAN, and its build spend is n·b per family.
- It is not one of the 7 `ARMS` and is not in either `POOLS` entry (`scripts/condensed_figs.py:27-34`).
- It is also removed from every figure by `HIDE_HALVING = True`, marked "TEMPORARY (2026-09-22, user request)" (`scripts/extra_figs.py:120,127`).
- The guide's statement "HIDE_HALVING has no effect on A/B" (`docs/FIGURE_DATA_GUIDE.md:2532`) is true mechanically. It hides the substantive point: the arm was never eligible to appear in A or B at all.
- Paired per seed, from the same rows `seed_tables.tables()` reads, with the same specialist and self-described filters:

| cell (b = 3, honest) | peer halving | MIDIAN | Δ | seeds where halving wins |
|---|---|---|---|---|
| live 10² | 0.810 | 0.782 | +0.029 | 10/10 |
| live 10³ | 0.860 | 0.813 | +0.047 | 10/10 |
| live 10⁴ | 0.863 | 0.811 | +0.052 | 3/3 |
| live 10⁵ | 0.863 | 0.836 | +0.028 | 3/3 |
| RouterEval 5k | 0.882 | 0.706 | +0.177 | 3/3 |
| bernoulli 10⁷ (matrix_success.csv, 100 seeds) | 0.846 (= oracle 0.846) | 0.801 | +0.045 | |
| replay 10⁶ (matrix_success.csv, 100 seeds) | 0.790 (= oracle 0.790) | 0.762 | +0.028 | |

- LLMRouterBench is the one exception: 0.664 vs 0.669.
- Under the β = 0.5 cartel at b = 3, peer halving collapses:

| cell | peer halving | MIDIAN |
|---|---|---|
| live 10² | 0.545 | 0.768 |
| live 10³ | 0.656 | 0.807 |
| live 10⁴ | 0.721 | 0.810 |
| live 10⁵ | 0.724 | 0.828 |
| RouterEval | 0.564 | 0.710 |
| LLMRouterBench | 0.464 | 0.666 |
| bernoulli | 0.708 | 0.794 |
| replay | 0.456 | 0.760 |

- It has no rows at b = 1 or 5 on live, RouterEval or LLMRouterBench; it was not in `rivals_b_*`.

**Affected.**
- The honest (solid) bars of A and B at b = 3.
- Any claim that MIDIAN is the best probe-budgeted method when there are no liars.

**Direction.** Favours MIDIAN: the strongest honest rival, which reaches the oracle, is absent from both headline figures. The true picture is a clean trade-off: halving wins honest, and MIDIAN wins under the cartel by 9–30 pp. Drawing both is more persuasive than hiding one.

**Fix.**
- Add peer halving as a drawn arm, or as a member of a "best probe-budgeted" pool.
- Add it to `rivals_b_*` for b = 1 and 5.
- State in the caption that the trusted-observer arm is withdrawn (erratum 26), not the peer arm.

### F2. HIGH: dropping the pre-registered warm-start bandit (n0 = 5) from the best-bandit pool lowers "best bandit" where it matters, and flips two bar orders against MIDIAN

**Evidence.**
- Today's change is `POOLS["best_bandit"] = [a for a in BANDIT if a != "warm_start_bandit"] + ["warm_start_bandit[n0=0.5]"]` (`scripts/condensed_figs.py:33-34`).
- n0 = 0.5 was tuned on live n = 10³, honest, b = 3, seeds 11-15 (`configs/grid.yaml:1029-1041`). It is applied unchanged to every family, both regimes and every b.
- I recomputed the cross-fitted pool with `seed_tables.crossfit`, using the same pool, the same `NOT_RUNNABLE` rule and the same rows, with and without n0 = 5 added back:

| cell | figure's best bandit (no n0 = 5) | with n0 = 5 in pool | MIDIAN |
|---|---|---|---|
| RouterEval 5k honest b = 1 | 0.740 | 0.822 | 0.606 |
| RouterEval 5k honest b = 3 | 0.709 | **0.822** | 0.706 |
| RouterEval 5k honest b = 5 | 0.743 | **0.829** | **0.758** (order flips) |
| live 10⁴ cartel b = 1 | 0.588 | **0.709** | **0.670** (order flips) |
| live 10⁴ honest b = 1 | 0.679 | 0.699 | 0.668 |
| live 10⁴ cartel b = 3 | 0.773 | 0.783 | 0.810 |
| live 10⁵ cartel b = 3 | 0.764 | 0.778 | 0.828 |
| LLMRouterBench honest b = 1 | 0.653 | 0.675 | 0.644 |

- RouterEval values are raw; B divides by the oracle, 0.902.
- The comparison is not one-sided. With n0 = 5 added, the cross-fit is lower in some cells: live 10³ cartel b = 3 0.764 vs 0.769; LLMRouterBench honest b = 3 and 5, 0.676 / 0.686 vs 0.679 / 0.693. The net effect on MIDIAN's margin is concentrated in RouterEval honest and live 10⁴ b = 1.
- On bernoulli 10⁷ and replay 10⁶ (matrix_success.csv), n0 = 5 is higher honest:

| cell (honest b = 3) | n0 = 5 | n0 = 0.5, figure |
|---|---|---|
| bernoulli | 0.839 | 0.818 |
| replay | 0.786 | 0.768 |

- n0 = 0.5 is higher in bernoulli cartel (0.779 vs 0.768).
- The tuning is a one-cell choice, live honest only, and it transfers badly to the declared-reliable RouterEval pool. There, a larger n0 (more trust in the declarations) is exactly right when there are no liars.
- CHANGES_AND_ERRATA §8d says "Reported numbers are unchanged (the rival is as pre-registered)" (`CHANGES_AND_ERRATA.md:411-423`). The figures now contradict it: the pre-registered arm is removed from the only place it could appear in A and B. §8d does not record the tuning run or the pool change (the guide already notes this at `FIGURE_DATA_GUIDE.md:1250,1428`).

**Affected.**
- A and B "best bandit" bars.
- Any claim of the form "MIDIAN ≥ best bandit".

**Direction.** Favours MIDIAN overall. The pool shrank by a pre-registered member, and in the cells where it matters most that member is the stronger one.

**How to present it.**
- The defensible pool is BANDIT plus the tuned arm: add the variant, never drop the pre-registered one. The cross-fit already guards against the winner's curse from a larger pool.
- If only one warm-start arm may be shown, show n0 = 5 (pre-registered). Label n0 = 0.5 "post hoc, tuned on live 10³ honest seeds 11-15" and report it in a supplementary panel.
- Record the change and its date in DEVIATIONS.md.

### F3. MEDIUM: budget fairness. Every rival in A and B spends exactly 1.00× n·K·b or less; MIDIAN spends 1.03–1.07×, and peer halving underspends

**Evidence.** `build_probes / (n·K·b)` from every non-framework row in the A and B source grids (rows.d):

| arm | b = 1 | b = 3 | b = 5 |
|---|---|---|---|
| MIDIAN, mean (live, RouterEval 5k and LLMRouterBench alike) | 1.048–1.050 | 1.029–1.033 | 1.038–1.041 |
| MIDIAN, maximum | 1.070 (LLMRouterBench) | 1.038 | 1.048 |

- MIDIAN w/o defenses, flat probe argmax online, kNN, LinUCB, UCB, Thompson and both warm-start arms: exactly 1.000 in every cell.
- peer-reported halving: 0.83–0.97×. It underspends and still beats MIDIAN honest (F1).
- verify_on_claim: 0 build probes, and 1.03 / 2.33 / 2.42 route-time probes per task at 10³ / 10⁴ / 10⁵. Over the whole stream that is still far below n·K·b.
- The overspend is by design: the audits are 5 % of level-0 instances (DEVIATIONS.md:692, "Audited builds exceed the n*K*b cap by the audit rate by design"). The budget check only logs a warning (guide §1.5).
- Size of the effect: on live, MIDIAN's b = 3 → 5 step (+67 % probes) is worth +0.02 to +0.03. So +3–5 % probes is worth roughly +0.001 to +0.002, which is immaterial to any bar order.
- MIDIAN w/o defenses at b = 1 equals MIDIAN at b = 1 in 4 cells (guide 0.3 #7). There MIDIAN spends 5 % more for nothing.
- Budget asymmetry for frameworks (E–H):
  - the frameworks spend 0 probes;
  - the MIDIAN reference line spends 1.03 × n·K·b probes plus reports;
  - the frameworks pay in supervisor calls instead.
- This is the design, but a caption must not call E–H a budget-matched comparison. The `retrieval=midian` (MIDIAN-cohort) shortlist arm inherits MIDIAN's probe spend, while the TF-IDF, BM25 and dense arms spend none.

**Direction.** Slightly favours MIDIAN against every probing rival. The effect is immaterial in size, but "same budget" is literally false.

**Fix.** Either:
- say "MIDIAN spends ≤ 1.07 × n·K·b (audits)" in every caption that says "same budget"; or
- give the rivals 1.05 × b.

### F4. MEDIUM: a rival is handicapped by the probe index. `World._probe` gives repeated agents in one call the SAME instance, and TrueSkill hits this about half the time

**Evidence.**
- `rte/world.py:317-318` computes `k = self._probe_idx[agents, families] + arange(reps)` and then `self._probe_idx[agents, families] += reps`.
- Under numpy fancy indexing, every duplicate (agent, family) in one call reads the same k, and the counter advances only once. Duplicates therefore get the same index-seeded instance, which yields the same memoised outcome on live, yet each is charged as a fresh probe.
- `trueskill_per_family.build` (`rte/methods/trueskill_per_family.py:41-48`) draws `a1`, `a2` with replacement: n·b/2 draws from n agents per family. At b = 3 roughly 48 % of its probes are repeats, and each repeat is an identical outcome.
- Every other prober passes unique agents per call (`_est.py`, `flat_probe_argmax.py:39`, `warm_start_bandit.py:25`, `sequential_halving.py:22`), so only TrueSkill is hit.
- Data at live 10³, honest, b = 3:

| rows | TrueSkill | written |
|---|---|---|
| seeds 1-5, `live_f1_n1000` | 0.774 / 0.676 / 0.792 / 0.773 / 0.711, mean 0.745 | 09-02 07:33-08:20, before the index-seeded instance change, DEVIATIONS.md:633 |
| seeds 6-10, `pool_seeds_n1000` | 0.512 / 0.619 / 0.600 / 0.643 / 0.659, mean 0.607 | 09-23 |

- UCB, Thompson, warm-start, cluster-head and flat-NSW are unchanged across the same two seed blocks (±0.01). `trueskill_per_family.py` was not edited after 09-02 02:02.
- The pooled 10-seed TrueSkill mean in A therefore mixes two behaviours.

**Affected.**
- The best-bandit pool at live 10², 10³ and 10⁴, RouterEval and LLMRouterBench. TrueSkill is rarely picked; it is picked once at live 10⁴ cartel b = 1.
- Any table that reports TrueSkill.
- The same trap would hit any future method that probes with replacement.

**Direction.** Handicaps a rival (favours MIDIAN), though the effect on the drawn bars is small.

**Fix.** Advance the counter with `np.add.at`, and give duplicates k, k+1, and so on. The alternative is to state the caveat and exclude TrueSkill.

### F5. MEDIUM: cross-grid disagreement is mostly the old per-method probe instances in `live_core_n100` and early `live_f1_n1000`, and it is averaged into MIDIAN w/o defenses, not into MIDIAN

**Evidence.**
- `seed_tables._table` and `bar_figs` average one (seed, arm)'s rows over every grid that has them.
- At live 10², β = 0.5 cartel, b = 3, MIDIAN w/o defenses per seed:

| seed | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| `live_core_n100` | 0.693 | 0.711 | 0.697 | 0.684 | 0.760 |
| `fw_live_n100_lowskill` and `learned_n100` (identical) | 0.769 | 0.723 | 0.725 | 0.701 | 0.766 |

- The seed-1 gap of 0.076 is the "0.076" in guide 0.3 #8.
- `live_core_n100` rows date from 09-02 06:54-18:36. That is the day of the "k-th probe is the same instance for every method" change (DEVIATIONS.md:633: "Live rows written before this change used per-method instances").
- `live_core_n100` carries only MIDIAN w/o defenses (r = 10 and 5), MIDIAN w/o audits (r = 10 and 5), flat frozen, declared, random and oracle. It has no MIDIAN. MIDIAN is bit-identical across its two grids at every seed.
- The effect on A:

| A bar at live 10², b = 3 | as drawn | post-change grids only |
|---|---|---|
| MIDIAN w/o defenses, cartel | 0.736 | 0.741 |
| MIDIAN w/o defenses, honest | 0.774 | 0.777 |
| MIDIAN w/o audits, honest | 0.785 | 0.782 |

- At live 10³, early `live_f1_n1000` rows (09-02) disagree with `variants_f1` by up to 0.020 (flat probe argmax online, seed 2). Frozen flat differs by up to 0.075.
- Residual disagreement between two post-change grids is ≤ 0.013 (MIDIAN w/o defenses, seeds 7-8, `fw_live_n100` vs `learned_n100`). This is consistent with non-memoised fresh generations on the batched vLLM fleet; the root cause was not proven.
- The pairing claim in guide 0.2 ("every method in a (cell, seed) unit sees … the same probe instances") is false for these early rows.

**Direction.**
- Makes MIDIAN w/o defenses look worse by 0.3-0.5 pp at live 10², which inflates "MIDIAN over MIDIAN w/o defenses".
- It is mixed for the rivals: frozen flat is not drawn, and flat online is only in `live_f1_n1000` / `variants_f1` at 10³.
- It does not flip any drawn order.

**Fix.** Drop the pre-change grids (`live_core_n100`, and `live_f1_n1000` rows written before the change) from `LIVE_GRIDS`, or at least exclude them for arms that also have post-change rows.

### F6. CRITICAL: the regime shown in A and B, together with the hidden peer halving, is the only configuration in which MIDIAN looks dominant

**Evidence.**
- Setup:
  - The comparison is peer halving minus MIDIAN, paired per seed.
  - Cells are live, specialist, self-described, b = 3.
  - Rows come from the same grids as A, and every (β, liar_select) regime the grids contain is included.

| n | β = 0 | β = 0.1 (lsf / rand) | β = 0.25 (lsf / rand) | β = 0.5 random | **β = 0.5 low-skill-first (A's "cartel")** |
|---|---|---|---|---|---|
| 10² (10 seeds) | +0.028 (10/10) | +0.048 / +0.049 (10/10, 10/10) | +0.041 / +0.044 (10/10, 10/10) | −0.030 (2/10) | **−0.223 (0/10)** |
| 10³ (10 seeds) | +0.047 (10/10) | +0.067 / +0.062 (10/10) | +0.064 / +0.067 (10/10) | +0.003 (5/10) | **−0.151 (0/10)** |
| 10⁴ (3 seeds) | +0.052 (3/3) | n/a | +0.072 / +0.060 (3/3) | +0.048 (3/3) | **−0.089 (0/3)** |
| 10⁵ (3 seeds) | +0.028 (3/3) | n/a | +0.076 / +0.043 (3/3) | +0.022 (3/3) | **−0.103 (0/3)** |

- In parentheses: the number of seeds where halving wins.
- MIDIAN beats the pre-registered peer-report rival in 5 of 24 regimes: β = 0.5 low-skill-first at every n, and β = 0.5 random at 10².
- It loses, usually on every seed, in the other 19.
- A and B show exactly two regimes, β = 0 and β = 0.5 low-skill-first (`condensed_figs.REG`, `seed_tables._regime`), and they do not draw halving (F1).
- At 10², flat probe argmax online (0.780) is above MIDIAN at β = 0.1 and at β = 0.25 (0.759-0.761).
- Status of the chosen regime:
  - SPEC.md:55 pre-registers β ∈ {0, 0.1, 0.25, 0.5} and both liar selections ("Both run"). The regime is therefore not invented.
  - Showing only the extreme cartel as "the" liar regime is a post-hoc presentation choice.
  - The grid comment calls it "the regime where the report channel matters most" (`configs/grid.yaml:145`).

**Affected.**
- Figures A and B, and E–H's reference line.
- Any sentence of the form "MIDIAN is robust and best".

**Direction.** Strongly favours MIDIAN.

**Fix.** Present the honest result and it holds. MIDIAN is the only probe-budgeted method that is flat in β and survives a majority low-skill cartel. It is not the best method at β ≤ 0.25, where peer-reported halving (same channel, ≤ same budget) is 3-8 pp better.
- Show peer halving.
- Add at least a β = 0.25 panel, or a β-sweep line figure.
- Label the cartel regime as the adversarial extreme.

### F2 (update, 16:10 tree). HIGH, bordering CRITICAL: `linucb_honest` is now ALSO dropped from the bandit pool; the two drops together flip 9 bar orders in MIDIAN's favour

**Evidence.**
- The pool is now `POOLS["best_bandit"] = [a for a in BANDIT if a not in ("warm_start_bandit", "linucb_honest")] + ["warm_start_bandit[n0=0.5]"]` (`scripts/condensed_figs.py:33-35`, mtime 16:10). The comment reads "linucb_honest out: its bonus picks the WEAKEST tied agent (audit 2026-09-23)".
- The 14:13 figure CSVs still use the older pool.
- LinUCB was the cross-fitted pick at:
  - live 10² (every b, both regimes);
  - LLMRouterBench (every cell);
  - RouterEval cartel b = 1 and 5.
- I recomputed with `seed_tables.crossfit` on the same rows, adding bernoulli 10⁷ and replay 10⁶ from their rows.csv (100 seeds; replay shapes pooled as in the figure).

| cell | MIDIAN | pool now (−n0 = 5, −LinUCB) | 14:13 figure (−n0 = 5) | pre-registered BANDIT | BANDIT + tuned n0 = 0.5 |
|---|---|---|---|---|---|
| live 10⁴ cartel b = 1 | 0.670 | **0.588** | 0.588 | 0.709 | **0.709** |
| RouterEval honest b = 5 | 0.758 | **0.743** | 0.743 | 0.829 | **0.829** |
| RouterEval cartel b = 1 | 0.600 | **0.579** | 0.668 | 0.668 | **0.668** |
| LLMRouterBench honest b = 3 | 0.669 | **0.648** | 0.679 | 0.676 | **0.676** |
| LLMRouterBench cartel b = 1 | 0.642 | **0.616** | 0.653 | 0.653 | **0.653** |
| LLMRouterBench cartel b = 3 | 0.666 | **0.643** | 0.679 | 0.679 | **0.679** |
| LLMRouterBench cartel b = 5 | 0.690 | **0.662** | 0.693 | 0.693 | **0.693** |
| bernoulli 10⁷ honest b = 5 | 0.826 | **0.822** | 0.822 | 0.839 | **0.839** |
| replay 10⁶ honest b = 5 | 0.777 | **0.760** | 0.760 | 0.787 | **0.787** |

- In each of these 9 cells, the current pool puts best bandit BELOW MIDIAN, and the full pool (BANDIT + tuned) puts it ABOVE.
- The drops also lower the bar without flipping it:

| cell | current pool | full pool |
|---|---|---|
| live 10² honest b = 3 | 0.742 | 0.764 |
| live 10² honest b = 5 | 0.757 | 0.793 |
| live 10² cartel b = 5 | 0.765 | 0.793 |
| RouterEval honest b = 3 | 0.709 | 0.822 |

- Among the rows compared, no pool change raises best bandit relative to MIDIAN in a way that flips an order against MIDIAN.
- On the LinUCB rationale:
  - A rival with a tie-breaking defect should be FIXED and rerun, for example with a random tie-break on the bonus. The same logic applies to MIDIAN w/o defenses, whose "argmax breaks ties blindly" was fixed by adding verification (DEVIATIONS.md:606).
  - Deleting the rival from the pool after seeing that it wins some cells is selection on test data.
  - LinUCB is a v2 labelled rival, and TARGETS_rte_v2 V2-5 says it "replaces UCB/Thompson in the headline". Removing it contradicts that pre-registration.
  - It is honest-only in its context and reads no declarations, so the "weak tied agent" pathology equally affects its cartel and honest bars. It is not a reason to exclude it from one regime's comparison.

**Direction.** Strongly favours MIDIAN.

**Recommendation.**
- The pool should be the pre-registered BANDIT plus labelled post-hoc variants. Cross-fitting already protects against the winner's curse, so a larger pool is safe.
- If a member is believed broken, fix it and rerun. Never subtract it.
- Every pool edit needs a dated DEVIATIONS entry and a figure caption note.

### F7. HIGH (labelling): the headline arm and several design choices are post hoc, and A and B carry no post-hoc marks

The post-hoc elements are listed below, with whether each is labelled as such anywhere a reader of A and B would see it.

| element | status | labelled? |
|---|---|---|
| MIDIAN as the headline arm (A/B first bar; the E–H reference line) | SPEC.md:257 names the plain tree (now MIDIAN w/o defenses) vs the frameworks as the headline. The full MIDIAN was pre-registered as a *labelled v2 variant* (TARGETS_rte_v2 V2-11, 2026-09-03 15:00, "before any run"), and it was designed after v1 showed the plain tree losing to peer halving and failing under the cartel. Its own pre-registered target V2-11 was scored **MISS** (RESULTS_rte_v2.md:664). Promoting it to the headline is post hoc. | METHODS.md says "v2 V2-11". Figures: no. |
| MIDIAN w/o audits (verification, inside MIDIAN) | post hoc, 2026-09-02 (DEVIATIONS.md:606) | DEVIATIONS: yes. Figures: no. |
| tuned warm-start bandit n0 = 0.5 | tuned on seeds 11-15, live 10³ honest (grid.yaml:1029-1041). **Seeds disjoint: verified.** Every reported live 10³ cell uses seeds 1-10; tune rows are seeds 11-15 only; tuning values reproduced (0.7624 for n0 = 0.5 vs 0.7448-0.7457 for n0 = 1/2/5). n0 = 0.5 is the edge of the searched grid {0.5, 1, 2, 5}. | grid.yaml comment only. Not in CHANGES §8d or DEVIATIONS. |
| dropping pre-registered warm-start n0 = 5 from the pool (2026-09-23) | post hoc | code comment only (F2) |
| dropping `linucb_honest` from the pool (2026-09-23, 16:10) | post hoc, contradicts V2-5 | code comment only (F2 update) |
| `HIDE_HALVING` (2026-09-22, "TEMPORARY, user request") | post hoc, removes a reported rival | code comment only (F1) |
| β = 0.5 low-skill-first as the only liar regime in A/B | regime pre-registered (SPEC.md:55); selecting it alone is post hoc (F6) | no |
| B: "largest n per family with both regimes" | a mechanical rule in code (`fig_B`), not chosen from results. It does pick the scale where MIDIAN's margin over flat is largest on live: 10⁵ honest b = 3 is +0.087, vs +0.002 at 10². On live that bar has only 3 seeds. | caption should say so |
| cross-fit "best of pool" (`seed_tables.crossfit`) | **correct: excludes the scored seed** (`T.drop(s)`; picks only among arms that ran on s). At 3 seeds the pick uses 2 seeds, so selection noise pulls the rival bar BELOW the ex-post best arm (see F8). | yes (docstring) |
| MIDIAN (every variant) r = 10, δ = 1/3 | pre-registered (SPEC §5 `build_midian(view, r=10, ...)`) | clean |
| retrieval shortlists other than TF-IDF (E–H) | post hoc, dated in METHODS/DEVIATIONS | METHODS: yes |

**Direction.** Every unlabelled post-hoc choice in this list either removes a rival or selects the regime in which MIDIAN wins.

### F8. MEDIUM: statistics

- **3-seed cells.** These are live 10⁴ and 10⁵, RouterEval 5k and the LLMRouterBench b-grids (5 seeds). Their percentile bootstrap has ≤ 10 distinct resample means and under-covers (guide 0.3 and 4.5 #18). A and B draw these CIs with the same glyph as 100-seed CIs.
- **The cross-fit penalises pooled rivals but not MIDIAN.**
  - The "best of pool" bar is an honest estimate of the *selection procedure*, and with few seeds the procedure picks wrong arms.
  - Measured as cross-fit minus the ex-post best single pool member on the same seeds: mean −0.007 at 3 seeds, −0.005 at 5, −0.003 at 10. The worst cases:

| cell | penalty |
|---|---|
| live 10⁴ honest b = 1, learned pool | −0.076 |
| live 10⁴ cartel b = 1, bandit pool (TrueSkill picked once) | −0.068 |
| live 10⁴ cartel b = 1, learned pool | −0.052 |
| LLMRouterBench cartel b = 1, learned pool | −0.044 |

  - MIDIAN is a single arm chosen once across all cells, from the MIDIAN family, by the authors (F7). It pays no such penalty.
  - The cross-fit is the right fix for the winner's curse, but on 3 seeds it biases the rival bars downward. The caption should say that the rival bar is "best-of-pool chosen out-of-sample", and the ex-post best member should be reported beside it.
- **No multiplicity control.** RESULTS and the README make dozens of "best at every n" or "every framework" claims across cells, regimes and b, with no correction and no paired-difference CIs. Every per-arm CI is a marginal CI, although the design is paired within a unit.
  - The fix is to report paired MIDIAN − rival differences with seed-bootstrap CIs, which is what `extra_figs.save`'s footer promises for "paired-delta bars".
  - These paired differences are much tighter than the marginal CIs. For example, peer halving − MIDIAN is positive on 10/10 seeds at every live 10² and 10³ β ≤ 0.25 regime (F6).
- **G's whisker** is 1.96·sd/√k across frameworks: a normal interval (≈ 17 % narrower than t₇) that measures framework spread, not seed uncertainty (guide 0.3 #14). Use a t interval, or bootstrap frameworks × seeds.
- **Unsupported or over-broad claims.**
  - RESULTS.md:474 "Four results, each on 200-1000 seeds": the n = 10⁷ / 10⁶ rungs have 100 seeds.
  - RESULTS.md:475 "MIDIAN is the best non-oracle arm under the cartel at every n": true at b = 3 for β = 0.5 low-skill-first only, silently excluding the withdrawn trusted halving (0.846 > 0.794 at bernoulli 10⁷) and route-to-3.
    - At b = 1 under the same cartel, MIDIAN (0.675) is below verify_on_claim (0.749), warm-start n0 = 5 (0.756), declared argmax (0.724) and flat online (0.677) on bernoulli 10⁷.
    - On replay 10⁶ at b = 1 it is below flat online too.
    - The claim needs "at b ≥ 3".
  - FIGURE_DATA_GUIDE.md §2.7.3 says the tuned bandit has "no rows yet". This is stale: rows exist for every live n at b = 1, 3 and 5, except live 10⁵ b = 5.

### F9. Hidden arms: which excluded arm beats MIDIAN in a cell that A or B shows (paired, same seeds)

**Method.** Every label in the per-seed tables was compared with MIDIAN on common seeds. The cells are live 10²–10⁵, RouterEval 5k and LLMRouterBench (rows.d), and bernoulli 10⁷ and replay 10⁶ (rows.csv), for b ∈ {1, 3, 5} and both regimes.

- **Peer-reported halving: HIGH, beats MIDIAN.**
  - At b = 3 honest in live 10², 10³, 10⁴ and 10⁵, RouterEval 5k, bernoulli 10⁷ and replay 10⁶.
  - By +0.028 to +0.177. See F1 and F6.
- **Trusted-observer halving.** Withdrawn by erratum 26; correctly excluded. Its values equal peer halving in the honest regime.
- **Warm-start n0 = 5 and LinUCB (pool members removed): HIGH.** See F2.
- **verify_on_claim: LOW** (reported per METHODS, never drawn in A/B).
  - It beats MIDIAN at b = 1 and 3 honest on bernoulli and replay, where declared ≈ truth. There declared argmax, which is drawn, is higher still.
  - It beats MIDIAN at b = 1 cartel on bernoulli (0.749 vs 0.675) and replay (0.692 vs 0.649).
  - On live it never beats MIDIAN.
  - Hiding it changes no drawn order, because declared argmax already exceeds MIDIAN in those B bars.
- **route_to_k_majority, cnp_self_bid, declared_softmax: LOW.**
  - They beat MIDIAN where declared argmax already does: honest bernoulli, replay and RouterEval, and b = 1 cartel.
  - Route-to-3 is excluded justifiably (3 executions per task).
- **MIDIAN w/o audits, MIDIAN w/o verification, the audited successive-halving variant (withdrawn 2026-09-24), MIDIAN w/o audits and w/o defenses at r = 5: LOW.** They beat MIDIAN by ≤ 0.011 in a few cells, including:
  - live 10² cartel b = 3: A +0.009 (6/10);
  - live 10³ honest b = 3: v_r5 +0.006;
  - LLMRouterBench cartel b = 3: A +0.011.

  Hiding them flatters "MIDIAN is the best variant of the tree", not MIDIAN versus the rivals.
- **Direction.** Only the peer-halving and pool-member exclusions change what a reader of A/B concludes, and both favour MIDIAN.

### F10. LOW / MEDIUM: data-hygiene details

- **rows.d + rows.csv de-duplication.**
  - Before today's 16:04 edit, `seed_tables.rows` deduplicated on a `rid` column that rows.d JSONs do not carry. Every rows.d row got rid = NaN, so `drop_duplicates("rid")` collapsed all rows.d rows into ONE and kept rows.csv.
  - Rows.csv holds the same rids in every A/B grid I checked (1:1, 0 extras), so the tables were rows.csv + 1 stray duplicate row per grid. That stray row double-weights one (seed, arm) average. The effect is negligible, but the 14:13 figure CSVs predate the fix: regenerate them.
  - The current code (rid = file name) deduplicates correctly: 100 % rid overlap and 0 duplicate rids in rows.csv.
- **Erratum 29 (applied symmetrically?).**
  - Framework rows written before the fix took the old "retry, then fallback" path for invalid actions. Rows written after it take "no retry, fallback".
  - The share of new-rule rows differs by regime at live 10² (cartel 0.56, honest 0.39) and is balanced at 10³ and 10⁴ (0.55 / 0.56, 0.76 / 0.78).
  - ADK is now present at live 10² honest, but with fewer seeds for some sources (dense I-comp 2 and sort I-comp 1, vs 3 in the cartel).
  - Direction: negligible and mixed.
- **Unequal seed sets.**
  - Replay 10⁶ honest b = 5: tuned warm-start has 79 seeds vs 100 for MIDIAN. The cross-fit uses skipna means, so pool members are judged on different seed sets.
  - Live 10³ pool candidates: seeds 1-5 are pre-instance-change rows and 6-10 are new (F4, F5).
  - E–H keep, per (framework, source), "the grid with the most seeds", so the frameworks in one bar average different seed sets.
- **Fallback in E–H.**
  - E–H plot lenient `success`. Magentic-One's rows have a 58 % mean fallback rate (declared argmax inside the shortlist) and ADK 20 %; the others are ≤ 4 %. So the "best single framework" dot and the framework means partly measure the adapter's declared-argmax fallback.
  - This biases frameworks UP, against MIDIAN. Report `success_strict` beside them.
  - Only 18 framework rows have fallback ≥ 0.9, all Magentic-One / smolagents in `_verified` (MIDIAN w/o audits cohort) grids, which E–H do not draw.

## Checked clean

- **Erratum-28 quarantine.**
  - 5,297 CSV-quarantined and 2,860 JSON-quarantined rows: none is present in rows.d with identical content.
  - The 747 quarantined rids that reappear in rows.d are all reruns with different content. Their mean fallback fell from 0.40-1.00 to 0.11-0.30.
- **Tuning disjointness.** `tune_wsb_n1000` rows are seeds 11-15 only. Every grid feeding A at live 10³ uses seeds 1-10.
- **Cross-fit.** It excludes the scored seed. There is no leakage of seed s into its own pick.
- **Same b per bar.** Every row in each b slot has that b, and every rival spends exactly 1.00 × n·K·b (F3 covers MIDIAN's overspend).
- **Oracle and task stream pairing.** The oracle success per (n, seed) is identical across all 18 live grids and across the RouterEval 5k and LLMRouterBench grids, so the task stream is shared per unit. `random` agrees across grids to ≤ 0.001.
- **The RouterEval family-order bug** (erratum 29 addendum) does not touch B: RouterEval 5k and LLMRouterBench have one oracle value per seed across every grid. It affects H's m ≤ 1,000 pools only, as documented.
- **Live 10² cartel MIDIAN b = 5 equal to honest.** Genuine; the guide checked it.
- **MIDIAN r = 10 and δ = 1/3** (every variant) are pre-registered. The r ≠ 10 exclusion hides nothing that beats MIDIAN by more than 0.006.

## Summary (severity order)

1. **CRITICAL, F6.** A and B show only the honest regime and the β = 0.5 low-skill cartel. This is the one liar regime in which MIDIAN beats peer-reported halving. Halving wins paired in 19 of 24 live regimes, by 3-8 pp and on every seed in most of them. Combined with F1 (halving is never drawn), the figures present a regime-dependent trade-off as dominance.
2. **HIGH, F1.** Peer halving, a pre-registered and reported rival reading the same channel on ≤ the same budget, is absent from A and B and hidden everywhere by `HIDE_HALVING`. It beats MIDIAN honest at b = 3 in 7 of 8 families and sizes.
3. **HIGH, F2 and F2 update.** Today's best-bandit pool edits drop the pre-registered warm-start n0 = 5 and `linucb_honest`. They flip 9 bar orders to put MIDIAN above best bandit, and lower that bar by 2-11 pp in others.
4. **HIGH, F7.** The headline arm (MIDIAN, whose own pre-registered target was a MISS), the tuned bandit, the pool edits, the hidden halving and the regime choice are all post hoc. None is marked in A or B.
5. **MEDIUM.**
   - F3: MIDIAN spends 1.03-1.07 × n·K·b while every rival spends exactly 1.00×. The effect is immaterial in size, but "same budget" is false.
   - F4: the probe-index bug gives repeated agents the same instance, handicapping TrueSkill (0.745 → 0.607 across seed blocks).
   - F5: the cross-grid disagreement comes from pre-instance-change grids averaged into MIDIAN w/o defenses only, which inflates MIDIAN − MIDIAN w/o defenses by 0.3-0.5 pp at 10².
   - F8: 3-seed CIs, cross-fit selection penalty on the rivals only, no paired or multiplicity-controlled inference, and over-broad claims in RESULTS.md.
6. **LOW, F9 and F10.** Other hidden arms, the fixed de-dup bug (regenerate the 14:13 CSVs), erratum-29 rule mixing, unequal seed sets, and the E–H fallback, which biases the frameworks up.
