# Audit: figures and aggregation (condensed_sample A-H)

Auditor: audit-figures2, 2026-09-23. Read-only on code/data. Findings appended as confirmed.
Names follow the 2026-09-24 rename (CHANGES_AND_ERRATA §8g): MIDIAN = both defenses (formerly VA); MIDIAN w/o defenses = the
plain tree; MIDIAN w/o audits / w/o verification = the single-defense ablations. Code references are to the code audited.

## Findings

### F1. [MEDIUM] Loader drops every not-yet-merged rows.d row but one (rid is NaN for rows.d) -- A 10^4 cartel b=1 pool shows "missing knn" although the rows are on disk
- Evidence: rows.d JSON files do not carry `rid` (rid = file name; checked `rivals_b_n100k/rows.d/0324fb67...json`: `'rid' in d == False`).
  `scripts/seed_tables.py:227,233` and `scripts/fw_variant_numbers.py:25,30` build a frame from rows.d (rid absent -> NaN), concat rows.csv,
  then `drop_duplicates("rid")`. pandas treats all NaN rids as equal, so every rows.d row not yet consolidated into rows.csv collapses to ONE row
  (and that survivor duplicates its own rows.csv copy when it is already merged).
- Measured (15:50 EDT): rivals_b_n10k rows.d 90 -> seed_tables.rows 85 / fw_variant_numbers.load 84; rivals_b_routereval5k 138 -> 133/132;
  pool_fill_routereval5k 76 -> 73/72; pool_fill_replay_1e6 4758 -> 4719/4718.
  Lost rows: rivals_b_n10k b=1 beta0.5 lsf knn_router{} and knn_router{online} seeds 1-3 (landed 15:03 EDT, before the 15:45 regen);
  rivals_b_routereval5k b=1 beta0 knn_router both variants seeds 1-3; pool_fill_routereval5k b=3 knn_router_online beta0 s1-3, beta0.5 s1;
  pool_fill_replay_1e6 b=5 heavy_tail beta0 oracle + warm_start_bandit[n0=0.5] seeds 60-79.
- Effect: A_live_allb.csv "n = 10,000, cartel, best learned/declared router, b=1" reads "INCOMPLETE POOL, missing knn_router, knn_router_online" and is
  scored on {flat_nsw, disrouter} although both knn arms have landed; B's RouterEval b=1 honest learned pool and replay 10^6 b=5 bandit pool
  likewise run on a reduced set. efficiency_figs.va_build (`scripts/efficiency_figs.py:52`) already handles NaN rid correctly; these two loaders do not.
- Numeric impact, recomputed with a loader that keys rows.d by file name (all other A/B pool bars match to 4 d.p.):
  A live 10^4 cartel b=1 best learned/declared router: plotted 0.5344 (flat_nsw x2, disrouter x1), correct 0.5156 (flat_nsw x2, knn_online x1): bar too HIGH by 0.019.
  B replay 10^6 honest b=5 best bandit: plotted 0.9462 of oracle (wsb[n0=0.5] x59, ucb x41), correct 0.9620 (wsb x79, ucb x21): bar too LOW by 0.016,
  and NOT flagged (wsb is present in the table, just with fewer seeds, so no "INCOMPLETE POOL" and no title-* contribution).
- Direction: transient but silent; pool bars are computed on fewer candidates/seeds than exist (either direction, see above), and the
  erratum-28 `landed()` check (`fw_variant_numbers.py:42,56`) treats an unmerged rerun as outstanding (asterisk stays on: conservative).
- Figures: A, B (pool bars), and every consumer of fw_variant_numbers.load (shortlist_figs -> E-H while a framework grid has unmerged rows.d).
- Fix: key rows.d rows by file name (`{**json, "rid": f[:-5]}`) as rte.run.consolidate does.

### F2. [MEDIUM] "Incomplete pool" only detects a candidate with ZERO rows; a candidate with a partial seed set silently swaps the pick on the missing seeds
- Evidence: `scripts/condensed_figs.py:72` `miss = [a for a in want if a not in T.columns]`; `seed_tables.crossfit` (`seed_tables.py:283-286`) only
  considers arms that ran on seed s, so on seeds a candidate lacks the next-best candidate is scored instead.
- Instance now: B replay 10^6 honest b=5 best bandit -- warm_start_bandit[n0=0.5] has 79 of 100 seeds on disk (59 as loaded, F1); the other seeds
  are scored on ucb_per_family. Plotted 0.9462 (x59 / x41) vs 0.9620 with the 79 seeds; with all 100 it would likely be ~0.972 like b=3 (0.9724) and b=1 (0.9663).
  So the b=5 bar sits visibly BELOW the b=1 and b=3 bars of the same arm (non-monotone in b) purely as an artefact of missing seeds, with no asterisk.
- The B title is starred anyway for other reasons, so the title-* is not wrong for the figure, but the rule "* iff data incomplete" is not implemented
  at the pool level: the detector would pass this cell as complete.
- Direction: biases the pooled rival DOWN where a strong candidate is partially landed (favours MIDIAN). Figure B (and A whenever a pool grid is partial).
- Note: FIGURE_DATA_GUIDE §0.3 #2 says live 10^3 b=3 has seven candidates with seeds 1-5 only; that is no longer true (all 10 seeds now) -- doc is stale.

### F3. [HIGH] E-H "mean of frameworks" bars average different framework sets and seed counts, and the title-* detector does not see it; E n=100 honest "declared top-k" is ONE framework x ONE seed
- Evidence: `scripts/shortlist_condensed.py:41` averages whatever frameworks exist (`k` column); `pair()` (`:58-66`) sets the title * only when a regime slot is
  empty or `rerun_outstanding` is true. Neither framework count k nor seed count enters the flag.
- E_shortlists_by_n.csv (15:49 regen): n=100 honest declared k=1 -> 0.609 = fw_autogen, seeds=1 (figures/shortlist/live.csv row 442), drawn as the TALLEST
  honest bar at n=100 next to a cartel bar of k=10 x 3 seeds (0.494). autogen's own cartel value is 0.496, so the drawn honest->cartel drop (0.115) is
  one unit vs a 10-framework mean. n=100 honest dense_icomp k=7, dense_idemo k=7, sota_icomp k=5, sota_idemo k=6 (1-2 seeds each, fw_live_n100_backfill
  still landing at 15:50) vs k=10 cartel. At 10^5 honest/cartel pairs differ (embed 10 vs 8, dense_idemo 10 vs 8, sota_idemo 10 vs 8, va_cohort 9 vs 8).
  H: RouterEval 5,000 sota_icomp/idemo k=8 vs 9 elsewhere. 19 of 74 E bars and 5 of H's bars have k below that n's maximum.
- The E title does carry " *" (other cells are starred), so the figure is not falsely clean, but the rule "* iff incomplete" is not what the code checks:
  a figure with only partial-k / partial-seed cells and no erratum-28 stars would get no asterisk.
- Direction: mixed. The n=100 declared honest bar (single best-case unit) overstates declared top-k at 10^2 and its cartel drop; dropping CrewAI/ADK
  (quarantined) from some bars but not TF-IDF shifts every non-TF-IDF mean. G is paired within framework and is not affected by set differences (only by k).
- Recomputed independently from raw fw rows (grid-with-most-seeds rule): all 74 E bars match to 4 d.p. at the 15:48 snapshot; two n=100 honest bars
  (dense_icomp 0.4681->0.4721, sota_icomp 0.5311->0.5342) have since moved as backfill rows landed at 15:50 (timing, not a bug).
- Fix: require the same framework set (intersection) for the honest/cartel pair and for every bar compared at one n, or print k under each bar and
  include k < k_max and seeds < seeds_max in the * condition.

### F4. [HIGH] C: the excluded constant-cost learned router (flat_nsw, 50 comparisons/query) is CHEAPER than MIDIAN w/o defenses at every n and than MIDIAN from 10^5 up, and it is the majority pick in 10 A/B bars -- the legend/code say "2 of 29"
- Evidence: `scripts/efficiency_figs.py:79-80` drops any pool member whose fitted slope is "constant" from the learned band. cost_by_n.csv:
  flat_nsw_router work = 50 at every n (0 messages + 50 comparisons); MIDIAN w/o defenses = 46, 69, 92, 115, 138, 161 and MIDIAN = 25, 36, 47, 58, 69, 80 at n = 10^2..10^7.
  C_routing_work_vs_n.csv draws "best learned/declared router (∝ n^0.97-0.99)" from cluster_head (24.6 -> 1.0e6) to disrouter (54 -> 3.7e6).
- Pick counts (A_live_allb.csv + B_families_allb.csv, 15:45): flat_nsw is the MAJORITY pick in 10 learned-router bars -- live 10^4 honest b=1,3 and cartel b=1,3,5;
  replay 10^6 cartel b=3,5; RouterEval 5,000 cartel b=1,3,5 -- and a minority pick in 3 more (live 10^4 honest b=5, LLMRouterBench cartel b=5).
  Unique learned bars: 48 (54 counting live 10^5 twice). The code comment (`efficiency_figs.py:79`) and FIGURE_DATA_GUIDE §0.3 #21 / §6 still say "2 of 29" (stale).
- Reading the figure as drawn: title "MIDIAN w/o defenses grows like log n", the learned band grows ~n, so MIDIAN w/o defenses looks orders of magnitude cheaper than the best learned/declared router.
  For the member A/B actually pick in 10 bars, the true per-query work is 50: below MIDIAN w/o defenses (69-161) for n >= 10^3 and below MIDIAN (58-80) for n >= 10^5.
  There is no on-figure mark that a member was excluded.
- Direction: favours MIDIAN w/o defenses / MIDIAN on the routing-work claim (C is the only figure making the "advantage over flat methods" point, guide #21).
- Fix: draw flat_nsw as its own dashed line (it is constant, so it costs one legend entry) or extend the band to include it; say in the legend that it is a
  centralised index with an n·K·b probe build (4.8e8 probes at 10^7, cost_by_n.csv).

### F5. [MEDIUM] "95%" error bars at 3 seeds are the min-max of the three seed means (~73% coverage); 10 seeds ~90%
- Method (`extra_figs.ci`, `extra_figs.py:89-97`; `fw_variant_numbers.ci`): percentile bootstrap of the mean over seed means, B = 2000. Unit = seed (correct).
- With 3 seeds (live 10^4, live 10^5, RouterEval 5,000 -- A's two right-hand groups and B's live and RouterEval groups) the 2.5/97.5 percentiles are
  exactly the smallest and largest seed: A live 10^5 honest b=3 MIDIAN seeds 0.810/0.840/0.857 -> bar CI [0.810, 0.857]; the t-interval is
  [0.777, 0.894]. MIDIAN w/o defenses [0.713, 0.790] vs t [0.657, 0.848]; flat [0.703, 0.807] vs t [0.618, 0.880].
  Simulated coverage of this interval for normal data: 73% at 3 seeds, 90% at 10 seeds (2,000 draws).
- The bars are not paired: every arm's interval is marginal, although arms share worlds per seed, so interval overlap is not a test in either direction.
- G's whisker is 1.96*sd/sqrt(k) across frameworks (`shortlist_condensed.py:111`) while the module docstring (`:8`) calls it a "95% t-interval"
  (guide §0.3 #14 already notes the ~17% under-width at k=8); it measures between-framework spread, not seed noise.
- No "significant" statement about A-H was found in RESULTS*.md / docs (grep); the v4 cohort significance claims are outside this figure set.
- Direction: overstates precision for the 3-seed groups (makes MIDIAN's lead at 10^4/10^5 look more certain than it is). Figures A, B, E-H.

### F6. [MEDIUM] C: three of the eight legend entries are invisible (flat probe argmax, best bandit, declared argmax are the identical line work = n)
- cost_by_n.csv / C csv: flat_probe_argmax_online, every bandit member and declared_argmax all equal n exactly (100, 1000, ... 1e7). draw_I plots them
  in ARMS order, so declared argmax (grey, drawn last) covers flat (blue) and the zero-width bandit band (purple). Legend lists all three with colours
  the reader cannot find. FIGURE_DATA_GUIDE §6 notes the coincidence, the figure does not.
- Direction: neutral on values, but it hides that the best bandit and flat probing cost exactly what declared argmax costs.
- Fix: one line labelled "flat probe argmax = best bandit = declared argmax (= n)" or small vertical offsets / markers.

### F7. [MEDIUM] _stacked: title says "b = 1, +gain to b = 3, +gain to b = 5", but bars are overlapped full-height, and a b=5 value at or below b=1 is completely hidden
- Evidence: `condensed_figs.py:105-107` sorts the three budgets tallest-first and draws them with zorder 2+depth, so the lowest is on top. When b5 <= b1 < b3
  the b=5 bar is drawn under the b=1 bar and cannot be seen; the reader sees light (b1) + mid (b3) only and reads "no further gain at b=5".
- Cases now: A/B live 10^5 best learned/declared router honest (b1 0.613, b3 0.741, b5 0.613) and cartel (0.588, 0.741, 0.588) -- b=5 there is an
  incomplete pool (only the two zero-probe routers), so the real message "b=5 bar is 0.13 LOWER" is invisible; B replay 10^6 honest best bandit
  (0.966, 0.972, 0.946 -- the F2 artefact) likewise hidden. Other non-monotone arms (LLMRouterBench cartel MIDIAN w/o defenses 0.866/0.835/0.850, learned
  0.796/0.948/0.919; RouterEval best bandit 0.820/0.786/0.824 honest, 0.740/0.700/0.773 cartel) show shades out of order, which the title's
  "+gain" wording does not prepare the reader for (the docstring mentions it; the figure does not).
- No error bars on _stacked (by design, "CIs in _allb"), so the 3-seed non-monotonicities (RouterEval) look as solid as the 100-seed ones.
- Direction: hides regressions with budget; mostly on rival pool bars. Figures A_live_stacked, B_families_stacked.

### F8. [MEDIUM] y-axis starts at 0.2 in A, B, E, F, H (bars are not zero-based)
- `condensed_figs.py:121` ylim (0.2, 0.95 / 1.05); `shortlist_condensed.py:97` ylim (0.2, 0.95). Bar length is then value - 0.2, so ratios are distorted:
  A n=10^3 honest b=3 MIDIAN 0.813 vs random 0.432 is a 1.9x ratio drawn as 2.6x; B replay random (0.242 of oracle) is drawn as a 0.04 sliver.
  E n=10^4 TF-IDF 0.293 vs declared 0.646 (2.2x) drawn as 4.8x.
- Nothing on the axis marks the break. Differences (not ratios) are what the figures are read for, so this is a presentation caveat, not a data error.
- Direction: exaggerates the gap between the top arms and the floor arms (random, declared under cartel, TF-IDF), i.e. favours whatever is tallest (MIDIAN in A/B).
- Fix: start at 0, or switch to dots/intervals (a truncated axis is conventional for points, not for bars), or add an axis-break mark.

### F9. [CRITICAL] Dropping the pre-registered warm_start_bandit (n0 = 5) from the "best bandit" pool flips MIDIAN vs best bandit at honest b = 5 in 3 of 5 B families
- Evidence: `condensed_figs.py:35-36` POOLS["best_bandit"] = BANDIT minus warm_start_bandit + warm_start_bandit[n0=0.5]. The n0 = 5 arm ran in every cell
  (rows in bernoulli_scale_v5, replay_scale_v5, routereval_mmlu5k, llmrouterbench_pool, the live grids and rivals_b_*). Cross-fitting is already
  winner's-curse-free, so a pool holding BOTH n0 values is the unbiased "best bandit"; removing one can only lower the bar.
- Recomputed (same cross-fit, same per-seed tables, seed_tables spec) with n0 = 5 added back vs the plotted pool vs MIDIAN (raw success):
    | cell | b | plotted best bandit | with n0=5 | MIDIAN |
    | RouterEval 5,000 honest | 1 / 3 / 5 | 0.740 / 0.709 / 0.743 | 0.822 / 0.822 / 0.829 | 0.606 / 0.706 / 0.758 |
    | bernoulli 10^7 honest | 1 / 3 / 5 | 0.813 / 0.818 / 0.822 | 0.837 / 0.839 / 0.839 | 0.678 / 0.801 / 0.826 |
    | replay 10^6 honest | 1 / 3 / 5 | 0.763 / 0.768 / 0.760 | 0.785 / 0.786 / 0.787 | 0.654 / 0.762 / 0.777 |
    | LLMRouterBench honest | 1 | 0.653 | 0.675 | 0.644 |
    | live 10^4 cartel | 1 / 3 | 0.588 / 0.773 | 0.709 / 0.783 | 0.670 / 0.810 |
    | live 10^5 cartel | 3 | 0.764 | 0.778 | 0.828 |
  n0 = 5 is the cross-fitted pick in every seed of bernoulli/replay honest (x100), RouterEval honest (x3), live 10^4 cartel b1/b3 (x3).
- Headline effect: as plotted, MIDIAN beats best bandit at honest b = 5 in bernoulli (0.826 vs 0.822), replay (0.777 vs 0.760) and RouterEval
  (0.758 vs 0.743); with the pre-registered arm restored best bandit wins all three (0.839, 0.787, 0.829), and on RouterEval honest it beats
  MIDIAN at every b by 0.07-0.22. At live 10^4 cartel b=1 the plotted bandit (0.588) is below MIDIAN (0.670); restored it is above (0.709).
  Cartel b = 3 / 5 orderings (the robustness claim) are unchanged except as listed.
- Where n0 = 0.5 was tuned is not recorded in condensed_figs; it helps on live 10^3 (x10 picks) and hurts on the B families. "Tuned" + "drop the
  pre-registered one" is a post-hoc pool change that favours MIDIAN.
- Direction: biases best bandit DOWN (favours MIDIAN), up to 0.11 raw (RouterEval honest b=3: 0.709 -> 0.822, i.e. 0.786 -> 0.911 of oracle in B).
- Fix: keep both n0 values as pool candidates (the cross-fit already guards against the extra choice), or plot the pre-registered arm and state the tuned one as an addition.

### F10. [HIGH] The TEMPORARY HIDE_HALVING switch removes peer-reported sequential halving, which beats MIDIAN in every honest b=3 cell of A and in 3 of the 4 other B families (all seeds)
- Evidence: `extra_figs.py:120,128` HIDE_HALVING = True drops every label containing "halving" (the peer-reported arm too, not only the withdrawn
  trusted-observer arm of erratum 26). Paired per seed, raw rows, b = 3 (only budget the arm ran at, plus b = 1 on bernoulli/replay):
    | cell | halving (peer) | MIDIAN | diff | seeds halving wins |
    | live 10^2 / 10^3 / 10^4 / 10^5 honest | 0.810 / 0.860 / 0.863 / 0.863 | 0.782 / 0.813 / 0.811 / 0.836 | +0.028 / +0.047 / +0.052 / +0.028 | 10/10, 10/10, 3/3, 3/3 |
    | RouterEval 5,000 honest | 0.882 | 0.706 | +0.177 | 3/3 |
    | bernoulli 10^7 honest | 0.846 | 0.801 | +0.045 | 100/100 |
    | replay 10^6 honest | 0.790 (= oracle) | 0.762 | +0.028 | 100/100 |
    | LLMRouterBench honest | 0.664 | 0.669 | -0.004 | 3/5 |
    | every cartel cell above | 0.20-0.72 | 0.67-0.83 | -0.09 to -0.45 | 0/all |
- Effect on headlines: with the arm drawn, A's honest bars would show a rival above MIDIAN at every n (and B at bernoulli, replay, RouterEval),
  while the cartel bars would show it collapsing -- i.e. it sharpens the robustness story but removes any "best in the honest regime" reading.
  The switch is labelled TEMPORARY (user request 2026-09-22); nothing on the figures says an arm is hidden.
- Direction: favours MIDIAN in the honest regime. Figures A, B (and C: halving's routing work is 1 comparison/task per energy.py, below MIDIAN w/o defenses).
- Other undrawn arms that beat MIDIAN at honest b=3 (figures/bars/*.csv, 13:41): cnp_self_bid, verify_on_claim (bernoulli, replay; e.g. 10^7:
  0.838 / 0.842 vs 0.801), MIDIAN w/o verification (RouterEval, LLMRouterBench, +0.01). None exceeds the drawn declared argmax / best learned/declared router there, so they
  do not change a headline (LOW; A/B's seven-arm selection is a stated design choice). route_to_k_majority is correctly excluded (3 executions).

### F11. [LOW] Stale documentation of the pick counts, seed counts and star placement
- `efficiency_figs.py:79` and FIGURE_DATA_GUIDE §0.3 #21 / §6: flat_nsw "majority pick in 2 of 29 A/B bars". It is now 10 (see F4).
- Guide §0.3 #9: "B's b = 5 bars use 51 / 36 seeds (bernoulli) and 12-38 (replay)". All are now 100 (verified).
- Guide §0.3 #2: seven live 10^3 b=3 candidates have seeds 1-5 only. All now have 10.
- Guide §0.1 (A, E) and the condensed_figs.py docstring (`:15-16`, `:92`) still describe a "*" above a bar or at the baseline. The code now puts only one "*" in the title.
- Guide §0.1 G says TF-IDF at 10^5 is 0.378889 for every framework and seed. Confirmed (E csv best = mean = 0.3789, k = 10).

### F12. [LOW] Presentation details
- Legend order (`extra_figs._ranked`) ranks each entry by the ONE bar that carries its label, which is the first group's honest b=3 bar. In A that is
  n=10^2, where flat probe argmax (0.781) ranks above MIDIAN w/o defenses (0.775), although MIDIAN w/o defenses is higher at 10^4. In B it is live 10^5. In E and H it is the
  n=10^2 / m=10 bar, which for "declared top-k" is the one-unit bar of F3. The rule holds, but it reflects the first group only.
- Colours mean different things in different figures. In A/B, #9467bd purple is "best bandit"; in E/F/H the same hex is "Qwen3 dense, I-comp".
  Blues (#3498db flat probe vs #1f77b4 MiniLM) and reds (#c0392b MIDIAN w/o defenses vs #d62728 / #8b0000 rerank) are close.
- D: the legend overlaps the y-label ("build amortised" is cut) and the title runs past the left edge. The 4-column row-major legend leaves a gap in the
  middle. The title says "estimated", so no "*" is needed.
- F: the black "best single framework" dot is the maximum of 8-10 framework means, each over 1-3 seeds at 10^5. That is the winner's curse A/B avoid by
  cross-fitting. It biases the dot UP, which is conservative for MIDIAN.
- B normalises each bar and its CI by the honest b=3 oracle mean (a constant). The oracle is identical across b and regime on the same seeds
  (verified for all five families), so this is sound. The interval ignores oracle variance, a negligible effect.
- A/B mix inputs from two snapshots: b=3 single arms from figures/bars/*.csv (bar_figs, 13:41) and b=1/5 and pools from raw rows at 15:45.
  I re-derived every b=3 bar from raw at 15:50 and all still match, so there is no current discrepancy. regen_figs.sh now runs bar_figs.py first,
  which fixes this going forward.
- beta=0 liar_select tags: bar_figs uses the "random"-tagged rows only (`bar_figs.py:80-81`), while seed_tables uses both tags. This matters only
  for arms that have both tags. The largest single-arm difference is 0.0006 (MIDIAN w/o defenses live 10^2 0.7747 vs 0.7741), so it is immaterial.

## Verified correct (independent recomputation from $RTE_DATA/results with my own loader; rows.d keyed by file name)
- A: all 5 single arms x 4 n x 2 regimes at b=3, plus every b=1/5 bar from va_b_* and rivals_b_*. That is 108 numbers, all matching to 4 d.p.,
  including MIDIAN (e.g. 10^5 honest 0.8356 / b5 0.8533; 10^3 cartel 0.8074).
- A: all 46 cross-fitted pool bars (learned and bandit, b = 1/3/5) match to 4 d.p., except 10^4 cartel b=1 learned (F1).
- B: all single-arm bars in the four non-live families, both regimes and all three b, match after division by the honest b=3 oracle:
  RouterEval 0.9022, LLMRouterBench 0.7250, bernoulli 0.8462, replay 0.7900.
  Replay pooled bars use only seeds that have all 3 shapes (100 seeds); e.g. MIDIAN replay honest b=5 0.7769 -> 0.9834 of oracle.
- B: all 24 non-live pool bars match after normalisation, except replay honest b=5 bandit (F1/F2).
- The oracle is identical across b and regime on shared seeds, so B's single normaliser is valid.
- MIDIAN b=5 at live 10^2 is identical honest vs cartel seed by seed. That is not a regime mix-up: n_liars is 0 vs 50 and misroute_to_liar is
  non-zero, so MIDIAN's verification simply overrides every declaration at that budget.
- Budget grids (va_b, rivals_b, pool_fill, tuned_wsb, pool_seeds) at live are specialist and self_described only, so add_budgets' lack of a
  dist/declared_source filter is harmless today. No live figure grid has a NaN declared_source.
- E: all 74 bars match at the 15:48 snapshot (F3). F is the n=10^5 subset of E. G lifts check out
  (e.g. declared honest 0.6272 - 0.3789 = 0.2483, k = 10). H's MIDIAN and oracle lines at 5,000 (0.7056 and 0.9022) equal B's raw values.
- Erratum-28 pending set: I recomputed it independently (a unit is landed when every param variant has a row for that method/dist/beta/ls/seed).
  It gives 333 cells, identical to fw_variant_numbers.pending_reruns(). Every E-H star traces to a real outstanding unit. beta=0 is counted once
  (random tag) where both tags exist.
- D: build-probe ledgers (16,805 / 49,427.5 / 83,007.5 at 10^3; 1,680,005 / 4,959,662 / 8,319,642 at 10^5; 831,998,807 at 10^7 b=5), the probe energy
  4.0388 J and the 7B call 20.588 J all reproduce. C's work values match cost_by_n.csv. bernoulli_scale_v5 has not changed since 09-17, so the
  12:26 cache is not stale.
- Title asterisks: A *, B *, E *, F *, G *, H * are all justified by at least one missing bar, missing pool candidate or outstanding erratum-28 unit.
  C and D carry none, which is correct (C is complete; D says "estimated"). The one gap is the detector itself: partial seeds or k are never flagged (F2, F3).
