# Figures

The paper's figures, the scripts that draw them, their inputs and the grids behind them (below), and the style
specification every figure follows (§1-§4). Every plotted value is traced to its rows in
[figure_provenance.md](figure_provenance.md).

## Figures and their grids

All figures are written to `figures/paper/` as vector PDF, a 300 dpi PNG and a CSV of every plotted value. The
scripts are in `scripts/figures/`, their shared code in `scripts/figures/lib/`, and the aggregate CSVs they draw from
in `results/aggregates/`.

| paper | file | script | inputs | grids |
|---|---|---|---|---|
| Fig. 1 | `A_live_stacked` | `condensed_figs.py` | `results/aggregates/bars/live*.csv` (b = 3); per-seed tables from rows (b = 1, 5; cross-fitted pools); `results/aggregates/shortlist/live.csv` (best framework) | `live_core_n100`, `live_f1_n1000`, `learned_n100`, `learned_f1`, `variants_f1`, `learned_n10k*`, `live_n10k_v2`, `live_n10k_cartel_random`, `fw_live_n10k_cartel`, `live_n100k*`, `va_b_n*`, `rivals_b_n*`, `pool_fill_n*`, `pool_seeds_n1000`, `tuned_wsb_n*`, `linucb_fix_n*`, `trueskill_fix_n*`; the live framework shortlist grids |
| Fig. 2 | `B_families_stacked` | `condensed_figs.py` | as Fig. 1 for live 10<sup>5</sup>; per-seed tables for the other families | live: as Fig. 1; bernoulli 10<sup>7</sup>: `bernoulli_scale_v5`, `va_b_` / `rivals_b_` / `pool_fill_` / `linucb_fix_bernoulli_1e7`, `bernoulli_1e7_cal`, `rivals5_bernoulli_1e7_cal`; replay 10<sup>6</sup>: `replay_1e6_split_cal`, `rivals5_replay_1e6_split_cal`; RouterEval 5,000: `routereval5k_norep_cal`, `rivals5_routereval5k_norep_cal`; LLMRouterBench: `llmrouterbench_norep_cal`, `rivals5_llmrouterbench_norep_cal` |
| Fig. 3 | `F_shortlists_1e5` | `shortlist_condensed.py` | `results/aggregates/shortlist/live.csv` | `fw_live_n100k_*` (dd, em, sota, dense / sota instruction probes, declared, verified_va), the live MIDIAN grids at 10<sup>5</sup> (reference lines) |
| Fig. 4 | `H_routereval_shortlists` | `shortlist_condensed.py` | `results/aggregates/shortlist/routereval.csv` | `fw_routereval_{small,1k,5k}{,_em,_va}_norep_cal`, `re_sl_{declared,embed}_{small,1k,5k}_norep_cal`, `routereval_mmlu_norep_cal`, `routereval5k_norep_cal` (reference lines) |
| Fig. 5 | `D_energy_per_query` | `efficiency_figs.py` | `results/aggregates/cost_by_n.csv` (ledger cache), `scripts/analysis/energy.py` (energy model, live supervisor measurement) | `bernoulli_scale_v5` |
| App. | `A_live_allb`, `B_families_allb` | `condensed_figs.py` | as Figs. 1-2 | as Figs. 1-2 |
| App. | `C_routing_work_vs_n` | `efficiency_figs.py` | `results/aggregates/cost_by_n.csv` | `bernoulli_scale_v5` |
| App. | `E_shortlists_by_n` | `shortlist_condensed.py` | `results/aggregates/shortlist/live.csv` | the live framework shortlist grids at 10<sup>2</sup>-10<sup>5</sup> |
| App. | `F_shortlists_1e5_appendix`, `G_shortlist_lift_1e5` | `shortlist_condensed.py` | as Fig. 3 | as Fig. 3 |
| App. | `I_max_lie` | `lie_max_fig.py` | rows; `A_live_allb.csv`; `results/aggregates/shortlist/live.csv` (redrawn from `results/aggregates/figures/I_max_lie.csv` and `refs.csv`) | `lie_max_n{1000,10k,100k}`, `lie_max_fw_n{1000,10k,100k}` |

`results/aggregates/bars/*.csv` are written from the rows by `bar_figs.py` and `results/aggregates/shortlist/*.csv` by
`shortlist_figs.py`; `make_all.py` runs the whole chain, and `make_all.py --from-csv` redraws every figure from
`results/aggregates/` alone, without reading any row ([reproducing.md](reproducing.md)). The arms no figure draws are one
predicate, `excluded()` in `scripts/figures/lib/exclusions.py` ([methods.md](methods.md) §6).

## Style specification

Implemented in `scripts/figures/lib/figspec.py` (canvas, fonts, colours, names, legend order, drawing helpers); every
figure script draws through it. The rule throughout: **a figure carries data, axes, and a legend; everything else goes
in the caption or the text.** No in-figure titles, no letters, no file names, no draft annotations.

### 1. Global rules (all figures)

**Canvas.** Body figures 5.5 × 1.9 in (D: 5.5 × 1.8 in), saved as vector PDF (`fonttype 42`, so text stays text) plus a 300 dpi PNG for previews. Placed at `width=\linewidth`; anything taller than 1.9 in pushes the conclusion onto page 10. Appendix figures may be 5.5 × 2.4 in.

**Fonts.** Serif to match the body (Times / `mathptmx`; matplotlib `font.family = serif`, `mathtext.fontset = stix`). Tick labels 8 pt, axis labels 9 pt, legend 8 pt. Nothing below 8 pt.

**Titles.** None. Remove `ax.set_title(...)` everywhere. The letter (A, B, …), the backend, the regime encoding, "CIs in _allb", "pre-registered", "hatched = cartel", the trailing "*" for incomplete data, and everything else now in a title moves to the caption (already written) or is dropped.

**Legends.** Inside the axes where there is room (top-left or top-right, `frameon=False`); otherwise a single row above the axes, no frame, no title. Entries in a fixed order that is the same in every figure that shares arms (Section 2 below), not ranked by value. One legend per figure; a figure with panels shares one legend.

**Axis labels.** Sentence case, no units in parentheses unless there is a unit: "task success", "success relative to oracle", "population size n", "queries served T", "messages + comparisons per query", "energy per query (J)", "gain in task success over hashed TF-IDF". No bold.

**Ticks.** Population sizes as $10^2, 10^3, \dots$ (mathtext), never "n = 100,000". Category ticks as the arm or shortlist name only. Rotate category labels 30° only when they overlap; prefer shorter names (Section 2).

**Colours.** Keep the current palette (MIDIAN green `#2ecc71`, undefended tree red `#c0392b`, flat scan blue `#3498db`, learned orange `#ff7f0e`, bandit purple `#9467bd`, declared slate `#5d6d7e`, random grey `#bbbbbb`). Colour-blind check: green/red pair is the risk; keep the hatch and the legend order so the figure reads in greyscale (ICLR asks that captions and body make sense printed in black and white).

**Regimes.** Solid = honest (β = 0); hatched `////` = β = 0.5 low-skill-first cartel. Stated once in each caption, never in the figure.

**Whiskers.** ±1 s.e. (over seeds; G across frameworks). Black, linewidth 0.6, cap 1.5 pt. Captions say "±1 s.e."; never "CI".

**Reference lines.** Oracle: grey dotted, linewidth 1.0, labelled "oracle" in the legend. MIDIAN on the whole population (E, F, H): green solid, labelled "MIDIAN, no framework".

**Markers for incompleteness.** None in the submission. Stars, "not yet run", "rerun outstanding", and the title "*" are draft-only; the draft build of the paper carries those notes in `\ifdraft` caption text, the figure must not.

**Grid.** Horizontal major grid only, linewidth 0.3, alpha 0.4. No top or right spines.

**Numbers.** Two decimals on value axes. Axis floor at 0.2 for success (as now) is fine; state "axes start at 0.2" in captions where bars are truncated.

### 2. Names (legend strings and tick labels)

After the rename (MIDIAN = full method), use exactly these strings everywhere:

| pre-rename string | submission string |
|---|---|
| `MIDIAN-VA` | `MIDIAN` |
| `MIDIAN` (plain tree) | `MIDIAN w/o defenses` |
| `MIDIAN-VA (whole population)` | `MIDIAN, no framework` |
| `MIDIAN-VA leaf cohort` / `VA cohort` / `MIDIAN cohort` | `MIDIAN shortlist` |
| `flat probe argmax (online)` | `flat probe argmax` |
| `best learned router` | `best learned/declared router` |
| `best bandit` | `best bandit` |
| `declared argmax` | `declared argmax` |
| `random` | `random` |
| `hashed TF-IDF (pre-registered)` / `TF-IDF (pre-reg.)` | `hashed TF-IDF` |
| `MiniLM` | `MiniLM` |
| `Qwen3-8B dense, I-competent` / `dense, I-comp` | `dense (Qwen3-8B)` in the body figures (best instruction only); `dense, competence instr.` / `dense, demonstration instr.` in the appendix |
| `fusion + reranker[, I-…]` / `rerank[, I-…]` | `fusion + reranker` in the body; `fusion + reranker, competence instr.` / `…, demonstration instr.` in the appendix |
| `declared-claim top-k` / `declared top-k` | `declared top-k` |
| `best single framework` | `best framework` |
| `frameworks: 1-10.7 supervisor-call equivalents per query (21-220 J)` | `frameworks (AutoGen to Magentic-One)` with the two band edges labelled at the right margin: `AutoGen, 20.6 J` and `Magentic-One, 220 J` |

Backend tick labels in B: `live`, `Bernoulli`, `RouterBench replay`, `RouterEval`, `LLMRouterBench` on one line, with the population on a second line as $n = 10^5$ etc.

### 3. Per-figure specification

#### Figure 1 (body): `A_live_stacked`
- Arms, in this legend order: oracle, MIDIAN, MIDIAN w/o defenses, flat probe argmax, best learned/declared router, best bandit, declared argmax, random.
- Every arm at b = 3. Budget overlay (b = 1 light in front, b = 3 mid, b = 5 dark behind) on MIDIAN only; the other arms as single b = 3 bars. This removes the shade legend and every empty slot. If you keep overlays on all arms, add one small inset legend "b = 1 / 3 / 5" (three swatches, grey) at the right of the main legend.
- x ticks: $n = 10^2$, $10^3$, $10^4$, $10^5$. y: "task success", 0.2–0.95.
- Optional eighth arm, cross-fitted per seed like the pooled arms: `best framework, best text shortlist`. Recommended: it puts claim (i) on the same axes as claims (ii)–(iii).

#### Figure 2 (body): `B_families_stacked`
- Same arms, same order, same overlay rule as Figure 1. y: "success relative to oracle", 0.2–1.05, oracle line at 1.
- x ticks: the five backends with $n$ on the second line.

#### Figure 3 (body): `F_shortlists_1e5`
- The body shortlists in the fixed order of §3b (not sorted by value): MIDIAN shortlist, hashed TF-IDF, BM25 (if it has rows at $10^5$; otherwise omit), MiniLM, dense (Qwen3-8B), fusion + reranker, declared top-k. The instruction variants go to the appendix version.
- Add a dotted grey `random` line at 0.44 (live $10^5$ random) alongside the oracle line; both in the legend.
- Best-framework dots stay (black, 3 pt), legend entry "best framework".
- x tick labels horizontal if they fit at seven bars; otherwise 30°.

#### Figure 4 (body): `H_routereval_shortlists`
- Shortlists (order of §3b): MIDIAN shortlist, hashed TF-IDF, MiniLM, dense (best instruction), fusion + reranker (best instruction), declared top-k. Instruction variants to the appendix.
- x ticks: $m = 10$, $10^2$, $10^3$, $5{,}000$. Reference lines as in Figure 3.

#### Figure 5 (body): `D_energy_per_query`
- MIDIAN only. Nine lines as now (n = $10^3$, $10^5$, $10^7$ light to dark; b = 1 dotted, 3 solid, 5 dashed). Legend inside the axes, bottom-left, two columns: three colour swatches labelled $n = 10^3$ / $10^5$ / $10^7$ and three line styles labelled b = 1 / 3 / 5, instead of nine separate entries.
- Band: light grey, both edges drawn and labelled at the right margin (AutoGen 20.6 J; Magentic-One 220 J). Break-even markers and the $T^\ast$ annotation: §3b.
- y: "energy per query (J)", log; x: "queries served T", log. The word "estimated" goes to the caption.
- Move the legend fully inside the axes so it no longer overlaps the y label.

#### Appendix figures
- `A_live_allb`, `B_families_allb`: every arm at every budget as separate bars with ±1 s.e. whiskers; keep the light/mid/dark encoding and add the three-swatch budget legend. Same names and order as Figures 1–2.
- `E_shortlists_by_n`: all shortlists, one legend row above the axes, names from Section 2 (appendix forms with the instruction variants). x ticks as $n = 10^2 \dots 10^5$.
- `G_shortlist_lift_1e5`: y "gain in task success over hashed TF-IDF"; whiskers ±1 s.e. across frameworks; no title.
- `C_routing_work_vs_n`: arms MIDIAN, MIDIAN w/o defenses, `any flat scan` (one line replacing flat probe argmax / declared argmax / best bandit, which coincide at exactly n), learned band; legend entries carry the fitted growth in parentheses as now. y "messages + comparisons per query", x "population size n", both log.
- `I_max_lie`: the Figure 1 arms without the oracle bar, all hatched (cartel), ±1 s.e.; the oracle as the dotted line; a
  short black tick on each bar for the same arm under the standard lie, with a "standard lie" legend entry.
- If C and D are to share the body: one figure 5.5 × 2.0 in, two panels side by side labelled (a) and (b) in the bottom-left corner of each panel (8 pt, bold), one legend each inside the panel.

### 3b. Shortlist palette, order and emphasis; D break-even marker

Implemented in `scripts/figures/lib/figspec.py` (`SHORTLIST_ORDER`, `SHORTLIST_COLOR`, `INSTR_TINT`, `SHORTLIST_EDGE`,
`ref`) and `scripts/figures/efficiency_figs.py` (`BREAK_EVEN`, `render_D`); applies to E, F, F appendix, G, H and D.

**Palette.** Greyscale values for reference (Rec. 601 luma $0.299R+0.587G+0.114B$ and WCAG relative luminance, sRGB 0–1):

| shortlist | colour | luma (601) | rel. luminance |
|---|---|---|---|
| MIDIAN shortlist | `#2ecc71` | 0.574 | 0.450 |
| hashed TF-IDF | `#7f8c8d` | 0.534 | 0.252 |
| MiniLM | `#2980b9` | 0.425 | 0.194 |
| dense (Qwen3-8B) | `#8e44ad` | 0.400 | 0.129 |
| fusion + reranker | `#a0522d` | 0.396 | 0.137 |
| declared top-k | `#f1c40f` | 0.740 | 0.582 |

The six colours are NOT all separable in greyscale: minimum pairwise luma gap 0.004 (dense vs fusion + reranker;
relative luminance 0.008); MiniLM / dense 0.025, MiniLM / fusion 0.029 and MIDIAN shortlist / hashed TF-IDF 0.040 are also
below 0.06. In print they are told apart by position (fixed order) and, for the MIDIAN shortlist, by the heavy edge.
BM25 (not drawn in the current figures) keeps `#17becf`.
Instruction variants: competence instr. = the base colour blended 40 % toward white (dense `#bb8fce`, fusion `#c69781`);
demonstration instr. = 65 % toward white (dense `#d7bee2`, fusion `#dec2b6`). Cartel bars keep the `////` hatch.
Reference lines: oracle grey dotted, random light-grey dotted (1.0 pt), `MIDIAN, no framework` green `#2ecc71` solid 1.6 pt.

**Order.** Left to right (F, F appendix, H; within each n group of E; G): MIDIAN shortlist, hashed TF-IDF, BM25 (when
present), MiniLM, dense (Qwen3-8B), fusion + reranker, declared top-k. Never sorted by value. Instruction variants follow
their base: dense, dense competence instr., dense demonstration instr., then fusion + reranker and its two variants in the
same pattern. In G hashed TF-IDF is the reference and has no bar; the rest keep the order.

**Emphasis.** MIDIAN-shortlist bars (and legend swatch) have a 0.8 pt black edge; every other bar 0.3 pt.

**D break-even marker.** Open black circles (`marker="o"`, no fill, edge black, size 3.5, edge width 0.8) where a MIDIAN
line meets a band edge, with the legend entry `break-even (line meets band edge)` in a fourth row under the b column
(legend inside the axes, bottom left, n swatches and b line styles as before). Annotation inside the axes bottom right,
above the x axis, 8 pt serif: $T^\ast = J_{\mathrm{build}}/(J_{\mathrm{fw}}-J_{\mathrm{route}})$. Band edge labels stay
at the right margin.

### 4. What the captions already carry (do not duplicate in figures)
Backend, population sizes and seed counts; regime encoding; the budget overlay rule; what pooled arms are; what a whisker is; that energy is estimated; that hashed TF-IDF returns clones at $10^5$; which frameworks are averaged.
