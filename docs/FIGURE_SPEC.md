# Figure specification for the ICLR 2027 submission

Implemented in `scripts/figspec.py` (canvas, fonts, colours, names, legend order, drawing helpers); every figure script
draws on it. Applies to every figure drawn by `scripts/condensed_figs.py`, `scripts/efficiency_figs.py`, and the
shortlist scripts. The rule throughout: **a figure carries data, axes, and a legend; everything else goes in the caption
or the text.** No in-figure titles, no letters, no file names, no draft annotations.

## 1. Global rules (all figures)

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

## 2. Names (legend strings and tick labels)

After the rename (MIDIAN = full method), use exactly these strings everywhere:

| pre-rename string | submission string |
|---|---|
| `MIDIAN-VA` | `MIDIAN` |
| `MIDIAN` (plain tree) | `MIDIAN w/o defenses` |
| `MIDIAN-VA (whole population)` | `MIDIAN, no framework` |
| `MIDIAN-VA leaf cohort` / `VA cohort` | `MIDIAN cohort` |
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

## 3. Per-figure specification

### Figure 1 (body): `A_live_stacked`
- Arms, in this legend order: oracle, MIDIAN, MIDIAN w/o defenses, flat probe argmax, best learned/declared router, best bandit, declared argmax, random.
- Every arm at b = 3. Budget overlay (b = 1 light in front, b = 3 mid, b = 5 dark behind) on MIDIAN only; the other arms as single b = 3 bars. This removes the shade legend and every empty slot. If you keep overlays on all arms, add one small inset legend "b = 1 / 3 / 5" (three swatches, grey) at the right of the main legend.
- x ticks: $n = 10^2$, $10^3$, $10^4$, $10^5$. y: "task success", 0.2–0.95.
- Optional eighth arm, cross-fitted per seed like the pooled arms: `best framework, best text shortlist`. Recommended: it puts claim (i) on the same axes as claims (ii)–(iii).

### Figure 2 (body): `B_families_stacked`
- Same arms, same order, same overlay rule as Figure 1. y: "success relative to oracle", 0.2–1.05, oracle line at 1.
- x ticks: the five backends with $n$ on the second line.

### Figure 3 (body): `F_shortlists_1e5`
- Seven shortlists, sorted by honest mean: declared top-k, MIDIAN cohort, dense (Qwen3-8B), MiniLM, fusion + reranker, BM25 (if it has rows at $10^5$; otherwise omit), hashed TF-IDF. The instruction variants go to the appendix version.
- Add a dotted grey `random` line at 0.44 (live $10^5$ random) alongside the oracle line; both in the legend.
- Best-framework dots stay (black, 3 pt), legend entry "best framework".
- x tick labels horizontal if they fit at seven bars; otherwise 30°.

### Figure 4 (body): `H_routereval_shortlists`
- Shortlists: hashed TF-IDF, MiniLM, dense (best instruction), fusion + reranker (best instruction), declared top-k, MIDIAN cohort. Instruction variants to the appendix.
- x ticks: $m = 10$, $10^2$, $10^3$, $5{,}000$. Reference lines as in Figure 3.

### Figure 5 (body): `D_energy_per_query`
- MIDIAN only. Nine lines as now (n = $10^3$, $10^5$, $10^7$ light to dark; b = 1 dotted, 3 solid, 5 dashed). Legend inside the axes, bottom-left, two columns: three colour swatches labelled $n = 10^3$ / $10^5$ / $10^7$ and three line styles labelled b = 1 / 3 / 5, instead of nine separate entries.
- Band: light grey, both edges drawn and labelled at the right margin (AutoGen 20.6 J; Magentic-One 220 J). Break-even crosses stay (black ×, 3 pt).
- y: "energy per query (J)", log; x: "queries served T", log. The word "estimated" goes to the caption.
- Move the legend fully inside the axes so it no longer overlaps the y label.

### Appendix figures
- `A_live_allb`, `B_families_allb`: every arm at every budget as separate bars with ±1 s.e. whiskers; keep the light/mid/dark encoding and add the three-swatch budget legend. Same names and order as Figures 1–2.
- `E_shortlists_by_n`: all shortlists, one legend row above the axes, names from Section 2 (appendix forms with the instruction variants). x ticks as $n = 10^2 \dots 10^5$.
- `G_shortlist_lift_1e5`: y "gain in task success over hashed TF-IDF"; whiskers ±1 s.e. across frameworks; no title.
- `C_routing_work_vs_n`: arms MIDIAN, MIDIAN w/o defenses, `any flat scan` (one line replacing flat probe argmax / declared argmax / best bandit, which coincide at exactly n), learned band; legend entries carry the fitted growth in parentheses as now. y "messages + comparisons per query", x "population size n", both log.
- If C and D are to share the body: one figure 5.5 × 2.0 in, two panels side by side labelled (a) and (b) in the bottom-left corner of each panel (8 pt, bold), one legend each inside the panel.

## 4. What the captions already carry (do not duplicate in figures)
Backend, population sizes and seed counts; regime encoding; the budget overlay rule; what pooled arms are; what a whisker is; that energy is estimated; that hashed TF-IDF returns clones at $10^5$; which frameworks are averaged.
