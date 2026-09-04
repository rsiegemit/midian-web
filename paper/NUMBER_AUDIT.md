# Number audit — mismatches only (2026-09-04)

Recomputed by `scripts/paper_numbers.py` into `paper/NUMBERS.json` (value, grid, units, 95% CI per entry).
Compared against the numbers as quoted in PAPER_WORKORDER.md, because `paper/` (main.tex, sections/, BUILD_NOTES.md)
is not present in this repository or anywhere on this filesystem — see the report.

| # | where | quoted | recomputed | grid | verdict |
|---|---|---|---|---|---|
| 1 | (c) real pools | their MLP router 0.620 / **0.607** at 1,000 real models | 0.620 / **0.620** | routereval_mmlu n=1000 | **wrong.** The MLP router reads neither declarations nor reports, so it is flat in β. 0.607 is MIDIAN-VA's cartel value from the row above. Fixed in RESULTS.md §III.5 and RESULTS_rte_v3.md D2. |
| 2 | (d) 14B orchestrator | lenient −0.013, **12 / 12 cells** | −0.013, strictly below in **4 of 12** cells, identical in the other 8 | fw_live_n1000 | **wrong.** On bimodal and heavy_tail both orchestrators pick the same agent from the same shortlist (difference exactly 0.0000); the whole gap is the 4 specialist cells. Fixed in RESULTS_rte_v2.md §1. |
| 3 | (d) shortlisted frameworks | trail MIDIAN-V by **0.03–0.11** | **0.030–0.122** (LlamaIndex −0.122, Magentic-One −0.030) | fw_live_n1000_verified | **wrong upper end.** RESULTS.md §II.2 already says 0.03–0.12. |
| 4 | (c) cartel | VA **+0.15** over every framework | **+0.141 … +0.159** (min Magentic-One, max LlamaIndex) | fw_live_n1000_lowskill | **rounds up.** "+0.14 … +0.16" is accurate. Per-framework values in RESULTS.md §II.5 are correct. |
| 5 | (b) cost exponents | MIDIAN route **n^0.14**, flat n^1.00 | n^0.136 [0.131, 0.152] on `combined_scale`; **n^0.112 [0.106, 0.118]** on `bernoulli_scale` | both | **ambiguous provenance, not wrong.** n^0.14 is the live cross-n fit; text quoting it should name `combined_scale`. Flat is n^1.000 in both. |
| 6 | (b) AutoGen crossing | 9,415 (H11) vs 9,416 (errata) | joules **9,415**, GPU-seconds **9,416** | energy.py | **two quantities, not a typo.** Fig. 1a, its CSV and the errata now use 9,415; the GPU-second table keeps 9,416, and RESULTS.md §II.6 now names both. |

Everything else in the work order's list matches within rounding. Spot values, for the record:
frameworks 0.524–0.546; MIDIAN 0.6672 / VA 0.6837 / oracle 0.7230 at β = 0; paired framework − MIDIAN −0.130 … −0.108;
specialist frameworks 0.3899 vs MIDIAN 0.7784 (0 / 400 pairs won); n = 10k frameworks 0.257–0.367, random 0.4167,
tree 0.786–0.813; VA − KNN +0.074 / +0.070 / +0.126; VA − MLP +0.020 … +0.036; halving − VA +0.031 / +0.038 / +0.052
and +0.177 on 5,000 real models; RouterBench 0.7074 vs 0.7135 at 36,850 / 277,915 labels; RouterEval tuned probe table
0.6296, best single 0.6290, LinearR 0.6669, 260,556 / 3,348,583 labels; LLMRouterBench probe 0.7040, Avengers 0.7088,
EmbedLLM 0.7017; declared 0.7095 → 0.4696 (1k) and 0.8644 → 0.6144 (5k); halving 0.7221 → 0.4016 and 0.8822 → 0.5644;
MIDIAN 0.6672 → 0.5686, A flat (0.6679 → 0.6655); 52.9 comparisons per task at 100k; 14B fallback 0.432 vs 0.604,
strict +0.116; V-shortlist lifts +0.030 … +0.087 with LlamaIndex +0.010 (CI covers 0); VA cohort +0.021 at β = 0.5;
20-model pool flat/MLP 0.687, MIDIAN-A 0.677, VA 0.669 at β = 0 (0.664 over all β); churn VA − MIDIAN −0.022 / −0.028.

## The two questions asked explicitly

**(i) "specialist 0.39 vs 0.78" is plain MIDIAN.** On the specialist cells of fw_live_n1000: frameworks 0.3899,
**MIDIAN 0.7784**, MIDIAN-V 0.7946, MIDIAN-VA 0.8019, oracle 0.8612. The README sentence is therefore correct as
written; if the paper wants the best arm the number is 0.80 (VA), not 0.78.

**(ii) "declared argmax 0.515" under the n = 1,000 cartel is the `fw_live_n1000_lowskill` value (0.5146)** — not the
`learned_f1` / `variants_f1` figure. Those grids carry declared_argmax only through `live_f1_n1000`, whose β = 0.5
low-skill-first value is **0.5489**. The two differ because the low-skill grid is a separate draw of the population at
β = 0.5 with a colluding cartel, while `live_f1_n1000` sweeps β on the F1 cells. Table 1's cartel column should keep
citing `fw_live_n1000_lowskill`.

## Convention note

Paired CIs in `paper/NUMBERS.json` bootstrap the per-seed means of the paired differences (10 seeds), which is the
convention `rte.analyze` uses for its own bars. Some paired CIs printed in RESULTS_rte_v2.md (for example
MIDIAN-VA − AutoGen under the cartel, [+0.082, +0.225]) bootstrap the 30 cell × seed units directly and are therefore
wider. Point estimates agree exactly (+0.154). Whichever convention the paper adopts should be stated once and used
throughout; NUMBERS.json carries the seed convention.
