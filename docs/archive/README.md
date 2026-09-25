# Archive: the dated records of the study

These files are kept **verbatim** as the record of what was specified, pre-registered, measured and corrected, and
when. They are not maintained. The current documentation is in [`docs/`](..) and the current numbers are the figure
CSVs in `figures/paper/`. The only edits made when these files were archived replace cluster account,
partition and host names and absolute user paths with placeholders (`<cpu-partition>`, `<gpu-account>`, `$RTE_DATA`, ...);
nothing else was changed (the pre-registration files needed no such edit). Apart from those, each file is the copy at tag
`submission-2026-09-24`, including the one-line names note it received on that date.

> **Names.** Nearly every file here predates the 2026-09-24 rename. In them, **MIDIAN-VA** is today's **MIDIAN**; plain
> **MIDIAN** (the pre-registered tree) is **MIDIAN w/o defenses**; **MIDIAN-V** is **MIDIAN w/o audits**; **MIDIAN-A** is
> **MIDIAN w/o verification**; **MIDIAN-SH / MIDIAN-SHA** are withdrawn. See [`../errata.md`](../errata.md). File paths
> and section cross-references inside these files are those of their date (e.g. `configs/grid.yaml`, `RESULTS II.2`).

## Pre-registration (`preregistration/`)

`TARGETS_*.md` is each file **as registered**: byte for byte the version of its first commit. Where the file was
changed before the rename, the last pre-rename version is shipped beside it as `TARGETS_*.amended.md`, byte for byte
from that commit. Registered text and amendments are never merged into one file. Every amendment only appended text
(no registered line was changed or deleted: 0 deleted lines in each commit's diff), and each dates itself in the
text. `git show <commit>:TARGETS_rte*.md` reproduces any version; the rename notes later added to the working copies
are in none of them.

| file | registered (first commit) | blob | first results of that phase |
|---|---|---|---|
| `TARGETS_rte.md` | `daa4a9a`, 2026-09-02 01:56 | `d578ed4` | `RESULTS_rte.md`, `4901c13`, 2026-09-02 20:44 |
| `TARGETS_rte_v2.md` | `625f8a3`, 2026-09-02 23:22 | `f77e70e` | `RESULTS_rte_v2.md`, `3d901c2`, 2026-09-03 00:21 (numbers for closed grids `c16ed61`, 13:34) |
| `TARGETS_rte_v3.md` | `f3b90b5`, 2026-09-03 17:38 | `437aeae` | `RESULTS_rte_v3.md`, `1759270`, 2026-09-03 18:24 (T3-1 to T3-3) |
| `TARGETS_rte_v4.md` | `8ed4f47`, 2026-09-08 00:27 | `47282b9` | `RESULTS_rte_v4.md`, `85dab5a`, 2026-09-08 11:39 |

`TARGETS_rte.md` and `TARGETS_rte_v4.md` were never amended. The amendments (times US Eastern; **after results** = the
phase already had results in the repository when the amendment was committed):

| file | commit | date | lines | what it added | after results? |
|---|---|---|---|---|---|
| `TARGETS_rte_v2.amended.md` (blob `d49b5c2`) | `edbc078` | 2026-09-03 14:35 | +5 | V2-11, the three expectations for MIDIAN-VA (today's MIDIAN), stated as written before any MIDIAN-VA run | **yes**: v2 results for other targets existed (`3d901c2`, `c16ed61`) |
| `TARGETS_rte_v3.amended.md` (blob `a19a4df`) | `58e052c` | 2026-09-03 18:03 | +28 | part C: RouterBench's kNN / MLP routers as methods on our terms, T3-6 to T3-10, with speeds measured before launch | no |
| | `6d95947` | 2026-09-03 18:50 | +29 | part D: RouterEval on its own terms (pools of 10 / 100 / 1,000 LLMs), T3-11 to T3-16 | **yes**: T3-1 to T3-3 results existed (`1759270`); part D had not run |
| | `e7a0f77` | 2026-09-03 20:32 | +11 | D2: RouterEval's 5,000-LLM leaderboard pool, T3-17, with the numbers of one smoke seed of that grid written into the text | **yes**: committed together with the part-D1 results (T3-11 to T3-13 verdicts), and after a smoke run of its own grid |
| | `185df61` | 2026-09-04 02:28 | +13 | part E: MIDIAN-VA added to every grid, scale to 10^4 / 10^5, T3-18 to T3-21 | **yes**: v3 parts A-D results existed |
| | `35a8017` | 2026-09-04 02:37 | +14 | part F: LLMRouterBench on its own terms, T3-22 to T3-24 | **yes**: v3 parts A-D results existed; part F had not run |

Intermediate versions: v3 after `58e052c` is blob `85cda71`, after `6d95947` `b077ab0`, after `e7a0f77` `02b22e5`; after
`185df61` it is `git show 185df61:TARGETS_rte_v3.md`.

## Specification, deviations, errata

| file | dates | what it is |
|---|---|---|
| `SPEC.md` | 2026-09-02 | the original study specification: world, the tree, the rivals, the frameworks (§6A), figures, compute plan |
| `DEVIATIONS.md` | 2026-09-02 to 2026-09-24 | every departure from the specification, as a dated bullet with its reason |
| `CHANGES_AND_ERRATA.md` | 2026-09-04 to 2026-09-24 | what earlier drafts had wrong or incomplete, errata 1-30, the rename (§8g); [`../errata.md`](../errata.md) is the current summary |
| `COVERAGE.md` | 2026-09-17 to 2026-09-24 | per-method-family audit of what was run, with justification verdicts and campaign state |

## Results write-ups (`results/`)

Generated from the stored rows at their date; superseded by the paper figures and their CSVs.

| file | dates | what it is |
|---|---|---|
| `RESULTS.md` | 2026-09-04 to 2026-09-24 | the consolidated dossier of phases 1-3, each table tagged with its status |
| `RESULTS_rte.md` | 2026-09-02 | the frozen phase-1 report, with verdicts on the six pre-registered targets |
| `RESULTS_rte_v2.md` | 2026-09-03 | v2: our benchmark, labeled variants and their targets |
| `RESULTS_rte_v3.md` | 2026-09-03 | v3: external comparisons (RouterBench, RouteLLM, RouterEval, LLMRouterBench) |
| `RESULTS_rte_v4.md` | 2026-09-08 | v4: cohort modes, pre-registered then measured |
| `RESULTS_channel_tables.md` | 2026-09-03 | per-declaration-channel tables, generated by `rte.analyze` |
| `RESULTS_energy.md` | 2026-09-03 | runtime and energy: the cost model and its estimates |

## Audits (`audits/`)

| file | dates | what it is |
|---|---|---|
| `NUMBER_AUDIT.md` | 2026-09-04 to 2026-09-24 | the audit of every quoted number against `paper/NUMBERS.json` |
