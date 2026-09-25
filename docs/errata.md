# Errata and changes

The current state of every correction that affects a reported number or how the stored rows are read. The full, dated
record (errata 1-30, every number that moved, the pre-registration verdicts) is
[archive/CHANGES_AND_ERRATA.md](archive/CHANGES_AND_ERRATA.md); every departure from the original specification is in
[archive/DEVIATIONS.md](archive/DEVIATIONS.md).

## Errata 25-30

| # | date | what was wrong | fix | effect on the paper's numbers |
|---|---|---|---|---|
| 25 | 2026-09-14 | The framework shortlist was clone-filled. Agents that share a prompt signature share a memoised self-description and memoised answers, so a stable TF-IDF top-10 fills with copies of one agent once a population has more copies than slots (every top-10 on the bimodal and heavy-tail populations; one signature at n = 10<sup>5</sup> specialist). | Labeled variant `dedup: true` (one agent per distinct description), run in separate `_dd` grids; later the dense, BM25, fusion and reranker shortlists. | Pre-registered rows unchanged and reported as such; the figures show each shortlist source separately (Figures E-H). |
| 26 | 2026-09-14 | The plain `sequential_halving` arm is scored by a trusted observer the setting never provides. | Withdrawn from every table and figure; only the peer-reported halving is a rival. | None drawn; its rows stay in the raw grids. |
| 27 | 2026-09-19 | The lie inflates the declared matrix but not the self-description text, so every text shortlist is identical across liar regimes. | Stated beside every cross-regime claim about a text shortlist; a partial lying-text condition (`lie_text`) was run separately. | Text-shortlist arms look liar-robust by construction; the declared-channel arms are the ones the lie attacks. |
| 28 | 2026-09-22 | Damaged framework environments made workers fail on some nodes; the adapter fell back to declared argmax and wrote rows that measured declared argmax under a framework's name (4,269 rows, mostly CrewAI and ADK). | Environments restored byte-identically; infrastructure errors now fail the unit; contaminated rows quarantined and rerun. | Only lowers framework numbers (the contamination had biased CrewAI and ADK upward). |
| 29 | 2026-09-23 | A supervisor's own invalid action (a tool that does not exist, a non-candidate name) was counted as an infrastructure error and failed units. Separately, RouterEval's MMLU family order depended on the hash seed (two subjects tie). | Invalid actions are non-picks (declared argmax inside the shortlist, 0 under `success_strict`, counted as `invalid_action`); ties broken by name. | Failed units rerun; RouterEval means unaffected. |
| 30 | 2026-09-23 | Three leaks on the non-live backends: declarations were a near-exact answer key (corr(S, D) 0.99 against 0.36 live); replay probed and routed on the same prompts; RouterEval / LLMRouterBench streams repeated test prompts. Also: RouterEval pools are stored weak to strong, so lowest-index tie-breaks favoured weak agents; and one probe call gave a repeated (agent, family) the same instance (affects only `trueskill_per_family`). | Opt-in cell values `declared_source: calibrated`, replay `split: true`, `no_repeat: true`, `shuffle: true`; the probe index fixed (changes default TrueSkill rows only). New `*_cal` / `*_norep_cal` grids. | Figure B's four non-live families and Figure H read only the new grids; declared argmax fell from 0.96-1.00 of oracle to 0.38-0.74. Live numbers unchanged. TrueSkill rows predating the fix are not read. |

Rival note (2026-09-22): the warm-start bandit scores higher under the cartel than honest at n = 10<sup>4</sup> and
10<sup>5</sup> (+0.04), because its declared-skill prior is too pessimistic on honest self-ratings; the lie clips liars'
prior failure count to zero. Reported as pre-registered; stated beside any honest-vs-cartel comparison of this arm.

## The MIDIAN rename (2026-09-24)

The full method is now called MIDIAN and the variants are its ablations. Nothing measured changed: the figure CSVs
regenerated on the renamed rows match the pre-rename ones value for value (737 rows, 0 differing), and 216 stored rows
rerun under the new keys are bit-identical.

| before | method key before | now | method key now | analysis label |
|---|---|---|---|---|
| MIDIAN-VA | `midian_va` | **MIDIAN** | `midian` (params `{}`) | `midian` |
| MIDIAN-A | `midian_a` | MIDIAN w/o verification | `midian{"verify": false}` | `midian_wo_verify` |
| MIDIAN-V | `midian_v`, `midian{"verify": true, "cached": true}` | MIDIAN w/o audits | `midian{"audit": false}` | `midian_wo_audit` |
| MIDIAN (plain, pre-registered) | `midian` | MIDIAN w/o defenses | `midian{"audit": false, "verify": false}` | `midian_wo_defenses` |
| MIDIAN-SH / MIDIAN-SHA | `midian_sh` / `midian_sha` | withdrawn | — | — |
| framework shortlist "VA cohort" | `retrieval: midian_va` | MIDIAN cohort | `retrieval: midian` | source key `va_cohort` (unchanged) |
| framework shortlist "V cohort" | `retrieval: midian` | cohort of MIDIAN w/o audits | `retrieval: midian_wo_audit` | not drawn |

- One class, `rte/methods/midian.py`, with two flags: `Midian(audit=True, verify=True)` is MIDIAN; each flag switches one
  defense off.
- `rte/methods/keys.py` is the one place that knows the old keys; every loader reads stored rows through it, and the
  runner seeds a world with the legacy key so a rerun reproduces the stored rows.
- The stored rows were re-keyed once (1,049,487 CSV rows and 35,407 row files); every changed or removed original was
  first copied to a `_premigration_v2/` backup in its results directory, and a sentinel file prevents a second pass. The
  withdrawn MIDIAN-SH / SHA rows exist only in those backups.
- The reference of every paired analysis is MIDIAN w/o defenses, the same arm as before the rename.
- Grid and directory names (`va_b_*`, `*_verified_va*`, `paired_vs_midian.csv`, ...) are identifiers and keep their
  spelling. Documents in [archive/](archive/README.md) keep the names of their date.

## What did not move

Through errata 25-30 and the rename: the oracle, random, every MIDIAN-family number, the flat-probe, declared and
supervisor arms on the live backend; the low-skill collusion results; the by-shape result (frameworks on the oracle
where skill is legible from a self-description, far below it on specialist populations); every external comparison on
RouterBench, RouteLLM, RouterEval and LLMRouterBench's own protocols; the MIDIAN-side energy numbers. Erratum 30 changes
only non-live rows, and on the live backend nothing changes.

## Row ids of grids that name data files

Eleven bernoulli grids pass a file under the data root in `backend_kwargs` (`calibrate_from: $RTE_DATA/populations/...`).
Their row ids used to hash the **expanded** path, so the same row had a different id on every machine and every stored
row carried an absolute path. `rte.run.row_id` now hashes `backend_kwargs` as the configuration writes them (the
unexpanded `$RTE_DATA/...` string). `scripts/checks/migrate_rte_data_ids.py` rewrites the stored rows of those eleven
grids once, dry run by default: each row gets the unexpanded `backend_kwargs` and its new row id, every other field is
written back byte-identical, the originals are first copied to `_premigration_rte_data_v1/` in the results directory,
and a sentinel prevents a second pass; it refuses to change anything if a stored id does not recompute or two rows would
collide. No value changes and no other grid's ids move. `tests/golden/grid_fingerprints.tsv` flags the eleven grids as
`rte_data_dep`, and `tests/golden/grid_fingerprints_d1.tsv` holds their new fingerprints.

## Reproducing stored rows exactly

The kNN rows of `rivals_b_n10k` (cartel), `rivals_b_n100k` and `routereval5k_norep_cal` (cartel) were produced with
batched MiniLM embeddings (`RTE_EMBED_BATCH=1`), which differ from the per-prompt path by at most 2.4e-7 and gave
identical neighbour sets on every query checked. Rerun those rows with `RTE_EMBED_BATCH=1` to reproduce them bit for bit;
the same reruns could also run MiniLM on a GPU in fp32 (`RTE_MINILM_CUDA=1`), which is the other setting to match.
`RTE_TEXT_PROCS`, `RTE_OUTCOME_CACHE`, `RTE_RG_CACHE` and `RTE_FW_PARALLEL` produce identical rows (same texts, outcomes,
tasks and picks; the framework prefetch is tested for identical picks).
