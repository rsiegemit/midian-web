# STATUS.md — RTE handoff (written 2026-09-02 13:55 ET, for context compaction)

Read with `SPEC.md` (the experiment), `CONTRACT.md` (interfaces + directives), `TARGETS_rte.md` (pre-registration),
`DEVIATIONS.md` (every departure, dated). Memory pointers: `~/.claude/projects/-n-home02-rsiegelmann/memory/project_rte_*.md`.

## 1. What exists (all committed in ~/rte, 64 commits; `RTE_DATA=/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte` holds data)
- Core: `rte/world.py` (View enforcement, lying model, report channel incl. float reports, paired streams, index-seeded
  probes shared across methods), `ledger.py`, `budget.py`, `methods/base.py`, `methods/_est.py` (probe_successes, BetaBandit,
  peer_reported_estimates, trimmed_by_reporter, observed_reports, greedy_walk), `_decl.py`.
- Backends: `bernoulli.py`, `replay.py` (RouterBench, K=67 real), `llm.py` + `families.py` (Reasoning Gym adapter, K=16 list
  final: basic_arithmetic chain_sum letter_counting syllogism family_relationships gcd lcm prime_factorization number_sorting
  simple_equations count_bits number_format time_intervals word_sequence_reversal binary_alternation calendar_arithmetic),
  `population.py`, `prompts.py`, `tools.py` (python tool only for agents ≥3B), `rte/measure.py`, `rte/llm_client.py`
  (content-hash memo, per-process shards + 30 s live refresh, replica endpoints "<model>#<job>", latency-aware pick, TTL 20 s).
- Methods: `midian.py` (plain = pre-registered, byte-identical; params r, delta, online; `verify=True` = MIDIAN-V with
  observers/b0/cached/top), 17 rivals (declared: random declared_argmax declared_softmax cnp_self_bid disrouter_cascade
  cluster_head_router route_to_k_majority; central verified: flat_probe_argmax ucb thompson sequential_halving[peer_reported]
  verify_on_claim warm_start_bandit trueskill; decentralized: referral_network gossip_reputation_greedy flat_nsw_router),
  LLM-native: llm_supervisor, midian_llm_descent; frameworks `methods/frameworks/fw_*.py` (10 working + fw_metagpt
  NotImplemented + fw_echo protocol check), shared `_common.py` (TF-IDF retrieval; `retrieval=midian` = MIDIAN-V leaf cohort),
  `_bridge.py` (JSON-lines worker subprocess per framework venv `$RTE_DATA/env/fw_*`).
- Runner/analysis: `rte/run.py` (grid → rows.d/*.json → rows.csv; `--seeds`, `--methods`, `--only k=v`), `configs/grid.yaml`,
  `rte/analyze.py` (aggregate, paired-vs-midian, F1–F7 + F3b cost panels, targets_check, summary.md), `scripts/check_methods.py`.
- Ops scripts: `serve_fleet.sbatch` + `_serve_env.sh` (7 models on 4 H100, max_model_len 8192, hermes tool parser),
  `serve_replica.sbatch`, `measure.sbatch`, `measure_all.sbatch`, `run_grid.sbatch`, `launch_live.sh` (RTE_ONLY, RTE_RUN_ARGS),
  `restart_wave.sh`, `fw_envs/*.sh`, `mock_openai_server.py`.
- Tests: `python -m pytest tests -q` (base python; framework tests need venvs + mock; llm tests need `$RTE_DATA/env/rte/bin/python`).

## 2. Runs
DONE (CPU): bernoulli mirrors of every live grid, bernoulli_cost_smoke, replay_scale (10^4–10^6), bernoulli_scale (10^2–10^7,
K=16, calibrated to measured S), plus midian-variant reruns of mirrors/scale. Results: `$RTE_DATA/results/<grid>/{rows.csv,summary.md,figs/}`.
RUNNING (live, one SLURM job per grid×method×seed, ~340 jobs, zero errors so far): live_core_n100, live_f1_n1000,
live_extra_n1000, budget_sweep, midian_internals (30 split jobs, --only per cell), live_n10k, fw_live_n100/n1000,
fw_live_n100_verified/n1000_verified, fw_k_sensitivity (done), fw_appendix (done). ETAs at 13:47: 0.5–8 h; fleet job 43867113
ends 2026-09-04 05:02. Populations: `$RTE_DATA/populations/<dist>_n<n>_K16_seed<s>/` (45, all gates passed; S per signature).
Monitor: `squeue -u $USER | grep rte_`; errors: `grep -l Traceback $RTE_DATA/logs/rte_*__*_*.err`; progress: the per-grid
rows-done script in this file's history (blocks/cells/method_specs/row_id over configs/grid.yaml).
GPU quota: 16/user incl. the user's other job; replicas pending on Priority start as GPUs free.

Sharding (2026-09-02 15:20): sequential per-method jobs for MIDIAN/halving/flat at n=1000 were ETA 50-110 h (each unit waits
behind ~1000 queued requests on the 7B and gemma-9b servers), so `launch_live.sh` now shards: `RTE_SHARD=dist,beta RTE_SEED_SHARD=1`
(one job per axis-value combo x seed, via `--only` + `--seeds`; `--only` values are yaml-typed). Slow fw jobs (magentic_one, verified
camel/langgraph/llamaindex) sharded by dist,beta. Idle 14B/1.5B replicas were replaced by gemma-9b + 7B replicas (RTE_MAX_NUM_SEQS 256/384;
the fleet's own servers cap at 64 seqs). midian_internals finished (300/300).

Profiling (17:30): with 100% memo hits a probe still cost 10-15 ms because every python tool call re-forked the 10 GB process;
tool runs are now memoised in the shared memo (`llm_client.memo_call`, commit 230f132) and `_refresh` skips unchanged shards by stat.
Shard jobs with >=2 units left were restarted to pick this up. Memo load on a compute node: ~107 s for 37M rows.


**2026-09-02 23:30 — everything complete (all grids 100%), RESULTS_rte.md audited against the rows, figures A–G in `figures/`, memo compacted (job 44095669: 2267 shards -> one `cache/memo_compact.sqlite`, 42.1M rows, 8.2 GB, 42 min), fleet + replicas kept up per user.**

(earlier note) **2026-09-02 22:00 — live programme complete; RESULTS_rte.md written.** All live grids closed except 5 budget_sweep rows
(halving b=10) and Magentic-One seeds 4-5 on the fw grids (24 shards running). Remaining: wait for those, rerun
`rte.analyze --grid budget_sweep fw_live_n100 fw_live_n1000`, refresh the §1/§7 numbers in RESULTS_rte.md if they move,
`python -m rte.llm_client compact`, keep the fleet and replicas UP (user request 2026-09-02 22:10) for follow-ups. Learning-curve script: `$RTE_DATA/scratch/curve.py`.

**2026-09-03 — v2 work order started.** TARGETS_rte_v2.md pre-registered (commit 625f8a3). Five parallel forks: fw-fixes (0.2),
analysis (0.1/0.5/0.6), midian-core (0.3/0.7/1.5 + Midian.churn), variants (1.1–1.3, 1.6), churn (2.1 infrastructure).
Grid blocks for 0.4 (fw grids 5 seeds / Q=1000), 1.4 midian_r20, 1.5 stratify, 2.2 live_n10k_v2, 2.3 midian_v_replication
(seeds 11-20), 2.4 budget_b10_shapes, 2.5 internals_v2, variants_f1 are in configs/grid.yaml; churn_n1000 block pending the
churn fork's API. All five forks landed (commits 160d55c, 5e40f9d; suite 371 passed, check_methods OK). LAUNCHED 2026-09-03: the ten
frameworks (+14B Magentic-One arm) on fw_live_n100/n1000 at Q=1000, 5 seeds (336 jobs; old Q=300 rows archived as
rows_v1_Q300.d); then, via the scratch launcher, the paired MIDIAN-side arms on the fw grids, both _verified grids,
midian_v_replication (360 units), internals_v2 (20), variants_f1 (120), midian_r20 (60), stratify (20), churn_n1000 (40),
live_n10k_v2 (6), budget_b10_shapes (12). Monitor as before; new code paths run live for the first time (midian_a probe_at,
churn redraw on the llm backend) so scan .err logs early. Then Phase 3 figures (H1-H9) and RESULTS_rte_v2.md.


**2026-09-03 01:45 — v2 HANDOFF (user going offline).** Running: ~2,300 CPU shard jobs (sapphire + shared) against the fleet
(43867113, 8 replicas incl. new 7B 44125204; fleet ends 2026-09-04 05:02 — v2 grids should close before that; if not, relaunch
`scripts/serve_fleet.sbatch` and re-run `launch_live.sh` for the grids with missing rows: it skips finished rows). Unattended
finisher `scripts/finish_v2.sh` (job 44125767, partition shared) waits for the last rte_ job, then runs rte.analyze on every v2
grid, the targets_v2 merge into results/v2_targets, scripts/extra_figs.py, copies figures/, and commits+pushes. Extra-seed
grids (fw grids seeds 6-10, variants_f1 seeds 6-10, live_f1_core_s6_10) are in the same queue. Background helpers in the
session scratchpad are all detached (nohup) and may be gone; nothing depends on them.
To resume in a new session: read this file, then (1) check `squeue -u $USER | grep rte_`, scan logs for Tracebacks (ignore
analyze.py-only tracebacks from old jobs), (2) for grids with rows missing, `RTE_GRIDS=<g> RTE_SHARD=dist,beta [RTE_SEED_SHARD=1]
RTE_PARTITION=shared scripts/launch_live.sh`, (3) after finish_v2 ran: fill every TODO(grid) in RESULTS_rte_v2.md from
results/<grid>/summary.md and results/v2_targets/summary.md, refresh README numbers, commit by explicit paths (author rsiegemit,
no Claude trailers), push to origin main. Known: MIDIAN-V v2 definition = per-probe reports (DEVIATIONS 2026-09-03); CrewAI
2026-09-02 rows were a corrupted-store artefact (archived as rows_v1_Q300.d).


**2026-09-03 15:05 — HANDOFF / COMPACT POINT.** Repo = GitHub rsiegemit/midian-web main (author rsiegemit, no Claude trailers).
DONE: v2 phases 0–1 code (all forks merged; suite 374+ green; `scripts/equivalence.py` guards MIDIAN refactors); grids complete:
variants_f1 (98.5%, +midian_va shards running), internals_v2, midian_r20, stratify, live_f1_core_s6_10, churn (461/480, gap-fill
running), budget_b10_shapes (103/108, tail-fill launched), live_n10k_v2 (120/126: the two peer-halving units died in the 12:52
registry gap after 8 h and were relaunched — ~8 h more), midian_v_replication (88%: midian_va added, seeds 11-20).
RUNNING (~1,650 CPU shards on shared+sapphire): fw_live_n100/n1000 (92%, 10 seeds, Q=1000; Magentic-One tail; ETA ~17:00),
fw_live_n100_verified 75% (~17:30), fw_live_n1000_verified 55% (~18:30), fw_live_n{100,1000}_lowskill (β=0.5 low-skill-first,
new, 39%/66%), midian_va on variants_f1 + replication. Serving: fleet 2 (44175863, all 7 models as "#44175863" aliases, ends
2026-09-05 ~10:30) + 8×7B replicas (4 of them until ~2026-09-04 13:00); primary fleet retired 12:52. endpoints.d = 15 entries;
`scripts/restore_endpoints.sh $RTE_DATA/endpoints_snapshot 44175863` re-registers after any wipe (serve_fleet no longer clears).
RESULTS_rte_v2.md: filled for every closed grid; §1/§2/App.E provisional (*) from ~90% fw grids; §1b verified and n=10k halving row
TODO; §6b energy/latency (H10/H11) final under stated assumptions; observe-time accounting (3415f03) applied everywhere.
KNOWN ISSUES: `scripts/finish_v2.sh` fired prematurely three times (squeue polls) — do NOT rely on it unattended; run it manually
once `squeue -u $USER | grep rte_ | grep -v serve` is empty. `midian.py` is 154 lines (guideline 150).
TO FINISH: (1) when fw grids close: rte.analyze both, `python scripts/extra_figs.py`, de-asterisk §1/§2/App.E, add the low-skill
block (fw_live_n*_lowskill) to §1 + an H1 panel, README headline numbers; (2) verified grids → §1b/H7; (3) midian_va → §4 + V2-11
verdict (targets_v2 merge: `--grid variants_f1 --grids stratify,internals_v2,live_f1_n1000,live_f1_core_s6_10,midian_v_replication,
churn_n1000,live_n10k_v2,budget_b10_shapes --out $RTE_DATA/results/v2_targets`); (4) n=10k halving row, churn/budget tails;
(5) final pass: RESULTS_rte_v2 status line, README, `python -m rte.llm_client compact` (memo ~9 GB, shards accumulate), commit, push.

## 3. Method variants in the grids (all paired on the same streams)
midian (v1, pre-registered), midian(online=false), midian(r=5), midian(verify,cached) = MIDIAN-V, midian(verify,cached,r=5),
midian_internals adds r∈{5,10,20}×δ∈{0,1/3} and V at r∈{5,10,20}; sequential_halving and sequential_halving(peer_reported);
flat_probe_argmax and (online); frameworks fw_* and fw_*(retrieval=midian, r∈{10,5}).

## 4. Findings so far (bernoulli/replay; live pending)
- Self-described channel at β=0 on the live population: overclaim +0.27, corr 0.36 with S, argmax-by-self-rating success 0.63
  vs oracle 0.85 vs random 0.38.
- Ties at b=3: ~116 agents/family tie at est=1.0 (n=1000) → flat argmax & plain MIDIAN pick blindly; halving's winner rests on ~12 probes.
- MIDIAN-V: +0.03..+0.14 over v1 at β≤0.25; loses at β=0.5 random liars; the only survivor at β=0.5 low-skill liars (bimodal
  0.85 vs peer-reported halving 0.36). Peer-reported halving > MIDIAN-V by ~0.03–0.07 at β≤0.25 (structural: early cohort elimination).
- Cost (calibrated bernoulli 10^2–10^7): MIDIAN comparisons/messages/hops ~ n^0.11, wall n^0.09; flat scans n^1.0.
  MIDIAN-V cached fetch = 1 comparison, 2 messages. Build: v1 48k probes/432k reports; V 47.8k/160k; peer halving 45k/404k (n=1000).
- Pre-registered targets on mirrors: 1 (declared collapse ≥0.25) MISS on bernoulli (channel nearly honest there; live is the test);
  2 MISS as written (online vs frozen; paired variants added); 3 consistent; 5 MISS (halving > flat by ~0.08); 4, 6 pending live.

## 5. To finish
1. When all live grids complete: `python -m rte.analyze --grid <g>` for each (run_grid.sbatch already does it per job; rerun once
   at the end) and combined: `--grids live_core_n100,live_f1_n1000,live_n10k,replay_scale,bernoulli_scale` for F2/F3.
2. Write `~/rte/RESULTS_rte.md`: headline table frameworks vs MIDIAN (fw_live_*: description-only vs verified shortlist),
   F1 by class with CIs, paired deltas + WITHIN_FLOOR, targets PASS/MISS with numbers, cost exponents, the b=3 tie caveat, the
   shared-S statistics caveat (seed CI excludes S measurement noise: ±0.035/±0.065 per cell), all DEVIATIONS summarized.
3. Cleanup: `python -m rte.llm_client compact` (merge memo shards) after the last job; scancel replicas; keep the fleet until done.

**2026-09-03 05:57 — finish_v2 (unattended): RAN PREMATURELY at 05:57 (its squeue poll failed once; grids were NOT closed — corrected in fdaae8f, resubmitted as job 44175864); analyses, targets_v2 merge (results/v2_targets) and figures regenerated. Next: fill RESULTS_rte_v2.md TODO(grid) markers from the summaries.**

**2026-09-03 11:30 — finish_v2 (unattended): RAN PREMATURELY AGAIN (three empty squeue listings under controller load while ~2,300 jobs were still queued; poll now also requires the finisher itself to appear in the listing); analyses, targets_v2 merge (results/v2_targets) and figures regenerated. Next: fill RESULTS_rte_v2.md TODO(grid) markers from the summaries.**

**2026-09-03 14:34 — finish_v2 (unattended): all v2 grids closed (3 consecutive empty polls); analyses, targets_v2 merge (results/v2_targets) and figures regenerated. Next: fill RESULTS_rte_v2.md TODO(grid) markers from the summaries.**
