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
