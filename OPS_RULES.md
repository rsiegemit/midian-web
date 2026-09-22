# OPS_RULES.md -- permanent operating rules for RTE campaigns

Every rule below was paid for by a real failure; the date and the damage are recorded so the rule is never "optimised
away". Each names where it is ENFORCED. A rule with no enforcement point is a wish -- add one or delete the rule.

## Cluster resources (FAS RC Job Defense Shield warnings, 2026-09-22)

| # | rule | why | enforced by |
|---|---|---|---|
| R1 | A framework / llm-backend unit requests **1 CPU**. | Units only wait on HTTP to the fleet. At 2 CPUs the campaign ran at 5% CPU efficiency over 129,533 CPU-hours (rank 13/931). | `scripts/launch_units.sh` |
| R2 | Walltime = **1.5 x measured p95** for that (population size, framework), never a flat multi-day limit. Floor 1 h, cap 3 d; p95s censored at an old cap get 2x. | Flat 1-3 day limits used 16% of allocated time across 9,571 jobs. Runtime varies 100x by framework (ADK p50 7 min, Magentic-One p95 23 h). | `scripts/job_time.py` <- `results/job_sizing.csv` |
| R3 | **Rebuild `job_sizing.csv` whenever the fleet changes** (replica count, placement). | Unit runtime tracks supervisor latency, i.e. fleet load -- not grid size. | manual; stated in `job_time.py` |
| R4 | The fleet requests **only the GPUs whose models get traffic**: supervisor-only replicas at 1 GPU (`configs/fleet_supervisor.yaml`) plus one all-models fleet packed on one H200 (`configs/fleet_packed_h200.yaml`) for memo misses and the 14B arm. 8 CPUs each. | The 4-GPU fleet left 3 GPUs at 0% for 384 GPU-hours: in 16 h the 7B supervisor took 487,222 requests, the other six models 8 in total (agent answers are memoised). | `RTE_FLEET_PLACEMENT` in `scripts/serve_fleet.sbatch` |
| R5 | Before sizing a new job type, **measure one** (runtime, MaxRSS, CPU, GPU util via `jobstats`), then size the rest from it. | Every JDS finding was a guess that was never checked against one real job. | practice; `jobstats <jobid>` |

## Fleet continuity

| # | rule | why | enforced by |
|---|---|---|---|
| F1 | Every fleet link is submitted with **two chained `afterany` successors**. | kempner's 2-day cap is shorter than a campaign. On 2026-09-19 both fleets hit it within hours; 13 h of framework jobs failed. | submission practice (see STATUS) |
| F2 | **Never fewer than 2 supervisor replicas**, on different partitions. | One replica is a single point of failure and halves throughput. | submission practice |
| F3 | Retire an old fleet **only after** its replacement serves (`endpoints.json` shows it). | Cancelling first opens a zero-replica window. | `scripts/ops/retire_old_fleet.sh` pattern |
| F4 | Keep `scripts/ops/fleet_watch.sh` armed during any campaign; it exits (notifies) on a replica-count change. | The 09-19 outage was found by noticing missing rows, 13 h late. | watcher |

## Data integrity

| # | rule | why | enforced by |
|---|---|---|---|
| D1 | **An infrastructure error never becomes a routing decision.** A supervisor/worker error is retried once, counted as `infra_errors`, and past 2% of calls the unit FAILS and writes no row. | A dead worker used to return `choice=None` and fall through to declared argmax, writing a normal row that measured declared argmax under the framework's name: ~4,500 rows (CrewAI, ADK). | `FrameworkMethod.fetch` + tests `test_infrastructure_errors_fail_*` |
| D2 | **Run `scripts/check_envs.sh` before every campaign**; it must print `ALL ENVS OK`. | 7 framework venvs had 98 dangling library symlinks each (targets deleted); jobs failed only on nodes lacking a compatible `/lib64` copy, so it looked like framework behaviour. | `scripts/check_envs.sh` (exit code) |
| D3 | Repair envs **byte-identically** (force-reinstall the owning package builds offline from the cache), never by rebuilding. | A rebuild can pull newer framework versions and silently change behaviour relative to rows already collected. | `scripts/ops/repair_fw_envs.sh` |
| D4 | After any campaign, **audit the non-pick rate** of every framework row (`scripts/ops/fallback_audit.py`). A row with fallback >= 0.9 is infrastructure until proven otherwise. | The CrewAI/ADK failure was bimodal per unit (0% or ~100% fallback), which is the signature of infrastructure, not behaviour. | audit script |
| D5 | Units come from the **grid loader**, never a hand-written list of filters. | A hand list enumerated 5 regimes where the grid had 6; one regime silently never ran. | `scripts/launch_units.sh` |
| D6 | **One warm-up unit** per new configuration, and check its rows AND its `method_stats` before launching the rest. | A warm-up that only checks "rows exist" would have passed every fallback-contaminated unit. | practice |
| D7 | Per-subset outputs (`--dist`, per-shape, per-b) **never share a path** with pooled outputs, and analyses **never pool across b**. | `scale_matrix --dist` overwrote the pooled matrix (bimodal read as pooled); a pivot over b=1 and b=3 invented a MIDIAN decay with n. | `scale_matrix.py` suffixes; `bar_figs` filters `b == 3` |
| D8 | A method never reads beta, the liar set, or S. Cache keys are **content hashes**, never regime labels. | The first `lie_text` cache key read `view.beta`, which the View rightly refuses to expose. | `View.__getattr__` (AccessError); `_ltag` hashes text |

## Operator hygiene

| # | rule | why |
|---|---|---|
| H1 | Kill processes **by PID only**, never `pkill -f` / `kill $(pgrep -f ...)`. | A pattern matched the invoking shell and killed the session. |
| H2 | Job scripts live under `$RTE_DATA/jobs/`, never the session scratchpad. | `/tmp` is login-node-local; compute nodes cannot see it. |
| H3 | Launchers are resumable; **the log is the state**. Re-run to continue; never restart with a fresh log. | A restarted launcher that could not see its first pass submitted 937 duplicates. |
| H4 | On `QOSMaxSubmitJobPerUserLimit`, wait and retry the **same** unit; never drop it. | Dropped units look like completed grids. |
| H5 | `kempner*` partitions need `-A kempner_sompolinsky_lab`; `kempner_requeue` and `test` cannot join a multi-partition submission; `scontrol update Partition=` drops a job's GPU request. | Each cost a failed submission. |
