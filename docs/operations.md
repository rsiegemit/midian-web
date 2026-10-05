# Operations: running a campaign

How the benchmark is run at scale: environments, the model fleet, the answer memo, sharding, merging, and the operating
rules learned from running tens of thousands of jobs against a shared GPU fleet. Nothing here changes a result; it is
about getting rows written correctly and cheaply. The scripts for one particular SLURM cluster are in `cluster/`
(`slurm/`: job files and launchers; `ops/`: campaign tools; `archive/`: one-off campaign drivers) and are not needed
to use the benchmark.

## The data root

Everything outside the repository lives under `$RTE_DATA` (`midian/config.py` documents every `RTE_*` switch):

```
$RTE_DATA/
  env/rte/              the base environment (vLLM, reasoning-gym, the [learned] and [llm] extras)
  env/fw_<framework>/   one isolated virtual environment per agent framework
  hf_cache/             model weights
  data/                 RouterBench cells, RouterEval / LLMRouterBench score tables
  populations/<dist>_n<n>_K<K>_seed<seed>/   live populations: S.npy, profiles, declarations, self-descriptions
  cache/                the answer memo (sharded SQLite) and embedding caches
  endpoints.json, endpoints.d/               the served models and their URLs
  results/<grid>/       rows.d/, rows.csv, summaries
```

## Environments

- The base environment holds the package, vLLM and the task generator. Build scripts are in `scripts/setup/`;
  `requirements.txt` pins the versions of the reported runs.
- **Each framework runs in its own virtual environment**, built from `requirements-frameworks/<framework>.txt` by
  `scripts/setup/fw_envs/<framework>.sh`, because the frameworks' dependencies conflict with each other. The adapter
  talks to a worker inside that environment over JSON lines, so no framework is imported into the benchmark process.
- Build with the user site-packages disabled (`PYTHONNOUSERSITE=1`); a user site on `sys.path` silently shadows a
  venv's own packages.
- **Check every environment before a campaign** (`cluster/ops/check_envs.sh`, which must print `ALL ENVS OK`). Damaged environments
  (for example dangling shared-library links after a cache cleanup) fail only on some nodes, which reads as framework
  behaviour. Repair byte-identically (force-reinstall the same package builds from the local cache), never by
  rebuilding: a rebuild can pull newer framework versions and silently change behaviour relative to rows already
  collected.

## The model fleet

- The live backend serves the model ladder of `configs/models.yaml` (seven models; id, parameter count, GPU share, tool
  gating) with vLLM, temperature 0. The framework arms need only the supervisor (Qwen2.5-7B).
- Each server registers itself as one file in `$RTE_DATA/endpoints.d/` (lock-free); `endpoints.json` is the merged view.
  Every server beyond the first registers as a replica, `<model>#<job>` (`RTE_AS_REPLICA=1`); a second server started
  without it overwrites the first one's entries and leaves it idle. A client picks the faster of two randomly chosen
  replicas by its own latency estimates, counts a failed or timed-out request at its elapsed time, and re-picks every
  20 s, so replicas added mid-run get used. (Always taking the fastest replica made every client herd onto one server
  while the others idled, and a stuck server kept looking fast because failures were never measured.)
- Before a stage, wait until every model answers its health check; never trust the registry alone, since a cancelled
  server can leave a stale entry or remove another server's.
- **Request only the GPUs that get traffic.** Agent answers are memoised, so after the first pass nearly all traffic is
  the supervisor: in one 16-hour window the 7B supervisor took 487,222 requests and the other six models 8. Serve
  supervisor-only replicas on one GPU each (`configs/fleet_supervisor.yaml`) and pack all models on one large GPU for memo
  misses (`configs/fleet_packed_h200.yaml`).
- **Never fewer than two supervisor replicas**, on different partitions or nodes; retire an old server only after its
  replacement is serving; resubmit each server with successors so a walltime limit never opens a gap.
- **When new seeds need fresh generations from every agent model** (not only the supervisor), serve each model group on
  its own GPUs: `configs/fleet_14b.yaml`, `fleet_mid.yaml` (9B / 7B / 3B), `fleet_small.yaml` (1.5B) and `fleet_tiny.yaml`
  (2B / 0.5B), with `RTE_MAX_NUM_SEQS` raised for the small models. The bottleneck moves from model to model as each gets
  capacity; measure queue lengths per model (`/metrics`) rather than guessing.
- **No model may ever have zero registered servers.** Servers end at the partition's walltime; submit replacements so
  they start before the old ones end, and keep a watcher running that alerts when any model has no server or the replica
  count changes.

## The answer memo

- Every model call goes through `midian/llm_client.py` and is memoised by a content hash of the full request (model,
  messages, max tokens). A reworded prompt therefore misses the cache automatically; a key built from the inputs of a
  prompt instead of its text would serve stale answers.
- The memo is **sharded per process**: each process writes only its own SQLite file under `$RTE_DATA/cache/` and reads
  every shard at start-up, so any number of processes share it without locks. Merge the shards between stages with
  `python -m midian.llm_client compact`.
- On network file systems where `fcntl` locks hang, every database is opened without locking; this is safe only because
  no two processes write the same file.
- Each process loads **every shard into memory** at start-up and keeps pulling rows other processes write, so its resident
  memory is about the size of the memo and grows during a long run (about 40 GB per process with a 43 GB memo). Size live
  units for the memo, not for the method: 64 GB for live units at n = 10<sup>5</sup>, 128 GB for kNN units with two cells.
- Tests use a private memo (`RTE_LLM_CACHE`), never the production one.

## Running grids

- A **unit** is one (cell, seed); a job runs one or more units of one grid, restricted with `--only k=v` (cells),
  `--seeds a-b` and `--methods`. Units come from the grid loader, never from a hand-written list of filters (a hand
  list once enumerated five liar regimes where the grid had six, and one regime silently never ran).
- Rows are written atomically, one file per row in `rows.d/`; a unit whose row ids are already on disk is skipped. Any
  job can be killed and resubmitted, and a launcher asked to submit only units with a missing row cannot disagree with
  the runner about which units are done.
- **One merger per grid.** Folding `rows.d/` into `rows.csv` is done by exactly one process per grid; while a
  `.merge_owner` file is present in the results directory, workers run with `RTE_CONSOLIDATE=0` and leave the fold to
  it (`python cluster/ops/progress.py --merge [--prune] [--every=S] <grid>...`).
- **Live units: one process per seed.** The runner forces one worker on the live backend, so pack seeds as separate
  processes. Non-LLM backends parallelise within a job with `RTE_WORKERS=N`; above n = 10<sup>5</sup> use one worker and
  one or two seeds per job (forked pools have deadlocked on long multi-wave jobs).
- **Framework units**: 1 CPU each (they wait on HTTP), `RTE_FW_PARALLEL=8` to send a unit's stateless framework
  requests concurrently (identical picks; MIDIAN-cohort shortlists and churn stay sequential), about 40 GB of memory.
- **Probe builds** keep `RTE_CONCURRENCY` requests in flight (default 64); a MIDIAN build costs about 16.5 n b probes, and
  only generations already in the memo are free.
- A unit covers every budget b of its cell in turn; `--only ...,b=5` runs one budget, so the cells of a unit can run as
  separate jobs in parallel. At n = 10<sup>5</sup> a kNN cell at b = 5 takes more than two days.
- **Completion is judged by rows, not by job states.** A unit can exit normally while one of its arms failed (an
  infrastructure error fails the arm, which writes no row), so compare the rows on disk with the plan and resubmit every
  missing unit that no queued job covers.
- Load any million-row CSV as a batch job, not on a shared login machine.

## Rules

Each rule was paid for by a real failure.

**Sizing.**
1. A framework or live unit requests **one CPU**; at two CPUs one campaign ran at 5% CPU efficiency.
2. Walltime is **1.5 x the measured p95** runtime for that (population size, framework), with a floor of one hour, never
   a flat multi-day limit. Runtimes differ by 100x between frameworks and track fleet load, so re-measure whenever the
   fleet changes.
3. Before sizing a new job type, **measure one job** (runtime, peak memory, CPU and GPU utilisation) and size the rest
   from it.

**Data integrity.**
4. **An infrastructure error never becomes a routing decision.** A worker or supervisor error is retried once and
   counted; past max(3, 2% of calls) the unit fails and writes no row. (A dead worker once returned "no choice", the
   adapter fell back to declared argmax, and about 4,500 rows measured declared argmax under a framework's name;
   [errata.md](errata.md), erratum 28.)
5. After a campaign, **audit every framework row's non-pick rate**; a unit with fallback >= 0.9 is infrastructure until
   proven otherwise. Infrastructure failures are bimodal per unit (0% or ~100%).
6. **One warm-up unit** per new configuration; check its rows and its `method_stats` before launching the rest.
7. Per-subset outputs (per shape, per b) never share a path with pooled outputs, and analyses **never pool across b**.
8. A method never reads β, the liar set or S, and cache keys are **content hashes**, never regime labels.

**Operator hygiene.**
9. Stop processes **by PID** (written to a pid file at launch), never by matching a command-line pattern: the pattern
   also matches the shell that types it.
10. Launchers are resumable and **the log is the state**; rerun to continue, never restart with a fresh log (a restarted
    launcher that could not see its first pass submitted 937 duplicates).
11. On a submission-limit rejection, wait and retry **the same** unit; a dropped unit looks like a completed grid.
12. Anything that must outlive an interactive session runs detached from it (a detached process or a self-rescheduling
    batch job), and concurrent drivers coordinate through heartbeat files, not process lookups.
13. Job scripts live on storage the compute nodes can see, never on node-local temporary space.
14. Before quoting an ETA, count **running** jobs, not only pending ones; under fair-share scheduling the number of slots,
    not job speed, is the bottleneck.
15. Long units never run on a **preemptible** partition: a preempted unit restarts from scratch (one lost 12 hours).
    Walltimes for long units come from measured cell times with margin; a partition's minimum and maximum walltime are
    checked before submitting (one partition rejects jobs shorter than three days).
16. An unattended campaign is supervised by a **batch job**, not by an interactive session: it resubmits failed and
    missing units, keeps every model served, removes registry entries of ended servers, and resubmits itself before its
    walltime ends. A watcher inside an interactive session stops with the session.
