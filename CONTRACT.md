# CONTRACT.md — frozen interfaces every agent builds against

Read `SPEC.md` for the science. This file is the engineering contract. The core files
below are ALREADY WRITTEN and pass a smoke test; do not change their public API without
saying so in your final report (append-only additions are fine).

## Environment
- Cluster: Harvard FAS RC (SLURM). Login node has internet; compute nodes generally do NOT (set `HF_HUB_OFFLINE=1` in jobs).
  GPUs are NVIDIA (partitions `kempner_h100`, `kempner_h200`, `gpu`, `gpu_test` for smoke); account `kempner_sompolinsky_lab`.
  The spec's `VLLM_ROCM_USE_AITER=0` is from an old AMD cluster and does not apply here.
- Code lives in `~/rte` (git repo). Home quota is nearly full: ALL data, caches, model weights, envs, logs and results go under
  `RTE_DATA=/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte` (exists). Use `os.environ.get("RTE_DATA", ...)`.
- Python: `~/miniconda3/bin/python` (3.13, numpy 2.5, matplotlib) for CPU work. A dedicated env for vLLM/reasoning-gym is the
  llm-backend agent's job: `$RTE_DATA/env/rte` (conda, python 3.12, `pip install vllm reasoning-gym openai`). Extra CPU deps
  (pytest, pandas, scipy, pyyaml, hnswlib, trueskill, scikit-learn) go into that same env; until it exists, `pip install --user` into
  the base python is acceptable for pure-python packages needed to run tests.
- Run everything from `~/rte` with `PYTHONPATH=~/rte` (or `pip install -e .` once pyproject exists). Tests: `python -m pytest tests -q`.
- Never run GPU work on the login node. Submit sbatch; scripts go in `~/rte/scripts/`.
- Always seed through `rte.stable_hash.stable_seed_32(*parts)`; never `hash()` of strings; never unseeded RNG.

## Core API (rte/world.py, rte/ledger.py, rte/budget.py, rte/methods/base.py)
```python
Task(id:int, family:int, instance:int)      # instance = seed; backend regenerates the concrete instance from (family, instance)
World(n, K, dist, beta, liar_select="random", collude=True, seed=0, backend="bernoulli",
      lie_mode="inflate", declared_source="programmatic", demand="uniform", backend_kwargs=None)
  .n .K .families .S (runner-only!) .D .liars (runner-only!) .ledger .demand
  .tasks(Q, stream_seed=None) -> list[Task]           # paired stream
  .execute(a, task) -> 0/1                            # charges ledger.tasks
  .oracle(task) -> agent ; .oracle_all() -> argmax S per family
  .view(needs) -> View ; .stats() -> dict (S summary incl. skill_excess_ratio, skill_excess_ratio_family)
View  (what a method sees; raises AccessError on anything outside `needs`)
  .n .K .families .ledger .rng .needs                 # always
  .declared -> D[n,K] read-only                      # needs "declared"
  .probe(a,f) -> 0/1 ; .probe_many(agents, families, reps) -> int8[len(agents), reps]     # needs "probe"; fresh instances; charged
  .report_channel(j,a,outcome) -> value ; .report_many(reporters, agents, outcomes) -> int8  # needs "reports"; liar j may corrupt
  .bus.send(src,dst,payload) / .bus.send_many(k) / .bus.broadcast(src,payload)               # needs "bus"; charges messages
Ledger: .probe(k) .report(k) .message(k) .hop(k) .compare(k) .task(k)  .snapshot() .diff(before) .reset()
   -> methods charge ONLY hops/comparisons via view.ledger.hop(k)/compare(k); probes/reports/messages/tasks are charged by
      the world/bus automatically. Do not touch ledger.probes etc. directly.
Budget(probes_per_agent_family=3) ; .b ; .total_probes(n,K)
Method: name, needs (frozenset), __init__(**params), build(view,budget), fetch(task)->int, observe(task,agent,outcome)
   load via rte.methods.load_method("midian") -> class; exactly ONE Method subclass per file; file name == method name.
```
Backends (rte/backends/): `make(name, n=, K=, dist=, seed=, rng=, **backend_kwargs)`; protocol in `rte/backends/__init__.py`.
`bernoulli.py` is the reference implementation. `replay.py` and `llm.py` must implement the same 6 members.
A backend may change n (e.g. rounding); World re-reads backend.n/K.

## Conventions
- `fetch` may return an int, OR a list of ints for route-to-many methods: the runner executes every agent (tasks += k) and
  scores majority-of-outcomes (ties -> 0). Document that this is an optimistic proxy for majority-of-answers.
- `build` budget: probing methods spend AT MOST `budget.total_probes(n, K)` probes in build (assert it in your test).
  Prefer `view.probe_many` (vectorized) — at n=1e6+ scalar probes are infeasible.
- Every method file ≤150 lines, self-contained (numpy only + its own optional dep imported lazily with a clear ImportError).
  Params via `__init__(**params)` with defaults from SPEC §5–6 (e.g. `Midian(r=10, delta=1/3, online=True)`).
- Comparisons: a flat scan over n candidates charges `compare(n)`; a max over r children charges `compare(r)` and `hop(1)`.
  O(1) cached-argmax variants take a param (e.g. `cached=True`) and charge `compare(1)`.
- Tests: `tests/test_each_method.py` (runner agent) auto-discovers every file in `rte/methods/` and checks needs-respect
  (View raises if you touch more; test also mutates needs to prove enforcement), valid agent on 100 tasks at n=100 bernoulli,
  ledger charged, build-probe budget respected. Method authors add method-specific tests in `tests/test_<method>.py` only if
  something non-trivial needs checking (e.g. MIDIAN tree invariants).
- LLM access for LLM-native methods: `rte/llm_client.py` (llm-backend agent) exposes
  `complete(model: str, messages: list[dict], max_tokens=512, cache_key=None) -> str` against the vLLM endpoints listed in
  `$RTE_DATA/endpoints.json` (`{"Qwen/Qwen2.5-7B-Instruct": "http://host:port/v1", ...}`), temperature 0, disk-memoized by cache_key.
  Methods that need it import lazily and degrade with a clear error if no endpoints are configured.
- Results: one CSV row per (n, beta, dist, liar_select, collude, declared_source, method, params, seed) at
  `$RTE_DATA/results/<grid_name>/rows.csv`, appended atomically; runner resumable (skip rows already present).
- Report honestly. If something in SPEC is impossible or ambiguous, implement the closest faithful thing, and write the deviation
  into `DEVIATIONS.md` (append a dated bullet). Do not silently change the spec.

## Simplicity & modularity (lead directive, 2026-09-02 — overrides everything above on style)
- MINIMUM lines. No speculative flexibility, no abstractions for single-use code, no defensive handling of impossible cases.
  If a file can be 40 lines, it is 40 lines. Prefer one vectorized numpy expression to a loop; prefer a function to a class.
- ZERO duplication. Shared logic lives in exactly one place: probe-then-estimate → `rte/methods/_est.py` helper (`probe_estimates(view, budget)`);
  framework plumbing → `frameworks/_common.py`; LLM calls → `rte/llm_client.py`. If two files share 5+ lines, factor them.
- SWAPPABLE BY CONFIG, not by code edits: (a) models — the ladder is a list in `configs/models.yaml` (id, params_b, tp, gpu share); nothing
  hardcodes a model id except that file and `SUPERVISOR` in one place; (b) selection algorithms — already `Method` files, discovered by name;
  (c) datasets/verifiers — a task family is one small adapter `{generate(instance_seed) -> entry, question(entry) -> str, score(answer, entry) -> float}`;
  Reasoning Gym is ONE such adapter in `rte/backends/families.py`; adding a new source = one new adapter, nothing else changes;
  (d) backends — already the 6-member protocol.
- Read like a paper appendix: a reader should be able to see the whole algorithm of any method on one screen.
