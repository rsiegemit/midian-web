# Architecture

The benchmark is a small core (world, view, ledger, budget, method interface, backends) that every method and every
experiment builds on. This page is its contract: the public API, the conventions a method must follow, what each method
is charged for, and the correctness checks every method passes. The methods themselves are in [methods.md](methods.md);
how experiments are laid out is in [experimental_design.md](experimental_design.md).

## The world model

A **World** holds n agents and K task families. Each agent a has a true skill `S[a, f]` in [0, 1] per family: the
probability that it solves a task of that family. S is never shown to a method; the runner uses it only for the oracle
line, for regret and for choosing low-skill liars. A method learns about agents only through four **channels**, each of
which it must declare in its `needs` set; its `View` raises `AccessError` on anything else.

| channel | what it gives | who can corrupt it |
|---|---|---|
| `declared` | `D[n, K]`: what agents say they can do (a number per family, or a self-description text on the live backend) | liars inflate their own row (`lie_mode`: +0.4 clipped, or 1.0 everywhere) |
| `probe(a, f)` | run agent a on a fresh instance of family f; 0/1 outcome | nobody: outcomes are real |
| `reports` | peer j's report of what it observed about agent a | a lying reporter reports 1 for fellow liars and 0 for the top-20% honest (collusion) |
| `bus` | a message between agents | nobody, but every message is counted |

A fraction β of agents are **liars**, chosen at random or low-skill first. Lying changes what agents *say*, never what
they *do*, so probe-only methods are flat in β by construction and declaration readers are not.

## Core API

Files: `rte/world.py`, `rte/ledger.py`, `rte/budget.py`, `rte/methods/base.py`, `rte/backends/__init__.py`.

```python
Task(id: int, family: int, instance: int)     # instance = seed; the backend regenerates the concrete task from (family, instance)

World(n, K, dist, beta, liar_select="random", collude=True, seed=0, backend="bernoulli",
      lie_mode="inflate", declared_source="programmatic", demand="uniform", backend_kwargs=None)
  .n .K .families .S (runner only) .D .liars (runner only) .ledger .demand
  .tasks(Q, stream_seed=None) -> list[Task]           # the paired task stream
  .execute(a, task) -> 0/1                            # charges ledger.tasks
  .oracle(task) -> agent ; .oracle_all() -> argmax S per family
  .view(needs) -> View ; .stats() -> dict             # summary of S (skill_excess_ratio, ...)

View                                                   # what a method sees; AccessError on anything outside `needs`
  .n .K .families .ledger .rng .needs                 # always
  .declared -> D[n, K], read-only                     # needs "declared"
  .probe(a, f) -> 0/1 ; .probe_many(agents, families, reps) -> int8[len(agents), reps]    # needs "probe"; charged
  .report_channel(j, a, outcome) ; .report_many(reporters, agents, outcomes)              # needs "reports"; charged
  .bus.send(src, dst, payload) / .bus.send_many(k) / .bus.broadcast(src, payload)         # needs "bus"; charged

Ledger: .probe(k) .report(k) .message(k) .hop(k) .compare(k) .task(k) .snapshot() .diff(before) .reset()
Budget(probes_per_agent_family=3): .b ; .total_probes(n, K) = n * K * b

class Method:                                          # one subclass per file in rte/methods/, file name == method name
    name: str; needs: frozenset                        # subset of {"declared", "probe", "reports", "bus"}
    def __init__(self, **params): ...
    def build(self, view, budget): ...                 # one-time setup; at most budget.total_probes(n, K) probes
    def fetch(self, task) -> int | list[int]: ...      # route one task (a list = route-to-many)
    def observe(self, task, agent, outcome): ...       # optional online update
rte.methods.load_method("midian") -> class
```

**Backends** implement one six-member protocol (`true_skill`, `declared`, `execute`, `execute_many`, `stats`, plus
`n` / `K` / `families`) behind `rte.backends.make(name, n=, K=, dist=, seed=, rng=, **backend_kwargs)`, so every method
runs unchanged on every backend. `bernoulli.py` is the reference implementation. A backend may change n (for example by
rounding a pool size); the World re-reads `backend.n` and `backend.K`.

| backend | agents | families | true skill |
|---|---|---|---|
| `llm` (live) | LLM agents served by vLLM, each a (model, per-family handicap, tool) signature from a 7-model ladder (`configs/models.yaml`) | K = 16 Reasoning Gym families (`rte/backends/families.py`) | measured per signature (200 probes); agents write their own self-descriptions |
| `replay` | RouterBench's recorded outcomes of 11 models | its evaluation categories | recorded accuracy |
| `routereval` | real LLM pools of RouterEval (10 to 5,000 models) and LLMRouterBench (20 models) | MMLU subjects / datasets | mean train-prompt score |
| `bernoulli` | synthetic | K | drawn per population shape, optionally calibrated to the measured live S |

Population shapes (`dist`): `specialist` (three strong families per agent), `heavy_tail` (one in ten is a big model),
`bimodal` (20% big with tools, 80% small), `correlated` (group-level skill over four family groups), `iid_uniform`.

## Conventions

- **Seeding.** All randomness goes through `rte.stable_hash.stable_seed_32(*parts)`; never `hash()` of a string, never
  an unseeded generator. Results are therefore identical across processes and machines.
- **Paired probes.** The k-th probe of (agent, family) is the same task instance for every method (index-seeded), so
  memoised LLM answers are shared and method differences within a cell and seed are paired.
- **Route-to-many.** `fetch` may return a list; the runner executes every agent (tasks += k) and scores the majority of
  outcomes (ties fail). This is an optimistic proxy for majority-of-answers.
- **Build budget.** A probing method spends at most `budget.total_probes(n, K)` probes in `build`; audited variants
  may add their audit rate. Prefer `view.probe_many` (vectorised); scalar probing is infeasible at n = 10<sup>6</sup>.
- **Method files.** At most about 150 lines, self-contained (numpy plus an optional dependency imported lazily with a
  clear ImportError), parameters through `__init__(**params)` with defaults from the specification. A reader should see
  the whole algorithm on one screen.
- **Shared logic lives once.** Probe-then-estimate and trimmed peer reports: `rte/methods/_est.py`; declared-channel
  helpers: `rte/methods/_decl.py`; framework plumbing: `rte/methods/frameworks/_common.py`; LLM calls:
  `rte/llm_client.py`. <!-- VERIFY-PATH: helper modules after the rte/ refactor -->
- **Swappable by configuration.** Models are a list in `configs/models.yaml`; methods are files discovered by name; a
  task source is one adapter `{generate(instance_seed) -> entry, question(entry) -> str, score(answer, entry) -> float}`
  in `rte/backends/families.py`; backends are the six-member protocol.
- **Results.** One row per (cell, method, params, seed), written atomically; the runner is resumable and skips rows it
  already has by row id ([experimental_design.md](experimental_design.md#rows-and-row-ids)).

## Accounting: what every method pays for

The `Ledger` has one increment site per counter, and the world charges the channels itself, so a method cannot
under-report.

| counter | charged by | one unit is |
|---|---|---|
| `probes` | the world | one agent execution during build |
| `reports` | the world | one peer report |
| `messages` | the method (`view.ledger.message(k)`) | one coordination message; a query and its answer are 2 |
| `hops` | the method | one tree level or graph step |
| `comparisons` | the method | one candidate compared: a flat scan over n is n, a max over r children is r, a cached argmax is 1 |
| `tasks` | the world | one execution at route time (the dispatch itself; identical for every method) |

Total communication = probes + reports + messages + tasks. Wall-clock is recorded too, but it mixes memo hits and misses,
so every cost claim uses the counters.

**Message rules**, applied identically to every method, at build and at fetch:

- A query to one agent and its answer = 2 messages. Reading a centrally held table = 0.
- **Reports**: one per (reporter, member, family, probe) in every arm that reads reports (MIDIAN and its ablations,
  peer-reported halving, referral, gossip), so `reports` = probes x reporters per probe. `_est.peer_estimate` is the one
  place that probes-then-reports a batch of cells; `trimmed_by_reporter(rep, delta, s, exclude)` is the one aggregator.
- **Build**: collecting declarations from n agents = n messages (every method with `declared` in `needs` pays this
  once). Structure construction charges what it sends: MIDIAN = (r - 1) member-to-leader messages per level-0 cohort plus
  one leader-to-parent message per node above (O(n) in total); graphs = edges x messages exchanged.
- **Fetch** (messages per routed task): MIDIAN w/o defenses and MIDIAN w/o verification, which descend the tree, 2 per
  level = 2 ceil(log_r n); MIDIAN and MIDIAN w/o audits, which cache the root's pick, 2; centralized table lookups (flat
  probe argmax, bandits, sequential halving, verify-on-claim's ranking) 0, plus any probes they spend at fetch;
  contract-net bidding 2n; cluster-head router 2 (ask the head) + 2 (ask the member); cascade one per forward;
  referral and gossip 2 per neighbour consulted per hop; agent frameworks and the LLM supervisor k (descriptions read)
  + 2 (the supervisor call). Comparisons: a flat scan charges n, a max over r children r, a cached argmax 1.

Measured per routed query at b = 3 on the calibrated bernoulli backend (`figures/paper/C_routing_work_vs_n.csv`),
MIDIAN's messages plus comparisons are 25, 36, 47, 58, 69 and 80 at n = 10<sup>2</sup> to 10<sup>7</sup>, MIDIAN w/o
defenses' 46 to 161, and any flat scan's exactly n. A fitted exponent depends on the range of n it is fitted over, so
every quoted exponent names its range. <!-- VERIFY-PATH -->

## Correctness checks

Every method passes, on the bernoulli world at n = 100 and n = 1,000 (`tests/test_each_method.py`, which discovers every
file in `rte/methods/`, and `scripts/checks/check_methods.py` <!-- VERIFY-PATH -->):

1. **View enforcement.** Touching a channel outside `needs` raises; the test also removes a declared need to prove the
   enforcement is real.
2. **Valid routes.** Every fetch returns valid agent ids on 100 tasks.
3. **Budget.** Build probes <= `budget.total_probes(n, K)` (plus the audit rate for audited variants); no probes or
   reports without the corresponding need.
4. **Exact ledger.** Probes, reports, messages, hops and comparisons equal the documented formula; per-fetch messages
   are asserted for at least one method per class (MIDIAN without the cached pick: exactly 2 x depth; contract-net: 2n),
   and build messages are recorded in the row (`build_messages`).
5. **Better than random.** Success >= random's on `specialist` at β = 0. A router that loses to random is a bug until
   explained.
6. **Exact estimates.** Where a method is an argmax over estimates, it returns the true argmax when `probe_many` is mocked
   to return S.

Beyond these, the repository checks its own invariants: grid fingerprints (every grid enumerates the same units and row
ids as the golden file, `tests/golden/grid_fingerprints.tsv`), stored-row ids (a row's id recomputes from its fields),
parity (re-running stored rows reproduces every non-wall-clock column), and figure CSVs (regenerated figures match value
for value). They are in `scripts/checks/`.

## Framework arms

Each framework arm (`rte/methods/frameworks/fw_*.py`, 6 to 11 lines) hands a shortlist and the task to a worker process
running inside that framework's own virtual environment (`workers/*_worker.py`, JSON lines over `_bridge.py`) and returns
the agent the framework's own selection primitive chose. The shared adapter (`_common.FrameworkMethod`) does what a
practitioner would: a retriever picks the top k = 10 agents by their self-descriptions, the framework's supervisor
(Qwen2.5-7B) picks one, and a response that names no valid candidate falls back to declared argmax over the shortlist,
counted in `fallback_rate` and scored 0 under `success_strict`. An infrastructure error (a dead worker, a failed call) is
retried once, counted as `infra_errors`, and past max(3, 2% of calls) fails the unit so that no row is written; a supervisor's
own invalid action (a tool that does not exist, a non-candidate name) is a non-pick, not an infrastructure error. How
each framework's primitive is intercepted is documented per framework in [frameworks/](frameworks/).
<!-- VERIFY-PATH: docs/frameworks/ is filled by the rte/ refactor -->
