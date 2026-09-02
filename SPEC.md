# RTE from scratch — plain MIDIAN vs. self-contained rivals, along three axes

Start clean. New package `rte/`. No dependence on the old repo except copying `stable_hash.py`. Plain MIDIAN only (no -P, no -H). Every method is one self-contained file with a `build()` and a `fetch()`. One runner sweeps **n × β × skill-distribution × method × seed** with paired task streams.

---

## 1. The world (`rte/world.py`) — LLM agents first

One `World` interface, three backends. **`llm` is the experiment.** The other two exist only to test invariants and to extrapolate scale.

| backend | agents | task execution | use |
|---|---|---|---|
| **`llm` (primary)** | real models served by vLLM, each with a skill profile (below) | the chosen agent generates an answer; a programmatic verifier scores it | all headline results, n ≤ 10⁴ |
| `replay` | RouterBench's 11 real models × 64 categories, pre-recorded outcomes (HF `withmartian/routerbench`) | outcome read from the cell table | real-model outcomes at zero inference cost; n up to 10⁴ by sharding models into specialty profiles |
| `bernoulli` | synthetic, `S` drawn from §3 | `outcome ~ Bernoulli(S[a,f])` | unit tests, tree invariants, and the n=10⁵ cost-scaling curve only (calibrated from the measured `S` of the llm world). Never a headline number. |

**Agents (llm backend).** `n` agents; agent `a` = (base model, per-family handicap, tool set), exactly the skill-profiled population of `MIDIAN_EVAL_GUIDE.md`: base model from a real ladder (Qwen2.5-{0.5,1.5,3,7,14}B + Llama-3.2-{1,3}B or Gemma-2-{2,9}B), a specialty set of families served at full capability, other families handicapped (exemplars withheld, difficulty capped, family tool removed), tools ∈ {calculator, python sandbox, none}. Served through vLLM's OpenAI-compatible endpoints, temperature 0, `VLLM_ROCM_USE_AITER=0` on our nodes.

**Task families.** `K ∈ {16, 64}` Reasoning Gym generators (`pip install reasoning-gym`; verifier per family; seeded fresh instances → no contamination, no cache leakage). A task = `(family f, instance)`. **Execution = a real generation** by the routed agent; `outcome = score_answer(answer, entry) ≥ 0.99` (binary), charged as one task call.

**True skill `S[n, K]`.** Measured once per population, offline: every agent on 200 fresh instances per family (60 for ≥14B; wider CI). Used only by the runner for the oracle line, regret, and the `skill_excess_ratio ≥ 1.5` gate. **Never exposed to any method.**

**Declared skill `D[a] ∈ [0,1]^K`.** Two sources, both run: (i) `programmatic`: `D = S + N(0,0.05)` for honest agents (a clean control); (ii) **`self_described`**: the agent's *own LLM* is shown one example per family and asked to rate its competence per family in [0,1] — this is the ecological declared channel, and it is already miscalibrated at β=0 (LLMs overclaim; that is a measured finding, not an assumption). Liars (fraction β) then inflate on top (§4).

**Probes.** `view.probe(a, f) -> outcome`: a fresh instance of `f` is generated, agent `a` answers (real generation), the verifier scores it, the ledger is charged one probe. The only way any method learns anything true.

**Memoization (what makes the llm grid affordable).** Agent responses are deterministic at temperature 0, and the task stream is paired across methods, so cache `(agent, instance) → answer`. Every method's fetch of the same task to the same agent costs one generation *once* across the whole grid. Build probes are cached the same way. Report cache hit rate.

**Peer reports.** Decentralized methods learn through *reports*: when peer `j` observes a probe of `a`, it reports a value. Honest `j` reports the true outcome. A lying `j` may corrupt it (§4). Centralized methods bypass reports and get outcomes directly (a trusted central observer — an advantage we state, not hide).

**View** (what a method may touch): `view.n, view.K, view.declared (D), view.probe(...), view.report_channel(...), view.bus (messages)`. Access is declared per method and enforced (raises otherwise).

**Ledger.** Counters per method per run: `probes`, `reports`, `messages`, `hops`, `comparisons`, `tasks`. One increment site each.

---

## 2. The method interface (`rte/methods/base.py`)

```python
class Method:
    name: str
    needs: set[str]                     # subset of {"declared", "probe", "reports", "bus"}
    def build(self, view, budget: Budget) -> None: ...     # pre-emptive step; may spend budget.probes
    def fetch(self, task) -> int: ...                      # returns agent id; charges hops/comparisons/messages
    def observe(self, task, agent, outcome) -> None: ...   # online update; default no-op
```
`Budget.probes_per_agent_family = b` (default 3) ⇒ every method that probes gets the same `n·K·b` build budget. Methods that don't probe have a zero build cost and are charged only at fetch.

---

## 3. The three axes

**n:** {10², 10³, 10⁴, 10⁵, 10⁶, 10⁷}. Backend by scale: **live LLM agents at 10²–10⁴** (all quality results); **`replay` at 10⁵–10⁶** (real-model outcomes; profiles = RouterBench model × specialty shard, table lookups only); **`bernoulli` calibrated to the measured `S` at 10⁶–10⁷** (cost-scaling curves and the tree's invariants at depth 6–7). Every figure labels which backend produced each point. Engineering at 10⁷: `S` is 10⁷×64 float32 = 2.6 GB (chunk it); build probes are vectorized draws in chunks of 10⁶ agents; MIDIAN's tree is built by padding n to r^L and reshaping (depth ⌈log₁₀ 10⁷⌉ = 7); flat scans use vectorized argmax (≈10 ms per task at 10⁷ — fine, and it *is* the point: their cost is linear); `flat_nsw_router` uses hnswlib at 10⁷ points (tens of GB RAM — run on a large-memory node); EigenTrust power iteration on the sparse 10⁷-row report matrix (~30 nnz/row).

**β (liar fraction):** {0, 0.1, 0.25, 0.5}. Liar selection: `random` (default) and `low_skill_first` (agents with the lowest mean skill lie — the ecological case). Both run.

**Skill distribution** (`world.skill_dist`), `K ∈ {16, 64}`. In the `llm` backend a distribution is *realized* by how profiles are drawn (model ladder × specialty sets × handicaps), then **measured** — report the measured `S` summary (spread, `skill_excess_ratio`) beside every result. The `bernoulli` backend samples the same shapes directly.

| name | llm-backend profile draw | bernoulli generator | what it tests |
|---|---|---|---|
| `specialist` | each agent: 3 families unhandicapped, rest handicapped; models mixed | 3 families ~ U(0.7, 0.95), rest ~ U(0.05, 0.3) | expertise exists to be discovered |
| `heavy_tail` | 1 in 10 agents is a 7–14B model unhandicapped; the rest 0.5–1.5B handicapped | per family `Beta(0.5, 3)` rescaled | few experts, most poor — argmax matters most |
| `bimodal` | 20% agents = 7B with tools, 80% = 0.5B without | 20% ~ U(0.75, 0.95), 80% ~ U(0.2, 0.4) | "good agents" vs "bad agents" |
| `correlated` | families in 4 groups; an agent's handicaps are group-level | group skill + N(0, 0.05) | summaries compress well |
| `iid_uniform` | per-(agent, family) handicap drawn independently | `U(0.2, 0.9)` | no structure; pure argmax-finding |

Live grid runs the first three at n ∈ {100, 1,000} (all four β, all methods); `correlated` and `iid_uniform` at n=1,000 only; n=10⁴ live for `specialist` at β ∈ {0, 0.25} with b=1, else `replay`; n=10⁵ cost curves on `bernoulli` calibrated to the measured `S`.

---

## 4. Lying model (`rte/world.py: apply_lying`)

Two channels, both simple and explicit:

- **Declared-channel lie (all liars):** `D[a] = clip(S[a] + δ, 0, 1)` with δ = 0.4 (`inflate`); variant `squat`: set `D[a, f*] = 1.0` on the top-3 highest-demand families. Default: `inflate`.
- **Report-channel lie (only matters for methods with `needs ⊇ {"reports"}`):** when liar `j` reports about agent `a`: if `a` is a liar → report 1; if `a` is in the top-20% honest by *j's* observed outcomes → report 0; else truthful. Toggle `collude ∈ {False, True}`; default True at β>0.

Liars execute tasks at their true skill — lying changes what they *say*, not what they can do.

---

## 5. Plain MIDIAN (`rte/methods/midian.py`) — `needs = {"probe", "reports"}`

**Idea.** A tree of cohorts. Leaves are agents. Every node carries, per family, a *summary* = the best estimated skill in its subtree and *which child* holds it. Routing descends from the root choosing the best child per family at each level: `⌈log_r n⌉` decisions, each over `r` children. Skill estimates come only from verified probes reported by cohort peers, aggregated by a trimmed mean.

```
build_midian(view, r=10, b=3, delta=1/3):
    agents -> random partition into cohorts of size r            # stratified-random by declared mean if flag set; default random
    # level 0: peer-probed estimates
    for cohort C, member m, family f:
        outcomes = [view.probe(m, f) for _ in range(b)]           # executed by / observed via peers
        reports  = [view.report_channel(j, m, o) for j in C\{m} for o in outcomes]   # each peer reports what it saw
        est[m, f] = trimmed_mean(reports, trim_each_side = floor(delta * (r-1)))
    for cohort C:
        summary[C, f]    = max_m est[m, f];   best_child[C, f] = argmax_m est[m, f]
        leader[C]        = argmax_m mean_f est[m, f]              # representative; carries the summary upward
    # level 1..L: leaders form cohorts of r; recurse on summaries
    nodes = cohorts
    while len(nodes) > 1:
        groups = random partition of nodes into groups of r
        for G, f: summary[G, f] = max_{C in G} summary[C, f];  best_child[G, f] = argmax_{C in G} summary[C, f]
        nodes = groups
    root = nodes[0];  depth = ceil(log_r n)

run_midian(task f):
    node = root
    while node is not a leaf cohort:
        node = best_child[node, f]            # 1 comparison-set of size r per level  -> charge hops += 1, comparisons += r
    return best_child[node, f]                # the agent

observe(task f, agent a, outcome):            # optional online mode, default on
    est[a, f] = running mean update
    walk up a's path: recompute summary/best_child at each ancestor   # log_r n updates
```
Notes. (i) With exact estimates this is a tournament (max-)tree: descent returns the global argmax per family — so MIDIAN's *quality* ceiling equals flat argmax, and its *cost* is O(r·log_r n) per task instead of O(n). (ii) Its robustness comes entirely from (a) never reading `D`, and (b) trimming ≤ ⌊δ(r−1)⌋ = 3 corrupted reports per cohort. (iii) Upper levels use summaries, not new probes, so build cost is exactly `n·K·b` probes + `n·K·b·(r−1)` reports. (iv) `r`, `b`, `delta` are sweepable; defaults fixed.

---

## 6. The rivals (one file each, `rte/methods/`)

**Floors / ceilings**
- `random.py` — uniform random agent. `needs = {}`.
- `oracle.py` — `argmax_a S[a, f]`. Reported as the ceiling line only; never in method tables. (Runner-only.)

**Declared-channel only (what practitioners deploy)**
- `declared_argmax.py` — `argmax_a D[a, f]`. `needs = {"declared"}`. Flat scan O(n).
- `declared_softmax.py` — sample ∝ exp(D[a,f]/τ), τ=0.1 (models an LLM-supervisor's soft pick).
- `cnp_self_bid.py` — broadcast task; each agent bids `D[a,f] + N(0,0.02)`; argmax. Decentralized; `messages += 2n`.
- `disrouter_cascade.py` — agents ordered by declared cost proxy (mean D); each answers if `D[a,f] ≥ τ=0.7` else forwards; `hops` = position of first taker. `needs = {"declared","bus"}`.
- `cluster_head_router.py` (AgentNet++-style prior art) — k-means on `D` into clusters of ~r; head = argmax declared mean; route: pick cluster by head's declared[f], then argmax declared within it. Two-level, self-reported.

**Verified outcomes, centralized (trusted observer)**
- `flat_probe_argmax.py` — probe every agent `b` times per family; `est = mean`; `argmax_a est[a, f]`. **The key control: same probes as MIDIAN, no hierarchy, no report channel.** O(n) scan per task (or O(1) via cached argmax — report both).
- `ucb_per_family.py` — arms (a, f); UCB1; warmup = the same `n·K·b` budget spread uniformly; online `observe`. `needs = {"probe"}`.
- `thompson_per_family.py` — Beta posteriors; same warmup.
- `sequential_halving.py` — per family, fixed-budget best-arm ID with budget `n·b`; returns the identified arm; no online update.
- `verify_on_claim.py` — rank by `D[·, f]`; probe the top candidate `k=3` times; accept if mean ≥ `D − 0.15`, else demote and try next (max 5 tries); cache verdicts per (a, f). `needs = {"declared","probe"}`. **The most dangerous baseline.**
- `warm_start_bandit.py` — Beta prior with pseudo-count `n0=5` at mean `D[a,f]`; update from probes/observations; Thompson pick. `needs = {"declared","probe"}`.
- `trueskill_per_family.py` — pairwise: sample pairs, probe both on the same instance, update TrueSkill (`pip install trueskill`) per family; fetch = argmax μ. Same probe budget.

**Verified outcomes, decentralized, non-hierarchical (the structural alternatives to MIDIAN's tree)**
- `referral_network.py` — random d-regular graph (d=10); each node keeps per-neighbor per-family beliefs from observed outcomes (via report channel); fetch: start at random node, ask neighbors, follow best referral up to depth 4, return best-believed agent. `needs = {"probe","reports","bus"}`.
- `gossip_reputation_greedy.py` — EigenTrust over the report matrix (power iteration, no pre-trusted seed), then greedy forwarding on trust×est over a T-Man similarity graph built from `est`. `needs = {"probe","reports","bus"}`.
- `flat_nsw_router.py` — hnswlib index over `est` vectors (from the same probes, trusted-observer mode); query vector = one-hot(f); greedy search returns nearest; `hops` = search hops. The E7 flat rival. `needs = {"probe"}`.

**Route-to-many**
- `route_to_k_majority.py` — top-k by `D` (k=3), execute all, majority; `tasks += k`. `needs = {"declared"}`.

That is 18 rivals + MIDIAN. Each file ≤150 lines. Each has a unit test that it (a) respects `needs`, (b) returns a valid agent, (c) charges the ledger.

---

## 7. Runner (`rte/run.py`) and metrics (`rte/analyze.py`)

```
for seed in 1..10:
  for n, beta, dist in grid:
     world = World(n, K, dist, beta, liar_select, collude, seed)
     stream = tasks(Q=2000, families ~ uniform or demand-skewed, seed)      # SAME for all methods
     for M in methods:
        view = world.view(needs=M.needs); M.build(view, Budget(b=3)); ledger.reset()
        for task in stream:
            a = M.fetch(task); o = world.execute(a, task); M.observe(task, a, o); ledger.log(...)
        write row(n, beta, dist, method, seed, metrics..., ledger...)
```

**Metrics per run:** `success` = mean outcome over the stream (primary); `regret` = success(oracle) − success; `misroute_to_liar` = fraction of tasks routed to a liar; `build_probes`, `build_reports`; per-task `hops`, `comparisons`, `messages`; `wall_clock_build`, `wall_clock_per_task`; for online methods, success over the last 500 tasks (`success_late`) to show learning.

**Stats.** 10 seeds, paired by stream; report mean ± 95% Wilson/bootstrap CI; paired sign test MIDIAN vs each rival per cell; a delta inside the seed envelope is `WITHIN_FLOOR`.

**Figures.**
- F1 `success vs β`, one line per method, panel per skill distribution, n=1,000. (Declared-channel methods should fall; probe methods flat; MIDIAN flat until report-channel capture.)
- F2 `success vs n` (log-x, 10² → 10⁷) at β=0.25, panel per distribution; marker shape = backend (live / replay / bernoulli-calibrated).
- F3 `per-task cost vs n` (log-log, 10² → 10⁷): comparisons, hops, and wall-clock per task. MIDIAN ~ r·log₁₀ n (depth 2 → 7); flat scans ~ n; bandits O(1) after warmup; NSW ~ log n. Fit exponents with CIs — this is the figure the 10⁷ point exists for.
- F4 `success vs build budget b ∈ {1,3,10}` at n=1,000, β=0.25.
- F5 heatmap method × distribution at β=0.25, n=1,000.
- F6 `misroute_to_liar vs β`.
- F7 MIDIAN internals: success vs r ∈ {5,10,20} and δ ∈ {0, 1/3}, with `collude` on/off — does trimming matter at all?

---

## 8. Pre-registered expectations (commit `TARGETS_rte.md` before running)

1. All declared-channel methods lose ≥0.25 success from β=0 → 0.5 under `inflate`; MIDIAN and all probe-only methods move ≤0.03.
2. MIDIAN's success equals `flat_probe_argmax` within 0.02 at β=0 (it is a max-tree over the same estimates) and its per-task comparisons scale as r·log_r n while flat scans scale as n (fit exponents; report).
3. With `collude=True`, MIDIAN degrades once expected liars per cohort exceeds 3 (β·r > ⌊δ(r−1)⌋ ⇒ β > 0.3); trimming vs no trimming (δ=0) separates *only* in that regime. If it doesn't, say so.
4. `verify_on_claim` matches oracle within 0.03 at β ≤ 0.1 and loses ≥0.10 by β=0.5 (verification budget drains on liars).
5. `sequential_halving` ≈ `flat_probe_argmax` at equal budget; bandits' `success_late` ≥ MIDIAN's when build budget b=1 (online learning wins when build is thin).
6. Under `heavy_tail`, the gap between argmax-finding methods and `random`/`route_to_k` is largest; under `iid_uniform`, smallest.

Misses are reported as misses. No parameter changes after the first run.

---

## 9. Layout, tests, order

```
rte/
  world.py  ledger.py  budget.py  stable_hash.py
  methods/base.py  methods/midian.py  methods/<rival>.py  (18 files)
  run.py  analyze.py  configs/grid.yaml
  tests/test_view_enforcement.py  test_ledger.py  test_midian_tree.py  test_each_method.py
```
Tests before the grid: `View` raises on undeclared access (mutation test); MIDIAN tree invariants (every agent in exactly one leaf cohort; depth = ⌈log_r n⌉; `summary[node,f] == max over children` exactly when estimates are exact — run with `b→∞` mock); ledger conservation; each method returns a valid agent on 100 random tasks at n=100.

Order: world (all three backends) + ledger + base + tests on `bernoulli` → population build + measured `S` on the `llm` backend (gate ≥ 1.5) → MIDIAN + random + declared_argmax + flat_probe_argmax live at n=100 (the four that define the story) → F1 live at n=1,000 → remaining rivals live → n=10⁴ live (`specialist`), `replay` 10⁵–10⁶, `bernoulli` 10⁶–10⁷ for F2/F3 → figures → `RESULTS_rte.md`.

**Two LLM-native additions to the method set** (llm backend only):
- `llm_supervisor.py` — the practitioner default, run for real: retrieve top-20 agents by description embedding, then a supervisor model (Qwen2.5-7B) reads the 20 descriptions and picks one. `needs = {"declared"}`. At n=1,000 the descriptions don't fit a context, so retrieval-then-pick is what real systems do — say so.
- `midian_llm_descent.py` — ablation of plain MIDIAN where each *leader* is an LLM shown its r children's summaries and asked to choose, instead of the arithmetic argmax. Same tree, same estimates. Tests whether an LLM in the loop adds or subtracts from the descent.

**Compute plan (live grid).** n ∈ {100, 1,000} × β 4 × dist 3 × ~20 methods × 5 seeds × Q=1,000 paired tasks ≈ 2.4M routed generations *before* memoization; with the (agent, instance) cache the unique generations are the union of routed pairs — typically 5–15% of that. Build probes: n·K·b = 48K (n=1,000, K=16, b=3) per population, cached across methods. Small-model mixes on 4×H100 sustain tens of thousands of tokens/s; expect hours, not days. n=10⁴ live: build 480K probes at b=1 → 160K; run only `specialist` at β ∈ {0, 0.25}. No live runs above 10⁴; those points are `replay`/`bernoulli` and labeled.

---

## 6A. The popular frameworks as rivals (`rte/methods/frameworks/`) — run the real libraries

The comparisons the paper leads with are the agent-management systems practitioners actually use, run **through their own libraries** against our vLLM endpoints, with their **own multi-agent selection primitive** deciding which agent gets the task. Versions pinned in `requirements-frameworks.txt`; all facts below verified 2026-09-02 against GitHub/PyPI/docs.

**Inventory (stars / downloads-per-month where found):**

| Rival file | Framework, package, version | Stars · DL/mo | License | Selection primitive (what it reads) |
|---|---|---|---|---|
| `fw_langgraph.py` | LangGraph `langgraph` 1.2.11 + `langgraph-supervisor` 0.0.31 | 40.9k · 64.5M | MIT | `create_supervisor(agents, model, prompt)` → `transfer_to_<name>` tools. **Descriptions are not injected by default** — put them in `prompt` (the docs' own pattern) |
| `fw_crewai.py` | CrewAI `crewai` 1.15.18 | 58.0k · 27.2M | MIT | `Crew(process=Process.hierarchical, manager_llm=…)` → `DelegateWorkTool(coworker,…)`; manager sees **only `role` strings** → put the self-description in `role` |
| `fw_autogen.py` | AutoGen `autogen-agentchat` 0.7.5 (maintenance mode) | 60.8k · 0.9M | MIT | `SelectorGroupChat(participants, selector_prompt, candidate_func, emit_team_events=True)`; roster = `name: description` |
| `fw_magentic_one.py` | Magentic-One (in AutoGen) | — | MIT | `MagenticOneGroupChat` progress-ledger JSON `next_speaker` |
| `fw_maf.py` | Microsoft Agent Framework `agent-framework` 1.16.0 (successor of AutoGen + Semantic Kernel) | 13.3k · 2.2M | MIT | `GroupChatBuilder(orchestrator_agent=…)` structured `AgentOrchestrationOutput.next_speaker`; `HandoffBuilder` (`handoff_to_<id>` tools with agent `description`) |
| `fw_openai_agents.py` | OpenAI Agents SDK `openai-agents` 0.22.0 | 29.1k | MIT | triage `Agent(handoffs=[…])` → `transfer_to_<name>` with `handoff_description` |
| `fw_google_adk.py` | Google ADK `google-adk` 2.8.0 | 21.4k · 19.7M | Apache-2.0 | `LlmAgent(sub_agents=[…])` auto-delegation → `transfer_to_agent(agent_name)`; roster = "Agent name / Agent description" |
| `fw_llamaindex.py` | LlamaIndex `llama-index-core` 0.14.24 | 52.0k · 12.8M | MIT | `LLMSingleSelector.select([ToolMetadata(name, description)], task)` (pure selection) and `AgentWorkflow` `handoff(to_agent)` |
| `fw_smolagents.py` | HF smolagents `smolagents` 1.26.0 | 29.1k | Apache-2.0 | `ToolCallingAgent(managed_agents=[…])`; roster = `def name(task): """description"""` |
| `fw_camel_workforce.py` | CAMEL/OWL `camel-ai` 0.2.90 | 17.7k / 20.1k | Apache-2.0 | `Workforce` coordinator `ASSIGN_TASK_PROMPT` → JSON `assignee_id`; roster = `id:description:toolkits` |
| appendix `fw_metagpt.py` | MetaGPT `metagpt` 0.8.2 (stale, Py<3.12) | 70.2k · 16k | MIT | `TeamLeader.publish_team_message(send_to)` — SOP-hardwired; report with caveat |
| appendix `fw_agentscope.py` | AgentScope 2.0.7 | 30.4k | Apache-2.0 | no selection primitive — DIY structured-output router; report as such |
| cite only | AWS Bedrock multi-agent (supervisor; hard limit 10 collaborators; Bedrock models only), Azure Foundry connected agents (classic; depth 2; Foundry models only) | — | — | not reproducible against vLLM — reference in text |

Every one reads **names + self-descriptions** (and nothing else) to select ⇒ `needs = {"declared"}` for all. That is the point: these are the systems that will route to whoever describes themselves best.

**Common scaling adapter (applied identically to all, and it is what their own docs prescribe):** frameworks enumerate agents in a prompt or a tool list; LangGraph's docs say enumeration is for "< 10 agents" and Bedrock hard-caps at 10. So at n ≥ 100: embed self-descriptions once; per task retrieve top-k (k=10) by description similarity; build the framework object over those k; let the framework's primitive select among them. Report k-sensitivity (k ∈ {5, 10, 20}) for two frameworks. The retrieval step reads descriptions too, so the information class is unchanged.

**`fetch(task) -> agent_id` interception recipes** (selection only; kill the run after the pick; ledger charges one supervisor call + k descriptions):
1. LangGraph — `create_supervisor(topk, model, prompt=roster)`; stream `updates`, return the first `transfer_to_*` tool-call name.
2. CrewAI — hierarchical crew of top-k, one task; subscribe `ToolUsageStartedEvent`, return `tool_args["coworker"]`.
3. AutoGen — `SelectorGroupChat(all, candidate_func=lambda _: topk_names, emit_team_events=True)`, `MaxMessageTermination(1)`; return `SelectSpeakerEvent.content[0]`. Magentic-One: parse `next_speaker` from the first ledger.
4. MAF — `GroupChatBuilder(participants=topk, orchestrator_agent=…)`; return first `GroupChatRequestSentEvent.participant_name` (or `HandoffSentEvent.target`).
5. OpenAI Agents SDK — triage agent with `handoff(a, is_enabled=lambda ctx,_: a.name in topk)`, `Runner.run(max_turns=1)`; return `to_agent.name` from `RunHooks.on_handoff`.
6. Google ADK — router `LlmAgent(sub_agents=topk)`; return the first `event.actions.transfer_to_agent`.
7. LlamaIndex — `LLMSingleSelector.select([ToolMetadata(...)]*k, task).ind` — no agent execution at all.
8. smolagents — `ToolCallingAgent(managed_agents=topk, max_steps=1, step_callbacks=[cb])`; return `ActionStep.tool_calls[0].name`.
9. CAMEL — `Workforce` with top-k workers; `WorkforceCallback.log_task_assigned(TaskAssignedEvent)` → `assignee_id`.

Model endpoints: every framework accepts an OpenAI-compatible `base_url` (`OpenAIChatCompletionClient`, `ChatOpenAI`, `LLM(model="openai/…")`, `OpenAIProvider`, `LiteLlm(api_base)`, `OpenAILike`, `OpenAIModel(api_base)`, `ModelFactory(VLLM)`). Supervisor/manager model = Qwen2.5-7B-Instruct for all (fixed), so the *only* thing that differs between framework rivals is their selection primitive and prompt.

**Comparative literature to cite:** MAFBench (arXiv:2602.03128, github CoDS-GCS/MAFBench) benchmarks these frameworks' orchestration overhead (framework choice alone changes latency >100×; pins autogen-agentchat 0.7.5, crewai, langgraph, openai-agents, agno); arXiv:2603.22651 compares sequential/parallel/hierarchical-supervisor patterns on 10k documents. **None of them test routing accuracy or description honesty** — that is our gap, stated in the intro.

The algorithmic rivals in §6 stay in the experiment as *mechanism controls* (what a probe-based centralized router, a bandit, or a flat graph achieves) — but the paper's headline comparison table is the framework table above vs MIDIAN.
