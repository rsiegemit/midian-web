# METHODS.md — every method in the benchmark: where it comes from, what it is, what it does

One file per method in `rte/methods/` (frameworks in `rte/methods/frameworks/`). Every method sees only what it pays
for: `needs` ⊆ {declared, probe, reports, bus}. `build` may spend the shared budget of n·K·b probes; `fetch` routes
one task and is charged comparisons / hops / messages; `observe` is the optional online update. True skill S and the
liar set are never exposed to any method. **Status**: *reported* = drawn and quoted; *do-not-add* = kept in the grids,
never drawn (`extra_figs.excluded`); *never reported* = withdrawn from every table (erratum 26). Origins: SPEC §5/§6/§9
(pre-registered, 2026-09-01), v2 = `TARGETS_rte_v2.md` (2026-09-03, labeled variants), v3 = `TARGETS_rte_v3.md`
(external routers), v4 = `TARGETS_rte_v4.md` (cohort modes), post-hoc = dated in `DEVIATIONS.md`.

## 1. Floors and ceiling

| method | origin | reads | what it does | status |
|---|---|---|---|---|
| `random` | SPEC §6 | nothing | uniform pick; the floor | reported |
| oracle | runner (SPEC §7) | true S | always the agent with the highest true skill for the task's family; re-picks after churn; the dotted line in every figure | reported (ceiling) |

## 2. MIDIAN family (SPEC §5; plain MIDIAN pre-registered and byte-identical since the first run)

| method | origin | reads | what it does | status |
|---|---|---|---|---|
| `midian` (plain) | SPEC §5 | probe, reports | Agents in random cohorts of r = 10. Level 0: each member is probed b times per family; the r−1 cohort peers report what they saw; the cohort's estimate of a member is a trimmed mean over reporters (drop ⌊δ(r−1)⌋ from each end, δ = 1/3), so up to that many liars per cohort are absorbed. Each node keeps, per family, the best estimate in its subtree and which child holds it; nodes are regrouped at random up a tree of depth ⌈log_r n⌉. A route descends from the root: r comparisons and 2 messages per level. After each outcome the routed agent's estimate and its path are updated (`online=True`). Build O(n) probes, reports and messages; route O(log n). Parameters never changed. | reported |
| `midian` r ≠ 10 (`r=5`, `r=20`) | v2 1.4 | as plain | the same tree with cohorts of 5 or 20; r-sweep control | **do-not-add** |
| `midian` δ sweep (`delta=0`) | v2 2.5 (`internals_v2`) | as plain | trimming off; internals ablation | do-not-add (r ≠ 10 arms) / appendix only |
| `midian_v` = `midian(verify=True, cached=True)` | post-hoc 2026-09-02, replicated as v2 V2-8 | probe, reports | verification at promotion: level 0 spends b0 = b−1 probes per cell; the saved n·K·(b−b0) probes re-probe, budget-exactly, every candidate a child forwards to its parent, by reporters drawn from sibling subtrees, and the verified value is written back. The root's per-family pick is cached: a route costs 1 comparison and 2 messages. Zero probes to verify at b = 1 (erratum 22). | reported (MIDIAN-V) |
| `midian_a` | v2 1.1 | probe, reports | plain MIDIAN plus report audits: 5% of level-0 probe instances are re-run by the auditor (same index-seeded instance), each peer's report about it is compared with the truth, and a reporter with two mismatches is excluded from every later aggregation; online, 5% of routed outcomes are audited the same way. Cost 1.05× build probes, nothing per task. | reported (MIDIAN-A) |
| `midian_va` | v2 V2-11 | probe, reports | MIDIAN-A's audited level 0 + MIDIAN-V's verified promotion and cached root pick; excluded reporters are also dropped from verification. The robust arm: flat in β at 10^3–10^7. | reported (MIDIAN-VA) |
| `midian_sh` | v2 negative control | probe, reports | level-0 estimation by successive halving inside each cohort (probe all, keep the top half by trimmed peer estimate, repeat), spending exactly plain MIDIAN's s·b probes; unfundable at b = 1 (erratum 23) | **do-not-add** |
| `midian_sha` | v2 negative control | probe, reports | `midian_sh` + MIDIAN-A's audits | **do-not-add** |
| `midian(stratify=True)` / `cohort=block` / `specialty` / `declared` | v4 (T4-1…T4-6) | probe, reports (+declared for `declared`) | how level-0 cohorts are formed, budget-neutral: `stratify` one member per ability stratum (diverse cohorts); `block` contiguous ability blocks; `specialty` grouped by argmax family of the measured profile; `declared` grouped by the declared profile. Applied to plain, A and VA. Results: RESULTS_rte_v4.md. | reported in the v4 tables only |
| `midian_llm_descent` | SPEC §9 | probe, reports, LLM | plain MIDIAN where an LLM (not the arithmetic argmax) chooses among a node's r children at each level; unparseable answers fall back to the argmax and are counted | reported (appendix) |

## 3. Self-contained rivals (SPEC §6), by what they read

**Declared channel only** (self-descriptions / declared skill D, inflated by liars; flat in β by construction because they never read reports):

| method | origin | what it does | cost per task | status |
|---|---|---|---|---|
| `declared_argmax` | SPEC §6 | argmax_a D[a, f]; the declared-channel floor | n comparisons (or O(1) cached) | reported |
| `declared_softmax` | SPEC §6 | sample ∝ exp(D[a, f]/τ), τ = 0.1; models a soft supervisor pick | n | reported |
| `cnp_self_bid` | SPEC §6 | Contract-Net Protocol: broadcast the task, every agent bids D[a, f] + noise, the highest bid wins | 2n messages | reported |
| `cluster_head_router` | SPEC §6 (AgentNet++-style prior art) | k-means on D into clusters of ~r (built inside random buckets above 10^5, DEVIATIONS); a head per cluster; route: best cluster by its head, then argmax within it | 2 hops, 4 messages | reported |
| `disrouter_cascade` | SPEC §6 | agents in ascending mean-declared order; each takes the task if D[a, f] ≥ τ = 0.7 else forwards; if everyone forwards, the highest declarer | hops = position of the taker | reported |
| `route_to_k_majority` | SPEC §6 (route-to-many) | top-3 by D; the runner executes ALL THREE and scores majority-of-outcomes (ties fail); charged 3 tasks. Executes three agents per task, so it can exceed the single-agent oracle (0.888 vs 0.845 on the calibrated backend); not comparable at equal per-task cost. | 3 executions | **do-not-add** (2026-09-17) |

**Verified outcomes, centralized** (the method itself probes; no report channel; MIDIAN's exact probe budget):

| method | origin | what it does | cost per task | status |
|---|---|---|---|---|
| `flat_probe_argmax` (frozen) | SPEC §6 | **the key control**: probe every agent b times per family, est = mean, argmax per family; MIDIAN's probes without the tree or the reports | n comparisons | reported |
| `flat_probe_argmax(online=True)` | SPEC §6 | the same, updating est from every routed outcome | n | reported (flat online) |
| `ucb_per_family` | SPEC §6 | UCB1 over arms (agent, family); warm-up = the shared budget, b pulls per arm; online | n | reported |
| `thompson_per_family` | SPEC §6 | Thompson sampling, Beta(1, 1) prior, same warm-up | n | reported |
| `warm_start_bandit` | SPEC §6 | Thompson sampling with a Beta prior of pseudo-count n0 = 5 centred on the DECLARED skill; reads declared + probe | n | reported |
| `linucb_honest` | v2 (labeled rival) | contextual bandit whose only context is the agent's own outcome history [1, mean, √count, mean over families]; ridge model per family; the honest-context control | n | reported |
| `trueskill_per_family` | SPEC §6 | pairwise: sample agent pairs, probe both on the same instance, TrueSkill update per family; argmax μ; not implemented above 10^5 (pure-Python loop) | n | reported where run |
| `verify_on_claim` | SPEC §6 | rank by D; probe the top unverdicted candidate k = 3 times, accept if mean ≥ D − 0.15 else try the next (5 tries); verdicts cached. The dangerous baseline: its verification budget drains on liars | route-time probes | reported |
| `sequential_halving(peer_reported=True)` | SPEC §6 | per family, fixed-budget best-arm identification: probe all, keep the top half, repeat, budget n·b per family; the probe outcomes are scored by PEER REPORTS (the same channel MIDIAN reads), so liars can steer the tournament; fetch is a cached lookup, no online update | 1 comparison | reported (peer halving) |
| `sequential_halving` (plain) | SPEC §6 | the same tournament scored by a trusted observer the setting never provides; ties the oracle from b ≥ 3 | 1 | **never reported** (erratum 26) |
| `flat_nsw_router` | SPEC §6 (E7) | hnswlib navigable-small-world index over the probed estimates; query one-hot(f); ~log n hops; charged hop(⌈log2 n⌉) and compare(ef) | log n hops | reported |

**Verified outcomes, decentralized** (probe + reports + bus):

| method | origin | what it does | status |
|---|---|---|---|
| `referral_network` | SPEC §6 | random d-regular graph (d = 10); each node holds beliefs only about its neighbours from outcomes it observed through the report channel (a lying node corrupts its own map); route = greedy referral walk | reported |
| `gossip_reputation_greedy` | SPEC §6 | EigenTrust over the report matrix (power iteration, no pre-trusted seed) weights every report; T-Man gossip builds a similarity overlay; route = greedy walk on trust × est | reported |

**Published routers as methods** (v3, part C: RouterBench's predictive routers on MIDIAN's probe budget, no report channel):

| method | origin | what it does | status |
|---|---|---|---|
| `knn_router` (+ `online=True`) | v3 (Hu et al. 2024, RouterBench) | predicted success of agent a on a task = mean outcome of a's k = b nearest probes by prompt-embedding cosine (all-MiniLM-L6-v2); argmax over agents, n comparisons; `online` adds every routed outcome to the store | reported |
| `mlp_router` | v3 (RouterBench) | one regressor (prompt embedding ⊕ agent one-hot) → success, fit on the n·K·b probes; scores every agent per task; offline only; does not fit above 10^4 (one-hot over 240k+ probes) | reported where run |

## 4. LLM-native arms (SPEC §9, live backend only)

| method | origin | what it does | status |
|---|---|---|---|
| `llm_supervisor` | SPEC §9 | the practitioner default: retrieve the top-k = 10 agents by self-description similarity, a supervisor model (Qwen2.5-7B) reads the descriptions and picks one; everything except the pick (retrieval, name mapping, ledger, declared-argmax fallback) is the frameworks' shared adapter; one direct call, no framework | reported |

## 5. The ten agent frameworks (SPEC §6A), each run inside its own venv through a JSON-lines worker

Each `fw_*.py` hands a shortlist and the task to the framework's OWN selection primitive and returns the agent it named
(`NOTES_<fw>.md` documents the intercepted code path per framework). Strict accounting: `success_strict` counts only
tasks where the framework itself named a candidate; a supervisor that answers the task itself, names nothing or names a
non-candidate falls back to declared argmax over the shortlist (counted in `fallback_rate`). Supervisor Qwen2.5-7B;
Magentic-One also has a 14B-orchestrator arm.

| framework | selection primitive intercepted |
|---|---|
| `fw_autogen` | `SelectorGroupChat`: the manager reads the roster (name + description) and names the next speaker |
| `fw_langgraph` | `create_react_agent` supervisor with one handoff tool per candidate |
| `fw_crewai` | hierarchical manager's `Delegate work to coworker` tool; the chosen coworker |
| `fw_magentic_one` | the orchestrator's progress ledger `next_speaker` |
| `fw_maf` (Microsoft Agent Framework) | group-chat manager's `next_speaker` JSON |
| `fw_openai_agents` | a triage agent with one `handoff` per candidate |
| `fw_google_adk` | auto-delegation among sub-agents (name + description) |
| `fw_llamaindex` | `LLMSingleSelector` over `ToolMetadata(name, description)` |
| `fw_smolagents` | managed agents listed in the system prompt and as tools; the called tool |
| `fw_camel_workforce` | Workforce task assignment over the worker roster |
| `fw_agentscope` | appendix only: the library has no multi-agent selection primitive; the worker uses its model API for a direct pick |
| `fw_metagpt` | NotImplemented: nothing to intercept that answers "which agent" (`NOTES_metagpt.md`) |
| `fw_echo` | protocol check, not a rival: echoes the first candidate |

**Shortlist sources** (the "common scaling adapter"; the frameworks read the whole roster and top out at tens of agents,
so a top-10 is put in front of them). Five, never pooled, each its own grid (RESULTS II.2, Appendix S):

| adapter parameter | origin | what the framework is handed | status |
|---|---|---|---|
| default (hashed TF-IDF) | SPEC §6A, pre-registered | top-10 by hashed-TF-IDF cosine between the agents' self-descriptions and the family description; stable sort → at scale ten clones of one agent (erratum 25) | reported (headline) |
| `dedup=True` | post-hoc 2026-09-14 | the same ranking over DISTINCT description texts, one agent per text | reported beside |
| `retrieval="embed"` (+dedup) | post-hoc 2026-09-15 | all-MiniLM-L6-v2 cosine over the deduped descriptions: the dense retriever a deployed stack would use | reported beside |
| `retrieval="midian"` (r = 10) | v2 (T3-19 lineage; H7) | MIDIAN-V's probed leaf cohort, V's pick first | reported beside |
| `retrieval="midian_va"` (r = 10) | v2 V2-? / post-hoc at 10^2, 10^4, 10^5 (2026-09-16) | MIDIAN-VA's audited leaf cohort, VA's pick first | reported beside |

## 6. The do-not-add list (`scripts/extra_figs.py`, `excluded()`), applied to every figure

`midian` with r ≠ 10 (`midian[r=5]`, `midian_v_r5`, `midian[r=20,…]`, …), `midian_sh`, `midian_sha`,
`route_to_k_majority` (three executions per task), and the trusted-observer `sequential_halving` (never reported anywhere,
erratum 26). They stay in the grids and in `paper/NUMBERS.json` (except the trusted arm) for the record.
