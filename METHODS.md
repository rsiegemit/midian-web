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

## 2. MIDIAN family (SPEC §5; MIDIAN w/o defenses is the pre-registered plain tree, reproduced bit for bit since the first run)

One class, `midian`, with two defenses as parameters, both on by default: `audit` (report audits with reporter
exclusion) and `verify` (verified promotion; `cached` defaults to `verify`). MIDIAN is the default; the ablations switch
a defense off. Renamed 2026-09-24 (CHANGES_AND_ERRATA §8g; `rte/methods/keys.py` maps the stored keys).

| method | origin | reads | what it does | status |
|---|---|---|---|---|
| `midian{"audit": false, "verify": false}` (MIDIAN w/o defenses) | SPEC §5 | probe, reports | Agents in random cohorts of r = 10. Level 0: each member is probed b times per family; the r−1 cohort peers report what they saw; the cohort's estimate of a member is a trimmed mean over reporters (drop ⌊δ(r−1)⌋ from each end, δ = 1/3), so up to that many liars per cohort are absorbed. Each node keeps, per family, the best estimate in its subtree and which child holds it; nodes are regrouped at random up a tree of depth ⌈log_r n⌉. A route descends from the root: r comparisons and 2 messages per level. After each outcome the routed agent's estimate and its path are updated (`online=True`). Build O(n) probes, reports and messages; route O(log n). Parameters never changed. Reference arm of every paired analysis (`analyze.REF`). | reported (MIDIAN w/o defenses) |
| `midian` r ≠ 10 (`r=5`, `r=20`) | v2 1.4 | as above | the same tree with cohorts of 5 or 20; r-sweep control | **do-not-add** |
| `midian` δ sweep (`delta=0`, r = 10) | v2 2.5 (`internals_v2`) | as above | trimming off; internals ablation (Fig. A_internals) | appendix only |
| `midian{"audit": false}` (MIDIAN w/o audits) | post-hoc 2026-09-02, replicated as v2 V2-8 | probe, reports | verification at promotion: level 0 spends b0 = b−1 probes per cell; the saved n·K·(b−b0) probes re-probe, budget-exactly, every candidate a child forwards to its parent, by reporters drawn from sibling subtrees, and the verified value is written back. The root's per-family pick is cached: a route costs 1 comparison and 2 messages. Level 0 trims by reporter. Zero probes to verify at b = 1 (erratum 22). | reported (MIDIAN w/o audits) |
| `midian{"verify": false}` (MIDIAN w/o verification) | v2 1.1 | probe, reports | MIDIAN w/o defenses plus report audits: 5% of level-0 probe instances are re-run by the auditor (same index-seeded instance), each peer's report about it is compared with the truth, and a reporter with two mismatches is excluded from every later aggregation; online, 5% of routed outcomes are audited the same way. Cost 1.05× build probes, nothing per task. | reported (MIDIAN w/o verification) |
| `midian` (MIDIAN, both defenses; the default) | v2 V2-11 | probe, reports | the audited level 0 + verified promotion and cached root pick; excluded reporters are also dropped from verification. The robust arm: flat in β at 10^3–10^7. | reported (MIDIAN) |
| successive halving inside cohorts, with and without audits | v2 negative controls | probe, reports | level-0 estimation by successive halving inside each cohort; unfundable at b = 1 (erratum 23) | **withdrawn** 2026-09-24: code deleted, stored rows moved to `_premigration_v2` backups (CHANGES_AND_ERRATA §8g) |
| `midian(stratify=True)` / `cohort=block` / `specialty` / `declared` | v4 (T4-1…T4-6) | probe, reports (+declared for `declared`) | how level-0 cohorts are formed, budget-neutral: `stratify` one member per ability stratum (diverse cohorts); `block` contiguous ability blocks; `specialty` grouped by argmax family of the measured profile; `declared` grouped by the declared profile. Applied to MIDIAN w/o defenses, MIDIAN w/o verification and MIDIAN. Results: RESULTS_rte_v4.md. | reported in the v4 tables only |
| `midian_llm_descent` | SPEC §9 | probe, reports, LLM | MIDIAN w/o defenses where an LLM (not the arithmetic argmax) chooses among a node's r children at each level; unparseable answers fall back to the argmax and are counted | reported (appendix) |

## 3. Self-contained rivals (SPEC §6), by what they read

**Declared channel only** (self-descriptions / declared skill D, inflated by liars; flat in β by construction because they never read reports):

| method | origin | what it does | cost per task | status |
|---|---|---|---|---|
| `declared_argmax` | SPEC §6 | argmax_a D[a, f]; the declared-channel floor | n comparisons (or O(1) cached) | reported |
| `declared_softmax` | SPEC §6 | sample ∝ exp(D[a, f]/τ), τ = 0.1; models a soft supervisor pick | n | reported |
| `cnp_self_bid` | SPEC §6 | Contract-Net Protocol: broadcast the task, every agent bids D[a, f] + noise, the highest bid wins | 2n messages | reported |
| `verbal_confidence` | 2026-09-24 (Tian et al. 2023 style self-knowledge routing; LLM backend only) | shortlist k = 10 (`shortlist=declared`: top-k by D[:, f]; `embed`: MiniLM over self-descriptions), ask each how confident it is (0–10) that it solves THIS task in its own solve prompt, route to the most confident (ties: earliest shortlist position; unparseable = 0, counted). Liars answer the top of the scale in every lie mode (`World.confidence`) | 2k messages, k comparisons | not yet run (grids `vc_live_n1000`, `vc_lie_max_n1000`) |
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
| `retrieval="midian_wo_audit"` (r = 10) | v2 (T3-19 lineage; H7) | the probed leaf cohort of MIDIAN w/o audits, its pick first | reported beside |
| `retrieval="midian"` (r = 10) | v3 T3-19 at 10^3; post-hoc at 10^2, 10^4, 10^5 and the cartel cells (2026-09-16) | MIDIAN's audited leaf cohort ("MIDIAN cohort"), MIDIAN's pick first | reported beside |
| `retrieval="bm25"` | post-hoc 2026-09-17 | Okapi BM25 (k1 = 1.5, b = 0.75) over an inverted index; the lexical half of a production stack | reported beside |
| `retrieval="hybrid"` | post-hoc 2026-09-17 | reciprocal-rank fusion (k = 60) of BM25 with the dense scores | reported beside |
| `retrieval="sota"` (+ `rerank_pool` = 50) | post-hoc 2026-09-17 | hybrid, then the top 50 reranked by a cross-encoder (`Qwen/Qwen3-Reranker-4B`), top k kept. Table is per-population and cached | reported beside |
| `embed_model` (`Qwen/Qwen3-Embedding-8B`) | post-hoc 2026-09-17 | the strong dense half; defaults to MiniLM so `retrieval="embed"` is bit-identical to what was already reported | reported beside |
| `embed_instruct` | post-hoc 2026-09-18 | the QUERY-side task instruction an instruction-tuned embedder expects; only the K family texts depend on it, so the n document embeddings are reused | reported beside |
| `retrieval="declared"` | post-hoc 2026-09-18 | top-k by the declared claim: no text retrieval at all, the cheap baseline every text arm should be measured against | grid written, NOT yet run |
| `shuffle=True` | post-hoc 2026-09-18 | position CONTROL: permutes a midian cohort deterministically so the pick is not first. Same members, ordering only | **control -- never a rival, never pooled with the MIDIAN-cohort rows** |

### 5b. What the shortlist diagnostics measure (`jobs/shortlist_skill.py`, `cohort_skill.py`, `position1_skill.py`)

Three statistics of a shortlist, all computed from the measured S matrix OFFLINE -- S is read for MEASUREMENT only and
is never visible to any method:

| statistic | what it answers | why it matters |
|---|---|---|
| top-k **mean** | what you score picking uniformly from the list | the right statistic only if the consumer picks blindly |
| **best in list** | the ceiling a perfect consumer could reach | the right statistic only if the consumer is competent |
| **position 1** | what the list's own top-ranked agent is worth | every retriever ranks best-first, and the shuffle control shows frameworks largely consume position 1 -- so this is the statistic that actually drives outcomes |

Measured recovery, `(routing - mean) / (best - mean)`, is 15-35 % across every source: frameworks sit much closer to
random-within-list than to competent, which is why mean and position-1 both matter and `best` alone never does.

## 6. The do-not-add list (`scripts/extra_figs.py`, `excluded()`), applied to every figure

`midian` with r ≠ 10 (`midian[r=5]`, `midian_wo_audit_r5`, `midian[r=20,…]`, …), `midian_llm_descent`, the online-off
ablation `midian[audit=False,online=False,verify=False]`,
`route_to_k_majority` (three executions per task), and the trusted-observer `sequential_halving` (never reported anywhere,
erratum 26). They stay in the grids and in `paper/NUMBERS.json` (except the trusted arm) for the record.
