# RESULTS_rte.md — RTE: plain MIDIAN vs self-contained rivals and ten real agent frameworks

Written 2026-09-02 (final numbers 23:30) from the completed live programme (see STATUS.md for the run log, DEVIATIONS.md for every
departure from SPEC.md, TARGETS_rte.md for the pre-registered expectations). Per-grid machine summaries with
bootstrap CIs, per-cell paired deltas and figures F1–F6 are under `$RTE_DATA/results/<grid>/summary.md`
(`combined_scale/summary.md/summary.md` holds the cross-n cost-exponent fits). Every number below is a mean over
paired (cell, seed) units unless stated; a *cell* fixes population shape, β, liar selection and declaration channel,
and every method in a cell sees the same agents, the same liars and the same task stream.

**Setup in one paragraph.** Live backend: n agents drawn from a 7-model ladder (Qwen2.5 0.5B–14B, Gemma-2 2B/9B) with
per-family handicaps and an optional python tool, K = 16 Reasoning Gym families, true skill S measured per prompt
signature (200 probes, 60 on the ≥9B rungs) and never shown to any method. Liars (fraction β) inflate their
declarations by +0.4 and, when colluding, report 1 for fellow liars and 0 for the top-20% honest. Budget b = 3 probes
per (agent, family) at build; Q = 1000 routed tasks per unit (300 on the framework grids). Seeds: 5 on the algorithmic
grids, 3 on the framework grids (5 for the MIDIAN variants and oracle/flat/declared, see caveats). Intervals are 95%
percentile bootstrap over units; "wins" are paired sign counts. Completed grids: live_core_n100, live_f1_n1000,
live_extra_n1000, live_n10k, budget_sweep, midian_internals,
fw_live_n100, fw_live_n1000, fw_live_n100_verified, fw_live_n1000_verified, fw_k_sensitivity, fw_appendix, replay
twins of the two live sweeps, replay_scale, bernoulli_scale.

---

## 1. Headline (SPEC §6A): the ten frameworks vs MIDIAN

Each framework runs through its own library and its own selection primitive (LangGraph supervisor, CrewAI
hierarchical manager, AutoGen GroupChat selector, Magentic-One orchestrator, MAF handoff, OpenAI Agents SDK
handoffs, Google ADK sub-agent delegation, LlamaIndex router, smolagents managed agents, CAMEL Workforce), fed the
same common scaling adapter: hashed TF-IDF over the agents' self-descriptions selects the top k = 10, the framework's
supervisor picks one. Paired on identical cells (3 shapes × 4 β × seeds), Q = 300.

| method | n=100 success [95% CI] | n=1000 success [95% CI] | Δ vs MIDIAN, n=1000 (paired) | cells rival > MIDIAN |
|---|---|---|---|---|
| oracle (best agent by true S) | 0.711 [0.683, 0.739] | 0.719 [0.689, 0.748] | +0.081 | 60/60 |
| MIDIAN-V r=10 (labeled variant) | 0.623 [0.597, 0.648] | 0.659 [0.630, 0.689] | +0.022 | 37/56 |
| MIDIAN-V r=5 (labeled variant) | 0.630 [0.603, 0.656] | 0.656 [0.623, 0.688] | +0.019 | 45/60 |
| **MIDIAN** | 0.626 [0.603, 0.648] | 0.637 [0.613, 0.663] | 0 | — |
| MIDIAN r=5 | 0.618 [0.594, 0.642] | 0.628 [0.601, 0.655] | −0.009 | 22/57 |
| flat probe argmax (same 48k probes, no tree) | 0.572 [0.552, 0.591] | 0.612 [0.583, 0.640] | −0.025 | 24/59 |
| declared argmax | 0.560 [0.545, 0.575] | 0.562 [0.549, 0.574] | −0.076 | 13/60 |
| llm_supervisor (one call over top-20) | 0.555 [0.545, 0.564] | 0.562 [0.546, 0.577] | −0.076 | 28/60 |
| Magentic-One | 0.546 [0.532, 0.559] | 0.541 [0.521, 0.562] | −0.096 | 28/59 |
| Google ADK | 0.518 [0.501, 0.535] | 0.536 [0.514, 0.557] | −0.102 | 29/60 |
| CrewAI | 0.536 [0.519, 0.552] | 0.535 [0.512, 0.558] | −0.102 | 28/60 |
| smolagents | 0.522 [0.505, 0.538] | 0.523 [0.496, 0.549] | −0.114 | 28/60 |
| LangGraph | 0.506 [0.486, 0.524] | 0.521 [0.494, 0.548] | −0.116 | 28/59 |
| AutoGen | 0.511 [0.493, 0.529] | 0.520 [0.493, 0.546] | −0.118 | 28/59 |
| LlamaIndex | 0.512 [0.494, 0.529] | 0.519 [0.491, 0.547] | −0.118 | 28/59 |
| OpenAI Agents SDK | 0.506 [0.488, 0.525] | 0.519 [0.491, 0.546] | −0.119 | 29/60 |
| MAF | 0.517 [0.496, 0.537] | 0.518 [0.489, 0.546] | −0.120 | 29/60 |
| CAMEL Workforce | 0.514 [0.495, 0.532] | 0.510 [0.479, 0.540] | −0.127 | 28/59 |
| random | 0.307 [0.289, 0.326] | 0.313 [0.291, 0.336] | −0.325 | 0/60 |

All 60 cells (3 shapes × 4 β × 5 seeds) have every listed method. The average hides a split that the per-shape
breakdown (figure A, right panel) makes plain:

| n=1000, by shape | oracle | MIDIAN-V | MIDIAN | flat probe | frameworks (mean) | cells framework > MIDIAN |
|---|---|---|---|---|---|---|
| specialist (3 strong families per agent) | 0.857 | 0.789 | 0.751 | 0.751 | 0.390 | 0 / 200 |
| heavy_tail (1 in 10 is a big model) | 0.733 | 0.664 | 0.634 | 0.592 | 0.617 | 83 / 200 |
| bimodal (20% big-with-tools, 80% small) | 0.566 | 0.525 | 0.527 | 0.493 | 0.566 | 200 / 200 |

(n=100: specialist 0.834 / 0.725 / 0.726 / 0.653 / 0.447, frameworks win 0/200; heavy_tail 0.733 / 0.626 / 0.623 /
0.537 / 0.550, 12/200; bimodal 0.566 / 0.517 / 0.529 / 0.524 / 0.559, 188/200.)

Three things the tables say. (i) Where skill is *legible from a description*, the frameworks are as good as it gets:
on the bimodal population the question is only "is this one of the 20% big models with a tool?", the agents'
self-descriptions answer it, and every framework sits on the oracle (0.566) while MIDIAN's 3-probe estimates trail by
0.04. On heavy_tail the two are close. (ii) Where skill is *family-specific*, descriptions fail and the frameworks
collapse: on the specialist population they score 0.39 against 0.75 for MIDIAN and 0.79 for MIDIAN-V, losing every
one of 200 cells by 0.28–0.36. Averaged over shapes that is −0.10 to −0.13 per framework. (iii) The frameworks are
flat in β (±0.02): self-descriptions overclaim by +0.27 and correlate 0.36 with S, so lying about a channel that
carries little family-level signal changes nothing. Per task a framework spends 12 messages, 10 comparisons and one
or more supervisor LLM calls (0.6–8.2 s wall); MIDIAN spends 6 messages and 30 comparisons with no LLM call,
MIDIAN-V 2 messages and 1 comparison (§6). At n=100 MIDIAN-V and plain MIDIAN are indistinguishable (Δ −0.003 and
+0.004); verification pays only once cohorts are deep enough to promote through.

**Fallbacks.** When a framework's supervisor returns no valid agent name the adapter falls back to declared-argmax
over its own shortlist and counts it (`method_stats`). Rates: CrewAI 79–84% of tasks (its hierarchical manager
rarely delegates; two worker defects fixed, see DEVIATIONS), Magentic-One 52–54%, Google ADK 21–23%, LangGraph 6%,
OpenAI Agents 5–7%, MAF 2–3%, CAMEL 0–2%, AutoGen / smolagents / LlamaIndex 0%. CrewAI's number therefore measures
the fallback, not CrewAI.

**k-sensitivity (fw_k_sensitivity, specialist, self-described).** LangGraph 0.35 / 0.39 / 0.38 and AutoGen
0.35 / 0.38 / 0.37 at k = 5 / 10 / 20: the shortlist size is not what limits them. **Appendix:** AgentScope 0.415 vs
random 0.405 vs MIDIAN 0.73 at n=100; MetaGPT not run (its team abstraction has no selection primitive to
intercept, NOTES_metagpt.md).

## 2. Frameworks given MIDIAN's verified shortlist (labeled variant, `retrieval: midian`)

Same frameworks, same supervisors, but the top-k shortlist is MIDIAN-V's leaf cohort (k = r = 10 or 5) instead of
TF-IDF over self-descriptions; the framework pays MIDIAN-V's build. Paired on identical cells (36 per n).

| framework | n=1000 own | + MIDIAN r=10 | Δ | wins | + MIDIAN r=5 | Δ |
|---|---|---|---|---|---|---|
| CrewAI* | 0.531 | 0.648 | +0.116 | 24/36 | 0.653 | +0.121 |
| AutoGen | 0.518 | 0.622 | +0.104 | 26/36 | 0.610 | +0.091 |
| Magentic-One* | 0.537 | 0.636 | +0.098 | 24/36 | 0.624 | +0.087 |
| Google ADK* | 0.539 | 0.635 | +0.096 | 28/36 | 0.624 | +0.085 |
| LangGraph | 0.519 | 0.611 | +0.092 | 22/36 | 0.583 | +0.064 |
| CAMEL | 0.502 | 0.585 | +0.083 | 24/36 | 0.601 | +0.098 |
| smolagents | 0.518 | 0.597 | +0.080 | 23/36 | 0.630 | +0.113 |
| OpenAI Agents | 0.511 | 0.586 | +0.075 | 21/36 | 0.576 | +0.065 |
| MAF | 0.514 | 0.565 | +0.051 | 19/36 | 0.593 | +0.078 |
| LlamaIndex | 0.519 | 0.555 | +0.036 | 15/36 | 0.538 | +0.019 |

(* = fallback ≥ 20%; CrewAI with the MIDIAN shortlist falls back on 100% of tasks, so that row is declared-argmax
over MIDIAN's cohort.) At n=100 the lifts are +0.00 (LlamaIndex) to +0.10 (CrewAI). Every framework improves, and
every one still trails MIDIAN-V alone by 0.01–0.10 at n=1000 (the supervisor second-guessing a verified pick), while
costing 12× MIDIAN-V's messages per task. The r=5 cohort gives the same lift at half the build reports.

## 3. F1 — every rival at n=1000 (live_f1_n1000: 3 shapes × 4 β × 2 liar selections × 2 declaration channels × 5 seeds)

Success by β; classes per SPEC §6. Full CIs, per-cell deltas and `WITHIN_FLOOR` flags in the grid summary.

| class | method | β=0 | β=0.1 | β=0.25 | β=0.5 | msgs/task | comps/task |
|---|---|---|---|---|---|---|---|
| ceiling | oracle | 0.723 | 0.723 | 0.723 | 0.723 | 0 | 0 |
| verified, central | sequential_halving (trusted observer) | 0.723 | 0.722 | 0.722 | 0.722 | 0 | 1 |
| verified, central | sequential_halving (peer-reported, fair control) | 0.722 | 0.722 | 0.718 | 0.545 | 0 | 1 |
| verified, central | warm_start_bandit | 0.694 | 0.684 | 0.674 | 0.673 | 0 | 1000 |
| midian | MIDIAN-V r=5 | 0.690 | 0.691 | 0.680 | 0.562 | 2 | 1 |
| verified, central | verify_on_claim | 0.687 | 0.658 | 0.643 | 0.650 | 0 | 17 |
| midian | MIDIAN-V r=10 | 0.681 | 0.685 | 0.677 | 0.575 | 2 | 1 |
| verified, central | flat_probe_argmax (online) | 0.666 | 0.666 | 0.667 | 0.667 | 0 | 1000 |
| midian | **MIDIAN** | 0.666 | 0.663 | 0.646 | 0.595 | 6 | 30 |
| midian | MIDIAN r=5 | 0.664 | 0.660 | 0.639 | 0.564 | 10 | 25 |
| declared | cluster_head_router | 0.661 | 0.602 | 0.592 | 0.587 | 4 | 110 |
| declared | route_to_k_majority | 0.659 | 0.600 | 0.590 | 0.583 | 0 | 1000 |
| declared | cnp_self_bid | 0.659 | 0.601 | 0.584 | 0.581 | 2000 | 1000 |
| declared | declared_argmax | 0.656 | 0.592 | 0.580 | 0.582 | 0 | 1000 |
| midian | midian_llm_descent | 0.652 | 0.650 | 0.634 | 0.574 | 6 | 30 |
| verified, central | flat_nsw_router | 0.633 | 0.630 | 0.627 | 0.627 | 0 | 50 |
| verified, central | trueskill_per_family | 0.626 | 0.621 | 0.623 | 0.623 | 0 | 1000 |
| verified, central | flat_probe_argmax (frozen) | 0.621 | 0.621 | 0.619 | 0.615 | 0 | 1000 |
| midian | MIDIAN, updates off | 0.612 | 0.598 | 0.584 | 0.551 | 6 | 30 |
| verified, central | thompson_per_family | 0.603 | 0.603 | 0.604 | 0.604 | 0 | 1000 |
| declared | declared_softmax | 0.600 | 0.551 | 0.508 | 0.477 | 0 | 1000 |
| verified, decentral | gossip_reputation_greedy | 0.596 | 0.600 | 0.477 | 0.360 | 120 | 60 |
| verified, central | ucb_per_family | 0.596 | 0.597 | 0.598 | 0.598 | 0 | 1000 |
| declared | disrouter_cascade | 0.588 | 0.455 | 0.458 | 0.476 | 571 | 0 |
| declared | llm_supervisor | 0.584 | 0.574 | 0.574 | 0.571 | 22 | 20 |
| verified, decentral | referral_network | 0.451 | 0.438 | 0.408 | 0.340 | 80 | 40 |
| floor | random | 0.311 | 0.311 | 0.311 | 0.311 | 0 | 0 |

Paired vs pre-registered MIDIAN over the 48 cells (mean Δ, cells rival-better / MIDIAN-better / within MIDIAN's own
seed envelope): sequential_halving −0.080 (40/0/8); warm_start_bandit −0.039 (20/3/25); peer-reported halving −0.035
(32/6/10); flat online −0.025 (9/0/39); verify_on_claim −0.017 (15/9/24); MIDIAN-V −0.012 (10/0/38); flat frozen
+0.023 (2/20/26); declared_argmax +0.040 (7/19/22); ucb +0.045; llm_supervisor +0.067; gossip +0.134;
disrouter +0.148; referral +0.233; random +0.331. By population: on *specialist* MIDIAN 0.772 vs flat 0.738 vs oracle
0.861; on *bimodal* everything sits at 0.52–0.57 (oracle 0.574). Extra shapes (live_extra_n1000): correlated /
iid_uniform MIDIAN 0.790 / 0.791, MIDIAN-V 0.802 / 0.798, peer halving 0.822 / 0.840, oracle 0.858 / 0.859.

Declaration channel: with programmatic declarations (S + noise) declared_argmax reaches 0.656 at β=0; with the
agents' own self-descriptions it drops to 0.549 while MIDIAN is unchanged (0.642 / 0.642).

## 4. MIDIAN vs sequential halving — the fair control

`sequential_halving(peer_reported=True)` spends MIDIAN's probe budget adaptively (halve the candidate set per family
each round) but learns outcomes only through the same trimmed peer-report channel MIDIAN uses. Paired, 240 units.

| MIDIAN-V minus peer-reported halving | β=0 | β=0.1 | β=0.25 | β=0.5 |
|---|---|---|---|---|
| random liars | −0.040 | −0.038 | −0.046 | −0.064 |
| low-skill-first liars | −0.040 | −0.037 | −0.037 | **+0.123** |
| wins / non-tied cells | 0/60 | 0/60 | 0/60 | 26/58 |

Plain MIDIAN: −0.056 / −0.060 / −0.072 at β ≤ 0.25 (0/60 each, sign-test p ≈ 2e-18) and +0.049 at β=0.5 (32/60,
p = 0.7). At β=0.5 with low-skill liars: MIDIAN 0.57, MIDIAN-V 0.53, peer halving 0.41. Trusted-observer halving sits
exactly on the oracle (0.722 vs 0.723) at every β, so halving's edge at β ≤ 0.25 is adaptive allocation, not
information MIDIAN lacks; its collapse at β=0.5 is early elimination on poisoned reports, which the tree's
per-cohort trimming survives. On RouterBench replay (real outcomes, K=64) the same pattern holds: peer halving leads
by 0.04 at β ≤ 0.25 and plain MIDIAN leads by +0.10 at β=0.5.

## 5. MIDIAN internals (midian_internals: n=1000 specialist, collude on/off, 5 seeds)

| variant | β=0.1 | β=0.25 | β=0.5 collude | β=0.5 no collude | build reports | msgs / comps per task |
|---|---|---|---|---|---|---|
| flat_probe_argmax | 0.745 | 0.745 | 0.745 | 0.745 | 0 | 0 / 1000 |
| MIDIAN r=10 δ=1/3 | 0.790 | 0.789 | 0.743 | 0.788 | 432k | 6 / 30 |
| MIDIAN r=10 δ=0 (no trimming) | 0.789 | 0.789 | 0.778 | 0.788 | 432k | 6 / 30 |
| MIDIAN r=5 δ=1/3 | 0.792 | 0.791 | 0.728 | 0.789 | 192k | 10 / 25 |
| MIDIAN r=20 δ=1/3 | 0.782 | 0.785 | 0.765 | 0.784 | 912k | 6 / 60 |
| MIDIAN-V r=5 | 0.822 | 0.821 | 0.730 | 0.820 | 80k | 2 / 1 |
| MIDIAN-V r=10 | 0.816 | 0.816 | 0.749 | 0.816 | 160k | 2 / 1 |
| MIDIAN-V r=20 | 0.785 | 0.786 | 0.764 | 0.785 | 320k | 2 / 1 |
| oracle | 0.861 | | | | | |

r and δ do not matter at β ≤ 0.25; lying without collusion is fully absorbed at every setting. Collusion at β=0.5 is
the one regime that separates variants: small cohorts lose 0.05–0.09, r=20 loses 0.02, and trimming *hurts* there
(δ=0 0.778 vs δ=1/3 0.743: with half the reporters colluding, trimming removes honest extremes). MIDIAN-V buys
+0.03 over plain at ≤ 1/3 of the reports and 30× fewer comparisons per task, but is the most exposed at β=0.5.

## 6. Cost: per task, build, and scaling

Per task at n=1000 (all methods; frameworks add 0.6–8.2 s of supervisor calls):

| method | build probes | build reports | build msgs | msgs/task | comps/task | hops/task |
|---|---|---|---|---|---|---|
| MIDIAN | 48,000 | 432,000 | 1,010 | 6 | 30 | 3 |
| MIDIAN-V r=10 | 47,840 | 159,840 | 1,010 | 2 | 1 | 0 |
| MIDIAN-V r=5 | 48,000 | 80,000 | 1,050 | 2 | 1 | 0 |
| flat probe argmax | 48,000 | 0 | 0 | 0 | 1,000 | 0 |
| sequential halving (peer) | 44,930 | 404,400 | 0 | 0 | 1 | 0 |
| declared argmax | 0 | 0 | 1,000 | 0 | 1,000 | 0 |
| llm_supervisor | 0 | 0 | 1,000 | 22 | 20 | 1 |
| any framework | 0 | 0 | 1,000 | 12 | 10 | 1 |
| framework + MIDIAN shortlist | 47,840 | 159,840 | 2,010 | 14 | 11 | 1 |

Fitted exponents cost ∝ n^k over n = 10²…10⁷ (bernoulli, calibrated to the measured S) plus 10⁴…10⁶ (replay) and
the live 10²…10⁴ (`combined_scale`): MIDIAN comparisons, messages and hops per task **n^0.14** [0.13, 0.15]
(2·⌈log_r n⌉ exact); MIDIAN-V per-task everything n^0.00; flat / declared / CNP comparisons n^1.00; CNP messages
n^1.00; build probes n^1.03 for every probe-based method (n·K·b) and n^0.94 for MIDIAN-V (b−1 at level 0 plus
budget-exact re-probes); MIDIAN build messages n^1.00, build reports n^1.03. Wall-clock per task: MIDIAN n^−0.05,
flat n^0.54, CNP n^0.83.

Break-even against a framework (build cost / per-task saving): messages and comparisons — after the first task;
messages + reports — ~15,900 tasks (r=10) or ~7,900 (r=5); LLM calls — ~48,000 tasks at b=3 against a one-call
framework (16,000 at b=1; 5,000–8,000 against the multi-call Magentic-One / CAMEL).

## 7. Budget and scale (budget_sweep: n=1000, β=0.25; live_n10k: n=10,000, b=1)

| method | b=1 | b=3 | b=10 | | n=10k b=1, β=0 | β=0.25 |
|---|---|---|---|---|---|---|
| oracle | 0.723 | 0.723 | 0.723 | | 0.862 | 0.862 |
| sequential_halving (trusted) | 0.479 | 0.722 | 0.723 | | 0.583 | 0.583 |
| MIDIAN-V r=10 | 0.593 | 0.675 | 0.698 | | 0.679 | 0.679 |
| MIDIAN | 0.585 | 0.650 | 0.702 | | 0.684 | 0.683 |
| flat probe argmax | 0.481 | 0.620 | 0.695 | | 0.580 | 0.580 |
| warm_start_bandit | 0.676 | 0.684 | 0.704 | | 0.847 | 0.807 |
| declared_argmax | 0.673 | 0.673 | 0.673 | | 0.853 | 0.767 |
| ucb / thompson (late quarter) | 0.449 / 0.477 | 0.572 / 0.606 | 0.671 / 0.679 | | 0.581 / 0.573 | 0.581 / 0.566 |

(budget_sweep, all 45 (shape, seed, b) cells, paired.)

At b=1 the tree is the best probe-only method (+0.10 over flat and halving: one probe per cell is too thin to halve
on), but any method that also reads the *programmatic* declaration channel (declared_argmax, warm-start bandit,
verify_on_claim) beats every probe-only method at b=1 and at n=10k, because that channel is S + N(0, 0.05) for honest
agents and 75% of agents are honest. This is a property of the synthetic declaration channel, not of the LLMs: with
the agents' real self-descriptions declared_argmax falls to 0.55 (§3). Halving needs b ≥ 3 to pay off, and from b=3 on it sits on the oracle (paired delta −0.001 at b=3, +0.001
at b=10). No method beats the oracle on paired cells; an unpaired mean can only appear to when rows are missing.

## 8. Learning over the stream

Per-task outcomes on 9 live cells (3 shapes × 3 seeds, n=1000, β=0.25), success per block of 100 tasks:

| method | 1–100 | 101–200 | 201–300 | 301–1000 |
|---|---|---|---|---|
| oracle | 0.711 | 0.719 | 0.739 | 0.718 |
| MIDIAN-V r=10 | 0.660 | 0.659 | 0.689 | 0.679 |
| MIDIAN | 0.623 | 0.648 | 0.663 | 0.651 |
| MIDIAN, updates off | 0.588 | 0.599 | 0.600 | 0.591 |
| flat probe, online | 0.636 | 0.653 | 0.690 | 0.666 |
| warm_start_bandit | 0.672 | 0.687 | 0.684 | 0.680 |
| llm_supervisor | 0.549 | 0.581 | 0.604 | 0.580 |

The online path recompute is worth +0.06 in total and half of it is banked within the first 100 tasks; the curve is
fast-then-flat, and MIDIAN-V starts where plain MIDIAN ends up because verification does at build time what the
online updates learn. Frameworks gain 0.00–0.015 from early to late (fw grids, Q=300).

## 9. Pre-registered targets (TARGETS_rte.md) — verdicts

| # | target | verdict | evidence (live_f1_n1000 unless stated) |
|---|---|---|---|
| 1 | declared methods lose ≥0.25 from β=0→0.5; MIDIAN and probe-only move ≤0.03 | **MISS** (both halves) | declared lose 0.07–0.12 (declared_argmax −0.074, softmax −0.124, disrouter −0.112; llm_supervisor −0.013); frameworks move ≤0.012; probe-only flat/bandits move ≤0.006 but MIDIAN loses 0.071 (0.104 at n=100) under collusion at β=0.5 |
| 2 | MIDIAN = flat_probe_argmax within 0.02 at β=0; comparisons r·log_r n vs n | **MISS** on the equality as stated: MIDIAN is +0.044 above flat at n=1000 (60 pairs), +0.072 at n=100, +0.104 at n=10k; **PASS** for the max-tree itself (updates off: −0.009) and for the exponents (0.14 vs 1.00) | the online path update, not the tree, is the difference; the analyzer's earlier +0.019 had averaged MIDIAN's variants together (fixed) |
| 3 | trimming separates from δ=0 only where β·r exceeds the trim (β > 0.3) | **MISS** | midian_internals, r=10 with collusion: δ=1/3 − δ=0 = +0.004 / −0.004 / −0.036 at β = 0.1 / 0.25 / 0.5 (averaged over r: −0.001 / −0.003 / −0.018); the separation is in the *predicted* regime but in the wrong direction |
| 4 | verify_on_claim within 0.03 of oracle at β ≤ 0.1, loses ≥0.10 by β=0.5 | **MISS** | gap 0.050 at β ≤ 0.1; loses 0.037 (0.687→0.650), not 0.10 (0.023 on extra) |
| 5 | sequential_halving ≈ flat at equal budget; bandits' late success ≥ MIDIAN's at b=1 | **MISS** (both halves) | halving − flat paired = +0.103 (f1), +0.148 (extra), +0.003 (n=10k, b=1); b=1 late-quarter success ucb − MIDIAN = −0.138, thompson −0.110 (budget_sweep), −0.086 / −0.107 (n=10k) |
| 6 | argmax-vs-floor gap largest under heavy_tail, smallest under iid_uniform | **MISS** (partial) | gaps: heavy_tail 0.175 ≈ specialist 0.173 > bimodal 0.140 (f1); iid_uniform 0.166 ≈ correlated 0.163 (extra) — heavy_tail is largest but iid_uniform is not smallest |

Six pre-registered expectations: five misses, and T2 split (its equality half missed because online MIDIAN
beats flat by 0.04, its max-tree and scaling halves passed).
No parameter of plain MIDIAN was changed after the first run; every improvement is a labeled variant.

## 10. Replay backend (RouterBench real outcomes, K=64 real categories, n=1000, 10 seeds, CPU)

replay_mirror_live_f1_n1000 at β = 0 / 0.25 / 0.5: oracle 0.783; trusted halving 0.779 / 0.779 / 0.779; peer halving
0.779 / 0.773 / 0.519; declared_argmax 0.779 / 0.730 / 0.690; MIDIAN-V 0.738 / 0.731 / 0.524; MIDIAN 0.705 / 0.688 /
0.623; flat online 0.707; flat frozen 0.682; random 0.189. With only ~11 distinct real models, many agents are exact
copies and a 3-probe estimate cannot separate them, so on replay MIDIAN ≈ flat ≈ MIDIAN-V at β ≤ 0.25 while the
honest programmatic declaration channel is near-oracle; the β=0.5 ordering (MIDIAN > MIDIAN-V > peer halving) is the
same as live. replay_scale (n = 10⁴…10⁶) and bernoulli_scale (10²…10⁷) supply the exponents in §6.

## 11. Caveats that bound these numbers

- **b=3 ties.** With 3 probes an estimate takes 4 values; at n=1000 ≈ 116 agents per family tie at 1.0, so flat
  argmax and plain MIDIAN pick among them blindly. This is why halving (≈ 12 probes behind its winner) and MIDIAN-V
  (verified promotion) win at β ≤ 0.25, and why b, not the tree, sets the ceiling (§7).
- **Shared S.** True skill is measured once per prompt signature and shared by every seed; the seed CIs above capture
  population draw and stream variation only. The binomial error of S itself (200 probes; 60 on the ≥9B rungs) is
  ±0.035 / ±0.065 per cell at p = 0.5 and is a common offset, not part of the intervals.
- **Frameworks.** Fallback rates in §1 (CrewAI mostly measures the fallback). Framework grids use Q = 300 and 3 seeds
  (MIDIAN variants, oracle, flat, declared, supervisor have 5; all 60 cells complete).
  Wall-clock per task for frameworks includes the supervisor call against a shared, sometimes saturated fleet.
- **Reports accounting.** Plain MIDIAN charges one report per (peer, member, family, probe) as the spec reads;
  MIDIAN-V and peer-reported halving charge one per (peer, member, family) — the peer's mean — so their report
  columns are ~3× lower for the same information (DEVIATIONS 2026-09-02 MIDIAN-V bullet).
- **Wall-clock columns** mix cache-hit and cache-miss runs (the LLM memo, and since 17:30 the python-tool memo);
  use counts (probes, messages, comparisons) for cost claims, wall-clock only for the frameworks' supervisor calls.

## 12. Deviations from SPEC.md (summary; all 100+ dated bullets in DEVIATIONS.md)

Backend: NVIDIA/CUDA fleet (not ROCm); Gemma-2 for the non-Qwen half of the ladder; six families swapped for ones the
ladder can actually separate (final K=16 list, all measured); python tool gated at ≥3B, calculator dropped; S measured
per prompt signature and shared across seeds; index-seeded shared probe instances (the k-th probe of an (agent,
family) is the same instance for every method); memo keyed on content hash, sharded per process, tool runs memoised.
Methods: declared lie is clip(D_honest + 0.4); collusion rule exists in scalar and vectorised forms; MIDIAN pads per
level (not to r^L), leaders not materialised, one report_many per cohort chunk, `stratify` not implemented; MIDIAN-V,
r=5, cached, peer-reported halving and the framework `retrieval: midian` variant are post-hoc labeled additions;
decentral rivals' message accounting per CONTRACT; NSW index parameters fixed after a transposition bug.
Runner/analysis: per-row JSON files, row-level `--only` sharding, success_late = last quarter, route-to-many majority
of binary outcomes, bernoulli_scale at K=16 and b=1 above 10⁶, replay twins programmatic-only (no self-description
channel on replay). Ops: flock hangs on the shared scratch filesystem (locks in /tmp, SQLite nolock), MIG partition unusable, FlashInfer
sampler off, replicas with client-side latency-aware re-pick, ~500 sharded jobs.

## 13. Reproduce

`configs/grid.yaml` holds every grid; `scripts/launch_live.sh` (RTE_GRIDS / RTE_ONLY / RTE_SHARD / RTE_SEED_SHARD)
launches sharded jobs against the fleet (`scripts/serve_fleet.sbatch`, replicas via `serve_replica.sbatch`);
`python -m rte.analyze --grid <g>` rewrites a grid's summary and figures; `scripts/check_methods.py` and `pytest`
(152 tests) verify ledger formulas and method correctness. Data, memo and rows: `$RTE_DATA` on the shared scratch filesystem.

Figures (v1, 2026-09-02 data, 5 seeds, pooled error bars): the seven synthesis figures A–G, now under `figures/v1/` (also `$RTE_DATA/results/extra_figs/` (`results/rte_figures_core.zip`; regenerate with
`scripts/extra_figs.py`): A headline frameworks vs MIDIAN, B verified-shortlist lift, C MIDIAN vs halving by β × liar selection,
D learning curve, E every rival vs β by class, F cost scaling 10²–10⁷, G budget sweep + MIDIAN internals. The analyzer's per-grid
F1–F7 (46 files, `results/rte_figures.zip`) are the machine-generated originals behind them.
