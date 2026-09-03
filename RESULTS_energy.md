# Runtime and energy — ESTIMATE (*)

Runtime/energy ESTIMATE (*) per method from LLM-call counts x measured per-call GPU cost.  python scripts/energy.py
Wall-clock in the rows is not used for probe methods (memo hits). Model: GPU-seconds per call = params_b * (A*prompt_tok + B*gen_tok),
B = 5A (decode is ~5x prefill per token on H100/vLLM), A calibrated so a 7B supervisor call (1,900 prompt + 65 gen tokens) costs
1/34 GPU-s = the throughput measured on the saturated 1-GPU 7B replicas (2026-09-03, 4 samples, 32-36 req/s). Energy = GPU-s * W.

Per-call GPU-seconds: 7B supervisor call 0.0294; expected probe/execution call by population shape: specialist 0.00577, heavy_tail 0.00176, bimodal 0.00203 (specialist mixes all 7 models uniformly; heavy_tail 90% 0.5-1.5B; bimodal 80% 0.5B / 20% 7B).
Framework supervisor cost = measured latency ratio to AutoGen (one call) x the 7B call cost; the 14B Magentic-One arm is scaled by 14/7. CPU-side routing (tree descent, TF-IDF) is microseconds per task and omitted. The routed task's own execution (0.0058 GPU-s per task on specialist) is common to every method and excluded.

CUMULATIVE LLM compute after t routed tasks = build + t x per-task (n=1000 specialist). Probe-based methods pay their build up front and then ~0 per task; frameworks pay 0 up front and a supervisor call per task, so the crossing of the two lines is the break-even.

| method | build GPU-s | per-task GPU-s | cumulative GPU-s @ t=1k | @ 10k | @ 100k | cumulative Wh @ 10k (700 W) | (400 W) | messages @ 10k (build + 10k/task) | comparisons @ 10k |
|---|---|---|---|---|---|---|---|---|---|
| declared_argmax | 0 | 0.0000 | 0 | 0 | 0 | 0.0 | 0.0 | 1,000 (1,000 + 0/task) | 10,000,000 (0 + 1000/task) |
| verify_on_claim | 0 | 0.0101 | 10 | 101 | 1,013 | 19.7 | 11.3 | 1,000 (1,000 + 0/task) | 169,840 (0 + 17/task) |
| llm_supervisor | 0 | 0.0138 | 14 | 138 | 1,376 | 26.8 | 15.3 | 221,000 (1,000 + 22/task) | 200,000 (0 + 20/task) |
| sequential_halving | 259 | 0.0000 | 259 | 259 | 259 | 50.4 | 28.8 | 0 (0 + 0/task) | 10,000 (0 + 1/task) |
| sequential_halving{"peer_reported":true} | 259 | 0.0000 | 259 | 259 | 259 | 50.4 | 28.8 | 0 (0 + 0/task) | 10,000 (0 + 1/task) |
| midian_v | 276 | 0.0000 | 276 | 276 | 276 | 53.7 | 30.7 | 51,010 (1,010 + 5/task) | 310,000 (0 + 31/task) |
| flat_probe_argmax{"online":true} | 277 | 0.0000 | 277 | 277 | 277 | 53.9 | 30.8 | 0 (0 + 0/task) | 10,000,000 (0 + 1000/task) |
| flat_probe_argmax | 277 | 0.0000 | 277 | 277 | 277 | 53.9 | 30.8 | 0 (0 + 0/task) | 10,000,000 (0 + 1000/task) |
| linucb_honest | 277 | 0.0000 | 277 | 277 | 277 | 53.9 | 30.8 | 0 (0 + 0/task) | 10,000,000 (0 + 1000/task) |
| warm_start_bandit | 277 | 0.0000 | 277 | 277 | 277 | 53.9 | 30.8 | 1,000 (1,000 + 0/task) | 10,000,000 (0 + 1000/task) |
| midian | 277 | 0.0000 | 277 | 277 | 277 | 53.9 | 30.8 | 91,010 (1,010 + 9/task) | 600,000 (0 + 60/task) |
| midian_sh | 277 | 0.0000 | 277 | 277 | 277 | 53.9 | 30.8 | 91,010 (1,010 + 9/task) | 600,000 (0 + 60/task) |
| midian_va | 285 | 0.0000 | 285 | 285 | 285 | 55.4 | 31.7 | 51,638 (1,010 + 5/task) | 316,280 (0 + 32/task) |
| midian_a | 291 | 0.0000 | 291 | 291 | 291 | 56.5 | 32.3 | 91,010 (1,010 + 9/task) | 600,000 (0 + 60/task) |
| midian_sha | 291 | 0.0000 | 291 | 291 | 291 | 56.5 | 32.3 | 91,010 (1,010 + 9/task) | 600,000 (0 + 60/task) |
| fw_autogen | 0 | 0.0294 | 29 | 294 | 2,941 | 57.2 | 32.7 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_maf | 0 | 0.0611 | 61 | 611 | 6,112 | 118.8 | 67.9 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_google_adk | 0 | 0.0737 | 74 | 737 | 7,373 | 143.4 | 81.9 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_smolagents | 0 | 0.0760 | 76 | 760 | 7,605 | 147.9 | 84.5 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_openai_agents | 0 | 0.0824 | 82 | 824 | 8,238 | 160.2 | 91.5 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_langgraph | 0 | 0.0835 | 84 | 835 | 8,354 | 162.4 | 92.8 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_crewai | 0 | 0.1836 | 184 | 1,836 | 18,364 | 357.1 | 204.0 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_llamaindex | 0 | 0.2202 | 220 | 2,202 | 22,022 | 428.2 | 244.7 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_camel_workforce | 0 | 0.2691 | 269 | 2,691 | 26,915 | 523.3 | 299.1 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_magentic_one | 0 | 0.5114 | 511 | 5,114 | 51,144 | 994.5 | 568.3 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_magentic_one{"supervisor":"Qwen/Qwen2.5-14B-Instruct"} | 0 | 0.8023 | 802 | 8,023 | 80,234 | 1,560.1 | 891.5 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |

**Crossing points (tasks routed before the probe-based method's cumulative cost drops below the framework's):**

| | autogen | maf | google_adk | smolagents | openai_agents | langgraph | crewai | llamaindex | camel_workforce | magentic_one | magentic_one{"supervisor":"Qwen/Qwen2.5-14B-Instruct"} |
|---|---|---|---|---|---|---|---|---|---|---|---|
| midian | 9,416 | 4,531 | 3,756 | 3,642 | 3,362 | 3,315 | 1,508 | 1,258 | 1,029 | 542 | 345 |
| midian_a | 9,884 | 4,756 | 3,943 | 3,822 | 3,528 | 3,480 | 1,583 | 1,320 | 1,080 | 568 | 362 |
| midian_v | 9,385 | 4,516 | 3,744 | 3,630 | 3,350 | 3,304 | 1,503 | 1,253 | 1,026 | 540 | 344 |

In MESSAGES (fetch 2 per level + observe-time update 1 per level, commit 3415f03) the crossing is immediate: MIDIAN 1,010 + 9t vs a framework 1,000 + 12t crosses at t = 3; MIDIAN-V (1,010 + 5t) at t = 1. In COMPARISONS MIDIAN pays 60 per task (30 descent + 30 observe-time update) vs a framework's 10 (MIDIAN-V 31 = 1 cached pick + 30 update; halving 1; flat 1,000), so no MIDIAN variant undercuts a framework on comparisons; MIDIAN-V's saving over MIDIAN is the descent, not the update. Per task, MIDIAN's cost is communication (messages and comparisons), not LLM compute: it makes no LLM call at route time.

MIDIAN's 48,000-probe build by population shape (GPU-s): specialist 277, heavy_tail 84, bimodal 98; the crossings scale with it (heavy_tail and bimodal cross ~3x sooner).
Reading: against a one-call framework (AutoGen) MIDIAN breaks even after ~9,400 tasks on specialist (~2,900 on heavy_tail), MIDIAN-A after ~9,900; against the multi-call frameworks (CrewAI, LlamaIndex, CAMEL) after 1,000-1,600 tasks; against Magentic-One after ~570 (7B) / ~340 (14B arm). Before the crossing the framework is cheaper; after it, the probe-based methods' cost is flat while every framework's keeps growing linearly.

## Combined currencies (*)

(a) ENERGY, joules per event: LLM call = GPU-s x 700 W (a 7B supervisor call = 20.6 J; a specialist probe = 4.04 J); message = one RPC handled in ~100 us on a ~10 W core = 0.001 J (pessimistic column: 0.01 J); comparison = one float compare = 1e-08 J.
(b) LATENCY on the critical path per task: each sequential message hop = 1 ms RTT (MIDIAN: the 2·depth = 6 fetch hops; the observe-time update propagation (1 message per level) is off the critical path and excluded; MIDIAN-V 2 ms; frameworks and llm_supervisor: 2 hops + the supervisor call at its measured median latency under shared-fleet load); comparisons 10 ns each (flat's 1,000 = 10 us); a route-time probe call (verify_on_claim) 0.3 s.

| method | J/task at t=10k (build amortised) | J/task, pessimistic messages | of which LLM J/task | messages J/task | comparisons J/task | latency s/task |
|---|---|---|---|---|---|---|
| declared_argmax | 0.000 | 0.001 | 0.000 | 0.0001 | 1.00e-05 | 0.0000 |
| verify_on_claim | 7.093 | 7.094 | 7.093 | 0.0001 | 1.70e-07 | 0.5269 |
| llm_supervisor | 9.653 | 9.852 | 9.631 | 0.0221 | 2.00e-07 | 0.5218 |
| sequential_halving | 18.146 | 18.146 | 18.146 | 0.0000 | 1.00e-08 | 0.0000 |
| sequential_halving{"peer_reported":true} | 18.146 | 18.146 | 18.146 | 0.0000 | 1.00e-08 | 0.0000 |
| midian_v | 19.327 | 19.373 | 19.322 | 0.0051 | 3.10e-07 | 0.0020 |
| flat_probe_argmax{"online":true} | 19.386 | 19.386 | 19.386 | 0.0000 | 1.00e-05 | 0.0000 |
| flat_probe_argmax | 19.386 | 19.386 | 19.386 | 0.0000 | 1.00e-05 | 0.0000 |
| linucb_honest | 19.386 | 19.386 | 19.386 | 0.0000 | 1.00e-05 | 0.0000 |
| warm_start_bandit | 19.386 | 19.387 | 19.386 | 0.0001 | 1.00e-05 | 0.0000 |
| midian | 19.395 | 19.477 | 19.386 | 0.0091 | 6.00e-07 | 0.0060 |
| midian_sh | 19.395 | 19.477 | 19.386 | 0.0091 | 6.00e-07 | 0.0060 |
| midian_va | 19.965 | 20.011 | 19.960 | 0.0052 | 3.16e-07 | 0.0021 |
| midian_a | 20.357 | 20.439 | 20.348 | 0.0091 | 6.00e-07 | 0.0060 |
| midian_sha | 20.357 | 20.439 | 20.348 | 0.0091 | 6.00e-07 | 0.0060 |
| fw_autogen | 20.600 | 20.709 | 20.588 | 0.0121 | 1.00e-07 | 1.1132 |
| fw_maf | 42.797 | 42.906 | 42.785 | 0.0121 | 1.00e-07 | 2.3111 |
| fw_google_adk | 51.624 | 51.733 | 51.612 | 0.0121 | 1.00e-07 | 2.7875 |
| fw_smolagents | 53.247 | 53.355 | 53.234 | 0.0121 | 1.00e-07 | 2.8751 |
| fw_openai_agents | 57.682 | 57.790 | 57.669 | 0.0121 | 1.00e-07 | 3.1144 |
| fw_langgraph | 58.490 | 58.598 | 58.477 | 0.0121 | 1.00e-07 | 3.1580 |
| fw_crewai | 128.558 | 128.666 | 128.545 | 0.0121 | 1.00e-07 | 6.9396 |
| fw_llamaindex | 154.168 | 154.277 | 154.156 | 0.0121 | 1.00e-07 | 8.3218 |
| fw_camel_workforce | 188.417 | 188.526 | 188.405 | 0.0121 | 1.00e-07 | 10.1703 |
| fw_magentic_one | 358.017 | 358.126 | 358.005 | 0.0121 | 1.00e-07 | 19.3236 |
| fw_magentic_one{"supervisor":"Qwen/Qwen2.5-14B-Instruct"} | 561.653 | 561.762 | 561.641 | 0.0121 | 1.00e-07 | 15.1580 |

Crossings in joules (tasks after which the probe-based method's cumulative energy falls below the framework's): midian: autogen 9,415, maf 4,531, google_adk 3,756, smolagents 3,641, openai_agents 3,361, langgraph 3,315, crewai 1,508, llamaindex 1,258, camel_workforce 1,029, magentic_one 542, magentic_one{"supervisor":"Qwen/Qwen2.5-14B-Instruct"} 345; midian_a: autogen 9,882, maf 4,756, google_adk 3,942, smolagents 3,822, openai_agents 3,528, langgraph 3,480, crewai 1,583, llamaindex 1,320, camel_workforce 1,080, magentic_one 568, magentic_one{"supervisor":"Qwen/Qwen2.5-14B-Instruct"} 362; midian_v: autogen 9,382, maf 4,515, google_adk 3,743, smolagents 3,629, openai_agents 3,350, langgraph 3,304, crewai 1,503, llamaindex 1,253, camel_workforce 1,026, magentic_one 540, magentic_one{"supervisor":"Qwen/Qwen2.5-14B-Instruct"} 344.
Under any sane weighting the LLM call dominates energy by 3-5 orders of magnitude (20.6 J per supervisor call vs 6e-3 J for MIDIAN's six messages and 3e-7 J for its thirty comparisons per task), so the joule crossings equal the GPU-second crossings to the task; messages dominate MIDIAN's latency (6 ms of fetch hops vs 0.6 us of comparisons) while the supervisor call dominates every framework's (0.5-19 s).
