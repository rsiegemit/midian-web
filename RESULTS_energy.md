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
| midian_v | 276 | 0.0000 | 276 | 276 | 276 | 53.7 | 30.7 | 21,010 (1,010 + 2/task) | 10,000 (0 + 1/task) |
| flat_probe_argmax{"online":true} | 277 | 0.0000 | 277 | 277 | 277 | 53.9 | 30.8 | 0 (0 + 0/task) | 10,000,000 (0 + 1000/task) |
| flat_probe_argmax | 277 | 0.0000 | 277 | 277 | 277 | 53.9 | 30.8 | 0 (0 + 0/task) | 10,000,000 (0 + 1000/task) |
| linucb_honest | 277 | 0.0000 | 277 | 277 | 277 | 53.9 | 30.8 | 0 (0 + 0/task) | 10,000,000 (0 + 1000/task) |
| warm_start_bandit | 277 | 0.0000 | 277 | 277 | 277 | 53.9 | 30.8 | 1,000 (1,000 + 0/task) | 10,000,000 (0 + 1000/task) |
| midian | 277 | 0.0000 | 277 | 277 | 277 | 53.9 | 30.8 | 61,010 (1,010 + 6/task) | 300,000 (0 + 30/task) |
| midian_sh | 277 | 0.0000 | 277 | 277 | 277 | 53.9 | 30.8 | 61,010 (1,010 + 6/task) | 300,000 (0 + 30/task) |
| midian_sha | 291 | 0.0000 | 291 | 291 | 291 | 56.5 | 32.3 | 61,010 (1,010 + 6/task) | 300,000 (0 + 30/task) |
| midian_a | 291 | 0.0000 | 291 | 291 | 291 | 56.5 | 32.3 | 61,010 (1,010 + 6/task) | 300,000 (0 + 30/task) |
| fw_autogen | 0 | 0.0294 | 29 | 294 | 2,941 | 57.2 | 32.7 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_maf | 0 | 0.0611 | 61 | 611 | 6,112 | 118.8 | 67.9 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_google_adk | 0 | 0.0729 | 73 | 729 | 7,286 | 141.7 | 81.0 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_smolagents | 0 | 0.0736 | 74 | 736 | 7,362 | 143.1 | 81.8 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_openai_agents | 0 | 0.0824 | 82 | 824 | 8,238 | 160.2 | 91.5 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_langgraph | 0 | 0.0835 | 84 | 835 | 8,354 | 162.4 | 92.8 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_crewai | 0 | 0.1702 | 170 | 1,702 | 17,025 | 331.0 | 189.2 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_llamaindex | 0 | 0.2088 | 209 | 2,088 | 20,879 | 406.0 | 232.0 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_camel_workforce | 0 | 0.2678 | 268 | 2,678 | 26,783 | 520.8 | 297.6 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_magentic_one | 0 | 0.5050 | 505 | 5,050 | 50,499 | 981.9 | 561.1 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_magentic_one{"supervisor":"Qwen/Qwen2.5-14B-Instruct"} | 0 | 0.8020 | 802 | 8,020 | 80,201 | 1,559.5 | 891.1 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |

**Crossing points (tasks routed before the probe-based method's cumulative cost drops below the framework's):**

| | autogen | maf | google_adk | smolagents | openai_agents | langgraph | crewai | llamaindex | camel_workforce | magentic_one | magentic_one{"supervisor":"Qwen/Qwen2.5-14B-Instruct"} |
|---|---|---|---|---|---|---|---|---|---|---|---|
| midian | 9,416 | 4,531 | 3,801 | 3,762 | 3,362 | 3,315 | 1,627 | 1,326 | 1,034 | 548 | 345 |
| midian_a | 9,884 | 4,756 | 3,990 | 3,949 | 3,528 | 3,480 | 1,707 | 1,392 | 1,085 | 576 | 362 |
| midian_v | 9,385 | 4,516 | 3,788 | 3,749 | 3,350 | 3,304 | 1,621 | 1,322 | 1,031 | 547 | 344 |

In MESSAGES the crossing is immediate: MIDIAN 1,010 + 6t vs a framework 1,000 + 12t crosses at t = 2; MIDIAN-V (1,010 + 2t) at t = 1. In COMPARISONS MIDIAN pays 30 per task vs a framework's 10 (MIDIAN-V 1, halving 1, flat 1,000), so MIDIAN never undercuts a framework on comparisons while MIDIAN-V does from the first task. Per task, MIDIAN's cost is communication (messages and comparisons), not LLM compute: it makes no LLM call at route time.

MIDIAN's 48,000-probe build by population shape (GPU-s): specialist 277, heavy_tail 84, bimodal 98; the crossings scale with it (heavy_tail and bimodal cross ~3x sooner).
Reading: against a one-call framework (AutoGen) MIDIAN breaks even after ~9,400 tasks on specialist (~2,900 on heavy_tail), MIDIAN-A after ~9,900; against the multi-call frameworks (CrewAI, LlamaIndex, CAMEL) after 1,000-1,600 tasks; against Magentic-One after ~570 (7B) / ~340 (14B arm). Before the crossing the framework is cheaper; after it, the probe-based methods' cost is flat while every framework's keeps growing linearly.

## Combined currencies (*)

(a) ENERGY, joules per event: LLM call = GPU-s x 700 W (a 7B supervisor call = 20.6 J; a specialist probe = 4.04 J); message = one RPC handled in ~100 us on a ~10 W core = 0.001 J (pessimistic column: 0.01 J); comparison = one float compare = 1e-08 J.
(b) LATENCY on the critical path per task: each sequential message hop = 1 ms RTT (MIDIAN: 2 per tree level = its messages/task; MIDIAN-V 2 ms; frameworks and llm_supervisor: 2 hops + the supervisor call at its measured median latency under shared-fleet load); comparisons 10 ns each (flat's 1,000 = 10 us); a route-time probe call (verify_on_claim) 0.3 s.

| method | J/task at t=10k (build amortised) | J/task, pessimistic messages | of which LLM J/task | messages J/task | comparisons J/task | latency s/task |
|---|---|---|---|---|---|---|
| declared_argmax | 0.000 | 0.001 | 0.000 | 0.0001 | 1.00e-05 | 0.0000 |
| verify_on_claim | 7.093 | 7.094 | 7.093 | 0.0001 | 1.70e-07 | 0.5269 |
| llm_supervisor | 9.653 | 9.852 | 9.631 | 0.0221 | 2.00e-07 | 0.5218 |
| sequential_halving | 18.146 | 18.146 | 18.146 | 0.0000 | 1.00e-08 | 0.0000 |
| sequential_halving{"peer_reported":true} | 18.146 | 18.146 | 18.146 | 0.0000 | 1.00e-08 | 0.0000 |
| midian_v | 19.324 | 19.343 | 19.322 | 0.0021 | 1.00e-08 | 0.0020 |
| flat_probe_argmax{"online":true} | 19.386 | 19.386 | 19.386 | 0.0000 | 1.00e-05 | 0.0000 |
| flat_probe_argmax | 19.386 | 19.386 | 19.386 | 0.0000 | 1.00e-05 | 0.0000 |
| linucb_honest | 19.386 | 19.386 | 19.386 | 0.0000 | 1.00e-05 | 0.0000 |
| warm_start_bandit | 19.386 | 19.387 | 19.386 | 0.0001 | 1.00e-05 | 0.0000 |
| midian | 19.392 | 19.447 | 19.386 | 0.0061 | 3.00e-07 | 0.0060 |
| midian_sh | 19.392 | 19.447 | 19.386 | 0.0061 | 3.00e-07 | 0.0060 |
| midian_sha | 20.354 | 20.409 | 20.348 | 0.0061 | 3.00e-07 | 0.0060 |
| midian_a | 20.354 | 20.409 | 20.348 | 0.0061 | 3.00e-07 | 0.0060 |
| fw_autogen | 20.600 | 20.709 | 20.588 | 0.0121 | 1.00e-07 | 1.1132 |
| fw_maf | 42.797 | 42.906 | 42.785 | 0.0121 | 1.00e-07 | 2.3111 |
| fw_google_adk | 51.016 | 51.125 | 51.004 | 0.0121 | 1.00e-07 | 2.7547 |
| fw_smolagents | 51.545 | 51.654 | 51.533 | 0.0121 | 1.00e-07 | 2.7832 |
| fw_openai_agents | 57.682 | 57.790 | 57.669 | 0.0121 | 1.00e-07 | 3.1144 |
| fw_langgraph | 58.490 | 58.598 | 58.477 | 0.0121 | 1.00e-07 | 3.1580 |
| fw_crewai | 119.185 | 119.294 | 119.173 | 0.0121 | 1.00e-07 | 6.4338 |
| fw_llamaindex | 146.168 | 146.277 | 146.156 | 0.0121 | 1.00e-07 | 7.8901 |
| fw_camel_workforce | 187.491 | 187.600 | 187.479 | 0.0121 | 1.00e-07 | 10.1203 |
| fw_magentic_one | 353.506 | 353.615 | 353.494 | 0.0121 | 1.00e-07 | 19.0802 |
| fw_magentic_one{"supervisor":"Qwen/Qwen2.5-14B-Instruct"} | 561.422 | 561.531 | 561.410 | 0.0121 | 1.00e-07 | 15.1517 |

Crossings in joules (tasks after which the probe-based method's cumulative energy falls below the framework's): midian: autogen 9,413, maf 4,530, google_adk 3,801, smolagents 3,761, openai_agents 3,361, langgraph 3,315, crewai 1,627, llamaindex 1,326, camel_workforce 1,034, magentic_one 548, magentic_one{"supervisor":"Qwen/Qwen2.5-14B-Instruct"} 345; midian_a: autogen 9,881, maf 4,755, google_adk 3,989, smolagents 3,948, openai_agents 3,528, langgraph 3,479, crewai 1,707, llamaindex 1,392, camel_workforce 1,085, magentic_one 576, magentic_one{"supervisor":"Qwen/Qwen2.5-14B-Instruct"} 362; midian_v: autogen 9,380, maf 4,515, google_adk 3,788, smolagents 3,749, openai_agents 3,350, langgraph 3,304, crewai 1,621, llamaindex 1,322, camel_workforce 1,031, magentic_one 547, magentic_one{"supervisor":"Qwen/Qwen2.5-14B-Instruct"} 344.
Under any sane weighting the LLM call dominates energy by 3-5 orders of magnitude (20.6 J per supervisor call vs 6e-3 J for MIDIAN's six messages and 3e-7 J for its thirty comparisons per task), so the joule crossings equal the GPU-second crossings to the task; messages dominate MIDIAN's latency (6 ms vs 0.3 us of comparisons) while the supervisor call dominates every framework's (0.5-18 s).
