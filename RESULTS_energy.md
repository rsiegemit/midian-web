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
| fw_google_adk | 0 | 0.0704 | 70 | 704 | 7,045 | 137.0 | 78.3 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_smolagents | 0 | 0.0736 | 74 | 736 | 7,362 | 143.1 | 81.8 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_openai_agents | 0 | 0.0824 | 82 | 824 | 8,238 | 160.2 | 91.5 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_langgraph | 0 | 0.0830 | 83 | 830 | 8,300 | 161.4 | 92.2 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_crewai | 0 | 0.1702 | 170 | 1,702 | 17,025 | 331.0 | 189.2 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_llamaindex | 0 | 0.2113 | 211 | 2,113 | 21,131 | 410.9 | 234.8 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_camel_workforce | 0 | 0.2678 | 268 | 2,678 | 26,783 | 520.8 | 297.6 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_magentic_one | 0 | 0.4873 | 487 | 4,873 | 48,733 | 947.6 | 541.5 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_magentic_one{"supervisor":"Qwen/Qwen2.5-14B-Instruct"} | 0 | 0.8057 | 806 | 8,057 | 80,573 | 1,566.7 | 895.3 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |

**Crossing points (tasks routed before the probe-based method's cumulative cost drops below the framework's):**

| | autogen | maf | google_adk | smolagents | openai_agents | langgraph | crewai | llamaindex | camel_workforce | magentic_one | magentic_one{"supervisor":"Qwen/Qwen2.5-14B-Instruct"} |
|---|---|---|---|---|---|---|---|---|---|---|---|
| midian | 9,416 | 4,531 | 3,931 | 3,762 | 3,362 | 3,337 | 1,627 | 1,311 | 1,034 | 568 | 344 |
| midian_a | 9,884 | 4,756 | 4,126 | 3,949 | 3,528 | 3,502 | 1,707 | 1,376 | 1,085 | 596 | 361 |
| midian_v | 9,385 | 4,516 | 3,918 | 3,749 | 3,350 | 3,325 | 1,621 | 1,306 | 1,031 | 566 | 343 |

In MESSAGES the crossing is immediate: MIDIAN 1,010 + 6t vs a framework 1,000 + 12t crosses at t = 2; MIDIAN-V (1,010 + 2t) at t = 1. In COMPARISONS MIDIAN pays 30 per task vs a framework's 10 (MIDIAN-V 1, halving 1, flat 1,000), so MIDIAN never undercuts a framework on comparisons while MIDIAN-V does from the first task. Per task, MIDIAN's cost is communication (messages and comparisons), not LLM compute: it makes no LLM call at route time.

MIDIAN's 48,000-probe build by population shape (GPU-s): specialist 277, heavy_tail 84, bimodal 98; the crossings scale with it (heavy_tail and bimodal cross ~3x sooner).
Reading: against a one-call framework (AutoGen) MIDIAN breaks even after ~9,400 tasks on specialist (~2,900 on heavy_tail), MIDIAN-A after ~9,900; against the multi-call frameworks (CrewAI, LlamaIndex, CAMEL) after 1,000-1,600 tasks; against Magentic-One after ~570 (7B) / ~340 (14B arm). Before the crossing the framework is cheaper; after it, the probe-based methods' cost is flat while every framework's keeps growing linearly.
