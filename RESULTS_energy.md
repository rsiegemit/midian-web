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
| llm_supervisor | 0 | 0.0080 | 8 | 80 | 796 | 15.5 | 8.8 | 221,000 (1,000 + 22/task) | 200,000 (0 + 20/task) |
| verify_on_claim | 0 | 0.0101 | 10 | 101 | 1,013 | 19.7 | 11.3 | 1,000 (1,000 + 0/task) | 169,840 (0 + 17/task) |
| sequential_halving | 259 | 0.0000 | 259 | 259 | 259 | 50.4 | 28.8 | 0 (0 + 0/task) | 10,000 (0 + 1/task) |
| sequential_halving{"peer_reported":true} | 259 | 0.0000 | 259 | 259 | 259 | 50.4 | 28.8 | 0 (0 + 0/task) | 10,000 (0 + 1/task) |
| midian_v | 276 | 0.0000 | 276 | 276 | 276 | 53.7 | 30.7 | 51,010 (1,010 + 5/task) | 310,000 (0 + 31/task) |
| flat_probe_argmax{"online":true} | 277 | 0.0000 | 277 | 277 | 277 | 53.9 | 30.8 | 0 (0 + 0/task) | 10,000,000 (0 + 1000/task) |
| flat_probe_argmax | 277 | 0.0000 | 277 | 277 | 277 | 53.9 | 30.8 | 0 (0 + 0/task) | 10,000,000 (0 + 1000/task) |
| linucb_honest | 277 | 0.0000 | 277 | 277 | 277 | 53.9 | 30.8 | 0 (0 + 0/task) | 10,000,000 (0 + 1000/task) |
| warm_start_bandit | 277 | 0.0000 | 277 | 277 | 277 | 53.9 | 30.8 | 1,000 (1,000 + 0/task) | 10,000,000 (0 + 1000/task) |
| midian | 277 | 0.0000 | 277 | 277 | 277 | 53.9 | 30.8 | 91,010 (1,010 + 9/task) | 600,000 (0 + 60/task) |
| midian_sh | 277 | 0.0000 | 277 | 277 | 277 | 53.9 | 30.8 | 91,010 (1,010 + 9/task) | 600,000 (0 + 60/task) |
| midian_va | 285 | 0.0000 | 285 | 285 | 285 | 55.4 | 31.7 | 51,772 (1,010 + 5/task) | 317,620 (0 + 32/task) |
| midian_a | 291 | 0.0000 | 291 | 291 | 291 | 56.5 | 32.3 | 91,194 (1,010 + 9/task) | 601,840 (0 + 60/task) |
| midian_sha | 291 | 0.0000 | 291 | 291 | 291 | 56.5 | 32.3 | 91,010 (1,010 + 9/task) | 600,000 (0 + 60/task) |
| fw_autogen | 0 | 0.0294 | 29 | 294 | 2,941 | 57.2 | 32.7 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_maf | 0 | 0.0353 | 35 | 353 | 3,534 | 68.7 | 39.3 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_smolagents | 0 | 0.0440 | 44 | 440 | 4,397 | 85.5 | 48.9 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_google_adk | 0 | 0.0473 | 47 | 473 | 4,732 | 92.0 | 52.6 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_langgraph | 0 | 0.0568 | 57 | 568 | 5,681 | 110.5 | 63.1 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_openai_agents | 0 | 0.0617 | 62 | 617 | 6,169 | 119.9 | 68.5 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_crewai | 0 | 0.1188 | 119 | 1,188 | 11,879 | 231.0 | 132.0 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_llamaindex | 0 | 0.1453 | 145 | 1,453 | 14,533 | 282.6 | 161.5 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_camel_workforce | 0 | 0.1484 | 148 | 1,484 | 14,835 | 288.5 | 164.8 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_magentic_one | 0 | 0.2806 | 281 | 2,806 | 28,058 | 545.6 | 311.8 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |
| fw_magentic_one{"supervisor":"Qwen/Qwen2.5-14B-Instruct"} | 0 | 0.4721 | 472 | 4,721 | 47,212 | 918.0 | 524.6 | 121,000 (1,000 + 12/task) | 100,000 (0 + 10/task) |

**Crossing points (tasks routed before the probe-based method's cumulative cost drops below the framework's):**

| | autogen | maf | smolagents | google_adk | langgraph | openai_agents | crewai | llamaindex | camel_workforce | magentic_one | magentic_one{"supervisor":"Qwen/Qwen2.5-14B-Instruct"} |
|---|---|---|---|---|---|---|---|---|---|---|---|
| midian | 9,416 | 7,836 | 6,298 | 5,852 | 4,875 | 4,489 | 2,331 | 1,906 | 1,867 | 987 | 587 |
| midian_a | 9,882 | 8,224 | 6,610 | 6,142 | 5,116 | 4,712 | 2,447 | 2,000 | 1,959 | 1,036 | 616 |
| midian_v | 9,385 | 7,810 | 6,277 | 5,833 | 4,858 | 4,474 | 2,324 | 1,899 | 1,861 | 984 | 585 |

In MESSAGES (fetch 2 per level + observe-time update 1 per level, commit 3415f03) the crossing is immediate: MIDIAN 1,010 + 9t vs a framework 1,000 + 12t crosses at t = 3; MIDIAN-V (1,010 + 5t) at t = 1. In COMPARISONS MIDIAN pays 60 per task (30 descent + 30 observe-time update) vs a framework's 10 (MIDIAN-V 31 = 1 cached pick + 30 update; halving 1; flat 1,000), so no MIDIAN variant undercuts a framework on comparisons; MIDIAN-V's saving over MIDIAN is the descent, not the update. Per task, MIDIAN's cost is communication (messages and comparisons), not LLM compute: it makes no LLM call at route time.

MIDIAN's 48,000-probe build by population shape (GPU-s): specialist 277, heavy_tail 84, bimodal 98; the crossings scale with it (heavy_tail and bimodal cross ~3x sooner).
Reading: against a one-call framework (AutoGen) MIDIAN breaks even after ~9,400 tasks on specialist (~2,900 on heavy_tail), MIDIAN-A after ~9,900; against the multi-call frameworks (CrewAI, LlamaIndex, CAMEL) after 1,900-2,300 tasks; against Magentic-One after ~990 (7B) / ~590 (14B arm). Before the crossing the framework is cheaper; after it, the probe-based methods' cost is flat while every framework's keeps growing linearly.

## Combined currencies (*)

(a) ENERGY, joules per event: LLM call = GPU-s x 700 W (a 7B supervisor call = 20.6 J; a specialist probe = 4.04 J); message = one RPC handled in ~100 us on a ~10 W core = 0.001 J (pessimistic column: 0.01 J); comparison = one float compare = 1e-08 J.
(b) LATENCY on the critical path per task: each sequential message hop = 1 ms RTT (MIDIAN: the 2·depth = 6 fetch hops; the observe-time update propagation (1 message per level) is off the critical path and excluded; MIDIAN-V 2 ms; frameworks and llm_supervisor: 2 hops + the supervisor call at its measured median latency under shared-fleet load); comparisons 10 ns each (flat's 1,000 = 10 us); a route-time probe call (verify_on_claim) 0.3 s.

| method | J/task at t=10k (build amortised) | J/task, pessimistic messages | of which LLM J/task | messages J/task | comparisons J/task | latency s/task |
|---|---|---|---|---|---|---|
| declared_argmax | 0.000 | 0.001 | 0.000 | 0.0001 | 1.00e-05 | 0.0000 |
| llm_supervisor | 5.591 | 5.790 | 5.569 | 0.0221 | 2.00e-07 | 0.5218 |
| verify_on_claim | 7.093 | 7.094 | 7.093 | 0.0001 | 1.70e-07 | 0.5269 |
| sequential_halving | 18.146 | 18.146 | 18.146 | 0.0000 | 1.00e-08 | 0.0000 |
| sequential_halving{"peer_reported":true} | 18.146 | 18.146 | 18.146 | 0.0000 | 1.00e-08 | 0.0000 |
| midian_v | 19.327 | 19.373 | 19.322 | 0.0051 | 3.10e-07 | 0.0020 |
| flat_probe_argmax{"online":true} | 19.386 | 19.386 | 19.386 | 0.0000 | 1.00e-05 | 0.0000 |
| flat_probe_argmax | 19.386 | 19.386 | 19.386 | 0.0000 | 1.00e-05 | 0.0000 |
| linucb_honest | 19.386 | 19.386 | 19.386 | 0.0000 | 1.00e-05 | 0.0000 |
| warm_start_bandit | 19.386 | 19.387 | 19.386 | 0.0001 | 1.00e-05 | 0.0000 |
| midian | 19.395 | 19.477 | 19.386 | 0.0091 | 6.00e-07 | 0.0060 |
| midian_sh | 19.395 | 19.477 | 19.386 | 0.0091 | 6.00e-07 | 0.0060 |
| midian_va | 19.963 | 20.009 | 19.958 | 0.0052 | 3.18e-07 | 0.0021 |
| midian_a | 20.355 | 20.437 | 20.346 | 0.0091 | 6.02e-07 | 0.0060 |
| midian_sha | 20.357 | 20.439 | 20.348 | 0.0091 | 6.00e-07 | 0.0060 |
| fw_autogen | 20.600 | 20.709 | 20.588 | 0.0121 | 1.00e-07 | 1.9236 |
| fw_maf | 24.752 | 24.860 | 24.739 | 0.0121 | 1.00e-07 | 2.3111 |
| fw_smolagents | 30.794 | 30.903 | 30.782 | 0.0121 | 1.00e-07 | 2.8751 |
| fw_google_adk | 33.138 | 33.247 | 33.126 | 0.0121 | 1.00e-07 | 3.0939 |
| fw_langgraph | 39.782 | 39.891 | 39.770 | 0.0121 | 1.00e-07 | 3.7140 |
| fw_openai_agents | 43.194 | 43.303 | 43.182 | 0.0121 | 1.00e-07 | 4.0325 |
| fw_crewai | 83.163 | 83.272 | 83.151 | 0.0121 | 1.00e-07 | 7.7630 |
| fw_llamaindex | 101.742 | 101.851 | 101.730 | 0.0121 | 1.00e-07 | 9.4971 |
| fw_camel_workforce | 103.858 | 103.967 | 103.846 | 0.0121 | 1.00e-07 | 9.6947 |
| fw_magentic_one | 196.420 | 196.529 | 196.408 | 0.0121 | 1.00e-07 | 18.3341 |
| fw_magentic_one{"supervisor":"Qwen/Qwen2.5-14B-Instruct"} | 330.494 | 330.603 | 330.482 | 0.0121 | 1.00e-07 | 15.4251 |

Crossings in joules (tasks after which the probe-based method's cumulative energy falls below the framework's): midian: autogen 9,415, maf 7,835, smolagents 6,297, google_adk 5,852, langgraph 4,874, openai_agents 4,489, crewai 2,331, llamaindex 1,906, camel_workforce 1,867, magentic_one 987, magentic_one{"supervisor":"Qwen/Qwen2.5-14B-Instruct"} 587; midian_a: autogen 9,881, maf 8,223, smolagents 6,609, google_adk 6,141, langgraph 5,115, openai_agents 4,711, crewai 2,447, llamaindex 2,000, camel_workforce 1,959, magentic_one 1,036, magentic_one{"supervisor":"Qwen/Qwen2.5-14B-Instruct"} 616; midian_v: autogen 9,382, maf 7,808, smolagents 6,276, google_adk 5,831, langgraph 4,858, openai_agents 4,474, crewai 2,323, llamaindex 1,899, camel_workforce 1,860, magentic_one 984, magentic_one{"supervisor":"Qwen/Qwen2.5-14B-Instruct"} 585.
Under any sane weighting the LLM call dominates energy by 3-5 orders of magnitude (20.6 J per supervisor call vs 6e-3 J for MIDIAN's six messages and 3e-7 J for its thirty comparisons per task), so the joule crossings equal the GPU-second crossings to the task; messages dominate MIDIAN's latency (6 ms of fetch hops vs 0.6 us of comparisons) while the supervisor call dominates every framework's (0.5-19 s).
