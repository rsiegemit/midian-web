# Runtime and energy — ESTIMATE (*)

Runtime/energy ESTIMATE (*) per method from LLM-call counts x measured per-call GPU cost.  python scripts/energy.py
Wall-clock in the rows is not used for probe methods (memo hits). Model: GPU-seconds per call = params_b * (A*prompt_tok + B*gen_tok),
B = 5A (decode is ~5x prefill per token on H100/vLLM), A calibrated so a 7B supervisor call (1,900 prompt + 65 gen tokens) costs
1/34 GPU-s = the throughput measured on the saturated 1-GPU 7B replicas (2026-09-03, 4 samples, 32-36 req/s). Energy = GPU-s * W.

Per-call GPU-seconds: 7B supervisor call 0.0294; expected probe/execution call by population shape: specialist 0.00577, heavy_tail 0.00176, bimodal 0.00203 (specialist mixes all 7 models uniformly; heavy_tail 90% 0.5-1.5B; bimodal 80% 0.5B / 20% 7B).
Framework supervisor cost = measured latency ratio to AutoGen (one call) x the 7B call cost; the 14B Magentic-One arm is scaled by 14/7. CPU-side routing (tree descent, TF-IDF) is microseconds per task and omitted.

| method | build GPU-s | route GPU-s/task | task-execution GPU-s/task (common) | supervisor call-equiv/task | GPU-s per 1k tasks @Q=300 | @Q=1000 | @Q=10000 | Wh per 1k tasks @Q=1000 (700 W) | Wh (400 W) | break-even Q vs AutoGen |
|---|---|---|---|---|---|---|---|---|---|---|
| declared_argmax | 0 | 0.0000 | 0.0058 | 0.0 | 0 | 0 | 0 | 0.0 | 0.0 | 0 |
| verify_on_claim | 0 | 0.0101 | 0.0058 | 0.0 | 10 | 10 | 10 | 2.0 | 1.1 | 0 |
| llm_supervisor | 0 | 0.0138 | 0.0058 | 0.5 | 14 | 14 | 14 | 2.7 | 1.5 | 0 |
| fw_autogen | 0 | 0.0294 | 0.0058 | 1.0 | 29 | 29 | 29 | 5.7 | 3.3 | never |
| fw_maf | 0 | 0.0611 | 0.0058 | 2.1 | 61 | 61 | 61 | 11.9 | 6.8 | never |
| fw_google_adk | 0 | 0.0704 | 0.0058 | 2.4 | 70 | 70 | 70 | 13.7 | 7.8 | never |
| fw_smolagents | 0 | 0.0736 | 0.0058 | 2.5 | 74 | 74 | 74 | 14.3 | 8.2 | never |
| fw_openai_agents | 0 | 0.0796 | 0.0058 | 2.7 | 80 | 80 | 80 | 15.5 | 8.8 | never |
| fw_langgraph | 0 | 0.0820 | 0.0058 | 2.8 | 82 | 82 | 82 | 15.9 | 9.1 | never |
| fw_crewai | 0 | 0.1721 | 0.0058 | 5.9 | 172 | 172 | 172 | 33.5 | 19.1 | never |
| fw_llamaindex | 0 | 0.2136 | 0.0058 | 7.3 | 214 | 214 | 214 | 41.5 | 23.7 | never |
| sequential_halving | 259 | 0.0000 | 0.0058 | 0.0 | 864 | 259 | 26 | 50.4 | 28.8 | 8814 |
| sequential_halving{"peer_reported":true} | 259 | 0.0000 | 0.0058 | 0.0 | 864 | 259 | 26 | 50.4 | 28.8 | 8814 |
| fw_camel_workforce | 0 | 0.2611 | 0.0058 | 8.9 | 261 | 261 | 261 | 50.8 | 29.0 | never |
| midian_v | 276 | 0.0000 | 0.0058 | 0.0 | 920 | 276 | 28 | 53.7 | 30.7 | 9385 |
| flat_probe_argmax{"online":true} | 277 | 0.0000 | 0.0058 | 0.0 | 923 | 277 | 28 | 53.9 | 30.8 | 9416 |
| midian_sh | 277 | 0.0000 | 0.0058 | 0.0 | 923 | 277 | 28 | 53.9 | 30.8 | 9416 |
| midian | 277 | 0.0000 | 0.0058 | 0.0 | 923 | 277 | 28 | 53.9 | 30.8 | 9416 |
| linucb_honest | 277 | 0.0000 | 0.0058 | 0.0 | 923 | 277 | 28 | 53.9 | 30.8 | 9416 |
| flat_probe_argmax | 277 | 0.0000 | 0.0058 | 0.0 | 923 | 277 | 28 | 53.9 | 30.8 | 9416 |
| warm_start_bandit | 277 | 0.0000 | 0.0058 | 0.0 | 923 | 277 | 28 | 53.9 | 30.8 | 9416 |
| midian_sha | 291 | 0.0000 | 0.0058 | 0.0 | 969 | 291 | 29 | 56.5 | 32.3 | 9884 |
| midian_a | 291 | 0.0000 | 0.0058 | 0.0 | 969 | 291 | 29 | 56.5 | 32.3 | 9884 |
| fw_magentic_one | 0 | 0.4890 | 0.0058 | 16.6 | 489 | 489 | 489 | 95.1 | 54.3 | never |
| fw_magentic_one{"supervisor":"Qwen/Qwen2.5-14B-Instruct"} | 0 | 0.8088 | 0.0058 | 13.7 | 809 | 809 | 809 | 157.3 | 89.9 | never |

MIDIAN's 48,000-probe build by population shape (GPU-s): specialist 277, heavy_tail 84, bimodal 98; one AutoGen supervisor call = 0.0294 GPU-s, so on specialist the build equals ~9,416 one-call framework tasks.
Reading: at Q = 1,000 every probe-based method spends ~9x a one-call framework's GPU-seconds (the build dominates); at Q = 10,000 they are level with AutoGen and ~10-30x cheaper than the multi-call frameworks; MIDIAN-A's audits add 5%; MIDIAN-V and halving are the cheapest builds (fewer probes).
