# Per-channel tables (v2 work order 0.1) — generated 2026-09-03 by `rte.analyze` (new "per declaration channel" section)

Programmatic = **upper bound (S + N(0,0.05))**: an honest declaration no live agent produces. Self-described = the live channel
(agents' own self-descriptions). Frameworks read self-descriptions only and appear in the fw_* grids. Labels: `flat_probe_argmax_frozen`,
`flat_probe_argmax_online`, `midian_v`, `midian_v_r5`, `sequential_halving_peer` (one name per arm everywhere; 0.5).

## Key numbers, n=1000 (live_f1_n1000, paired on (shape, β, liar selection, seed): 60 units per channel)

| method | self_described success | self_described Δ vs midian | self_described rival wins | programmatic success | programmatic Δ vs midian | programmatic rival wins | prog − self |
|---|---|---|---|---|---|---|---|
| warm_start_bandit | 0.678 | 0.036 | 93/120 | 0.685 | 0.043 | 113/120 | 0.007 |
| verify_on_claim | 0.647 | 0.005 | 76/119 | 0.672 | 0.03 | 92/118 | 0.025 |
| midian_llm_descent | 0.628 | -0.014 | 28/117 | 0.627 | -0.015 | 28/117 | -0.001 |
| cluster_head_router | 0.567 | -0.075 | 36/120 | 0.654 | 0.012 | 78/119 | 0.087 |
| llm_supervisor | 0.567 | -0.076 | 53/119 | 0.584 | -0.058 | 53/119 | 0.017 |
| cnp_self_bid | 0.56 | -0.082 | 26/118 | 0.652 | 0.01 | 77/120 | 0.092 |
| route_to_k_majority | 0.557 | -0.085 | 29/120 | 0.659 | 0.017 | 84/120 | 0.102 |
| declared_argmax | 0.549 | -0.094 | 23/120 | 0.656 | 0.014 | 80/120 | 0.107 |
| declared_softmax | 0.473 | -0.169 | 0/120 | 0.595 | -0.047 | 11/120 | 0.122 |
| disrouter_cascade | 0.369 | -0.273 | 0/120 | 0.619 | -0.023 | 36/117 | 0.25 |

Frameworks at n=1000 (fw_live_n1000, self-described channel, 60 paired cells):

| framework | success (self_described) | Δ vs midian | rival wins | cells |
|---|---|---|---|---|
| fw_autogen | 0.52 | -0.118 | 28/59 | 60 |
| fw_camel_workforce | 0.51 | -0.127 | 28/59 | 60 |
| fw_crewai | 0.535 | -0.102 | 28/60 | 60 |
| fw_google_adk | 0.536 | -0.102 | 29/60 | 60 |
| fw_langgraph | 0.521 | -0.116 | 28/59 | 60 |
| fw_llamaindex | 0.519 | -0.118 | 28/59 | 60 |
| fw_maf | 0.518 | -0.12 | 29/60 | 60 |
| fw_magentic_one | 0.541 | -0.096 | 28/59 | 60 |
| fw_openai_agents | 0.519 | -0.119 | 29/60 | 60 |
| fw_smolagents | 0.523 | -0.114 | 28/60 | 60 |

## live_f1_n1000 — the generated section (β and shape tables per channel, paired roll-ups per channel)

## Success by method, per declaration channel

### Declared-channel readers, success x beta [self_described]

declaration = self_described; the live channel: agents' own self-descriptions

| group | label | 0.0 | 0.1 | 0.25 | 0.5 |
| --- | --- | --- | --- | --- | --- |
| declared | cluster_head_router | 0.6081 | 0.5615 | 0.5549 | 0.5452 |
| declared | cnp_self_bid | 0.6014 | 0.5587 | 0.5420 | 0.5386 |
| declared | declared_argmax | 0.5935 | 0.5401 | 0.5305 | 0.5305 |
| declared | declared_softmax | 0.5477 | 0.4960 | 0.4432 | 0.4047 |
| declared | disrouter_cascade | 0.5240 | 0.3028 | 0.3099 | 0.3396 |
| declared | llm_supervisor | 0.5727 | 0.5666 | 0.5645 | 0.5631 |
| declared | route_to_k_majority | 0.5992 | 0.5506 | 0.5453 | 0.5334 |
| verified_central | verify_on_claim | 0.6557 | 0.6485 | 0.6345 | 0.6483 |
| verified_central | warm_start_bandit | 0.6804 | 0.6795 | 0.6739 | 0.6777 |


### Declared-channel readers, success x dist [self_described]

declaration = self_described; the live channel: agents' own self-descriptions

| group | label | bimodal | heavy_tail | specialist |
| --- | --- | --- | --- | --- |
| declared | cluster_head_router | 0.5350 | 0.5893 | 0.5779 |
| declared | cnp_self_bid | 0.5343 | 0.5872 | 0.5591 |
| declared | declared_argmax | 0.5325 | 0.5418 | 0.5716 |
| declared | declared_softmax | 0.4519 | 0.4492 | 0.5176 |
| declared | disrouter_cascade | 0.2932 | 0.3503 | 0.4638 |
| declared | llm_supervisor | 0.5746 | 0.6274 | 0.4982 |
| declared | route_to_k_majority | 0.5268 | 0.5879 | 0.5566 |
| verified_central | verify_on_claim | 0.5596 | 0.6712 | 0.7095 |
| verified_central | warm_start_bandit | 0.5700 | 0.6959 | 0.7676 |


### MIDIAN vs declared-channel readers, paired by seed [self_described]

declaration = self_described; the live channel: agents' own self-descriptions

| group | rival | cells | delta_mean | delta_lo | delta_hi | delta_min | delta_max | midian_better | rival_better | within_floor | min_sign_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| verified_central | warm_start_bandit | 24 | -0.0356 | -0.0477 | -0.0248 | -0.1860 | 0.0298 | 3 | 9 | 12 | 0.0625 |
| verified_central | verify_on_claim | 24 | -0.0045 | -0.0228 | 0.0143 | -0.1662 | 0.1334 | 6 | 5 | 13 | 0.0625 |
| declared | cluster_head_router | 24 | 0.0749 | 0.0460 | 0.1048 | -0.0432 | 0.2524 | 11 | 0 | 13 | 0.0625 |
| declared | llm_supervisor | 24 | 0.0755 | 0.0557 | 0.0950 | -0.1206 | 0.3090 | 8 | 5 | 11 | 0.0625 |
| declared | cnp_self_bid | 24 | 0.0821 | 0.0656 | 0.0971 | -0.0434 | 0.2688 | 12 | 0 | 12 | 0.0625 |
| declared | route_to_k_majority | 24 | 0.0852 | 0.0538 | 0.1164 | -0.0384 | 0.2568 | 13 | 0 | 11 | 0.0625 |
| declared | declared_argmax | 24 | 0.0936 | 0.0676 | 0.1191 | -0.0294 | 0.2696 | 16 | 0 | 8 | 0.0625 |
| declared | declared_softmax | 24 | 0.1694 | 0.1542 | 0.1832 | 0.0322 | 0.3084 | 22 | 0 | 2 | 0.0625 |
| declared | disrouter_cascade | 24 | 0.2732 | 0.2562 | 0.2884 | 0.0272 | 0.4268 | 20 | 0 | 4 | 0.0625 |


### Declared-channel readers, success x beta [programmatic]

declaration = programmatic; programmatic = upper bound (S + N(0,0.05)): an honest declaration no live agent produces

| group | label | 0.0 | 0.1 | 0.25 | 0.5 |
| --- | --- | --- | --- | --- | --- |
| declared | cluster_head_router | 0.7145 | 0.6421 | 0.6296 | 0.6281 |
| declared | cnp_self_bid | 0.7168 | 0.6439 | 0.6257 | 0.6235 |
| declared | declared_argmax | 0.7180 | 0.6443 | 0.6304 | 0.6328 |
| declared | declared_softmax | 0.6532 | 0.6054 | 0.5728 | 0.5489 |
| declared | disrouter_cascade | 0.6510 | 0.6063 | 0.6059 | 0.6119 |
| declared | llm_supervisor | 0.5946 | 0.5805 | 0.5831 | 0.5791 |
| declared | route_to_k_majority | 0.7197 | 0.6489 | 0.6345 | 0.6318 |
| verified_central | verify_on_claim | 0.7191 | 0.6680 | 0.6508 | 0.6509 |
| verified_central | warm_start_bandit | 0.7076 | 0.6877 | 0.6750 | 0.6686 |


### Declared-channel readers, success x dist [programmatic]

declaration = programmatic; programmatic = upper bound (S + N(0,0.05)): an honest declaration no live agent produces

| group | label | bimodal | heavy_tail | specialist |
| --- | --- | --- | --- | --- |
| declared | cluster_head_router | 0.5205 | 0.6622 | 0.7779 |
| declared | cnp_self_bid | 0.5192 | 0.6642 | 0.7740 |
| declared | declared_argmax | 0.5215 | 0.6668 | 0.7809 |
| declared | declared_softmax | 0.4939 | 0.5884 | 0.7029 |
| declared | disrouter_cascade | 0.5027 | 0.6136 | 0.7401 |
| declared | llm_supervisor | 0.5746 | 0.6274 | 0.5510 |
| declared | route_to_k_majority | 0.5219 | 0.6663 | 0.7880 |
| verified_central | verify_on_claim | 0.5352 | 0.6815 | 0.7998 |
| verified_central | warm_start_bandit | 0.5478 | 0.6907 | 0.8156 |


### MIDIAN vs declared-channel readers, paired by seed [programmatic]

declaration = programmatic; programmatic = upper bound (S + N(0,0.05)): an honest declaration no live agent produces

| group | rival | cells | delta_mean | delta_lo | delta_hi | delta_min | delta_max | midian_better | rival_better | within_floor | min_sign_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| verified_central | warm_start_bandit | 24 | -0.0427 | -0.0563 | -0.0294 | -0.1532 | 0.0050 | 0 | 11 | 13 | 0.0625 |
| verified_central | verify_on_claim | 24 | -0.0301 | -0.0491 | -0.0111 | -0.1106 | 0.0654 | 3 | 10 | 11 | 0.0625 |
| declared | route_to_k_majority | 24 | -0.0167 | -0.0326 | -0.0013 | -0.0972 | 0.1066 | 4 | 6 | 14 | 0.0625 |
| declared | declared_argmax | 24 | -0.0143 | -0.0321 | 0.0033 | -0.1030 | 0.1080 | 3 | 7 | 14 | 0.0625 |
| declared | cluster_head_router | 24 | -0.0115 | -0.0298 | 0.0063 | -0.0972 | 0.1058 | 3 | 6 | 15 | 0.0625 |
| declared | cnp_self_bid | 24 | -0.0104 | -0.0249 | 0.0034 | -0.1004 | 0.1046 | 4 | 6 | 14 | 0.0625 |
| declared | disrouter_cascade | 24 | 0.0233 | 0.0034 | 0.0441 | -0.0778 | 0.0966 | 9 | 1 | 14 | 0.0625 |
| declared | declared_softmax | 24 | 0.0470 | 0.0334 | 0.0595 | -0.0024 | 0.1170 | 13 | 0 | 11 | 0.0625 |
| declared | llm_supervisor | 24 | 0.0577 | 0.0389 | 0.0758 | -0.1206 | 0.2630 | 9 | 5 | 10 | 0.0625 |


### Success x beta

probe-only methods: identical across channels

| group | label | 0.0 | 0.1 | 0.25 | 0.5 |
| --- | --- | --- | --- | --- | --- |
| ceiling | oracle | 0.7228 | 0.7228 | 0.7228 | 0.7228 |
| floor | random | 0.3112 | 0.3112 | 0.3112 | 0.3112 |
| midian | midian | 0.6657 | 0.6627 | 0.6456 | 0.5946 |
| midian | midian[online=False] | 0.6124 | 0.5984 | 0.5844 | 0.5508 |
| midian | midian[r=5] | 0.6644 | 0.6597 | 0.6391 | 0.5643 |
| midian | midian_llm_descent | 0.6521 | 0.6505 | 0.6341 | 0.5740 |
| midian | midian_v | 0.6814 | 0.6846 | 0.6767 | 0.5750 |
| midian | midian_v_r5 | 0.6896 | 0.6914 | 0.6803 | 0.5622 |
| verified_central | flat_nsw_router | 0.6333 | 0.6303 | 0.6270 | 0.6270 |
| verified_central | flat_probe_argmax_frozen | 0.6214 | 0.6212 | 0.6186 | 0.6149 |
| verified_central | flat_probe_argmax_online | 0.6663 | 0.6664 | 0.6669 | 0.6672 |
| verified_central | sequential_halving | 0.7229 | 0.7225 | 0.7221 | 0.7224 |
| verified_central | sequential_halving_peer | 0.7217 | 0.7222 | 0.7181 | 0.5453 |
| verified_central | thompson_per_family | 0.6030 | 0.6034 | 0.6037 | 0.6037 |
| verified_central | trueskill_per_family | 0.6259 | 0.6213 | 0.6227 | 0.6227 |
| verified_central | ucb_per_family | 0.5960 | 0.5970 | 0.5978 | 0.5978 |
| verified_decentral | gossip_reputation_greedy | 0.5960 | 0.5996 | 0.4768 | 0.3604 |
| verified_decentral | referral_network | 0.4507 | 0.4381 | 0.4083 | 0.3403 |


## Frameworks' supervisor latency (the one wall-clock table; 0.6)

Framework supervisor calls go straight to vLLM through each framework's own OpenAI client and are never memoised, so
their per-task wall-clock is cache-consistent; every other wall-clock column is dropped (memo hits and misses mixed).
These are latencies under shared-fleet load, not compute costs.

## Frameworks' supervisor latency per task (seconds; cache-consistent, under shared-fleet load)

Every other wall-clock column is omitted: memo hits and misses are mixed and say nothing about cost.

| label | n | q25_s | median_s | q75_s |
| --- | --- | --- | --- | --- |
| fw_autogen | 1000 | 0.16 | 0.35 | 0.59 |
| fw_camel_workforce | 1000 | 1.82 | 2.79 | 5.85 |
| fw_crewai | 1000 | 0.11 | 0.13 | 0.45 |
| fw_google_adk | 1000 | 0.37 | 0.63 | 1.10 |
| fw_langgraph | 1000 | 0.33 | 0.59 | 1.14 |
| fw_llamaindex | 1000 | 1.35 | 2.19 | 4.13 |
| fw_maf | 1000 | 0.73 | 1.17 | 1.62 |
| fw_magentic_one | 1000 | 4.34 | 5.40 | 6.91 |
| fw_openai_agents | 1000 | 0.33 | 0.47 | 1.22 |
| fw_smolagents | 1000 | 0.87 | 1.40 | 2.32 |



## Frameworks' supervisor latency per task (seconds; cache-consistent, under shared-fleet load)

Every other wall-clock column is omitted: memo hits and misses are mixed and say nothing about cost.

| label | n | q25_s | median_s | q75_s |
| --- | --- | --- | --- | --- |
| fw_autogen | 100 | 0.16 | 0.35 | 0.63 |
| fw_camel_workforce | 100 | 1.96 | 3.73 | 5.79 |
| fw_crewai | 100 | 0.12 | 0.19 | 1.02 |
| fw_google_adk | 100 | 0.30 | 0.51 | 1.08 |
| fw_langgraph | 100 | 0.30 | 0.40 | 0.94 |
| fw_llamaindex | 100 | 1.59 | 2.25 | 4.60 |
| fw_maf | 100 | 0.77 | 1.19 | 1.67 |
| fw_magentic_one | 100 | 4.40 | 5.67 | 7.44 |
| fw_openai_agents | 100 | 0.31 | 0.51 | 1.36 |
| fw_smolagents | 100 | 0.90 | 1.50 | 2.14 |


