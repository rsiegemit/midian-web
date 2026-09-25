# TARGETS_rte.md — pre-registered expectations (committed before any run)

1. All declared-channel methods lose ≥0.25 success from β=0 → 0.5 under `inflate`; MIDIAN and all probe-only methods move ≤0.03.
2. MIDIAN's success equals `flat_probe_argmax` within 0.02 at β=0 (it is a max-tree over the same estimates) and its per-task comparisons scale as r·log_r n while flat scans scale as n (fit exponents; report).
3. With `collude=True`, MIDIAN degrades once expected liars per cohort exceeds 3 (β·r > ⌊δ(r−1)⌋ ⇒ β > 0.3); trimming vs no trimming (δ=0) separates *only* in that regime. If it doesn't, say so.
4. `verify_on_claim` matches oracle within 0.03 at β ≤ 0.1 and loses ≥0.10 by β=0.5 (verification budget drains on liars).
5. `sequential_halving` ≈ `flat_probe_argmax` at equal budget; bandits' `success_late` ≥ MIDIAN's when build budget b=1 (online learning wins when build is thin).
6. Under `heavy_tail`, the gap between argmax-finding methods and `random`/`route_to_k` is largest; under `iid_uniform`, smallest.

Misses are reported as misses. No parameter changes after the first run.

---
