# DEVIATIONS.md — where the implementation departs from SPEC.md, and why
- 2026-09-02: cluster is NVIDIA/CUDA (FAS RC), not ROCm; `VLLM_ROCM_USE_AITER=0` is irrelevant here.
- 2026-09-02: declared-channel lie uses `clip(D_honest + 0.4)` (so it "inflates on top" of the self-described channel too);
  for `programmatic` D_honest = S + N(0,0.05) this is the spec's `clip(S + 0.4)` up to the declaration noise.
- 2026-09-02: `skill_excess_ratio` (per-agent-mean variance / binomial floor) is structurally ~0 for `specialist`
  populations (everyone has 3 good families → flat per-agent mean). We report it AND `skill_excess_ratio_family`
  (median per-family ratio); the ≥1.5 gate is applied to the family version, stated beside every result.
- 2026-09-02: route-to-many majority is majority-of-binary-outcomes (optimistic proxy for majority-of-answers).
- 2026-09-02: `replay` backend's RouterBench normalization (`scripts/02_download_routerbench.py`) follows the old
  repo's `_normalize_routerbench` verbatim (category = raw `eval_name`, which already splits every MMLU subject
  into its own eval_name; keep categories with >= 60 rows; binarize a model's score at >= 0.5). At that threshold
  67 categories survive on the actual `withmartian/routerbench` 0-shot pickle, not the 64 named in SPEC.md/task
  prose — matches the old repo's own config comments ("N=500, K=67"), so 64 appears to be stale prose rather than
  a real target. K=64 is honored by taking the 16 (or up to 67) categories with the most prompts when a caller
  passes a smaller K; the full table carries K=67. 11 models present, matching SPEC's "11 models".
- 2026-09-02: `replay` backend's handicap rule (masked/handicapped category): the agent's recorded outcome is
  replaced by the outcome of the WEAKEST model (lowest overall mean accuracy) in that category, unless a coin
  flip w.p. 0.7 forces 0 regardless. i.e. `outcome = 0 if rng<0.7 else weakest_model_outcome`. Chosen because it
  (a) never lets a masked agent see its own true (possibly strong) outcome, (b) still occasionally reflects a
  real "some model got this right" case rather than being a pure zero mask, keeping cell-lookup semantics
  (real recorded outcomes only, never synthesized 0/1 draws outside the table).
- 2026-09-02: `replay` backend's `declared("self_described")` is identical to `declared("programmatic")`
  (`S + N(0,0.05)` clipped): there is no LLM in the replay backend to self-describe, matching `bernoulli.py`'s
  same documented deviation.
