"""The agent population: the model ladder (config) and how a skill distribution draws profiles.

A profile is `{id, model, specialty: [family index], tool}`. SPEC §3's llm column says a
distribution is *realized* by how profiles are drawn; the resulting S is MEASURED, never assumed.
Model ids come only from configs/models.yaml — the code selects by parameter-count band.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import yaml

from ..stable_hash import stable_seed_32

# The tool axis collapsed to one option: `calculator` measured at or BELOW the no-tool arm on the
# 7B (0.40 vs 0.47 on basic_arithmetic, 0.73 vs 0.80 on gcd) because one expression cannot carry a
# multi-step task and emitting it costs a turn. Every agent therefore holds `python`, and
# `signature()` withholds it on handicapped families -- which is exactly SPEC §1's "family tool
# removed". The field stays in the profile for the record.
TOOL = "python"


def tool_for(model: str, cfg: dict) -> str:
    """`python` only for agents at or above `tool_min_b`. MEASURED: below it the tool is a
    handicap, not a capability -- the mean specialty-minus-handicapped gap is -0.03 on the 0.5B and
    -0.11 on the 1.5B, worst on chain_sum where the 1.5B scores 0.30 with the tool and 0.65
    without. Gating it keeps the handicap monotone across the whole ladder."""
    b = next((m["params_b"] for m in cfg["models"] if m["id"] == model), 0.0)
    return TOOL if b >= cfg.get("tool_min_b", 0.0) else "none"

CONFIG = Path(os.environ.get("RTE_MODELS_YAML",
                             Path(__file__).resolve().parents[2] / "configs" / "models.yaml"))


def ladder() -> dict:
    """configs/models.yaml — the only place a model id is written (besides frameworks SUPERVISOR)."""
    return yaml.safe_load(CONFIG.read_text())


def bands(cfg: dict) -> tuple[list[str], list[str], list[str], frozenset[str]]:
    """(all ids, small, big, measured-with-fewer-probes) by parameter count. No ids in code."""
    m = cfg["models"]
    return ([x["id"] for x in m],
            [x["id"] for x in m if x["params_b"] <= cfg["small_max_b"]],
            [x["id"] for x in m if x["params_b"] >= cfg["big_min_b"]],
            frozenset(x["id"] for x in m if x["params_b"] >= cfg["large_min_b"]))


def draw_profiles(n: int, K: int, dist: str, seed: int, cfg: dict | None = None) -> list[dict]:
    cfg = cfg or ladder()
    ids, small, big, _ = bands(cfg)
    rng = np.random.default_rng(stable_seed_32(seed, "profiles", n, K, dist))
    pick = lambda pool: pool[int(rng.integers(len(pool)))]              # noqa: E731
    out = []
    # DRAW ORDER IS PART OF THE SEED CONTRACT: model, then specialty, for every distribution.
    # Reordering these rng calls changes every population.
    for a in range(n):
        if dist == "specialist":            # 3 families unhandicapped, models mixed
            model = pick(ids)
            spec = sorted(int(x) for x in rng.choice(K, min(3, K), replace=False))
        elif dist == "heavy_tail":          # 1 in 10 is a big model, unhandicapped everywhere
            e = bool(rng.random() < 0.1)
            model = pick(big) if e else pick(small)
            spec = list(range(K)) if e else []
        elif dist == "bimodal":             # 20% smallest big model with tools, 80% smallest without
            g = bool(rng.random() < 0.2)
            model = big[0] if g else small[0]
            spec = list(range(K)) if g else []
        elif dist == "correlated":          # handicaps are group-level over 4 family groups
            model = pick(ids)
            good = {g for g in range(4) if rng.random() < 0.5}
            spec = [f for f in range(K) if f % 4 in good]
        elif dist == "iid_uniform":         # per-(agent, family), independent
            model = pick(ids)
            spec = [f for f in range(K) if rng.random() < 0.5]
        else:
            raise ValueError(f"unknown skill_dist {dist!r}")
        out.append({"id": a, "model": model, "specialty": list(spec),
                    "tool": tool_for(model, cfg)})
    return out


def signature(profile: dict, f: int, max_tokens: int, handicap_max_tokens: int) -> tuple:
    """Everything that determines an agent's prompt on family f, hence (at temperature 0) its
    answer. Agents sharing a signature share memo entries — see rte.backends.llm."""
    hand = f not in profile["specialty"]
    return (profile["model"], hand, "none" if hand else profile["tool"],
            handicap_max_tokens if hand else max_tokens)


def pinned_cfg(models: list[str]) -> dict:
    """A one-band ladder over exactly `models`. Used when only part of the fleet is up (the smoke
    server, a calibration run, the live tests) so every profile draws from what is actually served.

    Per-model CAPABILITIES are inherited from the real ladder — dropping `system_role` here would
    silently reintroduce the Gemma-2 "System role not supported" failure."""
    real = {m["id"]: m for m in ladder()["models"]}
    return {"models": [dict(real.get(m, {}), id=m, params_b=1.0, tp=1, gpu=0, port=0, gpu_share=0.9)
                       for m in models],
            "small_max_b": 1.0, "big_min_b": 1.0, "large_min_b": 99.0}
