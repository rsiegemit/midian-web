"""rte.analyze on a toy frame: arm aliasing, framework accountings from method_stats, per-channel split, v2 targets."""
import json
import numpy as np
import pandas as pd
import pytest

from rte import analyze as A

CELL = dict(backend="llm", n=1000, K=16, dist="specialist", liar_select="random", collude=True, lie_mode="inflate", demand="uniform", b=3, Q=300)
COLS = ("success_late", "regret", "misroute_to_liar", "oracle_success", "build_probes", "comparisons_per_task", "messages_per_task", "seed")


def frame(rows):
    """rows: (method, params, beta, channel, seed, success, method_stats) -> a rows.csv-like frame."""
    out = []
    for m, p, beta, ch, seed, s, st in rows:
        out.append({**CELL, "method": m, "params": json.dumps(p), "beta": beta, "declared_source": ch, "seed": seed, "success": s,
                    "success_late": s, "regret": 0.8 - s, "misroute_to_liar": 0.0, "oracle_success": 0.8, "build_probes": 48000 * (1.05 if m == "midian_a" else 1),
                    "comparisons_per_task": 30.0, "messages_per_task": 6.0, "method_stats": st})
    return A.prepare(pd.DataFrame(out))


def toy():
    rows = []
    for ch in ("self_described", "programmatic"):
        for beta in (0.0, 0.25):
            for seed in (1, 2):
                base = 0.60 + 0.02 * seed
                rows += [("midian", {}, beta, ch, seed, base, ""), ("midian_sh", {}, beta, ch, seed, base + 0.05, ""),
                         ("midian_a", {}, beta, ch, seed, base + 0.005, ""), ("midian_sha", {}, beta, ch, seed, base + 0.05, ""),
                         ("sequential_halving", {"peer_reported": True}, beta, ch, seed, base + 0.06, ""),
                         ("flat_probe_argmax", {"online": True}, beta, ch, seed, base - 0.02, ""),
                         ("declared_argmax", {}, beta, ch, seed, base - (0.1 if ch == "self_described" else 0.0), ""),
                         ("fw_autogen", {}, beta, "self_described", seed, base - 0.1, json.dumps({"picks": 250, "fallbacks": 50, "success_strict": base - 0.2, "fallback_rate": 0.167}))]
    for seed in (1, 2):                                   # beta=0.5, colluding low-skill liars: the robustness cells
        base = 0.55 + 0.02 * seed
        for m, s in (("midian", base), ("midian_sh", base - 0.01), ("midian_a", base + 0.04), ("midian_sha", base + 0.04)):
            rows.append((m, {}, 0.5, "self_described", seed, s, ""))
    df = frame(rows); df.loc[np.isclose(df.beta, 0.5), "liar_select"] = "low_skill_first"
    return df


def test_aliases_stats_and_channels():
    df = toy()
    assert {"flat_probe_argmax_online", "sequential_halving_peer"} <= set(df.label)
    fw = df[df.method == "fw_autogen"]
    assert np.isclose(fw.fallback_rate, 0.167).all() and fw.success_strict.notna().all() and df[df.method == "midian"].fallback_rate.isna().all()
    text = "\n".join(A.by_channel(df, A.paired(df)))
    assert "[self_described]" in text and "[programmatic]" in text and A.UPPER in text
    assert "STRICT" in "\n".join(A.strict(df))


def test_targets_v2_verdicts():
    df, fits = toy(), pd.DataFrame()
    v = {t["target"]: t["verdict"] for t in A.targets_v2(df, fits)}
    assert v["V2-1"] == "HIT"            # SH within 0.02 of halving_peer, same per-task cost
    assert v["V2-3"] == "HIT"            # SH+A >= max(SH, A) - 0.01
    assert v["V2-6"] == "NO DATA" and v["V2-8"] == "NO DATA"
    assert v["V2-2"] == "HIT"            # A: loss beta 0->0.5 <= 0.02, unchanged within 0.01 at beta<=0.25, 1.05x probes
