"""The one place that knows MIDIAN's old method keys.

MIDIAN is one method with two defenses as parameters, both on by default:
    old key                          new key                                   display label
    midian_va                        midian                                    MIDIAN
    midian_a                         midian{"verify": false}                   MIDIAN w/o verification
    midian_v, midian{verify,cached}  midian{"audit": false}                    MIDIAN w/o audits
    midian                           midian{"audit": false, "verify": false}   MIDIAN w/o defenses
Framework shortlists named after a MIDIAN cohort follow: retrieval "midian_va" -> "midian" (the full method's cohort),
retrieval "midian" (the verified, unaudited cohort) -> "midian_wo_audit". midian_sh / midian_sha are withdrawn (no key).

to_new(method, params) translates a stored or configured key; it is applied on every read, so rows written before the
rename and after it load identically. No random stream depends on the method key (a View's rng is seeded by the world
seed and the method's `needs`), so a rerun of any cell under the new keys reproduces the rows stored under the old ones.
A results directory whose rows were rewritten carries SENTINEL; assert_v2 checks that it holds no old key."""
from __future__ import annotations

SENTINEL = ".method_keys_v2"
OLD = frozenset({"midian_va", "midian_a", "midian_v", "midian_sh", "midian_sha"})
WITHDRAWN = frozenset({"midian_sh", "midian_sha"})
RETRIEVAL = {"midian_va": "midian", "midian": "midian_wo_audit"}


def _canon(p: dict) -> dict:
    """Drop MIDIAN flags equal to their defaults (audit on at 5 %, verify on, cached = verify)."""
    p = dict(p)
    audit = p.get("audit", True)
    if audit is True or audit == 0.05: p.pop("audit", None)
    elif not audit: p["audit"] = False
    verify = bool(p.get("verify", True))
    if verify: p.pop("verify", None)
    else: p["verify"] = False
    if "cached" in p and bool(p["cached"]) == verify: p.pop("cached")
    return p


def to_new(method: str, params: dict | None) -> tuple[str, dict] | None:
    """(method, params) under the current keys; None for a withdrawn variant. Idempotent on new keys ONLY for directories
    that were never migrated -- a bare `midian` is the old undefended method there (see SENTINEL)."""
    p = dict(params or {})
    if method in WITHDRAWN: return None
    if method == "midian_va": return "midian", _canon(p)
    if method == "midian_a": return "midian", _canon({**p, "verify": False})
    if method == "midian_v":
        for k in ("verify", "cached"): p.pop(k, None)
        return "midian", _canon({**p, "audit": False})
    if method == "midian":                                                   # the old plain method: defenses off
        return "midian", _canon({**p, "audit": False, "verify": p.get("verify", False), "cached": p.get("cached", False)})
    if method.startswith("fw_") and p.get("retrieval") in RETRIEVAL:
        return method, {**p, "retrieval": RETRIEVAL[p["retrieval"]]}
    return method, p


def assert_v2(method: str, params: dict | None, where: str = "") -> None:
    """Rows of a migrated directory must carry current keys only: no old key, and no bare `midian` row that is the old
    undefended method (those were rewritten with audit/verify false)."""
    if method in OLD:
        raise AssertionError(f"{where}: old MIDIAN key {method!r} in a migrated ({SENTINEL}) directory")
    if method.startswith("fw_") and (params or {}).get("retrieval") == "midian_va":
        raise AssertionError(f"{where}: old retrieval 'midian_va' in a migrated directory")


def _jkey(d):
    import json
    return json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)


def normalize(df, results_dir: str):
    """Rows read from `results_dir` under the current keys. A migrated directory (SENTINEL) is checked, never coerced;
    an unmigrated one is translated on read and loses its withdrawn-variant rows. params are stored as compact JSON."""
    import json, os
    if df is None or len(df) == 0 or "method" not in df: return df
    old = list(zip(df.method.astype(str), (df["params"].fillna("{}").astype(str) if "params" in df else ["{}"] * len(df))))
    load = lambda p: json.loads(p) if p.startswith("{") else {}
    if os.path.exists(os.path.join(results_dir, SENTINEL)):
        for m, p in set(old): assert_v2(m, load(p), results_dir)
        return df
    new = {mp: to_new(mp[0], load(mp[1])) for mp in set(old)}
    if all(v is not None and v[0] == mp[0] and _jkey(v[1]) == _jkey(load(mp[1])) for mp, v in new.items()): return df
    keep = [new[mp] is not None for mp in old]
    df = df[keep].copy(); kept = [mp for mp, k in zip(old, keep) if k]
    df["method"] = [new[mp][0] for mp in kept]
    if "params" in df: df["params"] = [_jkey(new[mp][1]) for mp in kept]
    return df
