"""The one place that knows MIDIAN's old method keys (renamed 2026-09-24).

MIDIAN is one method with two defenses as parameters, both on by default:
    old key                          new key                                   display label
    midian_va                        midian                                    MIDIAN
    midian_a                         midian{"verify": false}                   MIDIAN w/o verification
    midian_v, midian{verify,cached}  midian{"audit": false}                    MIDIAN w/o audits
    midian                           midian{"audit": false, "verify": false}   MIDIAN w/o defenses
Framework shortlists named after a MIDIAN cohort follow: retrieval "midian_va" -> "midian" (the full method's cohort),
retrieval "midian" (the verified, unaudited cohort) -> "midian_wo_audit". midian_sh / midian_sha are withdrawn (no key).

to_new(method, params) translates a stored or configured key; it is applied on every read, so rows written before the
rename and after it load identically. legacy(method, params) is its inverse on the new keys: the world's random stream is
seeded with the LEGACY key (rte.run), so a rerun of any cell reproduces the rows stored under the old keys bit for bit.
A results directory whose rows were rewritten carries SENTINEL; assert_v2 checks that it holds no old key."""
from __future__ import annotations

SENTINEL = ".method_keys_v2"
OLD = frozenset({"midian_va", "midian_a", "midian_v", "midian_sh", "midian_sha"})
WITHDRAWN = frozenset({"midian_sh", "midian_sha"})
FLAGS = ("audit", "verify", "cached")
RETRIEVAL = {"midian_va": "midian", "midian": "midian_wo_audit"}
RETRIEVAL_LEGACY = {v: k for k, v in RETRIEVAL.items()}


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


def legacy(method: str, params: dict | None) -> tuple[str, dict]:
    """The pre-rename key of a CURRENT key (the world's RNG salt). Inverse of to_new on its image."""
    p = dict(params or {})
    if method.startswith("fw_") and p.get("retrieval") in RETRIEVAL_LEGACY:
        return method, {**p, "retrieval": RETRIEVAL_LEGACY[p["retrieval"]]}
    if method != "midian": return method, p
    audit, verify = p.pop("audit", True), bool(p.pop("verify", True))
    cached = bool(p.pop("cached", verify))
    if audit is not False and verify:                                        # the full method
        return "midian_va", ({**p, "audit": audit} if audit not in (True, 0.05) else p) | ({} if cached else {"cached": False})
    if audit is not False:                                                   # w/o verification
        return "midian_a", ({**p, "audit": audit} if audit not in (True, 0.05) else p) | ({"cached": True} if cached else {})
    if verify:                                                               # w/o audits
        return "midian_v", p if cached else {**p, "cached": False}
    return "midian", {**p, "cached": True} if cached else p                  # w/o defenses


def assert_v2(method: str, params: dict | None, where: str = "") -> None:
    """Rows of a migrated directory must carry current keys only: no old key, and no bare `midian` row that is the old
    undefended method (those were rewritten with audit/verify false)."""
    if method in OLD:
        raise AssertionError(f"{where}: old MIDIAN key {method!r} in a migrated ({SENTINEL}) directory")
    if method.startswith("fw_") and (params or {}).get("retrieval") == "midian_va":
        raise AssertionError(f"{where}: old retrieval 'midian_va' in a migrated directory")
