"""The pre-registered expectations (v1: six targets, v2: V2-2..V2-11), each checked against the rows and reported as
PASS/HIT, MISS, WITHIN_FLOOR, REPORTED or NO DATA with its numbers. Never a fix. See docs/archive/preregistration for
the target texts."""
import json

import numpy as np
import pandas as pd

from .load import FLAT, FLAT_ON, FLOOR, PLAIN, REF, fmt
from .stats import delta, envelope, pair


def _t1(df, fits):
    """declared/framework lose >=0.25 from beta 0->0.5, probe-only move <=0.03"""
    b0, b5 = df[np.isclose(df.beta, 0)], df[np.isclose(df.beta, 0.5)]
    if b0.empty or b5.empty:
        return None, "needs beta=0 and beta=0.5"
    drop = (b0.groupby("label").success.mean() - b5.groupby("label").success.mean()).dropna()
    grp = df.drop_duplicates("label").set_index("label").group
    dec = {m: v for m, v in drop.items() if grp.get(m) in ("declared", "framework")}
    prb = {m: v for m, v in drop.items() if grp.get(m) in ("verified_central", "verified_decentral", "midian")}
    return (bool(dec) and bool(prb) and all(v >= 0.25 for v in dec.values())
            and all(abs(v) <= 0.03 for v in prb.values()),
            f"declared+framework: {fmt(dec)} | probe-based: {fmt(prb)}")
def _t2(df, fits):
    """MIDIAN w/o defenses == flat_probe_argmax within 0.02 at beta=0; comparisons ~ r log_r n vs flat ~ n"""
    parts, ok = [], []
    b0 = df[np.isclose(df.beta, 0)]
    for flat in (FLAT, FLAT_ON):
        d = pair(b0, REF, flat)
        if d.empty:
            continue
        x = float((d[REF] - d[flat]).mean())
        if flat == FLAT:  # the pre-registered comparison is against the frozen scan
            ok.append(abs(x) <= 0.02)
        parts.append(f"{REF} - {flat} = {x:+.4f} over {len(d)} (cell, seed) pairs")
    frozen = "midian[audit=False,online=False,verify=False]"      # MIDIAN w/o defenses, online=False
    d = pair(b0, frozen, FLAT)
    if not d.empty:
        parts.append(f"{frozen} - {FLAT} = {float((d[frozen] - d[FLAT]).mean()):+.4f} "
                     f"(the max-tree alone, both frozen after build)")
    f = fits[fits.metric == "comparisons_per_task"] if not fits.empty else pd.DataFrame()
    if "b" in f.columns and not f.empty:  # one fit per label: the main budget (ties -> largest b)
        f = f[f.b == f.b.mode().max()]
    f = f.set_index("label") if not f.empty else f
    if {REF, FLAT} <= set(f.index):
        m, s = f.loc[REF], f.loc[FLAT]
        ok.append(bool(m.exponent < 0.4 and s.exponent > 0.8))
        parts.append(f"comparisons: MIDIAN w/o defenses k={m.exponent:.2f} [{m.exp_lo:.2f},{m.exp_hi:.2f}]; "
                     f"{FLAT} k={s.exponent:.2f} [{s.exp_lo:.2f},{s.exp_hi:.2f}]")
    return (all(ok) if ok else None), "; ".join(parts) or "needs beta=0 rows and >=2 values of n"
def _t3(df, fits):
    """trimming helps only where beta*r exceeds the trim"""
    mid = PLAIN(df).copy()
    mid["delta"] = [json.loads(p).get("delta", np.nan) for p in mid.params]
    c = mid[(mid.collude == True) & mid.delta.notna()]      # noqa: E712
    if c.delta.nunique() < 2:
        return None, "needs MIDIAN w/o defenses at two deltas with collude=True (grid midian_internals)"
    tab = c.pivot_table(index="beta", columns="delta", values="success", aggfunc="mean")
    sep = tab[tab.columns.max()] - tab[tab.columns.min()]
    below, above = [b for b in sep.index if b <= 0.3], [b for b in sep.index if b > 0.3]
    return (bool(above) and all(abs(sep[b]) <= 0.02 for b in below) and any(sep[b] > 0.02 for b in above),
            f"success(delta={tab.columns.max():.2g}) - success(delta={tab.columns.min():.2g}) by beta: "
            + ", ".join(f"{b}: {sep[b]:+.3f}" for b in sep.index))
def _t4(df, fits):
    """verify_on_claim: <=0.03 from oracle at beta<=0.1, loses >=0.10 by beta=0.5"""
    v = df[df.method == "verify_on_claim"]
    if v.empty:
        return None, "verify_on_claim rows missing"
    at_lo, at_hi = v.beta <= 0.1, np.isclose(v.beta, 0.5)
    gap = float((v.oracle_success - v.success)[at_lo].mean()) if at_lo.any() else np.nan
    lo = float(v[at_lo].success.mean()) if at_lo.any() else np.nan
    hi = float(v[at_hi].success.mean()) if at_hi.any() else np.nan
    return (None if not np.isfinite(gap - hi) else bool(gap <= 0.03 and lo - hi >= 0.10),
            f"oracle gap at beta<=0.1 = {gap:.3f}; success {lo:.3f} -> {hi:.3f} (loss {lo-hi:+.3f})")
def _t5(df, fits):
    """sequential_halving ~= flat_probe_argmax; bandits learn past MIDIAN w/o defenses at b=1"""
    parts, ok, miss = [], [], []
    d = pair(df, "sequential_halving", FLAT)
    if d.empty:
        miss.append(f"sequential_halving/{FLAT} rows")
    else:
        x = float((d.sequential_halving - d[FLAT]).mean())
        ok.append(abs(x) <= 0.03)
        parts.append(f"sequential_halving - {FLAT} = {x:+.3f}")
    for bandit in ("ucb_per_family", "thompson_per_family"):
        d = pair(df[df.b == 1], bandit, REF, "success_late")
        if d.empty:
            continue
        x = float((d[bandit] - d[REF]).mean())
        ok.append(x >= 0)
        parts.append(f"b=1 success_late {bandit} - MIDIAN w/o defenses = {x:+.3f}")
    if not any("b=1" in p for p in parts):
        miss.append("a b=1 block with bandit success_late")
    return ((all(ok) if ok else None),
            "; ".join(parts) + (f"  [not evaluated: {', '.join(miss)}]" if miss else ""))
def _t6(df, fits):
    """argmax-vs-floor gap largest under heavy_tail, smallest under iid_uniform"""
    finders = df[df.group.isin(["midian", "verified_central", "declared"])]
    floors = df[df.method.isin(["random", "route_to_k_majority"])]
    if finders.empty or floors.empty or df.dist.nunique() < 2:
        return None, "needs >=2 distributions"
    g = (finders.groupby("dist").success.mean() - floors.groupby("dist").success.mean()).dropna()
    return (bool(g.idxmax() == "heavy_tail" and g.idxmin() == "iid_uniform")
            if {"heavy_tail", "iid_uniform"} <= set(g.index) else None,
            "gap by dist: " + ", ".join(f"{k} {v:.3f}" for k, v in g.sort_values(ascending=False).items()))
def targets(df, fits):
    """The six pre-registered v1 expectations (texts in docs/archive/preregistration). PASS / MISS / NO DATA with the
    numbers. Never a fix."""
    out = []
    for i, fn in enumerate((_t1, _t2, _t3, _t4, _t5, _t6), 1):
        try:
            ok, detail = fn(df, fits)
        except (KeyError, ValueError, IndexError) as e:
            ok, detail = None, f"no data ({type(e).__name__}: {e})"
        out.append({"target": i, "verdict": "NO DATA" if ok is None else ("PASS" if ok else "MISS"),
                    "name": fn.__doc__, "detail": detail})
    return out

# ------------------------------------------------------------------ v2 targets


def _v2(df, fits):
    ("V2-2 MIDIAN w/o verification: beta=0.5-collude loss vs beta=0 <= 0.02 on specialist; unchanged within 0.01 "
     "at beta<=0.25; build probes <= 1.05x")
    a = df[(df.label == "midian_wo_verify") & (df.dist == "specialist") & (df.collude == True)]   # noqa: E712
    if a.empty:
        return None, "needs midian_wo_verify rows on specialist"
    loss = float(a[np.isclose(a.beta, 0)].success.mean() - a[np.isclose(a.beta, .5)].success.mean())
    x, n = delta(df[df.beta <= 0.25], "midian_wo_verify", REF)
    # same cells only
    d = pair(df, "midian_wo_verify", REF, "build_probes")
    ratio = float((d["midian_wo_verify"] / d[REF]).mean()) if not d.empty else float("nan")
    if not np.isfinite(loss) or not n:
        return None, ("needs midian_wo_verify at beta=0 and beta=0.5 (collude) on specialist and at beta<=0.25 "
                      "paired with midian_wo_defenses")
    return (bool(loss <= 0.02 and abs(x) <= 0.01 and ratio <= 1.05 + 1e-9),
            f"specialist loss beta 0->0.5 = {loss:+.3f}; vs MIDIAN w/o defenses at beta<=0.25 {x:+.3f} ({n} pairs); "
            f"build probes {ratio:.3f}x", x)
def _v4(df, fits):
    """V2-4 stratified cohorts vs random (no directional expectation; reported as measured)"""
    d = pair(df, "midian_stratified", REF)
    if d.empty:
        return None, "needs grid stratify"
    by = (d.midian_stratified - d[REF]).groupby(level=["dist", "beta"]).mean()
    return "REPORTED", "stratified - random by (shape, beta): " + ", ".join(f"{k} {v:+.3f}" for k, v in by.items())
def _v5(df, fits):
    ("V2-5 LinUCB-honest between flat_online and warm_start_bandit at beta=0; flat in beta (|beta=0.5 - beta=0| <= "
     "0.03)")
    b0 = df[np.isclose(df.beta, 0)]
    lo, hi = delta(b0, "linucb_honest", FLAT_ON)[0], delta(b0, "warm_start_bandit", "linucb_honest")[0]
    l = df[df.label == "linucb_honest"]
    if l.empty or not np.isfinite(lo):
        return None, "needs linucb_honest rows at beta=0 with flat_probe_argmax_online"
    flat = float(l[np.isclose(l.beta, .5)].success.mean() - l[np.isclose(l.beta, 0)].success.mean())
    return (bool(lo >= 0 and hi >= 0 and abs(flat) <= 0.03),
            f"linucb - flat_online {lo:+.3f}; warm_start - linucb {hi:+.3f}; beta 0.5 - 0: {flat:+.3f}")
def _v6(df, fits):
    ("V2-6 churn: MIDIAN w/o defenses within 0.03 of no-churn at 10% with repair <= 3% of build; halving-stale "
     "loses >= 0.05 at 30%; halving-rebuild matches at >= 10x MIDIAN's repair")
    c, base = df[df.churn_frac > 0], df[(df.churn_frac == 0) & (df.declared_source == "self_described")]
    if c.empty or base.empty:
        return None, "needs churn_n1000 rows and a no-churn baseline (live_f1_n1000 / variants_f1, self-described)"
    keys = ["dist", "beta", "seed"]
    def drop(label, frac):
        # no-churn success - churned success, paired on (shape, beta, seed)
        a = c[(c.label == label) & np.isclose(c.churn_frac, frac)].groupby(keys).success.mean()
        b = base[base.label == label].groupby(keys).success.mean()
        j = pd.concat([a, b], axis=1, join="inner")
        return float((j.iloc[:, 1] - j.iloc[:, 0]).mean()) if len(j) else np.nan
    rep = lambda label: float((c[c.label == label].repair_probes_per_event / c[c.label == label].build_probes).mean())
    m10 = drop(REF, .1)
    stale30, reb30 = drop("sequential_halving_peer_stale", .3), drop("sequential_halving_peer_rebuild", .3)
    rm, rh = rep(REF), rep("sequential_halving_peer_rebuild")
    ok = m10 <= 0.03 and rm <= 0.03 and stale30 >= 0.05 and reb30 <= 0.03 and rh >= 10 * rm
    return (bool(ok) if np.isfinite([m10, rm, stale30, reb30, rh]).all() else None,
            f"MIDIAN w/o defenses loss at 10% {m10:+.3f}, repair {100*rm:.2f}% of build/event; "
            f"halving-stale loss at 30% {stale30:+.3f}; halving-rebuild loss at 30% {reb30:+.3f}, "
            f"repair {rh/rm if rm else np.nan:.1f}x MIDIAN w/o defenses'")
def _v7(df, fits):
    ("V2-7 n=10k, b=3, self-described: MIDIAN w/o defenses >= flat_online - 0.02; frameworks below flat_online on "
     "specialist by >= 0.10")
    d = df[(df.n == 10000) & (df.b == 3)]
    x, n = delta(d, REF, FLAT_ON)
    fw = (d[d.group == "framework"].success.mean() - d[d.label == FLAT_ON].success.mean()
          if (d.group == "framework").any() else np.nan)
    if not n:
        return None, "needs live_n10k_v2 rows"
    return (bool(x >= -0.02 and fw <= -0.10),
            f"MIDIAN w/o defenses - flat_online {x:+.3f} ({n} pairs); frameworks - flat_online {fw:+.3f}", x)
def _v8(df, fits):
    ("V2-8 replication (seeds 11-20): MIDIAN w/o audits - MIDIAN w/o defenses = +0.02 +- 0.02 at beta<=0.25; "
     "beta=0.5 collude exposure reported")
    r = df[df.seed >= 11]
    x, n = delta(r[r.beta <= 0.25], "midian_wo_audit", REF)
    y, _ = delta(r[np.isclose(r.beta, .5)], "midian_wo_audit", REF)
    if not n:
        return None, "needs midian_v_replication rows"
    return (bool(0.0 <= x <= 0.04),
            f"midian_wo_audit - midian_wo_defenses at beta<=0.25 {x:+.3f} ({n} pairs); at beta=0.5 {y:+.3f} "
            f"(as measured)", x - 0.02)
def _v9(df, fits):
    ("V2-9 b=10: bimodal framework gap within +-0.02 of MIDIAN w/o defenses; heavy_tail MIDIAN w/o defenses >= "
     "frameworks + 0.03")
    d = df[df.b == 10]
    if d.empty or not (d.group == "framework").any():
        return None, "needs budget_b10_shapes rows"
    g = {s: float(x[x.group == "framework"].success.mean() - x[x.label == REF].success.mean())
         for s, x in d.groupby("dist")}
    ok = abs(g.get("bimodal", np.nan)) <= 0.02 and g.get("heavy_tail", np.nan) <= -0.03
    return ((bool(ok) if all(k in g for k in ("bimodal", "heavy_tail")) else None),
            "frameworks - MIDIAN w/o defenses at b=10: " + fmt(g))
def _v10(df, fits):
    ("V2-10 internals at beta=0.5 collude: trimming (delta=1/3 vs 0) hurts MIDIAN w/o defenses by >= 0.02, not "
     "MIDIAN w/o verification (|d| <= 0.02)")
    d = df[np.isclose(df.beta, .5) & (df.collude == True) & (df.method == "midian")].copy()   # noqa: E712
    par = d.params.map(json.loads)
    d["r"], d["delta"] = [p.get("r") for p in par], [p.get("delta") for p in par]
    d["arm"] = [REF if p.get("audit") is False and p.get("verify") is False
                else "midian_wo_verify" if p.get("verify") is False and "audit" not in p
                else None for p in par]                     # the two arms by their defense flags (r, delta vary)
    d = d[(d.r == 10) & d.delta.notna() & d.arm.notna()]
    if d.arm.nunique() < 2 or d.delta.nunique() < 2:
        return None, "needs internals_v2 rows (r=10, both deltas, midian_wo_defenses and midian_wo_verify)"
    t = d.pivot_table(index="arm", columns="delta", values="success")
    sep = t[t.columns.max()] - t[t.columns.min()]
    return (bool(sep[REF] <= -0.02 and abs(sep["midian_wo_verify"]) <= 0.02),
            f"delta=1/3 - delta=0 at r=10: MIDIAN w/o defenses {sep[REF]:+.3f}, "
            f"MIDIAN w/o verification {sep['midian_wo_verify']:+.3f}")
def _v11(df, fits):
    ("V2-11 MIDIAN >= max(MIDIAN w/o audits, MIDIAN w/o verification) - 0.01 at every beta; within 0.02 of MIDIAN "
     "w/o verification at beta=0.5 collude low-skill-first; build probes <= 1.05x w/o audits")
    s = df[df.label.isin(["midian", "midian_wo_audit", "midian_wo_verify"])]
    piv = s.pivot_table(index="beta", columns="label", values="success")
    if piv.shape[1] < 3:
        return None, "needs midian_wo_audit, midian_wo_verify and midian rows"
    gap = piv["midian"] - piv[["midian_wo_audit", "midian_wo_verify"]].max(axis=1)
    hi = s[np.isclose(s.beta, .5) & (s.collude == True) & (s.liar_select == "low_skill_first")]   # noqa: E712
    x, n = delta(hi, "midian", "midian_wo_verify")
    d = pair(df, "midian", "midian_wo_audit", "build_probes")
    ratio = float((d["midian"] / d["midian_wo_audit"]).mean()) if not d.empty else float("nan")
    ok = bool((gap >= -0.01).all() and x >= -0.02 and ratio <= 1.05 + 1e-9)
    return ok, ("MIDIAN - max(w/o audits, w/o verification) by beta: "
                + ", ".join(f"{b}: {v:+.3f}" for b, v in gap.items())
                + f"; MIDIAN - w/o verification at beta=0.5 low-skill {x:+.3f} ({n} pairs); "
                  f"build probes {ratio:.3f}x w/o audits"), float(gap.min())
def targets_v2(df, fits):
    """The pre-registered v2 expectations (texts in docs/archive/preregistration): HIT / MISS / WITHIN_FLOOR /
    REPORTED / NO DATA, with the numbers. V2-1 and V2-3 tested the withdrawn successive-halving variants (SH, SH+A);
    the remaining targets keep their numbers."""
    out, env = [], envelope(df)
    for i, fn in ((2, _v2), (4, _v4), (5, _v5), (6, _v6), (7, _v7), (8, _v8), (9, _v9), (10, _v10), (11, _v11)):
        try:  # key = the decisive paired delta, when the target is one
            ok, detail, *key = fn(df, fits)
        except (KeyError, ValueError, IndexError, TypeError) as e:
            ok, detail, key = None, f"no data ({type(e).__name__}: {e})", []
        v = ("NO DATA" if ok is None else ok if isinstance(ok, str) else "HIT" if ok else
             FLOOR if key and np.isfinite(env) and abs(key[0]) <= env else "MISS")
        tail = f"  [MIDIAN w/o defenses seed envelope {env:.3f}]" if np.isfinite(env) else ""
        out.append({"target": f"V2-{i}", "verdict": v, "name": fn.__doc__, "detail": detail + tail})
    return out
