"""Quarantine framework rows contaminated by infrastructure fallback (OPS_RULES D1/D4), and list the units to rerun.
    python scripts/ops/quarantine_fallback_rows.py [--apply]      # dry run by default
A row is contaminated when its supervisor never really answered and the adapter routed by declared argmax instead:
  * CrewAI / ADK (envs with deleted .so files): fallback > 0.05 -- their working units sit at ~0, the failure is per unit
  * every other framework: fallback >= 0.9 (the bimodal infrastructure signature)
Rows are MOVED to results/<grid>/quarantine/ (never deleted) and dropped from rows.csv; the unit list goes to
results/quarantine_units.tsv for scripts/ops/rerun_units.sh. rte.run recomputes a unit whose rids are absent."""
import glob, json, os, sys
import pandas as pd

R = os.environ.get("RTE_DATA", "/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte") + "/results"
BROKEN_ENV = {"fw_crewai", "fw_google_adk"}
APPLY = "--apply" in sys.argv


def contaminated(method, ms):
    if isinstance(ms, str):
        try: ms = json.loads(ms)
        except Exception: return False
    if not isinstance(ms, dict): return False
    tot = sum(ms.get(k, 0) for k in ("picks", "fallbacks", "failures", "bad_name"))
    if not tot: return False
    fb = ms.get("fallbacks", 0) / tot
    return fb > 0.05 if method in BROKEN_ENV else fb >= 0.9


units, moved = set(), 0
# Every grid that can hold framework / supervisor rows: those run only on the llm and routereval backends. Scanning
# bernoulli / replay (1 GB rows.csv, no frameworks) would only cost memory and time.
FW_PREFIXES = ("fw_", "live_", "learned_", "routereval", "llmrouterbench", "cohort_", "variants_", "churn_")
for gdir in sorted(glob.glob(f"{R}/*")):
    if not os.path.basename(gdir).startswith(FW_PREFIXES): continue
    if not (os.path.isdir(f"{gdir}/rows.d") or os.path.exists(f"{gdir}/rows.csv")): continue
    g = os.path.basename(gdir); q = f"{gdir}/quarantine"; bad = set()
    for f in glob.glob(f"{gdir}/rows.d/*.json"):
        try: r = json.load(open(f))
        except Exception: continue
        if contaminated(str(r.get("method")), r.get("method_stats")):
            bad.add(r.get("rid") or os.path.basename(f)[:-5])      # rows.d files are named <rid>.json
            units.add((g, r["method"], r["dist"], r["beta"], r["liar_select"], int(r["seed"])))
            if APPLY: os.makedirs(q, exist_ok=True); os.replace(f, f"{q}/{os.path.basename(f)}")
    p = f"{gdir}/rows.csv"
    if os.path.exists(p):
        d = pd.read_csv(p, low_memory=False)
        m = d.apply(lambda r: contaminated(str(r.get("method")), r.get("method_stats")), axis=1) if "method_stats" in d else pd.Series(False, index=d.index)
        for _, r in d[m].iterrows():
            bad.add(r["rid"]); units.add((g, r["method"], r["dist"], r["beta"], r["liar_select"], int(r["seed"])))
        if APPLY and m.any():
            os.makedirs(q, exist_ok=True)
            d[m].to_csv(f"{q}/rows_quarantined.csv", mode="a", index=False, header=not os.path.exists(f"{q}/rows_quarantined.csv"))
            tmp = f"{p}.{os.getpid()}.tmp"; d[~m].to_csv(tmp, index=False); os.replace(tmp, p)
    if bad: print(f"  {g:40s} {len(bad):5d} rows"); moved += len(bad)

out = pd.DataFrame(sorted(units), columns=["grid", "method", "dist", "beta", "liar_select", "seed"])
if APPLY:                                           # append: earlier passes' units may still be rerunning
    U = f"{R}/quarantine_units.tsv"
    prev = pd.read_csv(U, sep="\t") if os.path.exists(U) else out.iloc[0:0]
    pd.concat([prev, out]).drop_duplicates().to_csv(U, sep="\t", index=False)
print(f"\n{'QUARANTINED' if APPLY else 'WOULD QUARANTINE'} {moved} rows -> {len(out)} units to rerun"
      + ("" if APPLY else "   (dry run; pass --apply)"))
print(out.groupby("method").size().to_string())
