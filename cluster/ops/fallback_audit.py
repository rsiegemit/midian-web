"""Audit every live framework row for silent declared-argmax fallback. A framework whose supervisor call fails gets
choice=None and the adapter routes by declared argmax instead -- the row then measures declared argmax, not the
framework. fallback_rate comes from method_stats; failures = the framework answered instead of delegating."""

import glob, json, os
import pandas as pd

R = os.environ["RTE_DATA"] + "/results"
rows = []
for g in sorted(os.listdir(R)):
    if not (g.startswith("fw_") or g.startswith("live_") or g.startswith("learned_")):
        continue
    fs = glob.glob(f"{R}/{g}/rows.d/*.json")
    recs = []
    for f in fs:
        try:
            recs.append(json.load(open(f)))
        except Exception:
            pass
    p = f"{R}/{g}/rows.csv"
    if os.path.exists(p):
        try:
            recs += pd.read_csv(p, low_memory=False).to_dict("records")
        except Exception:
            pass
    seen = set()
    for r in recs:
        rid = r.get("rid")
        if rid in seen:
            continue
        seen.add(rid)
        m = str(r.get("method", ""))
        if not (m.startswith("fw_") or m == "llm_supervisor"):
            continue
        ms = r.get("method_stats")
        if isinstance(ms, str):
            try:
                ms = json.loads(ms)
            except Exception:
                ms = None
        if not isinstance(ms, dict):
            continue
        picks = ms.get("picks", 0)
        fb = ms.get("fallbacks", 0)
        fl = ms.get("failures", 0)
        bad = ms.get("bad_name", 0)
        tot = picks + fb + fl + bad
        if tot == 0:
            continue
        rows.append(
            dict(
                grid=g,
                method=m,
                fallback=fb / tot,
                failure=fl / tot,
                nonpick=(fb + fl + bad) / tot,
                errors=ms.get("errors", ms.get("bridge_errors", 0)),
                success=r.get("success"),
            )
        )
df = pd.DataFrame(rows)
print(f"{len(df)} framework rows audited across {df.grid.nunique()} grids\n")
print("distribution of the NON-PICK rate (fallback + failure + bad name) per row:")
print(df.nonpick.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).round(3).to_string())
for th in (0.5, 0.9, 0.99):
    print(f"  rows with non-pick >= {th:.2f}: {(df.nonpick >= th).sum():5d}  ({100*(df.nonpick >= th).mean():.1f}%)")
print("\nrows with PURE fallback (supervisor never answered, fallback >= 0.9), by grid:")
bad = df[df.fallback >= 0.9]
print(bad.groupby("grid").size().sort_values(ascending=False).head(20).to_string() if len(bad) else "  none")
print(
    "\nper-framework mean non-pick rate (behaviour, not failure -- e.g. a framework that answers instead of "
    "delegating):"
)
print(df.groupby("method")[["fallback", "failure", "nonpick"]].mean().round(3).to_string())
df.to_csv(os.environ["RTE_DATA"] + "/results/fallback_audit.csv", index=False)
