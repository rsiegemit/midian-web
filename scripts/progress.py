"""How far along is a run? Memo growth (LLM generations) and per-grid row counts, from the stored artefacts only.

    python scripts/progress.py                      # memo stats + every grid with rows
    python scripts/progress.py live_n100k cohort_rte # just these grids, with per-method detail
    python scripts/progress.py --memo               # memo stats alone (cheap; use while a build is running)
    python scripts/progress.py --memo --rows        # add row counts (SLOW: full scan; avoid while jobs write)

Replaces the ad-hoc row counters and `du`-based progress guesses used during the 10^5 build. Two lessons are baked in:
`du` on the cache is useless as a progress signal (SQLite grows in page chunks, so short windows read as stalls), and a
live shard must be opened read-only with nolock -- its row count is then a slight UNDER-count while a writer is active,
so treat single samples as a floor and compare over long windows.
"""
import glob, json, os, sqlite3, sys

RTE_DATA = os.environ.get("RTE_DATA", "/scratch/rte")
RESULTS = f"{RTE_DATA}/results"


def memo_stats(count_rows: bool = False) -> dict:
    """Bytes always; row counts ONLY under --rows. Two reasons they are opt-in. `count(*)` on the compacted file is a
    full table scan of tens of millions of rows (minutes). And while a build is running, dozens of jobs are writing
    shards to the same NFS directory, so even opening them read-only contends badly -- measured unusable with 25 jobs
    active on 2026-09-08. Sizes come from one stat() per file and are always safe."""
    out = {"compact_rows": None, "compact_bytes": 0, "shard_rows": 0, "shard_bytes": 0, "shards": 0}
    for f in glob.glob(f"{RTE_DATA}/cache/*.sqlite"):
        compact = f.endswith("memo_compact.sqlite")
        out["compact_bytes" if compact else "shard_bytes"] += os.path.getsize(f)
        out["shards"] += not compact
        if not count_rows:
            continue
        try:
            con = sqlite3.connect(f"file:{f}?nolock=1&mode=ro", uri=True)
            k = con.execute("select count(*) from memo").fetchone()[0]
            con.close()
            if compact:
                out["compact_rows"] = (out["compact_rows"] or 0) + k
            else:
                out["shard_rows"] += k
        except Exception:
            pass
    return out


def grid_rows(grid: str) -> dict:
    """method -> row count for one grid, read from the per-row JSON files (authoritative even mid-run)."""
    counts: dict[str, int] = {}
    for f in glob.glob(f"{RESULTS}/{grid}/rows.d/*.json"):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        key = r["method"] + ("" if r.get("params") in (None, "{}") else r["params"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def main(argv: list[str]) -> None:
    grids = [a for a in argv if not a.startswith("--")]
    m = memo_stats(count_rows="--rows" in argv)
    rows = (f"; rows: {m['compact_rows']:,} compacted + {m['shard_rows']:,} new" if m["compact_rows"] is not None else "")
    print(f"memo: compacted {m['compact_bytes'] / 1e9:.1f} GB, {m['shards']} new shard(s) "
          f"{m['shard_bytes'] / 1e9:.2f} GB, total {(m['compact_bytes'] + m['shard_bytes']) / 1e9:.1f} GB{rows}"
          + ("" if m["compact_rows"] is not None else "   (--rows to count rows; slow, and slower still under write load)"))
    if "--memo" in argv:
        return
    if not grids:
        grids = sorted(d for d in os.listdir(RESULTS) if os.path.isdir(f"{RESULTS}/{d}/rows.d") and os.listdir(f"{RESULTS}/{d}/rows.d"))
    for g in grids:
        c = grid_rows(g)
        if not c:
            print(f"\n{g}: no rows"); continue
        print(f"\n{g}: {sum(c.values()):,} rows, {len(c)} arms")
        if len(grids) <= 4:                                    # per-method detail only when the caller asked for few grids
            for k, v in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"   {v:5d}  {k}")


if __name__ == "__main__":
    main(sys.argv[1:])
