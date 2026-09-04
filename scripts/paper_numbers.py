"""Every number the paper quotes, recomputed from the result grids -> paper/NUMBERS.json ({"value", "grid", "units", "ci"} per entry).
    PAPER_CACHE=<dir of <grid>.pkl>  python scripts/paper_numbers.py        (without the cache every grid is read through rte.analyze.load)
Means are over (shape, β, liar-selection, seed) units; CIs are the 95% bootstrap over seeds (paired over seeds for differences)."""
import json, os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rte.analyze import RTE_DATA, load as _load, FLAT_ON, FLAT
from extra_figs import HAL, HALP, MAG14, ci as _ci, stat

CACHE = os.environ.get("PAPER_CACHE"); R = f"{RTE_DATA}/results"; OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "NUMBERS.json")
FWS = ["fw_autogen", "fw_camel_workforce", "fw_crewai", "fw_google_adk", "fw_langgraph", "fw_llamaindex", "fw_maf", "fw_magentic_one", "fw_openai_agents", "fw_smolagents"]
KEY = ["dist", "beta", "liar_select", "seed"]; N = {}; _mem = {}


def rows(*grids):
    for g in grids:
        if g not in _mem:
            p = f"{CACHE}/{g}.pkl" if CACHE else None; _mem[g] = pd.read_pickle(p) if p and os.path.exists(p) else _load([g])
    return pd.concat([_mem[g] for g in grids], ignore_index=True)


def W(df, labels, **f):
    q = df[df.label.isin(labels)]
    for k, v in f.items():
        if v is not None: q = q[np.isclose(q[k], v) if k == "beta" else q[k] == v]
    return q.pivot_table(index=KEY, columns="label", values="success")


def put(key, s, grid, note=None):
    s = s.dropna(); lo, hi = _ci(s); N[key] = dict(value=round(float(s.mean()), 4), grid=grid, units=int(len(s)), ci=[round(float(lo), 4), round(float(hi), 4)]) | ({"note": note} if note else {})
    return N[key]["value"]


def put_v(key, value, grid, note=None, units=None):
    N[key] = dict(value=value, grid=grid, units=units, ci=None) | ({"note": note} if note else {})


def paired(key, w, a, b, grid, note=None):
    d = (w[a] - w[b]).dropna(); return put(key, d, grid, note)


# ------------------------------------------------------------------------------------------------ (a) headline
fw = rows("fw_live_n1000"); wf = W(fw, FWS + ["midian", "midian_va", "midian_v", "oracle", "random", MAG14])
fmeans = wf[FWS].mean(); put_v("a.frameworks_all_beta_range_n1000", [round(fmeans.min(), 3), round(fmeans.max(), 3)], "fw_live_n1000", "min/max over the ten frameworks of the mean over 120 units", 120)
f1 = rows("variants_f1", "learned_f1", "live_f1_n1000"); w0 = W(f1, ["midian", "midian_va", "midian_v", "midian_a", "oracle", HALP, HAL, FLAT_ON, "knn_router", "mlp_router", "declared_argmax", "random"], beta=0.0)
for s in ("midian", "midian_va", "midian_v", "oracle"):
    put(f"a.{s}_beta0_n1000", w0[s], "variants_f1 (β = 0, both liar-selection cells, 3 shapes, 10 seeds)")
    put(f"a.{s}_beta0_n1000_random_cells_only", W(f1, [s], beta=0.0, liar_select="random")[s], "variants_f1 (β = 0, liar_select = random cells only)")
put("a.midian_all_beta_n1000_table1", wf["midian"], "fw_live_n1000 (all β, random liars)"); put("a.midian_va_all_beta_n1000_table1", wf["midian_va"], "fw_live_n1000 (all β)")
for f in FWS: paired(f"a.paired_{f}_minus_midian", wf, f, "midian", "fw_live_n1000, 120 units")
put_v("a.paired_framework_minus_midian_range", [round(min(N[f"a.paired_{f}_minus_midian"]["value"] for f in FWS), 3), round(max(N[f"a.paired_{f}_minus_midian"]["value"] for f in FWS), 3)], "fw_live_n1000")
sp = wf.xs("specialist", level="dist"); put("a.specialist_frameworks_mean", sp[FWS].stack(), "fw_live_n1000 specialist cells (4 β × 10 seeds × 10 frameworks)")
for s in ("midian", "midian_v", "midian_va", "oracle"): put(f"a.specialist_{s}", sp[s], "fw_live_n1000 specialist cells")
put_v("a.specialist_pairs_framework_beats_midian", f"{int((sp[FWS].gt(sp['midian'], axis=0)).sum().sum())} / {sp[FWS].size}", "fw_live_n1000 specialist cells")
n10 = rows("live_n10k_v2"); w10 = W(n10, FWS + ["midian", "midian_v", "midian_va", "random", "oracle"], beta=0.0)
put_v("a.n10k_frameworks_range", [round(w10[FWS].mean().min(), 3), round(w10[FWS].mean().max(), 3)], "live_n10k_v2 (β = 0, specialist, 3 seeds, Q = 300)", units=3)
put("a.n10k_random", w10["random"], "live_n10k_v2 β = 0"); put("a.n10k_midian", w10["midian"], "live_n10k_v2 β = 0"); put("a.n10k_midian_v", w10["midian_v"], "live_n10k_v2 β = 0"); put("a.n10k_midian_va", w10["midian_va"], "live_n10k_v2 β = 0")
w10b = W(n10, FWS + ["midian", "midian_v", "midian_va", "random"], beta=0.25); put_v("a.n10k_frameworks_range_beta025", [round(w10b[FWS].mean().min(), 3), round(w10b[FWS].mean().max(), 3)], "live_n10k_v2 (β = 0.25)")
# VA − KNN / MLP
w100 = W(rows("learned_n100"), ["midian_va", "knn_router", "mlp_router", HALP, "oracle"], beta=0.0); paired("a.va_minus_knn_n100_beta0", w100, "midian_va", "knn_router", "learned_n100 β = 0 (3 shapes × 2 liar-selection cells × 10 seeds)")
paired("a.va_minus_knn_n1000_beta0", w0, "midian_va", "knn_router", "variants_f1 + learned_f1, β = 0"); w10k = W(rows("learned_n10k"), ["midian_va", "knn_router", HALP, "oracle", "midian", FLAT_ON], beta=0.0)
paired("a.va_minus_knn_n10k_beta0", w10k, "midian_va", "knn_router", "learned_n10k β = 0 (specialist, 2 cells × 3 seeds, Q = 300)")
for b in (0.0, 0.1, 0.25, 0.5):
    wb = W(f1, ["midian_va", "mlp_router"], beta=b); paired(f"a.va_minus_mlp_n1000_beta{b}", wb, "midian_va", "mlp_router", f"variants_f1 + learned_f1, β = {b}")
    wb2 = W(f1, ["midian_va", "mlp_router"], beta=b, liar_select="low_skill_first"); paired(f"a.va_minus_mlp_n1000_beta{b}_lowskill", wb2, "midian_va", "mlp_router", f"variants_f1 + learned_f1, β = {b}, low-skill-first")
put_v("a.mlp_at_n10k", "absent (mlp_router excluded at n = 10,000: agent one-hot over 480k probes)", "learned_n10k")
# halving − VA
paired("a.halving_minus_va_n100_beta0", w100, HALP, "midian_va", "learned_n100 β = 0"); paired("a.halving_minus_va_n1000_beta0", w0, HALP, "midian_va", "variants_f1 β = 0"); paired("a.halving_minus_va_n10k_beta0", w10k, HALP, "midian_va", "learned_n10k β = 0")
re5 = rows("routereval_mmlu5k"); w5 = W(re5, [HALP, HAL, "midian_va", "midian_v", "midian", "declared_argmax", "knn_router", FLAT_ON, "oracle", "random", "warm_start_bandit", "linucb_honest"], beta=0.0)
paired("a.halving_peer_minus_va_5000real_beta0", w5, HALP, "midian_va", "routereval_mmlu5k β = 0 (2 liar-selection cells × 3 seeds)"); paired("a.halving_trusted_minus_va_5000real_beta0", w5, HAL, "midian_va", "routereval_mmlu5k β = 0")
# external benchmarks on their terms: parse the summaries
def grab(path, pattern, key, note):
    txt = open(path).read(); m = re.search(pattern, txt); put_v(key, m.group(1) if m else f"NOT FOUND ({pattern})", path.replace(RTE_DATA + "/", ""), note)
S = f"{R}/routerbench_terms/summary.md"
if os.path.exists(S): put_v("a.routerbench_summary_head", [l for l in open(S).read().splitlines() if "|" in l][:14], "results/routerbench_terms/summary.md", "AIQ table verbatim; the paper's 0.707 / 0.713 / 36,850 / 277,915 are read from here")
S = f"{R}/routereval_terms/summary.md"
if os.path.exists(S): put_v("a.routereval_terms_tuned_rows", [l for l in open(S).read().splitlines() if "tuned" in l.lower() or "probe" in l.lower() or "single" in l.lower() or "LinearR" in l][:20], "results/routereval_terms/summary.md", "tuned rows verbatim (0.630 / 0.629 / 0.667; labels 0.26M / 3.35M)")
S = f"{R}/llmrouterbench_terms/summary.md"
if os.path.exists(S): put_v("a.llmrouterbench_terms_rows", [l for l in open(S).read().splitlines() if "|" in l][:24], "results/llmrouterbench_terms/summary.md", "their-protocol table verbatim (probe 0.704 / Avengers 0.709 / EmbedLLM 0.702; 9,000 / 161k labels)")

# ------------------------------------------------------------------------------------------------ (b) cost
import energy
t = pd.read_pickle(f"{CACHE}/energy_table.pkl") if CACHE and os.path.exists(f"{CACHE}/energy_table.pkl") else energy.table()
put_v("b.autogen_J_per_task", round(float(t.loc["fw_autogen", "per_task_J"]), 2), "scripts/energy.py cost model × fw_live_n1000 ledger"); put_v("b.autogen_latency_s", round(float(t.loc["fw_autogen", "latency_s"]), 3), "fw_live_n1000 median supervisor wall-clock")
put_v("b.midian_build_probes", int(round(t.loc["midian", "build_msgs"] * 0 + rows("variants_f1").query("label == 'midian'").build_probes.mean())), "variants_f1 ledger"); put_v("b.midian_build_gpu_s", round(float(t.loc["midian", "build_gpu_s"]), 1), "energy.py (specialist)")
put_v("b.midian_v_comparisons_per_task", round(float(t.loc["midian_v", "cmp_per_task"]), 2), "ledger"); put_v("b.midian_v_messages_per_task", round(float(t.loc["midian_v", "msgs_per_task"]), 2), "ledger"); put_v("b.midian_v_latency_s", round(float(t.loc["midian_v", "latency_s"]), 4), "energy.py latency model")
put_v("b.midian_va_comparisons_per_task", round(float(t.loc["midian_va", "cmp_per_task"]), 2), "ledger"); put_v("b.midian_va_messages_per_task", round(float(t.loc["midian_va", "msgs_per_task"]), 2), "ledger")
for b in ("fw_magentic_one", "fw_crewai", "fw_autogen"):
    put_v(f"b.crossing_joules_midian_vs_{b}", round(float(energy.crossing(t, "midian", b, "build_J", "per_task_J"))), "energy.py joules (H11 / F1_energy)"); put_v(f"b.crossing_gpu_s_midian_vs_{b}", round(float(energy.crossing(t, "midian", b))), "energy.py GPU-s (RESULTS_energy.md)")
for g in ("bernoulli_scale", "combined_scale"):
    p = f"{R}/{g}/summary.md"; p = f"{p}/summary.md" if os.path.isdir(p) else p
    if os.path.isfile(p):
        txt = open(p).read(); i = txt.find("Cost exponents"); put_v(f"b.cost_exponents_{g}", txt[i:i + 1500].splitlines()[:25] if i >= 0 else "section not found", f"results/{g}/summary.md", "n^k fits (paper: MIDIAN n^0.14, flat n^1.00)")

# ------------------------------------------------------------------------------------------------ (c) robustness
re1 = rows("routereval_mmlu")
for n in (1000,):
    for s in ("declared_argmax", HALP, "midian", "midian_a", "midian_va", "mlp_router", FLAT_ON, "knn_router"):
        put(f"c.real{n}_{s}_beta0", W(re1, [s], beta=0.0, n=n)[s], f"routereval_mmlu n = {n} β = 0 (3 pool types × 2 cells × 5 seeds)"); put(f"c.real{n}_{s}_beta05_lowskill", W(re1, [s], beta=0.5, liar_select="low_skill_first", n=n)[s], f"routereval_mmlu n = {n} β = 0.5 low-skill-first")
for s in ("declared_argmax", HALP, "midian_va", "midian_v", "midian", "knn_router", FLAT_ON):
    put(f"c.real5000_{s}_beta0", W(re5, [s], beta=0.0)[s], "routereval_mmlu5k β = 0"); put(f"c.real5000_{s}_beta05_lowskill", W(re5, [s], beta=0.5, liar_select="low_skill_first")[s], "routereval_mmlu5k β = 0.5 low-skill-first")
for s in (HALP, "midian", "midian_a", "midian_va", "midian_v", "declared_argmax"):
    put(f"c.rte_{s}_beta0", W(f1, [s], beta=0.0)[s], "variants_f1 (+ live_f1_n1000 for declared) β = 0"); put(f"c.rte_{s}_beta05_lowskill", W(f1, [s], beta=0.5, liar_select="low_skill_first")[s], "variants_f1 β = 0.5 low-skill-first")
sc = rows("scale_100k"); put_v("c.va_comparisons_per_task_100k", round(float(sc[(sc.label == "midian_va") & (sc.n == 100000)].comparisons_per_task.mean()), 1), "scale_100k ledger, n = 100,000")
put_v("c.flat_comparisons_per_task_100k", round(float(sc[(sc.label == FLAT_ON) & (sc.n == 100000)].comparisons_per_task.mean()), 1), "scale_100k ledger, n = 100,000")
# VA within 0.05 of its honest score per population; VA +0.15 over every framework under the cartel
low = rows("fw_live_n1000_lowskill"); wl = W(low, FWS + ["midian_va", "midian", "midian_v", "midian_a", FLAT, "declared_argmax", "random", "oracle"]); wv0 = W(f1, ["midian_va"], beta=0.0)
for d in ("specialist", "heavy_tail", "bimodal"):
    put_v(f"c.va_honest_vs_cartel_{d}", [round(float(wv0["midian_va"].xs(d, level="dist").mean()), 3), round(float(wl["midian_va"].xs(d, level="dist").mean()), 3)], "variants_f1 β = 0 vs fw_live_n1000_lowskill β = 0.5 low-skill", "VA honest, VA under the cartel")
for f in FWS: paired(f"c.cartel_va_minus_{f}", wl, "midian_va", f, "fw_live_n1000_lowskill, 30 units")
put_v("c.cartel_va_minus_framework_min", round(min(N[f"c.cartel_va_minus_{f}"]["value"] for f in FWS), 3), "fw_live_n1000_lowskill")

# ------------------------------------------------------------------------------------------------ (d) 14B and shortlists
m = fw[fw.label.isin(["fw_magentic_one", MAG14])].copy(); m["strict"] = stat(m, "success_strict"); m["fallback"] = stat(m, "fallback_rate")
for arm, tag in (("fw_magentic_one", "7B"), (MAG14, "14B")):
    q = m[m.label == arm].set_index(["dist", "beta", "seed"]); put(f"d.magentic_{tag}_fallback_rate", q["fallback"], "fw_live_n1000"); put(f"d.magentic_{tag}_strict", q["strict"], "fw_live_n1000"); put(f"d.magentic_{tag}_lenient", q["success"], "fw_live_n1000")
a, b = m[m.label == "fw_magentic_one"].set_index(["dist", "beta", "seed"]), m[m.label == MAG14].set_index(["dist", "beta", "seed"])
for col in ("success", "strict", "fallback"):
    diff = (b[col] - a[col]).dropna(); cells = diff.groupby(level=["dist", "beta"]).mean(); put(f"d.magentic_14B_minus_7B_{col}", diff, "fw_live_n1000, 120 paired units", f"cells 14B lower / equal / higher: {int((cells < -1e-9).sum())} / {int((cells.abs() <= 1e-9).sum())} / {int((cells > 1e-9).sum())} of {len(cells)}")
ver = rows("fw_live_n1000", "fw_live_n1000_verified", "fw_live_n1000_verified_va"); cols = [f + s for f in FWS for s in ("", "[r=10,retrieval=midian]", "[r=10,retrieval=midian_va]")] + ["midian_v", "midian_va"]
wv = ver[ver.label.isin(cols)].pivot_table(index=["dist", "beta", "seed"], columns="label", values="success")
for f in FWS:
    paired(f"d.lift_V_r10_{f}", wv, f + "[r=10,retrieval=midian]", f, "fw_live_n1000_verified vs fw_live_n1000, 120 units"); paired(f"d.trail_{f}_Vcohort_minus_midian_v", wv, f + "[r=10,retrieval=midian]", "midian_v", "fw_live_n1000_verified vs fw_live_n1000")
    paired(f"d.lift_VA_r10_{f}", wv, f + "[r=10,retrieval=midian_va]", f, "fw_live_n1000_verified_va vs fw_live_n1000"); paired(f"d.VAcohort_minus_Vcohort_{f}", wv, f + "[r=10,retrieval=midian_va]", f + "[r=10,retrieval=midian]", "fw_live_n1000_verified_va vs _verified")
lifts = {f: N[f"d.lift_V_r10_{f}"]["value"] for f in FWS}; put_v("d.lift_V_r10_range_excl_llamaindex", [round(min(v for f, v in lifts.items() if f != "fw_llamaindex"), 3), round(max(lifts.values()), 3)], "fw_live_n1000_verified")
trail = {f: N[f"d.trail_{f}_Vcohort_minus_midian_v"]["value"] for f in FWS}; put_v("d.trail_Vcohort_minus_midian_v_range", [round(min(trail.values()), 3), round(max(trail.values()), 3)], "fw_live_n1000_verified")
wvb = wv.xs(0.5, level="beta"); put("d.VAcohort_minus_Vcohort_beta05_mean_over_frameworks", pd.concat([(wvb[f + "[r=10,retrieval=midian_va]"] - wvb[f + "[r=10,retrieval=midian]"]) for f in FWS]), "fw_live_n1000_verified_va vs _verified, β = 0.5, 10 frameworks × 3 shapes × 10 seeds")

# ------------------------------------------------------------------------------------------------ §4
pool = rows("llmrouterbench_pool"); wp = W(pool, [FLAT_ON, "mlp_router", "midian_va", "midian_a", "midian", "midian_v", HALP, "declared_argmax", "knn_router", "oracle"])
for s in (FLAT_ON, "mlp_router", "midian_va", "midian_a", "midian", "midian_v", HALP, "declared_argmax"): put(f"s4.pool20_{s}_all_beta", wp[s], "llmrouterbench_pool (all β, both liar selections, 5 seeds)")
ch = rows("churn_n1000")
if "churn" in ch:
    import ast
    ch = ch.assign(frac=ch.churn.map(lambda c: (c if isinstance(c, dict) else ast.literal_eval(str(c))).get("frac")))
    for fr in sorted(ch.frac.dropna().unique()):
        q = ch[ch.frac == fr].pivot_table(index=["dist", "seed"], columns="label", values="success")
        if {"midian", "midian_va"} <= set(q.columns): put(f"s4.churn_{int(fr*100)}pct_va_minus_midian", (q["midian_va"] - q["midian"]).dropna(), f"churn_n1000, churn {int(fr*100)}%")
put_v("s4.phase1_targets_missed", "5 of 6 (T2 split) — TARGETS_rte.md / RESULTS_rte.md §8", "static")

# ------------------------------------------------------------------------------------------------ appendix source tables
def table(key, df, labels, by, grid, **f):
    w = W(df, labels, **f); N[key] = dict(value={l: {str(k): round(float(v), 4) for k, v in w[l].groupby(level=by).mean().items()} for l in labels if l in w}, grid=grid, units=int(len(w)), ci=None)
table("T2.fw_live_n1000_by_beta", fw, FWS + [MAG14, "midian", "midian_v", "midian_va", "midian_a", "oracle", "random", "declared_argmax", "llm_supervisor", FLAT], "beta", "fw_live_n1000")
table("T2.n10k_by_beta", n10, FWS + ["midian", "midian_v", "midian_va", "midian_a", "midian_sh", FLAT_ON, "warm_start_bandit", "linucb_honest", "verify_on_claim", HALP, "oracle", "random"], "beta", "live_n10k_v2")
table("T3.variants_f1_by_beta_random", rows("variants_f1"), ["oracle", HALP, "midian_va", "midian_a", "midian_v", "midian", "midian_sh", "midian_sha", FLAT_ON, "linucb_honest"], "beta", "variants_f1", liar_select="random")
table("T3.variants_f1_by_beta_lowskill", rows("variants_f1"), ["oracle", HALP, "midian_va", "midian_a", "midian_v", "midian", "midian_sh", "midian_sha", FLAT_ON, "linucb_honest"], "beta", "variants_f1", liar_select="low_skill_first")
table("T3.learned_f1_by_beta_random", rows("learned_f1"), ["knn_router", "knn_router_online", "mlp_router"], "beta", "learned_f1", liar_select="random"); table("T3.learned_f1_by_beta_lowskill", rows("learned_f1"), ["knn_router", "knn_router_online", "mlp_router"], "beta", "learned_f1", liar_select="low_skill_first")
table("T5.learned_n100_by_beta_lowskill", rows("learned_n100"), ["oracle", HALP, HAL, "midian_va", "midian_a", "midian_v", "midian", FLAT_ON, FLAT, "knn_router", "mlp_router", "declared_argmax", "warm_start_bandit"], "beta", "learned_n100", liar_select="low_skill_first")
table("T5.learned_n10k_by_beta_lowskill", rows("learned_n10k"), ["oracle", HALP, "midian_va", "midian_a", "midian_v", "midian", FLAT_ON, FLAT, "knn_router", "declared_argmax", "warm_start_bandit"], "beta", "learned_n10k", liar_select="low_skill_first")
for n in (10000, 100000): table(f"T5.scale_100k_n{n}_by_beta_lowskill", sc, ["oracle", HALP, HAL, "midian_va", "midian_a", "midian_v", "midian", FLAT_ON, FLAT, "declared_argmax", "warm_start_bandit", "linucb_honest", "random"], "beta", "scale_100k", n=n, liar_select="low_skill_first")
for n in (10, 100, 1000): table(f"T7.routereval_mmlu_n{n}_by_beta_lowskill", re1, ["oracle", HAL, HALP, "declared_argmax", "warm_start_bandit", "midian_va", "midian_v", "midian_a", "midian", "mlp_router", FLAT_ON, "knn_router", "linucb_honest", "random"], "beta", "routereval_mmlu", n=n, liar_select="low_skill_first")
table("T7.routereval_mmlu5k_by_beta_lowskill", re5, ["oracle", HAL, HALP, "declared_argmax", "warm_start_bandit", "midian_va", "midian_v", "midian_a", "midian", FLAT_ON, "knn_router", "linucb_honest", "random"], "beta", "routereval_mmlu5k", liar_select="low_skill_first")
table("T8.llmrouterbench_pool_by_beta_lowskill", pool, ["oracle", HAL, HALP, "declared_argmax", "warm_start_bandit", "midian_va", "midian_v", "midian_a", "midian", "mlp_router", FLAT_ON, "knn_router", "random"], "beta", "llmrouterbench_pool", liar_select="low_skill_first")
bud = rows("budget_sweep"); N["D.budget_sweep_by_b"] = dict(value={l: {str(k): round(float(v), 4) for k, v in bud[bud.label == l].groupby("b").success.mean().items()} for l in ["oracle", HAL, "midian_va", "midian_a", "midian_v", "midian", FLAT, "declared_argmax", "warm_start_bandit"]}, grid="budget_sweep (programmatic, β = 0.25)", units=int(len(bud)), ci=None)

os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(N, open(OUT, "w"), indent=1, ensure_ascii=False); print(f"{len(N)} entries -> {OUT}")
for k, v in N.items():
    if not k[:2] in ("T2", "T3", "T5", "T7", "T8", "D.") and not isinstance(v["value"], list) or k.startswith("a.paired_framework") or k.startswith("a.frameworks"): print(f"{k:60s} {v['value']}  ci={v.get('ci')}  [{v['grid']}]" + (f"  {v['note']}" if v.get("note") else ""))
