"""Seven synthesis figures for RESULTS_rte.md (A-G) from the finished grids.  python scripts/extra_figs.py"""
import glob, json, os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.environ.get("RTE_DATA", "/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte") + "/results"; O = R + "/extra_figs"; os.makedirs(O, exist_ok=True)
RED, ORG, YEL, BLU, GRY = "#c0392b", "#e67e22", "#f1c40f", "#3498db", "#999"


def load(g, raw=False):
    df = pd.DataFrame([json.load(open(f)) for f in glob.glob(f"{R}/{g}/rows.d/*.json")])
    df["m"] = df.method + (df.params if raw else df.params.str.replace('{}', '').str.replace('"', '').str.replace('cached:true,', '')
                           .str.replace('verify:true', 'V').str.replace('peer_reported:true', 'peer'))
    return df


def ci(x, B=2000):
    rng = np.random.default_rng(0); x = np.asarray(x, float); return np.percentile([rng.choice(x, len(x)).mean() for _ in range(B)], [2.5, 97.5])


def line(ax, s, **kw):
    ax.errorbar(s.mean().index, s.mean().values, yerr=1.96 * s.sem().values, marker="o", ms=4, **kw)


NICE = {"midian": "MIDIAN", "midian{V}": "MIDIAN-V r=10", "midian{r:5,V}": "MIDIAN-V r=5", "midian{r:5}": "MIDIAN r=5",
        "flat_probe_argmax": "flat probe argmax", "declared_argmax": "declared argmax", "llm_supervisor": "LLM supervisor", "oracle": "oracle", "random": "random"}
fw = lambda m: m.replace("fw_", "").replace("_", " ")

# A: headline bars
fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
for ax, n in zip(axes, ["n100", "n1000"]):
    w = load(f"fw_live_{n}").pivot_table(index=["dist", "beta", "seed"], columns="m", values="success")
    ms = [m for m in ["oracle", "midian{r:5,V}", "midian{V}", "midian", "midian{r:5}", "flat_probe_argmax", "declared_argmax", "llm_supervisor"]
          + sorted(c for c in w.columns if c.startswith("fw_")) + ["random"] if m in w]
    core = w[ms].dropna(); mean = core.mean(); err = np.array([ci(core[m]) for m in ms]); y = np.arange(len(ms))[::-1]
    ax.barh(y, mean[ms], color=[GRY if m == "oracle" else RED if m.startswith("midian") else "#2980b9" if m.startswith("fw_") else "#7f8c8d" for m in ms],
            xerr=[mean[ms] - err[:, 0], err[:, 1] - mean[ms]], capsize=3)
    ax.set_yticks(y); ax.set_yticklabels([NICE.get(m, fw(m)) for m in ms]); ax.set_xlim(0.25, 0.8); ax.grid(axis="x", alpha=.3)
    ax.set_xlabel("success (paired cells, 95% bootstrap CI)"); ax.set_title(f"Frameworks vs MIDIAN, n={n[1:]} ({len(core)} cells)")
plt.tight_layout(); plt.savefig(f"{O}/A_headline_frameworks_vs_midian.png", dpi=150); plt.close()

# B: verified-shortlist lift
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, n in zip(axes, ["n100", "n1000"]):
    df = pd.concat([load(f"fw_live_{n}", raw=True), load(f"fw_live_{n}_verified", raw=True)])
    w = df.pivot_table(index=["dist", "beta", "seed"], columns="m", values="success"); fws = sorted(set(df[df.method.str.startswith("fw_")].method))
    own = [w[f + "{}"].mean() for f in fws]; x = np.arange(len(fws))
    ax.bar(x - .27, own, .27, label="own selection (TF-IDF top-10)", color="#2980b9")
    for off, r, c in ((0, 10, ORG), (.27, 5, YEL)):
        ax.bar(x + off, [w[[f + '{"r":%d,"retrieval":"midian"}' % r, f + "{}"]].dropna().iloc[:, 0].mean() for f in fws], .27, label=f"+ MIDIAN-V cohort r={r}", color=c)
    mv = w['midian{"cached":true,"verify":true}'].mean(); ax.axhline(mv, color=RED, ls="--", label=f"MIDIAN-V alone ({mv:.2f})"); ax.axhline(w["oracle{}"].mean(), color=GRY, ls=":", label="oracle")
    ax.set_xticks(x); ax.set_xticklabels([fw(f) for f in fws], rotation=35, ha="right"); ax.set_ylim(0.4, 0.8); ax.grid(axis="y", alpha=.3); ax.set_title(f"Frameworks with MIDIAN's verified shortlist, n={n[1:]}")
axes[0].set_ylabel("success"); axes[1].legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{O}/B_verified_shortlist_lift.png", dpi=150); plt.close()

# C: MIDIAN vs halving by beta x liar selection;  E: all rivals by class
w = load("live_f1_n1000").pivot_table(index=["dist", "beta", "liar_select", "declared_source", "seed"], columns="m", values="success")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, ls in zip(axes, ["random", "low_skill_first"]):
    sub = w.xs(ls, level="liar_select")
    for m, c, lab in [("oracle", GRY, "oracle"), ("sequential_halving", "#2c3e50", "halving, trusted observer"), ("sequential_halving{peer}", "#8e44ad", "halving, peer-reported (fair control)"),
                      ("midian{V}", ORG, "MIDIAN-V r=10"), ("midian{r:5,V}", YEL, "MIDIAN-V r=5"), ("midian", RED, "MIDIAN"), ("flat_probe_argmax", BLU, "flat probe argmax")]:
        line(ax, sub[m].groupby(level="beta"), label=lab, color=c)
    ax.set_title(f"MIDIAN vs halving, liars = {ls}"); ax.set_xlabel("β (liar fraction)"); ax.grid(alpha=.3)
axes[0].set_ylabel("success (n=1000, 3 shapes x 2 channels x 5 seeds)"); axes[1].legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{O}/C_midian_vs_halving_beta_liars.png", dpi=150); plt.close()
CLASSES = {"declared-channel rivals": ["declared_argmax", "declared_softmax", "cnp_self_bid", "route_to_k_majority", "cluster_head_router", "disrouter_cascade", "llm_supervisor"],
           "verified, centralized": ["sequential_halving", "sequential_halving{peer}", "warm_start_bandit", "verify_on_claim", "flat_probe_argmax{online:true}", "flat_probe_argmax", "flat_nsw_router", "trueskill_per_family", "thompson_per_family", "ucb_per_family"],
           "verified, decentralized": ["gossip_reputation_greedy", "referral_network"],
           "MIDIAN family": ["midian{r:5}", "midian{V}", "midian{r:5,V}", "midian_llm_descent", "midian{online:false}"]}
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
for ax, (title, ms) in zip(axes, CLASSES.items()):
    for m, c in (("oracle", GRY), ("random", "#ccc")): g = w[m].groupby(level="beta").mean(); ax.plot(g.index, g.values, color=c, ls=":", label=m)
    line(ax, w["midian"].groupby(level="beta"), label="MIDIAN", lw=2.5, color=RED, zorder=10)
    for m in ms: line(ax, w[m].groupby(level="beta"), label=m, lw=1.2)
    ax.set_title(title); ax.set_xlabel("β (liar fraction)"); ax.grid(alpha=.3); ax.legend(fontsize=7, loc="lower left")
axes[0].set_ylabel("success (live_f1_n1000: 48 cells x 5 seeds)"); plt.suptitle("F1: every rival vs liar fraction, n=1000 (MIDIAN in red in every panel)")
plt.tight_layout(); plt.savefig(f"{O}/E_all_rivals_vs_beta_by_class.png", dpi=150); plt.close()

# D: learning curve (per-task outcomes from $RTE_DATA/scratch/curve.py)
runs = [json.load(open(f)) for f in sorted(glob.glob(R.replace("/results", "") + "/scratch/curve_*.json"))]
if runs:
    B = 50; fig, ax = plt.subplots(figsize=(9, 5))
    for m, lab, c in [("oracle", "oracle", GRY), ('midian{"cached":true,"verify":true}', "MIDIAN-V r=10", ORG), ("midian{}", "MIDIAN", RED), ('midian{"online":false}', "MIDIAN, updates off", "#e74c3c"),
                      ('flat_probe_argmax{"online":true}', "flat probe, online", BLU), ("warm_start_bandit{}", "warm-start bandit", "#27ae60"), ("llm_supervisor{}", "LLM supervisor", "#7f8c8d")]:
        arr = np.array([r[m] for r in runs], float).reshape(len(runs), -1, B).mean(axis=(0, 2)); ax.plot(np.arange(len(arr)) * B + B / 2, arr, marker=".", label=lab, color=c, ls="--" if "off" in lab else "-")
    ax.set_xlabel("task index (blocks of 50)"); ax.set_ylabel(f"success ({len(runs)} live cells, n=1000, β=0.25)"); ax.set_title("Learning over the stream"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{O}/D_learning_curve.png", dpi=150); plt.close()

# F: cost scaling across backends
d = pd.concat([load(g) for g in ["live_core_n100", "live_f1_n1000", "live_n10k", "replay_scale", "bernoulli_scale"]]); d = d[d.beta.isin([0.0, 0.25])]
KEEP = {"midian": ("MIDIAN", RED), "midian{V}": ("MIDIAN-V", ORG), "flat_probe_argmax": ("flat probe argmax", BLU), "declared_argmax": ("declared argmax", "#7f8c8d"),
        "cnp_self_bid": ("CNP self-bid", "#8e44ad"), "sequential_halving": ("seq. halving", "#2c3e50"), "gossip_reputation_greedy": ("gossip", "#27ae60")}
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, (col, title) in zip(axes, [("comparisons_per_task", "comparisons per task"), ("messages_per_task", "messages per task"), ("build_probes", "build probes")]):
    for m, (lab, c) in KEEP.items():
        s = d[d.m == m].groupby("n")[col].mean(); s = s[s > 0]
        if len(s): ax.plot(s.index, s.values, marker="o", label=lab, color=c, lw=2.5 if m.startswith("midian") else 1.2)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("n (agents)"); ax.set_title(title + "  (methods at 0 omitted)"); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
plt.suptitle("F3: cost vs n, 10² to 10⁷ (live 10²–10⁴, replay 10⁴–10⁶, bernoulli 10²–10⁷); MIDIAN per task ∝ n^0.14, MIDIAN-V ∝ n^0, flat/declared ∝ n^1")
plt.tight_layout(); plt.savefig(f"{O}/F_cost_scaling.png", dpi=150); plt.close()

# G: budget sweep + internals
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw={"width_ratios": [1, 1.25]})
db = load("budget_sweep"); ax = axes[0]
keep_b = db.pivot_table(index=["dist", "seed", "b"], columns="m", values="success").dropna().index
db = db.set_index(["dist", "seed", "b"]).loc[keep_b].reset_index()    # paired cells only
for m, lab, c in [("oracle", "oracle", GRY), ("sequential_halving", "seq. halving (trusted)", "#2c3e50"), ("midian{V}", "MIDIAN-V r=10", ORG), ("midian", "MIDIAN", RED), ("flat_probe_argmax", "flat probe argmax", BLU),
                  ("warm_start_bandit", "warm-start bandit", "#27ae60"), ("declared_argmax", "declared argmax", "#16a085"), ("ucb_per_family", "UCB", "#8e44ad")]:
    line(ax, db[db.m == m].groupby("b").success, label=lab, color=c, lw=2.5 if m == "midian" else 1.2)
ax.set_xscale("log"); ax.set_xticks([1, 3, 10]); ax.set_xticklabels(["1", "3", "10"]); ax.set_xlabel("probe budget b per (agent, family)"); ax.set_ylabel("success (n=1000, β=0.25, 3 shapes x 5 seeds)")
ax.set_title("F4: success vs build budget"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower right")
di = load("midian_internals"); di = di[di.collude == True]; ax = axes[1]
for m, lab, c in [("midian{delta:0.3333333333333333,r:5}", "MIDIAN r=5 δ=1/3", "#e74c3c"), ("midian{delta:0.3333333333333333,r:10}", "MIDIAN r=10 δ=1/3", RED), ("midian{delta:0.3333333333333333,r:20}", "MIDIAN r=20 δ=1/3", "#7b241c"),
                  ("midian{delta:0.0,r:10}", "MIDIAN r=10 δ=0 (no trim)", ORG), ("midian{r:5,V}", "MIDIAN-V r=5", YEL), ("midian{r:10,V}", "MIDIAN-V r=10", "#f39c12"), ("midian{r:20,V}", "MIDIAN-V r=20", "#b9770e"),
                  ("flat_probe_argmax", "flat probe argmax", BLU), ("oracle", "oracle", GRY)]:
    line(ax, di[di.m == m].groupby("beta").success, label=lab, color=c, lw=2.5 if lab == "MIDIAN r=10 δ=1/3" else 1.2)
ax.set_xlabel("β (colluding liars)"); ax.set_title("F7: MIDIAN internals (n=1000 specialist): cohort r, trim δ, verification"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
plt.tight_layout(); plt.savefig(f"{O}/G_budget_and_internals.png", dpi=150, bbox_inches="tight"); plt.close()
print("wrote", sorted(os.listdir(O)))
