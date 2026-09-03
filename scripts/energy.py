"""Runtime/energy ESTIMATE (*) per method from LLM-call counts x measured per-call GPU cost.  python scripts/energy.py
Wall-clock in the rows is not used for probe methods (memo hits). Model: GPU-seconds per call = params_b * (A*prompt_tok + B*gen_tok),
B = 5A (decode is ~5x prefill per token on H100/vLLM), A calibrated so a 7B supervisor call (1,900 prompt + 65 gen tokens) costs
1/34 GPU-s = the throughput measured on the saturated 1-GPU 7B replicas (2026-09-03, 4 samples, 32-36 req/s). Energy = GPU-s * W."""
import glob, json, os, yaml, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.environ.get("RTE_DATA", "/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte") + "/results"
REQ_PER_S_7B, SUP_TOK = 34.0, (1900, 65)                          # measured: saturated 1-GPU 7B replica; supervisor prompt/gen tokens
A = (1 / REQ_PER_S_7B) / (7.0 * (SUP_TOK[0] + 5 * SUP_TOK[1])); B = 5 * A
WATTS = {"700 W (H100 TDP)": 700, "400 W (typical draw)": 400}
LADDER = yaml.safe_load(open(os.path.dirname(__file__) + "/../configs/models.yaml"))["models"]; P = {m["id"]: m["params_b"] for m in LADDER}
PROBE_TOK = {"Qwen/Qwen2.5-0.5B-Instruct": (96, 12), "Qwen/Qwen2.5-1.5B-Instruct": (101, 66), "google/gemma-2-2b-it": (115, 15),   # measured per-server lifetime means
             "Qwen/Qwen2.5-3B-Instruct": (294, 86), "Qwen/Qwen2.5-7B-Instruct": (294, 86), "google/gemma-2-9b-it": (158, 33), "Qwen/Qwen2.5-14B-Instruct": (294, 86)}  # 7B/14B: 3B proxy (tool-enabled)
small, big = [m for m in P if P[m] <= 1.5], [m for m in P if P[m] >= 7.0]
MIX = {"specialist": {m: 1 / 7 for m in P}, "heavy_tail": {**{m: 0.9 / len(small) for m in small}, **{m: 0.1 / len(big) for m in big}}, "bimodal": {small[0]: 0.8, big[0]: 0.2}}
call = lambda model, tok: P[model] * (A * tok[0] + B * tok[1])                   # GPU-seconds for one call
probe_cost = {d: sum(w * call(m, PROBE_TOK[m]) for m, w in mix.items()) for d, mix in MIX.items()}   # expected GPU-s per probe / per task execution
SUP = call("Qwen/Qwen2.5-7B-Instruct", SUP_TOK)

def rows(*grids):
    df = pd.concat([pd.DataFrame([json.load(open(f)) for f in glob.glob(f"{R}/{g}/rows.d/*.json")]) for g in grids])
    df = df[(df.declared_source == "self_described") & (df.n == 1000)]; df["m"] = df.method + df.params.str.replace("{}", "")
    return df.groupby("m")[["build_probes", "probes_per_task", "tasks_per_task", "wall_clock_per_task"]].mean()

def table(Q=1000, dist="specialist", watts=700):
    c = rows("live_f1_n1000", "variants_f1", "fw_live_n1000"); base = c.loc["fw_autogen", "wall_clock_per_task"]
    out = []
    for m, r in c.iterrows():
        if m in ("oracle", "random") or m.startswith("midian{") or m.startswith("flat_nsw") or m in ("gossip_reputation_greedy", "referral_network", "thompson_per_family", "ucb_per_family", "trueskill_per_family", "midian_llm_descent", "cluster_head_router", "cnp_self_bid", "declared_softmax", "disrouter_cascade", "route_to_k_majority"): continue
        sup_calls = (r.wall_clock_per_task / base) if (m.startswith("fw_") or m == "llm_supervisor") else 0.0   # call-equivalents (latency ratio to the one-call AutoGen)
        sup_scale = P["Qwen/Qwen2.5-14B-Instruct"] / 7.0 if "14B" in m else 1.0
        build = r.build_probes * probe_cost[dist]; per_task = r.probes_per_task * probe_cost[dist] + sup_calls * SUP * sup_scale
        exec_ = r.tasks_per_task * probe_cost[dist]                                        # the routed task itself, common to every method
        out.append(dict(method=m, build_gpu_s=build, route_gpu_s_per_task=per_task, exec_gpu_s_per_task=exec_, sup_call_equiv=sup_calls,
                        **{f"gpu_s_per_1k_tasks_Q{q}": (build / q + per_task) * 1000 for q in (300, 1000, 10000)}))
    t = pd.DataFrame(out).set_index("method"); t["Wh_per_1k_tasks_Q1000"] = t.gpu_s_per_1k_tasks_Q1000 * watts / 3600
    t["breakeven_Q_vs_autogen"] = np.where(t.route_gpu_s_per_task < SUP, t.build_gpu_s / np.maximum(SUP - t.route_gpu_s_per_task, 1e-12), np.inf)
    return t.sort_values("gpu_s_per_1k_tasks_Q1000")

if __name__ == "__main__":
    t = table(); pd.set_option("display.width", 250)
    md = ["# Runtime and energy — ESTIMATE (*)", "", __doc__.strip(), "",
          f"Per-call GPU-seconds: 7B supervisor call {SUP:.4f}; expected probe/execution call by population shape: " + ", ".join(f"{d} {v:.5f}" for d, v in probe_cost.items()) + " (specialist mixes all 7 models uniformly; heavy_tail 90% 0.5-1.5B; bimodal 80% 0.5B / 20% 7B).",
          "Framework supervisor cost = measured latency ratio to AutoGen (one call) x the 7B call cost; the 14B Magentic-One arm is scaled by 14/7. CPU-side routing (tree descent, TF-IDF) is microseconds per task and omitted.", "",
          "| method | build GPU-s | route GPU-s/task | task-execution GPU-s/task (common) | supervisor call-equiv/task | GPU-s per 1k tasks @Q=300 | @Q=1000 | @Q=10000 | Wh per 1k tasks @Q=1000 (700 W) | Wh (400 W) | break-even Q vs AutoGen |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, r in t.iterrows():
        md.append(f"| {m} | {r.build_gpu_s:,.0f} | {r.route_gpu_s_per_task:.4f} | {r.exec_gpu_s_per_task:.4f} | {r.sup_call_equiv:.1f} | {r.gpu_s_per_1k_tasks_Q300:,.0f} | {r.gpu_s_per_1k_tasks_Q1000:,.0f} | {r.gpu_s_per_1k_tasks_Q10000:,.0f} | {r.Wh_per_1k_tasks_Q1000:,.1f} | {r.Wh_per_1k_tasks_Q1000*400/700:,.1f} | {('%.0f' % r.breakeven_Q_vs_autogen) if np.isfinite(r.breakeven_Q_vs_autogen) else 'never'} |")
    md += ["", "MIDIAN's 48,000-probe build by population shape (GPU-s): " + ", ".join(f"{d} {48000 * v:,.0f}" for d, v in probe_cost.items())
           + f"; one AutoGen supervisor call = {SUP:.4f} GPU-s, so on specialist the build equals ~{48000 * probe_cost['specialist'] / SUP:,.0f} one-call framework tasks.",
           "Reading: at Q = 1,000 every probe-based method spends ~9x a one-call framework's GPU-seconds (the build dominates); at Q = 10,000 they are level "
           "with AutoGen and ~10-30x cheaper than the multi-call frameworks; MIDIAN-A's audits add 5%; MIDIAN-V and halving are the cheapest builds (fewer probes)."]
    open(os.path.dirname(__file__) + "/../RESULTS_energy.md", "w").write("\n".join(md) + "\n")
    Qs = np.logspace(2, 5, 25); fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for m, r in t.iterrows():
        y = (r.build_gpu_s / Qs + r.route_gpu_s_per_task) * 1000; lw = 2.5 if m in ("midian", "midian_a", "midian_v") else 1.2
        axes[0].plot(Qs, y, label=m, lw=lw); axes[1].plot(Qs, y * 700 / 3600, lw=lw)
    for ax, yl in zip(axes, ["GPU-seconds per 1,000 routed tasks", "Wh per 1,000 routed tasks (700 W per H100)"]):
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Q = tasks routed over the build's lifetime"); ax.set_ylabel(yl); ax.grid(alpha=.3, which="both")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7, title="declared_argmax: 0 LLM calls (off the log axis)", title_fontsize=7)
    fig.suptitle("H10*  Estimated LLM compute per 1,000 tasks (build amortised over Q); n=1000 specialist population; the routed task's own execution excluded")
    plt.tight_layout(); plt.savefig(os.path.dirname(__file__) + "/../figures/H10_runtime_energy.png", dpi=300, bbox_inches="tight"); print(t.round(4).to_string())
