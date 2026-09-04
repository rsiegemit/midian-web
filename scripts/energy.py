"""Runtime/energy ESTIMATE (*) per method from LLM-call counts x measured per-call GPU cost.  python scripts/energy.py
Wall-clock in the rows is not used for probe methods (memo hits). Model: GPU-seconds per call = params_b * (A*prompt_tok + B*gen_tok),
B = 5A (decode is ~5x prefill per token on H100/vLLM), A calibrated so a 7B supervisor call (1,900 prompt + 65 gen tokens) costs
1/34 GPU-s = the throughput measured on the saturated 1-GPU 7B replicas (2026-09-03, 4 samples, 32-36 req/s). Energy = GPU-s * W."""
import glob, json, os, yaml, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.environ.get("RTE_DATA", "/n/netscratch/sompolinsky_lab/Lab/rsiegelmann/rte") + "/results"
REQ_PER_S_7B, SUP_TOK = 34.0, (1900, 65)                          # measured: saturated 1-GPU 7B replica; supervisor prompt/gen tokens
A = (1 / REQ_PER_S_7B) / (7.0 * (SUP_TOK[0] + 5 * SUP_TOK[1])); B = 5 * A
WATTS = {"700 W (H100 TDP)": 700, "400 W (typical draw)": 400}
J_MSG, J_CMP, RTT, T_CMP, PROBE_LAT = 1e-3, 1e-8, 1e-3, 1e-8, 0.3    # J per message, J per comparison, s per hop, s per comparison, s per route-time probe call
DEPTH = 3                                                             # MIDIAN tree depth at n=1000, r=10: the observe-time update sends 1 message per level, OFF the critical path
LADDER = yaml.safe_load(open(os.path.dirname(__file__) + "/../configs/models.yaml"))["models"]; P = {m["id"]: m["params_b"] for m in LADDER}
PROBE_TOK = {"Qwen/Qwen2.5-0.5B-Instruct": (96, 12), "Qwen/Qwen2.5-1.5B-Instruct": (101, 66), "google/gemma-2-2b-it": (115, 15),   # measured per-server lifetime means
             "Qwen/Qwen2.5-3B-Instruct": (294, 86), "Qwen/Qwen2.5-7B-Instruct": (294, 86), "google/gemma-2-9b-it": (158, 33), "Qwen/Qwen2.5-14B-Instruct": (294, 86)}  # 7B/14B: 3B proxy (tool-enabled)
small, big = [m for m in P if P[m] <= 1.5], [m for m in P if P[m] >= 7.0]
MIX = {"specialist": {m: 1 / 7 for m in P}, "heavy_tail": {**{m: 0.9 / len(small) for m in small}, **{m: 0.1 / len(big) for m in big}}, "bimodal": {small[0]: 0.8, big[0]: 0.2}}
call = lambda model, tok: P[model] * (A * tok[0] + B * tok[1])                   # GPU-seconds for one call
probe_cost = {d: sum(w * call(m, PROBE_TOK[m]) for m, w in mix.items()) for d, mix in MIX.items()}   # expected GPU-s per probe / per task execution
SUP = call("Qwen/Qwen2.5-7B-Instruct", SUP_TOK)

def rows(*grids):
    """Per-method means over the n=1000 self-described cells, read through rte.analyze.load so legacy rows get MIDIAN's
    observe-time recompute charged (commit 3415f03: r comparisons + 1 message per level per task)."""
    from rte.analyze import load
    df = load(list(grids)); df = df[(df.declared_source == "self_described") & (df.n == 1000)]; df["m"] = df.method + df.params.str.replace("{}", "")
    return df.groupby("m")[["build_probes", "probes_per_task", "tasks_per_task", "wall_clock_per_task", "build_messages", "messages_per_task", "build_comparisons", "comparisons_per_task", "hops_per_task"]].mean()

def table(dist="specialist", watts=700):
    c = rows("live_f1_n1000", "variants_f1", "fw_live_n1000"); base = c.loc["fw_autogen", "wall_clock_per_task"]
    skip = ("oracle", "random", "gossip_reputation_greedy", "referral_network", "thompson_per_family", "ucb_per_family", "trueskill_per_family", "midian_llm_descent",
            "cluster_head_router", "cnp_self_bid", "declared_softmax", "disrouter_cascade", "route_to_k_majority")
    out = []
    for m, r in c.iterrows():
        if m in skip or m.startswith("midian{") or m.startswith("flat_nsw"): continue
        sup_calls = (r.wall_clock_per_task / base) if (m.startswith("fw_") or m == "llm_supervisor") else 0.0   # call-equivalents (latency ratio to one-call AutoGen)
        sup_scale = P["Qwen/Qwen2.5-14B-Instruct"] / 7.0 if "14B" in m else 1.0
        out.append(dict(method=m, build_gpu_s=r.build_probes * probe_cost[dist], per_task_gpu_s=r.probes_per_task * probe_cost[dist] + sup_calls * SUP * sup_scale,
                        exec_gpu_s_per_task=r.tasks_per_task * probe_cost[dist], sup_call_equiv=sup_calls, build_msgs=r.build_messages, msgs_per_task=r.messages_per_task,
                        build_cmp=r.build_comparisons, cmp_per_task=r.comparisons_per_task, sup_latency=(r.wall_clock_per_task if sup_calls else 0.0),
                        probes_per_task=r.probes_per_task))
    t = pd.DataFrame(out).set_index("method")
    for T in (1_000, 10_000, 100_000): t[f"cum_gpu_s_t{T}"] = t.build_gpu_s + T * t.per_task_gpu_s
    t["cum_Wh_t10000"] = t.cum_gpu_s_t10000 * watts / 3600; t["cum_msgs_t10000"] = t.build_msgs + 10_000 * t.msgs_per_task; t["cum_cmp_t10000"] = t.build_cmp + 10_000 * t.cmp_per_task
    # (a) one energy currency, joules: LLM call = GPU-s x 700 W; message = 1 RPC ~100 us on a ~10 W core = 1e-3 J (x10 pessimistic column); comparison = 1e-8 J
    for J_msg, tag in ((J_MSG, ""), (10 * J_MSG, "_pess")):
        t["build_J" + tag] = t.build_gpu_s * watts + t.build_msgs * J_msg + t.build_cmp * J_CMP
        t["per_task_J" + tag] = t.per_task_gpu_s * watts + t.msgs_per_task * J_msg + t.cmp_per_task * J_CMP
        t["J_per_task_t10000" + tag] = t["build_J" + tag] / 10_000 + t["per_task_J" + tag]
    # (b) critical-path latency per task: sequential message hops at 1 ms RTT (MIDIAN's 2 per level; frameworks 2 + the supervisor call at its
    #     measured median latency under shared-fleet load; probes at route time ~0.3 s each), comparisons at 10 ns (flat: 1,000 -> 10 us)
    is_fw = t.index.str.startswith("fw_") | (t.index == "llm_supervisor")
    crit_msgs = np.where(t.index.str.startswith("midian"), t.msgs_per_task - DEPTH, t.msgs_per_task)     # fetch hops only: the update propagation is off the critical path
    t["latency_s"] = np.where(is_fw, 2 * RTT + t.sup_latency, crit_msgs * RTT) + t.cmp_per_task * T_CMP + t.probes_per_task * PROBE_LAT
    return t.sort_values("cum_gpu_s_t10000")

def crossing(t, a, b, build="build_gpu_s", slope="per_task_gpu_s"):
    """Tasks after which method a (higher build, lower slope) becomes cheaper than b; inf if never."""
    db, ds = t.loc[a, build] - t.loc[b, build], t.loc[b, slope] - t.loc[a, slope]
    return db / ds if ds > 0 and db > 0 else (0.0 if db <= 0 else float("inf"))

if __name__ == "__main__":
    t = table(); pd.set_option("display.width", 250)
    fws = [m for m in t.index if m.startswith("fw_")]; mids = ["midian", "midian_a", "midian_v"]
    md = ["# Runtime and energy — ESTIMATE (*)", "", __doc__.strip(), "",
          f"Per-call GPU-seconds: 7B supervisor call {SUP:.4f}; expected probe/execution call by population shape: " + ", ".join(f"{d} {v:.5f}" for d, v in probe_cost.items())
          + " (specialist mixes all 7 models uniformly; heavy_tail 90% 0.5-1.5B; bimodal 80% 0.5B / 20% 7B).",
          "Framework supervisor cost = measured latency ratio to AutoGen (one call) x the 7B call cost; the 14B Magentic-One arm is scaled by 14/7. CPU-side routing (tree descent, TF-IDF) is microseconds per task and omitted. "
          "The routed task's own execution (0.0058 GPU-s per task on specialist) is common to every method and excluded.", "",
          "CUMULATIVE LLM compute after t routed tasks = build + t x per-task (n=1000 specialist). Probe-based methods pay their build up front and then ~0 per task; frameworks pay 0 up front and a supervisor call per task, so the crossing of the two lines is the break-even.", "",
          "| method | build GPU-s | per-task GPU-s | cumulative GPU-s @ t=1k | @ 10k | @ 100k | cumulative Wh @ 10k (700 W) | (400 W) | messages @ 10k (build + 10k/task) | comparisons @ 10k |", "|---|---|---|---|---|---|---|---|---|---|"]
    for m, r in t.iterrows():
        md.append(f"| {m} | {r.build_gpu_s:,.0f} | {r.per_task_gpu_s:.4f} | {r.cum_gpu_s_t1000:,.0f} | {r.cum_gpu_s_t10000:,.0f} | {r.cum_gpu_s_t100000:,.0f} | {r.cum_Wh_t10000:,.1f} | {r.cum_Wh_t10000*400/700:,.1f} | {r.cum_msgs_t10000:,.0f} ({r.build_msgs:,.0f} + {r.msgs_per_task:.0f}/task) | {r.cum_cmp_t10000:,.0f} ({r.build_cmp:,.0f} + {r.cmp_per_task:.0f}/task) |")
    md += ["", "**Crossing points (tasks routed before the probe-based method's cumulative cost drops below the framework's):**", "", "| | " + " | ".join(m.replace("fw_", "") for m in fws) + " |", "|---|" + "---|" * len(fws)]
    for a in mids: md.append(f"| {a} | " + " | ".join(f"{crossing(t, a, b):,.0f}" for b in fws) + " |")
    md += ["", "In MESSAGES (fetch 2 per level + observe-time update 1 per level, commit 3415f03) the crossing is immediate: MIDIAN 1,010 + 9t vs a framework 1,000 + 12t crosses at t = " + f"{crossing(t, 'midian', 'fw_autogen', 'build_msgs', 'msgs_per_task'):.0f}; MIDIAN-V (1,010 + 5t) at t = {crossing(t, 'midian_v', 'fw_autogen', 'build_msgs', 'msgs_per_task'):.0f}. "
           "In COMPARISONS MIDIAN pays 60 per task (30 descent + 30 observe-time update) vs a framework's 10 (MIDIAN-V 31 = 1 cached pick + 30 update; halving 1; flat 1,000), so no MIDIAN variant undercuts a framework on comparisons; MIDIAN-V's saving over MIDIAN is the descent, not the update. "
           "Per task, MIDIAN's cost is communication (messages and comparisons), not LLM compute: it makes no LLM call at route time."]
    md += ["", "MIDIAN's 48,000-probe build by population shape (GPU-s): " + ", ".join(f"{d} {48000 * v:,.0f}" for d, v in probe_cost.items())
           + "; the crossings scale with it (heavy_tail and bimodal cross ~3x sooner).",
           "Reading: against a one-call framework (AutoGen) MIDIAN breaks even after ~9,400 tasks on specialist (~2,900 on heavy_tail), MIDIAN-A after ~9,900; against the multi-call frameworks (CrewAI, LlamaIndex, CAMEL) after 1,900-2,300 tasks; against Magentic-One after ~990 (7B) / ~590 (14B arm). Before the crossing the framework is cheaper; after it, the probe-based methods' cost is flat while every framework's keeps growing linearly."]
    md += ["", "## Combined currencies (*)", "",
           f"(a) ENERGY, joules per event: LLM call = GPU-s x 700 W (a 7B supervisor call = {SUP*700:.1f} J; a specialist probe = {probe_cost['specialist']*700:.2f} J); message = one RPC handled in ~100 us on a ~10 W core = {J_MSG:g} J (pessimistic column: {10*J_MSG:g} J); comparison = one float compare = {J_CMP:g} J.",
           f"(b) LATENCY on the critical path per task: each sequential message hop = {RTT*1e3:.0f} ms RTT (MIDIAN: the 2·depth = 6 fetch hops; the observe-time update propagation (1 message per level) is off the critical path and excluded; MIDIAN-V 2 ms; frameworks and llm_supervisor: 2 hops + the supervisor call at its measured median latency under shared-fleet load); comparisons {T_CMP*1e9:.0f} ns each (flat's 1,000 = 10 us); a route-time probe call (verify_on_claim) {PROBE_LAT:.1f} s.", "",
           "| method | J/task at t=10k (build amortised) | J/task, pessimistic messages | of which LLM J/task | messages J/task | comparisons J/task | latency s/task |", "|---|---|---|---|---|---|---|"]
    for m, r in t.sort_values("J_per_task_t10000").iterrows():
        md.append(f"| {m} | {r.J_per_task_t10000:,.3f} | {r.J_per_task_t10000_pess:,.3f} | {(r.build_gpu_s/1e4 + r.per_task_gpu_s)*700:,.3f} | {(r.build_msgs/1e4 + r.msgs_per_task)*J_MSG:.4f} | {(r.build_cmp/1e4 + r.cmp_per_task)*J_CMP:.2e} | {r.latency_s:.4f} |")
    md += ["", "Crossings in joules (tasks after which the probe-based method's cumulative energy falls below the framework's): " + "; ".join(
           f"{a}: " + ", ".join(f"{b.replace('fw_', '')} {crossing(t, a, b, 'build_J', 'per_task_J'):,.0f}" for b in fws) for a in mids) + ".",
           "Under any sane weighting the LLM call dominates energy by 3-5 orders of magnitude (20.6 J per supervisor call vs 6e-3 J for MIDIAN's six messages and 3e-7 J for its thirty comparisons per task), so the joule crossings equal the GPU-second crossings to the task; "
           "messages dominate MIDIAN's latency (6 ms of fetch hops vs 0.6 us of comparisons) while the supervisor call dominates every framework's (0.5-19 s)."]
    open(os.path.dirname(__file__) + "/../RESULTS_energy.md", "w").write("\n".join(md) + "\n")
    T = np.logspace(2, 5, 300); fig, axes = plt.subplots(2, 2, figsize=(15, 11)); axes = axes.ravel()
    show = ["midian", "midian_a", "midian_v", 'sequential_halving{"peer_reported":true}', 'flat_probe_argmax{"online":true}', "linucb_honest", "verify_on_claim", "llm_supervisor"] + fws
    style = {"midian": ("#c0392b", 3.0), "midian_a": ("#e74c3c", 2.2), "midian_v": ("#e67e22", 2.2), "fw_autogen": ("#2980b9", 2.0), "fw_magentic_one": ("#8e44ad", 2.0)}
    panels = [("build_gpu_s", "per_task_gpu_s", 1.0, "cumulative LLM GPU-seconds"), ("build_gpu_s", "per_task_gpu_s", 700 / 3600, "cumulative Wh (700 W per H100)"),
              ("build_msgs", "msgs_per_task", 1.0, "cumulative messages"), ("build_cmp", "cmp_per_task", 1.0, "cumulative comparisons")]
    for ax, (bcol, scol, scale, yl) in zip(axes, panels):
        for m in show:
            r = t.loc[m]; col, lw = style.get(m, (None, 1.0)); y = (r[bcol] + T * r[scol]) * scale
            if y.max() <= 0: continue                                                            # zero-cost arms are off a log axis
            ax.plot(T, y, label=m, color=col, lw=lw, ls="-" if not m.startswith("fw_") or m in style else "--")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("tasks routed so far (t)"); ax.set_ylabel(yl); ax.grid(alpha=.3, which="both")
    for i, b in enumerate(("fw_magentic_one", "fw_crewai", "fw_langgraph", "fw_autogen")):           # MIDIAN's break-even points, LLM compute
        x = crossing(t, "midian", b); y = t.loc["midian", "build_gpu_s"]
        axes[0].plot([x], [y], "kx", ms=10, mew=2); axes[0].annotate(f"{b.replace('fw_', '')}: {x:,.0f} tasks", (x, y), textcoords="offset points", xytext=(4, -14 - 11 * i), fontsize=8)
    for i, a_ in enumerate(("midian", "midian_v")):                                                 # messages: MIDIAN vs a framework crosses almost at once
        x = max(crossing(t, a_, "fw_autogen", "build_msgs", "msgs_per_task"), T[0]); y = t.loc[a_, "build_msgs"] + x * t.loc[a_, "msgs_per_task"]
        axes[2].plot([x], [y], "kx", ms=10, mew=2); axes[2].annotate(f"{a_} vs any framework: t={crossing(t, a_, 'fw_autogen', 'build_msgs', 'msgs_per_task'):.0f}", (x, y), textcoords="offset points", xytext=(8, 14 + 16 * i), fontsize=8)
    axes[0].set_title("LLM compute (x = MIDIAN's break-even vs a framework)"); axes[1].set_title("energy"); axes[2].set_title("messages: frameworks 1,000+12t, MIDIAN 1,010+9t, MIDIAN-V 1,010+5t\n(2 per level to fetch + 1 per level to update; flat, halving, LinUCB send none)", fontsize=10); axes[2].set_ylim(bottom=5e2); axes[3].set_title("comparisons per task: flat 1,000, MIDIAN 60 (30 descent + 30 update), MIDIAN-V 31,\nframeworks 10, halving 1", fontsize=10)
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7, title="dashed = frameworks;\nmidian_sh/sha, flat frozen, warm-start coincide with midian/midian_a", title_fontsize=7)
    fig.suptitle("H10*  Cumulative cost vs tasks routed, n=1000 specialist, self-described channel. LLM compute is an estimate: GPU-s per call = params x (A*prompt + 5A*gen tokens),\n"
                 "A from a saturated 7B replica (34 req/s); 700 W per H100; the routed task's own execution excluded. Messages and comparisons are exact ledger counts from the rows.", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.dirname(__file__) + "/../figures/H10_runtime_energy.png", dpi=300, bbox_inches="tight")
    print(t.round(4).to_string()); print({a: {b: round(crossing(t, a, b)) for b in fws} for a in mids})
    # ---- H11: combined currencies
    fig2, ax2 = plt.subplots(1, 2, figsize=(15, 6))
    for m in show:
        r = t.loc[m]; col, lw = style.get(m, (None, 1.0)); ax2[0].plot(T, r.build_J + T * r.per_task_J, label=m, color=col, lw=lw, ls="-" if not m.startswith("fw_") or m in style else "--")
    for i, b in enumerate(("fw_magentic_one", "fw_crewai", "fw_autogen")):
        x = crossing(t, "midian", b, "build_J", "per_task_J"); y = t.loc["midian", "build_J"] + x * t.loc["midian", "per_task_J"]
        ax2[0].plot([x], [y], "kx", ms=10, mew=2); ax2[0].annotate(f"{b.replace('fw_', '')}: {x:,.0f}", (x, y), textcoords="offset points", xytext=(4, -14 - 11 * i), fontsize=8)
    ax2[0].set_xscale("log"); ax2[0].set_yscale("log"); ax2[0].set_xlabel("tasks routed so far (t)"); ax2[0].set_ylabel("cumulative joules (LLM + messages + comparisons)"); ax2[0].grid(alpha=.3, which="both")
    ax2[0].set_title("one energy currency: LLM call = GPU-s x 700 W, message 1e-3 J, comparison 1e-8 J")
    lat = t.loc[show, "latency_s"].sort_values(); ax2[1].barh(range(len(lat)), lat.values, color=[style.get(m, ("#7f8c8d", 1))[0] or "#7f8c8d" for m in lat.index])
    ax2[1].set_yticks(range(len(lat))); ax2[1].set_yticklabels(lat.index, fontsize=8); ax2[1].set_xscale("log"); ax2[1].set_xlabel("critical-path latency per task (s)"); ax2[1].grid(axis="x", alpha=.3, which="both")
    ax2[1].set_title("latency: 1 ms per sequential hop, 10 ns per comparison, supervisor call at measured median")
    h2, l2 = ax2[0].get_legend_handles_labels(); fig2.legend(h2, l2, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)
    fig2.suptitle("H11*  Combined currencies, n=1000 specialist: cumulative joules vs tasks routed (x = MIDIAN's break-even) and per-task critical-path latency", fontsize=10)
    plt.tight_layout(); fig2.savefig(os.path.dirname(__file__) + "/../figures/H11_joules_latency.png", dpi=300, bbox_inches="tight")
