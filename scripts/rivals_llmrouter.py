"""GraphRouter (Feng et al., ICLR 2025) through the released LLMRouter library, on RouterBench's part-A splits and on a
RouterEval pool — TARGETS_rte_v3.md part D3.   $RTE_DATA/env/llmrouter/bin/python scripts/rivals_llmrouter.py [routerbench|routereval] [n_splits]

Their pipeline, unchanged: routing JSONL (one row per (query, model) with `performance` and an `embedding_id`), a
query-embedding tensor, an LLM json (name -> {feature, embedding}), a yaml config; `GraphRouter(yaml)` builds the
query/LLM graph, `GraphTrainer.train()` fits the GNN (their defaults: hidden 64, AdamW 1e-3, 100 epochs, 4 masked
samples per step, mask rate 0.3, 20% validation). Scoring: their `route_single` re-embeds each query with Longformer
and appends it to the training graph; here every test query is appended at once with the SAME embedding model the
graph was built with (MiniLM on RouterBench, RouterEval's RoBERTa on RouterEval) and scored by `GNNPredictor.predict`;
LLM node features are their random init (no description embeddings; seeded)
— the batched form of their `route_single`, without the embedding-model mismatch. RouterDC fine-tunes a DeBERTa
encoder (GPU): NOT RUN. Output: results/rivals_llmrouter/summary.md (quality at λ = 0 next to part A's routers)."""
import os, sys, json, pickle, warnings, numpy as np, pandas as pd, torch, yaml
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from routerbench_terms import data, EMB, SEEDS, TEST, R
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from llmrouter.models.graphrouter.router import GraphRouter
from llmrouter.models.graphrouter.trainer import GraphTrainer

OUT = f"{R}/results/rivals_llmrouter"; os.makedirs(OUT, exist_ok=True)
HP = {"hidden_dim": 64, "learning_rate": 0.001, "weight_decay": 0.0001, "train_epoch": 100, "batch_size": 4, "train_mask_rate": 0.3, "val_split_ratio": 0.2, "random_state": 42}


def fit_and_score(work, prompts_tr, perf_tr, E_tr, prompts_te, E_te, models, task_tr):
    """Write their inputs, train GraphRouter, score every test prompt (batched route_single). Returns picks (n_te,)."""
    os.makedirs(work, exist_ok=True); n_tr, m = perf_tr.shape
    with open(f"{work}/routing_train.jsonl", "w") as f:
        for i in range(n_tr):
            for j in range(m):
                f.write(json.dumps({"task_name": str(task_tr[i]), "query": prompts_tr[i], "ground_truth": "", "metric": "em", "model_name": models[j],
                                    "response": "", "performance": float(perf_tr[i, j]), "embedding_id": i, "token_num": 0}) + "\n")
    torch.save(torch.tensor(E_tr, dtype=torch.float32), f"{work}/query_embeddings.pt")
    json.dump({mn: {"size": "", "feature": mn} for mn in models}, open(f"{work}/llm.json", "w"))   # no description embeddings: their code random-inits LLM node features (seeded below)
    cfg = {"data_path": {"routing_data_train": f"{work}/routing_train.jsonl", "query_embedding_data": f"{work}/query_embeddings.pt", "llm_data": f"{work}/llm.json", "llm_embedding_data": f"{work}/llm.json"},
           "model_path": {"ini_model_path": "", "save_model_path": f"{work}/graphrouter.pt", "load_model_path": f"{work}/graphrouter.pt"},
           "metric": {"weights": {"performance": 1, "cost": 0, "llm_judge": 0}}, "hparam": HP}
    yaml.safe_dump(cfg, open(f"{work}/config.yaml", "w"))
    np.random.seed(0); torch.manual_seed(0); router = GraphRouter(f"{work}/config.yaml"); GraphTrainer(router).train()
    router.gnn_predictor.model.load_state_dict(torch.load(f"{work}/graphrouter.pt", map_location="cpu"))
    # batched route_single: their scaled train embeddings + the test embeddings scaled by the train scaler; zero edges for test
    order = [int(router.routing_data_train[router.routing_data_train["query"] == q]["embedding_id"].iloc[0]) for q in router.routing_data_train["query"].unique()]
    scaler = MinMaxScaler().fit(E_tr[order]); q_all = np.vstack([router.query_embedding_list, scaler.transform(E_te)])
    n_all, L = len(q_all), router.num_llms
    perf_all = np.concatenate([router.performance_list, np.zeros(len(E_te) * L)])
    org = [q for q in range(n_all) for _ in range(L)]; des = list(range(L)) * n_all
    label = np.eye(L)[np.argmax(perf_all.reshape(n_all, L), 1)].flatten().reshape(-1, 1)
    mask_tr = torch.zeros(n_all * L); mask_tr[: len(router.performance_list)] = 1; mask_te = torch.zeros(n_all * L); mask_te[len(router.performance_list):] = 1
    d = router.form_data.formulation(query_feature=q_all, llm_feature=router.llm_embedding, org_node=org, des_node=des, edge_feature=perf_all, label=label,
                                     edge_mask=mask_te, train_mask=mask_tr, valide_mask=torch.zeros(n_all * L), test_mask=mask_te)
    return router.gnn_predictor.predict(d).cpu().numpy()[-len(E_te):], router.model_names


def routerbench(n_splits):
    prompts, fam, perf, cost, M = data(); E = np.load(EMB); rows = []
    for seed, (tr, te) in enumerate(StratifiedShuffleSplit(n_splits=SEEDS, test_size=TEST, random_state=0).split(E, fam)):
        if seed >= n_splits: break
        pick, names = fit_and_score(f"{OUT}/routerbench_split{seed}", list(prompts[tr]), perf[tr], E[tr], list(prompts[te]), E[te], M, fam[tr])
        idx = np.array([M.index(n) for n in names])[pick]                        # router's model order -> RouterBench column order
        q, c = perf[te][np.arange(len(te)), idx].mean(), cost[te][np.arange(len(te)), idx].mean()
        rows.append({"seed": seed, "router": "GraphRouter (ICLR25, LLMRouter lib)", "quality_at_lambda0": q, "cost": c}); print(f"[routerbench split {seed}] quality {q:.4f} cost {c:.5f}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/routerbench_rows.csv", index=False)
    a = pd.read_csv(f"{R}/results/routerbench_terms/aiq_rows.csv"); a = a[a.seed < n_splits].groupby("router").quality_at_lambda0.mean()
    md = [f"# GraphRouter via LLMRouter on RouterBench's part-A splits ({n_splits} splits), quality at λ = 0 (performance-only routing)", "", "| router | quality at λ=0 |", "|---|---|",
          f"| GraphRouter (ICLR25) | {df.quality_at_lambda0.mean():.4f} |"] + [f"| {r} | {v:.4f} |" for r, v in a.sort_values(ascending=False).items() if r in ("oracle", "knn", "mlp", "probe_family_b50", "probe_family_b20", "knn_b20", "zero_router")]
    open(f"{OUT}/summary_routerbench.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))


def routereval(m=100, cfg="strong_to_weak", n_train=2000):
    d = pickle.load(open(f"{R}/data/routereval/router_dataset/mmlu_router_dataset.pkl", "rb")); c = d["hard"][m][cfg]["data"]
    Ytr, Yte = np.asarray(c["train_score"], float), np.asarray(c["test_score"], float); Etr, Ete = np.asarray(d["embedding"]["train_embed"]), np.asarray(d["embedding"]["test_embed"])
    Ptr, Pte = [str(p) for p in d["prompt"]["train_prompt"]], [str(p) for p in d["prompt"]["test_prompt"]]
    sub = np.random.default_rng(0).choice(len(Ytr), size=min(n_train, len(Ytr)), replace=False)     # their per-query pandas filtering is O(N^2): subsample the train split (stated)
    models = [f"llm{j}" for j in range(Ytr.shape[1])]
    pick, names = fit_and_score(f"{OUT}/routereval_mmlu_m{m}_{cfg}", [Ptr[i] for i in sub], Ytr[sub], Etr[sub], Pte, Ete, models, ["mmlu"] * len(sub))
    idx = np.array([models.index(n) for n in names])[pick]; mu = Yte[np.arange(len(Yte)), idx].mean(); bsm = Yte.mean(0).max()
    line = f"| GraphRouter (ICLR25), mmlu m={m} {cfg}, {len(sub)} train queries | μ {mu:.4f} | V_B {mu / bsm:.3f} | best single {bsm:.4f} | oracle {Yte.max(1).mean():.4f} |"
    open(f"{OUT}/summary_routereval.md", "a").write(line + "\n"); print(line)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "routerbench"
    if what == "routerbench": routerbench(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    else: routereval(int(sys.argv[2]) if len(sys.argv) > 2 else 100)
