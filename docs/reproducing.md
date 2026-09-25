# Reproducing the paper

Four tiers, from minutes on a laptop to the full live campaign. Each tier checks the one below it: redrawn figures must
match the shipped figure CSVs, aggregates recomputed from rows must match the shipped aggregates, and rerun rows must
match the stored rows. All commands run from the repository root with `$RTE_DATA` set
([README](../README.md#data)).

| tier | needs | time | reproduces |
|---|---|---|---|
| 1. Redraw | `pip install -e ".[figures]"` | minutes | every figure, from the shipped aggregates in `results/aggregates/` |
| 2. Aggregate | the stored result rows | hours | the figure CSVs and `paper/NUMBERS.json`, from rows |
| 3. CPU grids | `[learned]` extra, the downloaded datasets | minutes to CPU-days | every non-live row: bernoulli, replay, RouterEval, LLMRouterBench |
| 4. Live | a GPU fleet, the model weights, the framework environments | GPU-weeks | the live rows, from the model answers |

## Tier 1: redraw every figure from the shipped CSVs

```bash
python scripts/figures/make_all.py --from-csv
```

Each script in `scripts/figures/` writes a figure and the CSV of every value it plots (`figures/paper/<name>.csv`).
With `--from-csv`, `make_all.py` skips the computation and draws each figure from `results/aggregates/` alone:
`bars/` and `shortlist/` for A, B and E-H, `cost_by_n.csv` for C, and `figures/` for the drawn values of A, B, D and I
(Figure I from `figures/I_max_lie.csv` and its oracle lines in `figures/refs.csv`). The output depends only on the
shipped aggregates and the style module (`scripts/figures/lib/figspec.py`); no result row and no `$RTE_DATA` is read.
Compare with the shipped PDFs, or draw into another directory (`--out DIR`, or `RTE_FIG_OUT=DIR`) and diff.

## Tier 2: figure CSVs and quoted numbers from the stored rows

The stored rows (4.5 M method rows under `$RTE_DATA/results/<grid>/`) are not in the repository. With them:

```bash
python scripts/figures/bar_figs.py            # rows -> results/aggregates/bars/*.csv (b = 3 bars of every family)
python scripts/figures/shortlist_figs.py      # rows -> results/aggregates/shortlist/{live,routereval}.csv
python scripts/figures/make_all.py --out new  # the whole chain (the two above included) -> new/*
python scripts/analysis/paper_numbers.py      # every quoted number -> paper/NUMBERS.json (value, grid, units, CI)
python scripts/checks/figure_csvs.py figures/paper new   # value-identical to the shipped figure CSVs (atol 1e-12)?
```

Scripts that read millions of rows need a machine with enough memory (the largest grids hold over a million rows).
`C` and `D` read the ledger of `bernoulli_scale_v5` from `results/aggregates/cost_by_n.csv`; the full `make_all.py`
run re-reads it from the rows (`efficiency_figs.py --refresh`). Figure A's
per-seed tables and cross-fitted pools are built from the rows of every grid listed in
[figures.md](figures.md#figures-and-their-grids).

## Tier 3: the non-live rows

Every bernoulli, replay, RouterEval and LLMRouterBench grid runs on CPU:

```bash
python -m midian.run --grid reviewer_bernoulli                              # the paper's arms, laptop-sized
RTE_WORKERS=16 python -m midian.run --grid replay_1e6_split_cal --seeds 1-3  # a Figure 2 family, a few seeds
python -m midian.run --grid bernoulli_scale_v5 --only n=100000 --seeds 1-10  # one slice of the scale grid
python -m midian.analyze --grid replay_1e6_split_cal
```

- **Data.** replay needs RouterBench (`scripts/data/02_download_routerbench.py`); `routereval` needs RouterEval's and
  LLMRouterBench's released score tables under `$RTE_DATA/data/routereval/`. The bernoulli grids that are calibrated to
  the live skill matrix read one live population's `S.npy` (`calibrate_from`), which the live tier produces.
- **Sharding.** `--only k=v[,k=v]` restricts a grid to matching cells and `--seeds a-b` to a seed range; rows are
  written per unit and existing ones are skipped, so any number of jobs can share a grid. `RTE_WORKERS=N` runs units in
  N processes.
- **Exactness.** A rerun reproduces every stored non-wall-clock column. Check a sample with
  `python scripts/checks/parity.py` (reruns stored rows per method and backend and compares) and
  `python scripts/checks/rid_check.py` (stored row ids recompute from their fields). The kNN rows of `rivals_b_n10k`,
  `rivals_b_n100k` and `routereval5k_norep_cal` were produced with batched embeddings; rerun them with
  `RTE_EMBED_BATCH=1` for bit-identical rows ([errata.md](errata.md)).
- **Scale.** The 10<sup>6</sup>-10<sup>7</sup> grids are sharded by n and seed; above n = 10<sup>5</sup> use one
  worker per job with one or two seeds.

## Tier 4: the live rows

The live backend's agents are LLM calls against a vLLM fleet serving the seven-model ladder of `configs/models.yaml`,
and the framework arms call a Qwen2.5-7B supervisor through each framework's own environment. The full procedure (fleet,
replicas, memo, sharding, and the rules for running a campaign) is [operations.md](operations.md). In short:

```bash
bash scripts/setup/00_build_env.sh                     # vLLM + reasoning-gym environment under $RTE_DATA/env/rte
python scripts/setup/01_download_weights.py            # the model ladder (internet access needed)
bash scripts/setup/fw_envs/<framework>.sh              # one isolated venv per framework
# serve the fleet (docs/operations.md), then:
python -m midian.measure --dist specialist --n 1000 --K 16 --seed 1   # build a population and measure its S
python -m midian.run --grid live_f1_n1000 --only dist=specialist --seeds 1
```

- **Memo.** Every model answer is memoised by a content hash of the full request (model, messages, max tokens; decoding is greedy) in a sharded SQLite
  store under `$RTE_DATA/cache/`. With the memo of the original campaign, rerunning a stored live cell makes no model
  calls and reproduces its rows exactly.
- **Frameworks.** Framework rows are reproducible in distribution only: the calls a framework makes inside its own
  environment are not memoised. Their rows are excluded from the exact parity check.
- **Cost.** A cold MIDIAN build at n = 10<sup>5</sup>, b = 3 took 16.6 h of the fleet; a framework unit at
  10<sup>2</sup>-10<sup>3</sup> takes minutes to hours depending on the framework.
