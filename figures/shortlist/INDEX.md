# figures/shortlist -- every framework under every shortlist source, per condition

One figure per (family, n, population shape, liar regime); one panel and one row. The oracle (dotted) and MIDIAN-VA routing
the whole population (solid) are horizontal lines with their 95% seed-bootstrap band; each framework group carries one bar per
shortlist source. Error bars are the 95% seed bootstrap; * marks a bar with an erratum-28 rerun outstanding. b = 3 only.
Do-not-add arms (extra_figs.excluded) are never drawn.

Shortlist sources: `tfidf` = hashed TF-IDF (pre-registered); `bm25` = BM25; `embed` = MiniLM; `dense` = Qwen3-8B dense; `dense_icomp` = Qwen3-8B dense, I-competent; `dense_idemo` = Qwen3-8B dense, I-demonstrated; `sota` = fusion + reranker; `sota_icomp` = fusion + reranker, I-competent; `sota_idemo` = fusion + reranker, I-demonstrated; `declared` = declared-claim top-k; `va_cohort` = MIDIAN-VA leaf cohort.


## live
- `live__n100__bimodal__beta0.png`
- `live__n100__bimodal__beta01_random.png`
- `live__n100__bimodal__beta025_random.png`
- `live__n100__bimodal__beta05_random.png`
- `live__n100__bimodal__cartel.png`
- `live__n100__heavy_tail__beta0.png`
- `live__n100__heavy_tail__beta01_random.png`
- `live__n100__heavy_tail__beta025_random.png`
- `live__n100__heavy_tail__beta05_random.png`
- `live__n100__heavy_tail__cartel.png`
- `live__n100__specialist__beta0.png`
- `live__n100__specialist__beta01_random.png`
- `live__n100__specialist__beta025_random.png`
- `live__n100__specialist__beta05_random.png`
- `live__n100__specialist__cartel.png`
- `live__n1000__bimodal__beta0.png`
- `live__n1000__bimodal__beta01_random.png`
- `live__n1000__bimodal__beta025_random.png`
- `live__n1000__bimodal__beta05_random.png`
- `live__n1000__bimodal__cartel.png`
- `live__n1000__heavy_tail__beta0.png`
- `live__n1000__heavy_tail__beta01_random.png`
- `live__n1000__heavy_tail__beta025_random.png`
- `live__n1000__heavy_tail__beta05_random.png`
- `live__n1000__heavy_tail__cartel.png`
- `live__n1000__specialist__beta0.png`
- `live__n1000__specialist__beta01_random.png`
- `live__n1000__specialist__beta025_random.png`
- `live__n1000__specialist__beta05_random.png`
- `live__n1000__specialist__cartel.png`
- `live__n10000__specialist__beta0.png`
- `live__n10000__specialist__beta025_random.png`
- `live__n10000__specialist__cartel.png`
- `live__n100000__specialist__beta0.png`
- `live__n100000__specialist__beta025_cartel.png`
- `live__n100000__specialist__beta025_random.png`
- `live__n100000__specialist__beta05_random.png`
- `live__n100000__specialist__cartel.png`

## routereval
- `routereval__n10__all_strong__beta0.png`
- `routereval__n10__all_strong__beta05_random.png`
- `routereval__n10__all_strong__cartel.png`
- `routereval__n10__all_weak__beta0.png`
- `routereval__n10__all_weak__beta05_random.png`
- `routereval__n10__all_weak__cartel.png`
- `routereval__n10__strong_to_weak__beta0.png`
- `routereval__n10__strong_to_weak__beta05_random.png`
- `routereval__n10__strong_to_weak__cartel.png`
- `routereval__n100__all_strong__beta0.png`
- `routereval__n100__all_strong__beta05_random.png`
- `routereval__n100__all_strong__cartel.png`
- `routereval__n100__all_weak__beta0.png`
- `routereval__n100__all_weak__beta05_random.png`
- `routereval__n100__all_weak__cartel.png`
- `routereval__n100__strong_to_weak__beta0.png`
- `routereval__n100__strong_to_weak__beta05_random.png`
- `routereval__n100__strong_to_weak__cartel.png`
- `routereval__n1000__all_strong__beta0.png`
- `routereval__n1000__all_strong__beta05_random.png`
- `routereval__n1000__all_strong__cartel.png`
- `routereval__n1000__all_weak__beta0.png`
- `routereval__n1000__all_weak__beta05_random.png`
- `routereval__n1000__all_weak__cartel.png`
- `routereval__n1000__strong_to_weak__beta0.png`
- `routereval__n1000__strong_to_weak__beta05_random.png`
- `routereval__n1000__strong_to_weak__cartel.png`
- `routereval__n5000__all__beta0.png`
- `routereval__n5000__all__beta05_random.png`
- `routereval__n5000__all__cartel.png`
