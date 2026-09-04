# Figure colours (fixed; reuse in every new figure)

Assignments inherited from `scripts/extra_figs.py` (H1–H9) and used unchanged by `scripts/paper_figs.py`:

| method | hex | note |
|---|---|---|
| oracle | #999999 | grey, dotted line |
| random | #cccccc | light grey, dashed line |
| MIDIAN | #c0392b | red |
| MIDIAN-V | #e67e22 | orange |
| MIDIAN-V r=5 | #f1c40f | yellow |
| MIDIAN-A | #7b241c | dark red |
| MIDIAN-VA | #2ecc71 | green, drawn bold |
| MIDIAN-SH / SHA | #d35400 / #e74c3c | |
| flat probe argmax (online) | #3498db | blue |
| flat probe argmax (frozen) | #7f8c8d | |
| sequential halving, peer-reported | #8e44ad | purple |
| sequential halving, trusted | #2c3e50 | midnight blue |
| warm-start bandit | #27ae60 | |
| LinUCB (honest) | #16a085 | |
| declared argmax | #5d6d7e | slate, dashed (report-free) |
| LLM supervisor | #34495e | |
| AutoGen / Magentic-One 7B / 14B arm | #2980b9 / #1f618d / #5dade2 | framework blues (H1 band #2980b9) |

Added 2026-09-04 for the paper figures (H1 fixes no colour for the learned routers; the remaining flat-UI entries
of the H1 palette — turquoise #1abc9c, amethyst #9b59b6, orange #f39c12 — were checked and the last two clash with
the halving-peer purple and the MIDIAN-V orange, so the second choice is a brown outside the palette):

| method | hex | note |
|---|---|---|
| RouterBench KNN router | #1abc9c | turquoise, dashed (report-free) |
| RouterBench MLP router | #795548 | brown, dashed (report-free) |
| ten-framework band | #bbbbbb at 45 % alpha | M1 |

Line-style convention in M1 / M2: MIDIAN family solid; report-free methods (flat, KNN, MLP, declared) dashed; oracle
dotted; random dashed light grey. Simulation points (scale_100k) are hollow markers joined by dashed lines.
