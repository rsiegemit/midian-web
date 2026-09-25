"""The exploratory figures' legend rule (scripts/figures/lib/legend_rank.py): entries ranked best first by what they
plot, read row-major."""
import os, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.figures.lib.legend_rank import _rowmajor, install  # noqa: E402

install()                                   # opt-in: importing the module changes nothing


def _grid(leg, ncol):
    """legend texts as displayed: matplotlib fills columns top to bottom."""
    t = [x.get_text() for x in leg.get_texts()]; n = len(t); q, rem = divmod(n, ncol)
    rows = [q + 1] * rem + [q] * (ncol - rem); cols, i = [], 0
    for r in rows: cols.append(t[i:i + r]); i += r
    return [[c[r] for c in cols if r < len(c)] for r in range(max(rows))]


def test_best_is_top_left_and_reads_row_major():
    fig, ax = plt.subplots()
    for i, v in enumerate([0.3, 0.9, 0.5, 0.7, 0.1]): ax.bar([i], [v], label=f"v{v}")
    ax.bar([], [], hatch="//", label="style key")
    rows = _grid(ax.legend(ncol=3), 3)
    assert rows == [["v0.9", "v0.7", "v0.5"], ["v0.3", "v0.1", "style key"]]


def test_ascending_for_costs_and_uneven_last_row():
    fig, ax = plt.subplots()
    for v in (5, 1, 3, 2): ax.plot([0, 1], [v, v], label=f"c{v}")
    assert _grid(ax.legend(ncol=3, rank="asc"), 3) == [["c1", "c2", "c3"], ["c5"]]


def test_rowmajor_permutation():
    assert _rowmajor(list("abcde"), 2) == list("acebd")
