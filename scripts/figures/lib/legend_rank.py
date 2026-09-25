"""The value-ranked legend of the exploratory figures (figures/bars, figures/shortlist): entries ranked best first by
the value each one plots, laid out ROW-MAJOR -- best top left, then left to right, then the next row. matplotlib fills
legend columns top to bottom, so the ranked list is permuted to read row-major. Entries that plot nothing (style keys
such as "hollow = honest") keep their order after the ranked ones. rank="asc" where lower is better (cost axes);
rank=None keeps the given order (still row-major).

Opt-in: install() patches Axes.legend and Figure.legend for the whole process; nothing happens at import. The paper
figures never use it (figspec.legend builds its Legend directly, in the fixed figspec order).
"""

from __future__ import annotations

import numpy as np


def _plotted(h):
    from matplotlib.collections import LineCollection, PathCollection
    from matplotlib.container import BarContainer, ErrorbarContainer

    try:
        if isinstance(h, BarContainer):
            v = [p.get_height() for p in h.patches]
        elif isinstance(h, ErrorbarContainer):
            v = list(h[0].get_ydata()) if h[0] is not None else []
        elif isinstance(h, LineCollection):
            v = [y for s in h.get_segments() for _, y in s]
        elif isinstance(h, PathCollection):
            v = [y for _, y in h.get_offsets()]
        elif hasattr(h, "get_ydata"):
            v = list(h.get_ydata())
        else:
            v = []
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else None
    except Exception:
        return None


def _rowmajor(items, ncol):
    n = len(items)
    ncol = max(1, min(ncol, n))
    q, rem = divmod(n, ncol)
    rows = [q + 1] * rem + [q] * (ncol - rem)
    return [items[r * ncol + c] for c in range(ncol) for r in range(rows[c])]


def _ranked(orig):
    def legend(self, *args, rank="desc", **kw):
        if len(args) == 1:
            return orig(self, *args, **kw)  # labels only: nothing to pair with
        if len(args) >= 2:
            hs, ls, args = list(args[0]), list(args[1]), args[2:]
        elif "handles" in kw:
            hs = list(kw.pop("handles"))
            ls = list(kw.pop("labels", [h.get_label() for h in hs]))
        else:
            hs, ls = self.get_legend_handles_labels() if hasattr(self, "get_legend_handles_labels") else ([], [])
        if not hs:
            return orig(self, *args, **kw)
        items = list(zip(hs, ls))
        if rank:
            val = [_plotted(h) for h, _ in items]
            data = sorted(
                [i for i, v in enumerate(val) if v is not None], key=lambda i: val[i], reverse=(rank == "desc")
            )
            items = [items[i] for i in data] + [it for i, it in enumerate(items) if val[i] is None]
        items = _rowmajor(items, kw.get("ncol", kw.get("ncols", 1)))
        return orig(self, [h for h, _ in items], [lab for _, lab in items], *args, **kw)

    return legend


def install():
    """Patch Axes.legend and Figure.legend with the ranked legend (idempotent)."""
    import matplotlib.axes
    import matplotlib.figure

    if getattr(matplotlib.axes.Axes.legend, "_rte_ranked", False):
        return
    for cls in (matplotlib.axes.Axes, matplotlib.figure.Figure):
        cls.legend = _ranked(cls.legend)
        cls.legend._rte_ranked = True
