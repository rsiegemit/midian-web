"""Liar regimes: (beta, liar_select) -> a key, and the key's names in every place that prints one.

    tag(beta, ls)    beta0 | cartel (beta 0.5 low-skill-first) | beta<x>_random | beta<x>_cartel     (every beta)
    ab(beta, ls)     beta0 | cartel | None: the two regimes the condensed A / B figures draw
    matrix_title(beta, ls)  the regime title a scale matrix (scripts/analysis/scale_matrix.py) writes
"""
from __future__ import annotations

REGIMES = [("beta0", 0.0, "random", "honest (β = 0)"), ("beta01_random", 0.1, "random", "β = 0.1, random liars"),
           ("beta025_random", 0.25, "random", "β = 0.25, random liars"),
           ("beta025_cartel", 0.25, "low_skill_first", "β = 0.25, low-skill cartel"),
           ("beta05_random", 0.5, "random", "β = 0.5, random liars"), ("cartel", 0.5, "low_skill_first", "β = 0.5, low-skill cartel")]
NAME = {k: title for k, _, _, title in REGIMES}
AB = {"beta0": "honest", "cartel": "β=0.5 cartel"}          # the two regimes of A / B, as their CSVs name them


def tag(beta, ls):
    """Regime key of one (beta, liar_select) cell; beta = 0 is liar-free whatever the selection."""
    beta = float(beta)
    if beta == 0:
        return "beta0"
    t = "cartel" if ls == "low_skill_first" else "random"
    return "cartel" if (beta == 0.5 and t == "cartel") else f"beta{beta:g}_{t}".replace(".", "")


def ab(beta, ls):
    """beta0 / cartel, None for every other regime."""
    return "beta0" if float(beta) == 0 else ("cartel" if float(beta) == 0.5 and ls == "low_skill_first" else None)


def matrix_title(beta, ls):
    """The regime title of a scale matrix row: 'beta=0 (no liars)', 'beta=0.5 CARTEL (low-skill-first)', ..."""
    if beta == 0:
        return "beta=0 (no liars)"
    return f"beta={beta:g} {'CARTEL (low-skill-first)' if ls == 'low_skill_first' else 'random liars'}"


MATRIX_REGIME = {k: matrix_title(b, ls) for k, b, ls, _ in REGIMES if k != "beta01_random"}   # the matrices have no beta 0.1
