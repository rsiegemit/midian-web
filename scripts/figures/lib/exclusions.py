"""The arms no figure ever draws, in one place (user directives 2026-09-17 and 2026-09-22).

    from scripts.figures.lib.exclusions import excluded
    labels = [l for l in labels if not excluded(l)]

Permanent exclusions:
  - every sequential-halving arm (2026-09-22). The trusted-observer arm reads a signal no deployed router has
    (erratum 26); the peer-reported arm was dropped from the figures with it. Halving numbers stay in the tables.
  - MIDIAN variants with r != 10 or delta != 1/3 (midian[...,r=5,...], midian_wo_audit_r5, midian[r=20], ...).
  - MIDIAN cohort-mode (v4, their own tables), churn-mode and online-off variants.
  - route_to_k_majority: executes THREE agents per task and majority-votes, not one execution per task like every
    other arm.
  - midian_llm_descent: the SPEC section 9 descent ablation (appendix only).
"""

from __future__ import annotations

import re

DO_NOT_ADD = {
    "route_to_k_majority",
    "midian_llm_descent",
    "midian[audit=False,online=False,verify=False]",
}  # online updates off: internals ablation
_R = re.compile(r"(?:\[|,)r=(\d+)|_r(\d+)$")
_DELTA = re.compile(r"delta=([0-9.]+)")
_VARIANT = re.compile(r"cohort=|stratify=True|churn_mode=|online=False")


def excluded(label) -> bool:
    """True for an arm on the do-not-add list (module docstring)."""
    label = str(label)
    if label in DO_NOT_ADD or "halving" in label.lower():
        return True
    if label.startswith("midian"):
        if _VARIANT.search(label):
            return True
        m = _R.search(label)
        if m and int(m.group(1) or m.group(2)) != 10:
            return True
        d = _DELTA.search(label)
        if d and abs(float(d.group(1)) - 1 / 3) > 1e-6:
            return True
    return False
