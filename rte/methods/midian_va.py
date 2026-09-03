"""MIDIAN-VA (labeled variant, 2026-09-03): MIDIAN-V's verification-at-promotion and cached root pick PLUS MIDIAN-A's
report audits with reporter exclusion. Level 0 is MIDIAN-A's audited engine (b-1 probes per cell, 5% of instances re-run,
reporters excluded at two mismatches); the promotion re-probes then aggregate over non-excluded sibling-subtree reporters
(`Midian._verify` passes the exclusion mask); the root pick is cached, so a route costs 1 comparison and 2 messages.
Cost vs MIDIAN-V: +audit probes (<= 5% of build) and A's online audits (5% of routed outcomes, s-1 reports each).
Pre-registered as V2-11 in TARGETS_rte_v2.md."""
from .midian_a import MidianA


class MidianVA(MidianA):
    name = "midian_va"

    def __init__(self, **p):
        super().__init__(verify=True, cached=True, **p)
