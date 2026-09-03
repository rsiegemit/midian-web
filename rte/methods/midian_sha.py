"""MIDIAN-SH+A (labeled variant, 2026-09-03): successive halving inside cohorts (midian_sh) AND report audits with
reporter exclusion (midian_a). Everything is inherited; this file only turns halving on in MIDIAN-A."""
from .midian_a import MidianA


class MidianSHA(MidianA):
    name = "midian_sha"

    def __init__(self, halving=True, **p):
        super().__init__(halving=halving, **p)
