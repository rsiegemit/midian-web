"""rte.run.load_config: configs/grids/ layout, named sets, the duplicate-key loader, and D1 row ids.

The repo-config tests are cheap (no cells are enumerated); the full enumeration gate is
scripts/checks/grid_fingerprint.py against tests/golden/grid_fingerprints.tsv + grid_fingerprints_d1.tsv."""
import os
import re

import pytest
import yaml

from rte import run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = """\
defaults: {backend: bernoulli, K: [16], liar_select: [random], collude: [true], declared_source: [programmatic],
           lie_mode: [inflate], demand: [uniform], b: [3], Q: 100, seeds: 1-2, exclude: []}
_sets:
  cells: {n: [100], dist: [specialist], beta: [0, 0.5]}
  pair: [random, {name: midian, params: {audit: false}}]
  one: {name: flat_probe_argmax, params: {online: true}}
  both: [{set: pair}, declared_argmax]
"""


def write(tmp_path, files):
    d = tmp_path / "grids"
    d.mkdir()
    for name, text in files.items():
        (d / name).write_text(text)
    return str(d)


def specs(cfg, grid):
    return [run.method_specs(b) for b in run.blocks(cfg, grid)]


def test_set_reference_expands_like_a_yaml_alias(tmp_path):
    """{set: X} in a method list == *X there (a nested list, flattened one level by method_specs); a set may use a
    set (`both`), as the whole method list, like fw10_backfill."""
    alias = yaml.safe_load("""
defaults: {backend: bernoulli, K: [16], liar_select: [random], collude: [true], declared_source: [programmatic],
           lie_mode: [inflate], demand: [uniform], b: [3], Q: 100, seeds: 1-2, exclude: []}
_sets:
  pair: &pair [random, {name: midian, params: {audit: false}}]
  one: &one {name: flat_probe_argmax, params: {online: true}}
grids:
  g: {n: [100], dist: [specialist], beta: [0], methods: [*pair, *one, declared_argmax]}
  h: {n: [100], dist: [specialist], beta: [0], methods: [*pair, declared_argmax]}
""")
    d = write(tmp_path, {"00_base.yaml": BASE, "a.yaml": """grids:
  g: {n: [100], dist: [specialist], beta: [0], methods: [{set: pair}, {set: one}, declared_argmax]}
  h: {n: [100], dist: [specialist], beta: [0], methods: {set: both}}
"""})
    cfg = run.load_config(d)
    assert cfg["grids"]["g"]["methods"] == alias["grids"]["g"]["methods"]
    assert cfg["grids"]["h"] == alias["grids"]["h"]
    assert specs(cfg, "g") == specs(alias, "g") and len(specs(cfg, "g")[0]) == 4
    assert specs(cfg, "h") == specs(alias, "h") and len(specs(cfg, "h")[0]) == 3


def test_use_merges_like_yaml_merge_key(tmp_path):
    """`use: X` == `<<: *X`: the set's keys first, the grid's own keys win."""
    alias = yaml.safe_load("""
_sets: {cells: &cells {n: [100], dist: [specialist], beta: [0, 0.5]}}
grids: {g: {<<: *cells, beta: [0.25], methods: [random]}}
""")
    grid = "grids:\n  g: {use: cells, beta: [0.25], methods: [random]}\n"
    d = write(tmp_path, {"00_base.yaml": BASE, "a.yaml": grid})
    assert run.load_config(d)["grids"]["g"] == alias["grids"]["g"]


def test_mirror_across_files(tmp_path):
    d = write(tmp_path, {"00_base.yaml": BASE,
                         "a.yaml": "grids:\n  src: {use: cells, methods: {set: pair}}\n",
                         "b.yaml": "grids:\n  twin: {mirror_of: src, beta: [0.5]}\n"})
    cfg = run.load_config(d)
    (blk,) = run.blocks(cfg, "twin")
    assert blk["beta"] == [0.5] and blk["n"] == [100]
    assert [s["name"] for s in run.method_specs(blk)] == ["random", "midian"]


@pytest.mark.parametrize("files, err", [
    ({"a.yaml": "grids:\n  g: {use: cells, methods: [random]}\n  g: {use: cells}\n"}, "duplicate key"),
    ({"a.yaml": "grids:\n  g: {use: cells, methods: [random], methods: [midian]}\n"}, "duplicate key"),
    ({"a.yaml": "grids:\n  g: {use: cells, methods: [random]}\n",
      "b.yaml": "grids:\n  g: {use: cells, methods: [random]}\n"}, "defined twice"),
    ({"a.yaml": "defaults: {Q: 5}\ngrids: {}\n"}, "only `grids:`"),
    ({"a.yaml": "grids:\n  g: {use: cells, methods: [{set: nope}]}\n"}, "unknown set"),
])
def test_rejects(tmp_path, files, err):
    d = write(tmp_path, {"00_base.yaml": BASE, **files})
    with pytest.raises((yaml.YAMLError, ValueError, KeyError), match=err):
        run.load_config(d)


def test_row_ids_do_not_depend_on_rte_data(tmp_path, monkeypatch):
    """D1: the id hashes "$RTE_DATA/..." as written; only expand() (the World's view) sees the path."""
    blk = {**yaml.safe_load(BASE)["defaults"], "n": [100], "dist": ["specialist"], "beta": [0.0],
           "backend_kwargs": {"calibrate_from": "$RTE_DATA/S.npy"}}
    blk = run.blocks({"defaults": blk, "grids": {"g": {}}}, "g")[0]
    ids = []
    for root in ("one", "two"):
        (tmp_path / root).mkdir()
        (tmp_path / root / "S.npy").write_bytes(b"")
        monkeypatch.setattr(run, "RTE_DATA", str(tmp_path / root))
        (cell,) = run.cells(blk)
        assert cell["backend_kwargs"] == {"calibrate_from": "$RTE_DATA/S.npy"}
        assert run.expand(cell["backend_kwargs"]) == {"calibrate_from": str(tmp_path / root / "S.npy")}
        ids.append(run.row_id(cell, "random", {}, 1))
        stored = {**{f: cell[f] for f in run.CELL}, "backend_kwargs": run.jkey(cell["backend_kwargs"]),
                  "method": "random", "params": "{}", "seed": 1}
        assert run.rid_of_row(stored) == ids[-1]
    assert ids[0] == ids[1]


# ---------------------------------------------------------------- the repo's config
@pytest.fixture(scope="module")
def cfg():
    return run.load_config()


def test_repo_config_resolves(cfg):
    """Every grid yields blocks and method specs; no grid still says `methods: all`."""
    for g in cfg["grids"]:
        for blk in run.blocks(cfg, g):
            assert blk["methods"] != "all", g
            assert run.method_specs(blk), g


def test_flat_grid_yaml_in_sync(cfg):
    """configs/grid.yaml is generated from configs/grids/ (python configs/export_grid_yaml.py)."""
    flat = run.load_config(os.path.join(ROOT, "configs", "grid.yaml"))
    assert flat["grids"] == cfg["grids"] and flat["defaults"] == cfg["defaults"]


def test_every_grid_has_a_header_tag():
    first = r"(FIG:[A-I](,[A-I])*|NUM|HIST|SUPERSEDED-BY:\w+|PENDING|SMOKE)"
    tag = re.compile(r"^  # \[" + first + r"( \| [^\]]+)?\] .+; read by: .+")
    for f in sorted(os.listdir(os.path.join(ROOT, "configs", "grids")))[1:]:
        lines = open(os.path.join(ROOT, "configs", "grids", f)).read().split("\n")
        for i, ln in enumerate(lines):
            if re.match(r"^  \w+:", ln):
                assert tag.match(lines[i - 1]), f"{f}: {ln.split(':')[0].strip()} has no header tag line"
