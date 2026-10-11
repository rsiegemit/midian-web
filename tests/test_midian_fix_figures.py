"""The 10-seed build's fixed MIDIAN: rows.swap_fixed (fixed rows drawn as MIDIAN in place of midian{} where they exist)
and grids.with_s4_10 (adds the <g>_fix mirrors only under RTE_SEEDS_4_10=1)."""
from __future__ import annotations

import pandas as pd
import pytest

import scripts.figures.lib as lib
from scripts.figures.lib import grids, seed_tables
from scripts.figures.lib.rows import swap_fixed


def row(method, params, beta, seed, success, ls="low_skill_first", **kw):
    return dict(n=1000, b=3, dist="specialist", beta=beta, liar_select=ls, declared_source="self_described",
                seed=seed, method=method, params=params, success=success, **kw)


def frame():
    return pd.DataFrame([
        row("midian", "{}", 0.0, 1, 0.80),                    # beta = 0: no fixed row, untouched
        row("midian", "{}", 0.5, 1, 0.50),                    # replaced by the fixed row below
        row("midian", '{"reagg":"keep"}', 0.5, 1, 0.70),
        row("midian", "{}", 0.5, 2, 0.55),                    # seed 2 has no fixed row: kept
        row("midian", '{"audit":false}', 0.5, 1, 0.40),       # another MIDIAN variant: untouched
        row("oracle", "{}", 0.5, 1, 0.90),
    ])


def test_substitutes_only_where_a_fixed_row_exists():
    out = swap_fixed(frame())
    mid = out[(out.method == "midian") & (out.params == "{}")]
    assert sorted(zip(mid.beta, mid.seed, mid.success)) == [(0.0, 1, 0.80), (0.5, 1, 0.70), (0.5, 2, 0.55)]
    assert set(out.params) == {"{}", '{"audit":false}'}
    assert len(out) == 5 and (out.method == "oracle").sum() == 1


def test_other_cell_axes_keep_rows_apart():
    df = frame()
    df.loc[2, "liar_select"] = "random"  # the fixed row is now in another cell
    out = swap_fixed(df)
    assert ((out.method == "midian") & (out.params == "{}") & (out.beta == 0.5) & (out.seed == 1)).sum() == 2


def test_noop_without_fixed_rows():
    df = frame().drop(index=2)
    out = swap_fixed(df)
    assert out is df
    assert swap_fixed(pd.DataFrame()).empty


@pytest.mark.parametrize("fixed", ['{"reagg":"keep"}', '{"reagg": "keep"}', {"reagg": "keep"}])
def test_params_formats_and_label(fixed):
    df = frame().assign(label="x")
    df["params"] = df.params.astype(object)
    df.at[2, "params"] = fixed
    out = swap_fixed(df)
    r = out[out.success == 0.70].iloc[0]
    assert r.params == "{}" and r.label == "midian"
    assert 0.50 not in set(out.success)


def test_backend_kwargs_dict_matches_json():
    df = frame().assign(backend_kwargs='{"b":2,"a":1}')
    df["backend_kwargs"] = df.backend_kwargs.astype(object)
    df.at[2, "backend_kwargs"] = {"a": 1, "b": 2}  # a rows.d row
    assert 0.50 not in set(swap_fixed(df).success)


def test_with_s4_10_adds_fix_mirrors_only_under_seeds_4_10(monkeypatch):
    src, mirror = "learned_n10k", "learned_n10k_s4_10"
    monkeypatch.setattr(grids, "fixes", lambda: [f"{src}_fix", f"{mirror}_fix", "other_fix"])
    monkeypatch.setattr(seed_tables, "complete", lambda g, load=None: True)
    monkeypatch.setattr(lib, "SEEDS_4_10", False)
    assert grids.with_s4_10([src]) == [src]
    monkeypatch.setattr(lib, "SEEDS_4_10", True)
    assert grids.with_s4_10([src]) == [src, mirror, f"{src}_fix", f"{mirror}_fix"]
    assert grids.with_s4_10(["fw_live_n10k_cartel"]) == ["fw_live_n10k_cartel"]
    monkeypatch.setattr(seed_tables, "complete", lambda g, load=None: g != "other_fix")
    with pytest.raises(SystemExit, match="other_fix"):
        grids.with_s4_10([src])


def test_fix_grids_complete_on_beta_positive_rows(monkeypatch):
    k0 = (1000, 3, "specialist", 0.0, "random", 1, "midian[reagg=keep]")
    k5 = (1000, 3, "specialist", 0.5, "low_skill_first", 1, "midian[reagg=keep]")
    have = pd.DataFrame([dict(n=1000, b=3, dist="specialist", beta=0.5, liar_select="low_skill_first", seed=1,
                              method="midian", params='{"reagg":"keep"}')])
    monkeypatch.setattr(seed_tables, "planned", lambda g: {k0, k5})
    monkeypatch.setattr(seed_tables, "_done", {})
    monkeypatch.setattr(grids, "fixes", lambda: ["a_fix"])
    assert seed_tables.complete("a_fix", load=lambda g: have)
    assert not seed_tables.complete("a", load=lambda g: have)  # any other grid needs its beta = 0 rows too
