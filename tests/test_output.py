"""Tests for src.output: computed-grid build_result and CSV writing."""
from __future__ import annotations

import pandas as pd
import pytest

from src.allocation import TOTAL_COL
from src.output import save_result, save_snapshots


def _alloc_cols(df):
    return [c for c in df.columns if "배분금액" in c]


# SUCCESS cases


def test_build_result_row_count_matches_computed_pairs(pipeline_outputs):
    # The grid is the computed result itself — the (전기COA, 기존COA, CC) groups that
    # actually carry an allocation or a direct cost — NOT the master's (COA, CC)
    # pairs. In sample_data the master has 15 pairs but only 13 are computed
    # (2 master-only pairs with neither allocation nor direct cost are absent).
    #
    # Asserted against a fixed expected count rather than a value derived from the
    # same compute path, so a build_result bug cannot mask itself. Trade-off: this
    # number must be updated whenever sample_data changes.
    result = pipeline_outputs["result"]
    assert len(result) == 13
    # No duplicate key rows: groupby collapses each (전기COA, 기존COA, CC) once.
    assert not result.duplicated(["전기COA", "기존COA", "코스트센터"]).any()


def test_direct_rows_have_zero_allocation(pipeline_outputs):
    result = pipeline_outputs["result"]
    # Direct-cost rows carry an empty 전기COA and must be zero in every cycle col.
    direct = result[result["전기COA"] == ""]
    assert len(direct) > 0
    assert (direct[_alloc_cols(result)] == 0).all().all()


def test_common_rows_total_equals_alloc_sum(pipeline_outputs):
    result = pipeline_outputs["result"]
    # For common-cost rows, 배부합계 is the sum of the per-cycle allocations.
    # (Direct rows instead carry their original Amounts in 배부합계 with zero
    # allocation columns, so the identity is asserted on common rows only.)
    common = result[result["전기COA"] != ""]
    assert len(common) > 0
    expected = common[_alloc_cols(result)].sum(axis=1)
    assert common[TOTAL_COL].to_numpy() == pytest.approx(expected.to_numpy())


def test_save_result_writes_utf8_sig_csv(pipeline_outputs, tmp_path):
    out_path = save_result(pipeline_outputs["result"], tmp_path)

    assert out_path.exists()
    assert out_path.name == "result.csv"
    # utf-8-sig writes a BOM prefix for Excel compatibility.
    assert out_path.read_bytes().startswith(b"\xef\xbb\xbf")
    # Round-trips back to the same shape with utf-8-sig decoding.
    reloaded = pd.read_csv(out_path, encoding="utf-8-sig")
    assert len(reloaded) == len(pipeline_outputs["result"])
    assert list(reloaded.columns) == list(pipeline_outputs["result"].columns)


def test_save_snapshots_creates_per_cycle_files(pipeline_outputs, tmp_path):
    # Sample data runs 2 allocation cycles.
    paths = save_snapshots(pipeline_outputs["result"], tmp_path, n_cycles=2)

    assert len(paths) == 2
    assert {p.name for p in paths} == {"result_1차.csv", "result_2차.csv"}
    assert (tmp_path / "result_1차.csv").exists()
    assert (tmp_path / "result_2차.csv").exists()


def test_save_snapshots_column_composition(pipeline_outputs, tmp_path):
    save_snapshots(pipeline_outputs["result"], tmp_path, n_cycles=2)

    snap1 = pd.read_csv(tmp_path / "result_1차.csv", encoding="utf-8-sig")
    assert list(snap1.columns) == [
        "전기COA", "기존COA", "코스트센터", "1차배분금액", TOTAL_COL,
    ]

    snap2 = pd.read_csv(tmp_path / "result_2차.csv", encoding="utf-8-sig")
    assert list(snap2.columns) == [
        "전기COA", "기존COA", "코스트센터", "1차배분금액", "2차배분금액", TOTAL_COL,
    ]


def test_save_snapshots_total_recomputed_per_cycle(pipeline_outputs, tmp_path):
    save_snapshots(pipeline_outputs["result"], tmp_path, n_cycles=2)

    # result_1차: 배부합계 == 1차배분금액
    snap1 = pd.read_csv(tmp_path / "result_1차.csv", encoding="utf-8-sig")
    assert snap1[TOTAL_COL].to_numpy() == pytest.approx(
        snap1["1차배분금액"].to_numpy()
    )

    # result_2차: 배부합계 == 1차배분금액 + 2차배분금액
    snap2 = pd.read_csv(tmp_path / "result_2차.csv", encoding="utf-8-sig")
    expected = snap2[["1차배분금액", "2차배분금액"]].sum(axis=1)
    assert snap2[TOTAL_COL].to_numpy() == pytest.approx(expected.to_numpy())
