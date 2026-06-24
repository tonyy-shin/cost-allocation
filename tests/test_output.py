"""Tests for src.output: the 배부금액 / 잔액 directory tree and CSV writing."""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.output import append_total_row, save_results


# append_total_row -----------------------------------------------------------


def _by_cc_sample() -> pd.DataFrame:
    return pd.DataFrame({
        "전기COA": ["E6100", "E6100"],
        "기존COA": ["6100", "6100"],
        "CC": ["1001", "1002"],
        "배부전금액": [500_000.0, 300_000.0],
        "1차후금액": [100.0, 200.0],
        "2차후금액": [10.0, 20.0],
    })


def test_append_total_row_adds_row_at_bottom():
    df = _by_cc_sample()
    result = append_total_row(df)

    assert len(result) == len(df) + 1
    total = result.iloc[-1]
    # 배부전금액 and every 후금액 column are summed.
    assert total["배부전금액"] == 800_000
    assert total["1차후금액"] == 300
    assert total["2차후금액"] == 30


def test_append_total_row_labels_cc_and_sums_amount_columns():
    result = append_total_row(_by_cc_sample())
    total = result.iloc[-1]
    assert total["CC"] == "합계"
    # 후금액 columns are summed in the totals row (not blanked).
    assert total["1차후금액"] == 300
    assert total["2차후금액"] == 30
    # Only the COA key columns are display-blank in the totals row.
    assert total["전기COA"] == ""
    assert total["기존COA"] == ""


def test_append_total_row_rounds_to_integer():
    df = pd.DataFrame({
        "CC": ["a", "b"],
        "배부전금액": [100.4, 100.4],
        "1차후금액": [0.5, 0.5],
    })
    total = append_total_row(df).iloc[-1]
    # 200.8 -> 201; 1.0 -> 1. Values are whole numbers (no float noise).
    assert total["배부전금액"] == 201
    assert total["1차후금액"] == 1


def test_append_total_row_leaves_individual_rows_and_input_untouched():
    df = _by_cc_sample()
    before = df.copy(deep=True)
    result = append_total_row(df)

    # Original row values are preserved in the result (the total row's blanks
    # upcast some columns to object, so compare values, not dtypes)...
    pd.testing.assert_frame_equal(
        result.iloc[:-1].reset_index(drop=True), df, check_dtype=False
    )
    # ...and the input frame itself is not mutated.
    pd.testing.assert_frame_equal(df, before)


def test_save_results_writes_directory_tree(pipeline_outputs, tmp_path):
    out = save_results(
        pipeline_outputs["by_coa_df"],
        pipeline_outputs["by_cc_files"],
        tmp_path,
    )

    assert out == tmp_path
    by_coa = tmp_path / "배부금액" / "result.csv"
    assert by_coa.exists()
    # One by_cc file per cycle, named {n}차배부후.csv.
    for n in pipeline_outputs["by_cc_files"]:
        assert (tmp_path / "잔액" / f"{n}차배부후.csv").exists()


def test_save_results_uses_utf8_sig(pipeline_outputs, tmp_path):
    save_results(
        pipeline_outputs["by_coa_df"],
        pipeline_outputs["by_cc_files"],
        tmp_path,
    )
    # utf-8-sig writes a BOM prefix for Excel compatibility.
    by_coa = tmp_path / "배부금액" / "result.csv"
    assert by_coa.read_bytes().startswith(b"\xef\xbb\xbf")


def test_save_results_by_cc_round_trips(pipeline_outputs, tmp_path):
    save_results(
        pipeline_outputs["by_coa_df"],
        pipeline_outputs["by_cc_files"],
        tmp_path,
    )
    file1 = pd.read_csv(tmp_path / "잔액" / "1차배부후.csv", encoding="utf-8-sig")
    expected = pipeline_outputs["by_cc_files"][1]
    assert list(file1.columns) == list(expected.columns)
    # The written file carries an extra totals row at the bottom.
    assert len(file1) == len(expected) + 1
    total = file1.iloc[-1]
    assert total["배부전금액"] == pytest.approx(expected["배부전금액"].sum())
    assert total["1차후금액"] == pytest.approx(expected["1차후금액"].sum())
    assert total["CC"] == "합계"
