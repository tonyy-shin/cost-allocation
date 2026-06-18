"""Tests for src.allocation: by_coa table and by_cc settled snapshots."""
from __future__ import annotations

import pandas as pd
import pytest

from src.allocation import build_by_cc, build_by_coa


def _row(df: pd.DataFrame, cc: str) -> pd.Series:
    return df[df["CC"] == cc].iloc[0]


# by_coa ---------------------------------------------------------------------


def test_by_coa_column_layout(pipeline_outputs):
    # Keys, per-cycle 배부금액, one empty separator, then per-cycle 배부합계.
    cols = list(pipeline_outputs["by_coa_df"].columns)
    assert cols == [
        "전기COA", "기존COA", "Sender CC",
        "1차배부금액", "2차배부금액",
        "",
        "1차배부합계", "2차배부합계",
    ]


def test_by_coa_amounts_keyed_by_sender(pipeline_outputs):
    df = pipeline_outputs["by_coa_df"]
    # Sender 1001 holds common cost E6100/6100 = 5,000,000 in cycle 1.
    row = df[(df["기존COA"] == "6100") & (df["Sender CC"] == "1001")].iloc[0]
    assert row["1차배부금액"] == pytest.approx(5_000_000.0)
    assert row["2차배부금액"] == pytest.approx(0.0)


def test_by_coa_total_is_column_wide_scalar_first_row_only(pipeline_outputs):
    df = pipeline_outputs["by_coa_df"]
    # Row 0 carries the column-wide total; every other row is blank.
    assert df.iloc[0]["1차배부합계"] == pytest.approx(df["1차배부금액"].sum())
    assert df.iloc[0]["2차배부합계"] == pytest.approx(df["2차배부금액"].sum())
    assert (df["1차배부합계"].iloc[1:] == "").all()
    assert (df["2차배부합계"].iloc[1:] == "").all()


def test_by_coa_sender_totals_are_sender_level(pipeline_outputs):
    # sender_totals is per Sender CC (drives by_cc), not the column-wide scalar.
    sender_totals = pipeline_outputs["sender_totals"]
    assert sender_totals[1] == {"1001": pytest.approx(7_000_000.0)}
    assert sender_totals[2] == {"2001": pytest.approx(3_500_000.0)}


# by_cc ----------------------------------------------------------------------


def test_by_cc_conservation_per_file(pipeline_outputs):
    # 배부전 total == 배부합계 total for every cycle file.
    for df in pipeline_outputs["by_cc_files"].values():
        assert df["배부합계"].sum() == pytest.approx(df["배부전금액"].sum())


def test_by_cc_static_money_lands_in_last_column(pipeline_outputs):
    # CC 2002 is never a sender/receiver but holds 400,000; it must surface in
    # the file's last 후금액 column, never in an earlier one.
    file2 = pipeline_outputs["by_cc_files"][2]
    row = _row(file2, "2002")
    assert row["1차후금액"] == pytest.approx(0.0)
    assert row["2차후금액"] == pytest.approx(400_000.0)
    assert row["배부합계"] == pytest.approx(400_000.0)


def test_by_cc_sender_and_receiver_flow(pipeline_outputs):
    # Cycle 1: sender 1001 pushes 7,000,000; receiver 1003 gets 70%.
    file1 = pipeline_outputs["by_cc_files"][1]
    assert _row(file1, "1001")["1차후금액"] == pytest.approx(
        1_000_000.0 - 7_000_000.0
    )
    assert _row(file1, "1003")["1차후금액"] == pytest.approx(
        300_000.0 + 7_000_000.0 * 0.7
    )


def test_by_cc_file_columns(pipeline_outputs):
    files = pipeline_outputs["by_cc_files"]
    assert list(files[1].columns) == ["CC", "배부전금액", "1차후금액", "배부합계"]
    assert list(files[2].columns) == [
        "CC", "배부전금액", "1차후금액", "2차후금액", "배부합계"
    ]


def test_by_cc_one_row_per_unique_cc(pipeline_outputs, loaded_inputs):
    file1 = pipeline_outputs["by_cc_files"][1]
    expected = set(loaded_inputs["cc_list"])
    assert set(file1["CC"]) == expected
    assert len(file1) == len(expected)


# Pass-through worked example ------------------------------------------------


def test_by_cc_pass_through_nets_out():
    # CC_A sends 1,000 in cycle 1 to CC_B; CC_B sends that 1,000 in cycle 2 to
    # CC_C. Money passing through CC_B nets to 0 in its 1차후금액.
    cc_list = ["A", "B", "C"]
    pre_alloc_cc = {"A": 1000.0}
    cycle_df = pd.DataFrame({
        "차수": [1, 2],
        "Sender CC": ["A", "B"],
        "Receiver CC": ["B", "C"],
        "%": [1.0, 1.0],
    })
    sender_totals = {1: {"A": 1000.0}, 2: {"B": 1000.0}}

    files = build_by_cc(cc_list, pre_alloc_cc, cycle_df, sender_totals)

    file1, file2 = files[1], files[2]
    # file 1: B holds the passed-through 1,000.
    assert _row(file1, "A")["배부합계"] == pytest.approx(0.0)
    assert _row(file1, "B")["1차후금액"] == pytest.approx(1000.0)
    assert _row(file1, "C")["배부합계"] == pytest.approx(0.0)
    # file 2: B has forwarded it; the 1,000 now rests at C in 2차후금액.
    assert _row(file2, "B")["1차후금액"] == pytest.approx(0.0)
    assert _row(file2, "B")["2차후금액"] == pytest.approx(0.0)
    assert _row(file2, "C")["2차후금액"] == pytest.approx(1000.0)
    # Conservation holds in both files.
    for f in (file1, file2):
        assert f["배부합계"].sum() == pytest.approx(f["배부전금액"].sum())


def test_by_cc_negative_balance_subtracted_unconditionally():
    # A sender whose 배부전 (200) is below its sender_total (500) goes negative;
    # no clamping, no warning.
    cc_list = ["S", "R"]
    pre_alloc_cc = {"S": 200.0}
    cycle_df = pd.DataFrame({
        "차수": [1],
        "Sender CC": ["S"],
        "Receiver CC": ["R"],
        "%": [1.0],
    })
    sender_totals = {1: {"S": 500.0}}

    files = build_by_cc(cc_list, pre_alloc_cc, cycle_df, sender_totals)
    assert _row(files[1], "S")["1차후금액"] == pytest.approx(-300.0)
    assert _row(files[1], "R")["1차후금액"] == pytest.approx(500.0)
