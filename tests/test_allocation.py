"""Tests for src.allocation: by_coa table and by_cc settled snapshots."""
from __future__ import annotations

import warnings

import pandas as pd
import pytest

from src.core.allocation import build_by_cc, build_by_coa


def _row(
    df: pd.DataFrame,
    cc: str,
    e: str | None = None,
    b: str | None = None,
) -> pd.Series:
    """Pick a by_cc row by CC, optionally narrowing to a (전기COA, 기존COA) pair."""
    sub = df[df["CC"] == cc]
    if e is not None:
        sub = sub[(sub["전기COA"] == e) & (sub["기존COA"] == b)]
    return sub.iloc[0]


# by_coa ---------------------------------------------------------------------


def test_by_coa_column_layout(pipeline_outputs):
    # Keys, per-cycle 배부금액, one empty separator, then per-cycle 배부합계.
    # The sample cycle.csv has 3 차수 (cycle 3 routes to the master-absent CC 4001).
    cols = list(pipeline_outputs["by_coa_df"].columns)
    assert cols == [
        "전기COA", "기존COA", "Sender CC", "Receiver CC",
        "1차배부금액", "2차배부금액", "3차배부금액",
        "",
        "1차배부합계", "2차배부합계", "3차배부합계",
    ]


def test_by_coa_amounts_split_by_receiver(pipeline_outputs):
    df = pipeline_outputs["by_coa_df"]
    # Sender 1001 holds common cost E6100/6100 = 5,000,000 in cycle 1, split
    # across its receivers (1002 → 0.3, 1003 → 0.7) per cycle.csv.
    rows = df[(df["기존COA"] == "6100") & (df["Sender CC"] == "1001")]
    assert rows["1차배부금액"].sum() == pytest.approx(5_000_000.0)
    assert rows["2차배부금액"].sum() == pytest.approx(0.0)

    r1002 = rows[rows["Receiver CC"] == "1002"].iloc[0]
    r1003 = rows[rows["Receiver CC"] == "1003"].iloc[0]
    assert r1002["1차배부금액"] == pytest.approx(5_000_000.0 * 0.3)
    assert r1003["1차배부금액"] == pytest.approx(5_000_000.0 * 0.7)


def test_by_coa_total_is_column_wide_scalar_first_row_only(pipeline_outputs):
    df = pipeline_outputs["by_coa_df"]
    # Row 0 carries the column-wide total; every other row is blank.
    assert df.iloc[0]["1차배부합계"] == pytest.approx(df["1차배부금액"].sum())
    assert df.iloc[0]["2차배부합계"] == pytest.approx(df["2차배부금액"].sum())
    assert (df["1차배부합계"].iloc[1:] == "").all()
    assert (df["2차배부합계"].iloc[1:] == "").all()


def test_by_coa_sender_totals_are_keyed_by_coa_pair(pipeline_outputs):
    # sender_totals is per (전기COA, 기존COA, Sender CC) so by_cc can carry the
    # money's COA identity; it is pre-explode (ratio-free), not the column scalar.
    sender_totals = pipeline_outputs["sender_totals"]
    assert sender_totals[1] == {
        ("E6100", "6100", "1001"): pytest.approx(5_000_000.0),
        ("E6200", "6200", "1001"): pytest.approx(2_000_000.0),
    }
    assert sender_totals[2] == {
        ("E6100", "6100", "2001"): pytest.approx(3_000_000.0),
        ("E6200", "6200", "2001"): pytest.approx(500_000.0),
    }


# by_coa: unmapped-COA warning ----------------------------------------------


def _enriched(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    """Build an enriched frame from (전기COA, 기존COA, Cost Center, amount) rows."""
    return pd.DataFrame(
        rows, columns=["전기COA", "기존COA", "Cost Center", "Amounts"]
    )


def test_by_coa_warns_for_unmapped_coa_on_sender():
    # D9 has no 전기COA mapping ("") and sits on sender S, so its 700 is excluded
    # from allocation: build_by_coa warns with the per-(COA, Sender CC) amount.
    # D9 on the non-sender X, and the zero-amount unmapped row, must not be flagged.
    enriched = _enriched([
        ("E1", "C1", "S", 1000.0),
        ("", "D9", "S", 700.0),
        ("", "D9", "X", 500.0),
        ("", "D8", "S", 0.0),
    ])
    cycle_df = pd.DataFrame({
        "차수": [1], "Sender CC": ["S"], "Receiver CC": ["R"], "%": [1.0],
    })

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_by_coa(enriched, cycle_df)

    messages = [str(w.message) for w in caught]
    assert len(messages) == 1
    msg = messages[0]
    assert msg.startswith("전기COA 매핑이 없어 배부에서 제외된 항목이 있습니다.")
    assert "COA D9 / Sender CC S: 700" in msg
    assert "Sender CC X" not in msg
    assert "D8" not in msg


def test_by_coa_no_warning_when_all_sender_coas_mapped():
    # Every COA on the sender has a 전기COA mapping, so nothing is excluded.
    enriched = _enriched([("E1", "C1", "S", 1000.0)])
    cycle_df = pd.DataFrame({
        "차수": [1], "Sender CC": ["S"], "Receiver CC": ["R"], "%": [1.0],
    })

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_by_coa(enriched, cycle_df)

    assert [str(w.message) for w in caught] == []


# by_cc: structure over the sample pipeline -----------------------------------


def test_by_cc_file_columns(pipeline_outputs):
    files = pipeline_outputs["by_cc_files"]
    assert list(files[1].columns) == [
        "전기COA", "기존COA", "CC", "배부전금액", "1차후금액", "배부합계"
    ]
    assert list(files[2].columns) == [
        "전기COA", "기존COA", "CC", "배부전금액", "1차후금액", "2차후금액", "배부합계"
    ]


def test_by_cc_rows_sorted_by_coa_then_cc(pipeline_outputs):
    # Rows are ordered by (전기COA, 기존COA, CC), like by_coa.
    file2 = pipeline_outputs["by_cc_files"][2]
    keys = list(zip(file2["전기COA"], file2["기존COA"], file2["CC"]))
    assert keys == sorted(keys)


def test_by_cc_first_sender_balance_is_zero(pipeline_outputs):
    # 1001 is the only 1차 sender (minimum 차수). Every one of its rows reports
    # 배부전금액 = 0 in every file, regardless of its 원본 amount.
    for df in pipeline_outputs["by_cc_files"].values():
        rows_1001 = df[df["CC"] == "1001"]
        assert (rows_1001["배부전금액"] == 0.0).all()
        # A non-1차 CC keeps its 원본 amount as 배부전금액 (2001 holds E6100/6100).
        assert _row(df, "2001", "E6100", "6100")["배부전금액"] == pytest.approx(
            3_000_000.0
        )


def test_by_cc_every_cc_has_at_least_one_row(pipeline_outputs, loaded_inputs):
    # Every CC in cc_list appears in at least one row.
    file1 = pipeline_outputs["by_cc_files"][1]
    expected = set(loaded_inputs["cc_list"])
    assert set(file1["CC"]) >= expected


# by_cc: 원본/복사본 model over consistent hand-built data ----------------------


def _en(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    """Build an enriched (원본) frame from (전기COA, 기존COA, CC, amount) rows."""
    return pd.DataFrame(
        rows, columns=["전기COA", "기존COA", "Cost Center", "Amounts"]
    )


def test_by_cc_first_sender_zero_others_keep_원본():
    # Cycle 1 has two senders (A, X) at the minimum 차수, so both are 1차 senders
    # and report 배부전금액 = 0. Every other CC reports its 원본 amount.
    enriched = _en([
        ("E1", "C1", "A", 1000.0),
        ("E1", "C1", "X", 500.0),
        ("E1", "C1", "B", 1000.0),
        ("E1", "C1", "Y", 500.0),
    ])
    cycle_df = pd.DataFrame({
        "차수": [1, 1],
        "Sender CC": ["A", "X"],
        "Receiver CC": ["B", "Y"],
        "%": [1.0, 1.0],
    })
    sender_totals = {1: {("E1", "C1", "A"): 1000.0, ("E1", "C1", "X"): 500.0}}

    file1 = build_by_cc(["A", "X", "B", "Y"], enriched, cycle_df, sender_totals)[1]
    assert _row(file1, "A", "E1", "C1")["배부전금액"] == pytest.approx(0.0)
    assert _row(file1, "X", "E1", "C1")["배부전금액"] == pytest.approx(0.0)
    assert _row(file1, "B", "E1", "C1")["배부전금액"] == pytest.approx(1000.0)
    assert _row(file1, "Y", "E1", "C1")["배부전금액"] == pytest.approx(500.0)


def test_by_cc_sender_rows_zero_receiver_rows_match_원본():
    # Consistent single-cycle flow: A (1차 sender) sends its full 1,000 to B,
    # whose 원본 balance is exactly the 1,000 it receives.
    enriched = _en([("E1", "C1", "A", 1000.0), ("E1", "C1", "B", 1000.0)])
    cycle_df = pd.DataFrame({
        "차수": [1], "Sender CC": ["A"], "Receiver CC": ["B"], "%": [1.0],
    })
    sender_totals = {1: {("E1", "C1", "A"): 1000.0}}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        file1 = build_by_cc(["A", "B"], enriched, cycle_df, sender_totals)[1]

    # Consistent data: no validation warning.
    assert [str(w.message) for w in caught] == []
    # Sender row drops to 0 in its own cycle; receiver row takes its 원본 value.
    assert _row(file1, "A", "E1", "C1")["1차후금액"] == pytest.approx(0.0)
    assert _row(file1, "B", "E1", "C1")["1차후금액"] == pytest.approx(1000.0)
    # 배부합계 is the final resting balance (the last 후금액 column).
    assert _row(file1, "A", "E1", "C1")["배부합계"] == pytest.approx(0.0)
    assert _row(file1, "B", "E1", "C1")["배부합계"] == pytest.approx(1000.0)


def test_by_cc_pass_through_chain():
    # Consistent two-cycle chain A -> B -> C, all in (E1, C1). A is the 1차 sender.
    # B holds the money after cycle 1 (1차후금액=1000) then forwards it in cycle 2
    # (2차후금액=0); C receives it in cycle 2 (2차후금액=1000).
    enriched = _en([
        ("E1", "C1", "A", 1000.0),
        ("E1", "C1", "B", 1000.0),
        ("E1", "C1", "C", 1000.0),
    ])
    cycle_df = pd.DataFrame({
        "차수": [1, 2],
        "Sender CC": ["A", "B"],
        "Receiver CC": ["B", "C"],
        "%": [1.0, 1.0],
    })
    sender_totals = {1: {("E1", "C1", "A"): 1000.0}, 2: {("E1", "C1", "B"): 1000.0}}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        files = build_by_cc(["A", "B", "C"], enriched, cycle_df, sender_totals)

    assert [str(w.message) for w in caught] == []
    file2 = files[2]
    assert _row(file2, "A", "E1", "C1")["배부합계"] == pytest.approx(0.0)
    assert _row(file2, "B", "E1", "C1")["1차후금액"] == pytest.approx(1000.0)
    assert _row(file2, "B", "E1", "C1")["2차후금액"] == pytest.approx(0.0)
    assert _row(file2, "B", "E1", "C1")["배부합계"] == pytest.approx(0.0)
    assert _row(file2, "C", "E1", "C1")["1차후금액"] == pytest.approx(0.0)
    assert _row(file2, "C", "E1", "C1")["2차후금액"] == pytest.approx(1000.0)
    assert _row(file2, "C", "E1", "C1")["배부합계"] == pytest.approx(1000.0)


def test_by_cc_validation_warns_on_mismatch():
    # A (1차 sender) drains 1,000 but its receiver B holds only 500 in 원본, so the
    # per-cycle validation fails and warns (it does not raise).
    enriched = _en([("E1", "C1", "A", 1000.0), ("E1", "C1", "B", 500.0)])
    cycle_df = pd.DataFrame({
        "차수": [1], "Sender CC": ["A"], "Receiver CC": ["B"], "%": [1.0],
    })
    sender_totals = {1: {("E1", "C1", "A"): 1000.0}}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_by_cc(["A", "B"], enriched, cycle_df, sender_totals)

    messages = [str(w.message) for w in caught]
    assert len(messages) == 1
    msg = messages[0]
    assert msg.startswith("1차 배부 검증 실패")
    assert "1,000" in msg   # amount drained from 복사본
    assert "500" in msg     # 원본 amount at the receiver CC


def test_by_cc_option_b_empty_cc_gets_blank_coa_row():
    # CC_Z is absent from 원본 (never holds anything, never receives). Option B
    # guarantees it one all-zero, blank-COA row so it is not dropped.
    enriched = _en([("E1", "C1", "A", 1000.0), ("E1", "C1", "B", 1000.0)])
    cycle_df = pd.DataFrame({
        "차수": [1], "Sender CC": ["A"], "Receiver CC": ["B"], "%": [1.0],
    })
    sender_totals = {1: {("E1", "C1", "A"): 1000.0}}

    file1 = build_by_cc(["A", "B", "Z"], enriched, cycle_df, sender_totals)[1]

    z_rows = file1[file1["CC"] == "Z"]
    assert len(z_rows) == 1
    z = z_rows.iloc[0]
    assert z["전기COA"] == ""
    assert z["기존COA"] == ""
    assert z["배부전금액"] == pytest.approx(0.0)
    assert z["1차후금액"] == pytest.approx(0.0)
    assert z["배부합계"] == pytest.approx(0.0)
