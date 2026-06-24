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


# by_cc ----------------------------------------------------------------------


def test_by_cc_conservation_per_file(pipeline_outputs):
    # 배부전 total == 배부합계 total for every cycle file.
    for df in pipeline_outputs["by_cc_files"].values():
        assert df["배부합계"].sum() == pytest.approx(df["배부전금액"].sum())


def test_by_cc_conservation_per_coa_pair(pipeline_outputs):
    # Money keeps its COA identity, so conservation also holds within each
    # (전기COA, 기존COA) pair, not just overall.
    for df in pipeline_outputs["by_cc_files"].values():
        grouped = df.groupby(["전기COA", "기존COA"])[["배부전금액", "배부합계"]].sum()
        for pre, total in zip(grouped["배부전금액"], grouped["배부합계"]):
            assert total == pytest.approx(pre)


def test_by_cc_static_money_lands_in_last_column(pipeline_outputs):
    # CC 2002 is never a sender/receiver but holds 400,000 under (E6100, 6100);
    # it must surface in the file's last 후금액 column, never in an earlier one.
    file2 = pipeline_outputs["by_cc_files"][2]
    row = _row(file2, "2002", "E6100", "6100")
    assert row["1차후금액"] == pytest.approx(0.0)
    assert row["2차후금액"] == pytest.approx(400_000.0)
    assert row["배부합계"] == pytest.approx(400_000.0)


def test_by_cc_sender_and_receiver_flow(pipeline_outputs):
    # Cycle 1: sender 1001 pushes its common cost per COA pair; the flow is now
    # split by (전기COA, 기존COA). 1001 holds E6100/6100=600,000 + E6200/6200=400,000
    # of 배부전 and sends 5,000,000 (E6100) + 2,000,000 (E6200).
    file1 = pipeline_outputs["by_cc_files"][1]
    assert _row(file1, "1001", "E6100", "6100")["1차후금액"] == pytest.approx(
        600_000.0 - 5_000_000.0
    )
    assert _row(file1, "1001", "E6200", "6200")["1차후금액"] == pytest.approx(
        400_000.0 - 2_000_000.0
    )
    # Receiver 1003 gets 70% of each pair.
    assert _row(file1, "1003", "E6100", "6100")["1차후금액"] == pytest.approx(
        300_000.0 + 5_000_000.0 * 0.7
    )
    assert _row(file1, "1003", "E6200", "6200")["1차후금액"] == pytest.approx(
        2_000_000.0 * 0.7
    )


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


def test_by_cc_every_cc_has_at_least_one_row(pipeline_outputs, loaded_inputs):
    # Option B (minimum guarantee): every CC in cc_list appears in at least one
    # row, even before it has acquired any COA pair of its own.
    file1 = pipeline_outputs["by_cc_files"][1]
    expected = set(loaded_inputs["cc_list"])
    assert set(file1["CC"]) >= expected

    # In cycle 1, receivers 3001/4001 have not been credited yet, so each is held
    # by a single blank-COA zero row rather than being dropped.
    for cc in ("3001", "4001"):
        rows = file1[file1["CC"] == cc]
        assert len(rows) == 1
        r = rows.iloc[0]
        assert r["전기COA"] == ""
        assert r["기존COA"] == ""
        assert r["배부전금액"] == pytest.approx(0.0)
        assert r["배부합계"] == pytest.approx(0.0)


# Pass-through worked example ------------------------------------------------


def _pre_alloc(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    """Build a pre_alloc_enriched frame from (전기COA, 기존COA, CC, amount) rows."""
    return pd.DataFrame(
        rows, columns=["전기COA", "기존COA", "Cost Center", "Amounts"]
    )


def test_by_cc_pass_through_nets_out():
    # CC_A sends 1,000 in cycle 1 to CC_B; CC_B sends that 1,000 in cycle 2 to
    # CC_C. CC_B has no book common cost of its own (no sender_totals entry) — the
    # live cascade forwards what it received with no hand-crafted sender total.
    # Money passing through CC_B nets to 0 in its 1차후금액 and stays in (E1, C1).
    cc_list = ["A", "B", "C"]
    pre_alloc = _pre_alloc([("E1", "C1", "A", 1000.0)])
    cycle_df = pd.DataFrame({
        "차수": [1, 2],
        "Sender CC": ["A", "B"],
        "Receiver CC": ["B", "C"],
        "%": [1.0, 1.0],
    })
    sender_totals = {1: {("E1", "C1", "A"): 1000.0}}

    files = build_by_cc(cc_list, pre_alloc, cycle_df, sender_totals)

    file1, file2 = files[1], files[2]
    # file 1: B holds the passed-through 1,000.
    assert _row(file1, "A", "E1", "C1")["배부합계"] == pytest.approx(0.0)
    assert _row(file1, "B", "E1", "C1")["1차후금액"] == pytest.approx(1000.0)
    # C has not been credited yet -> option B blank-COA zero row.
    assert _row(file1, "C")["배부합계"] == pytest.approx(0.0)
    # file 2: B has forwarded it; the 1,000 now rests at C in 2차후금액.
    assert _row(file2, "B", "E1", "C1")["1차후금액"] == pytest.approx(0.0)
    assert _row(file2, "B", "E1", "C1")["2차후금액"] == pytest.approx(0.0)
    assert _row(file2, "C", "E1", "C1")["2차후금액"] == pytest.approx(1000.0)
    # Conservation holds in both files.
    for f in (file1, file2):
        assert f["배부합계"].sum() == pytest.approx(f["배부전금액"].sum())


def test_by_cc_book_cost_distributed_once_then_only_receipts():
    # X has its own book common cost (1,000) and sends it in cycle 1. In cycle 2
    # X receives 400 from Y. In cycle 3 X is a sender again: under the live
    # cascade it forwards only the 400 it received since its last send — its book
    # cost is not re-distributed.
    cc_list = ["X", "Y", "R1", "R2"]
    pre_alloc = _pre_alloc([("E1", "C1", "X", 1000.0), ("E1", "C1", "Y", 400.0)])
    cycle_df = pd.DataFrame({
        "차수": [1, 2, 3],
        "Sender CC": ["X", "Y", "X"],
        "Receiver CC": ["R1", "X", "R2"],
        "%": [1.0, 1.0, 1.0],
    })
    sender_totals = {
        1: {("E1", "C1", "X"): 1000.0},
        2: {("E1", "C1", "Y"): 400.0},
        3: {("E1", "C1", "X"): 1000.0},  # static; ignored by the cascade in c3
    }

    files = build_by_cc(cc_list, pre_alloc, cycle_df, sender_totals)

    # Cycle 1: X forwards its 1,000 book cost to R1.
    assert _row(files[1], "R1", "E1", "C1")["1차후금액"] == pytest.approx(1000.0)
    # Cycle 3: X forwards only the 400 received in cycle 2 to R2 (not 1,000).
    assert _row(files[3], "R2", "E1", "C1")["3차후금액"] == pytest.approx(400.0)
    assert _row(files[3], "R2", "E1", "C1")["배부합계"] == pytest.approx(400.0)
    # Conservation holds in every file.
    for f in files.values():
        assert f["배부합계"].sum() == pytest.approx(f["배부전금액"].sum())


def test_by_cc_negative_balance_subtracted_unconditionally():
    # A sender whose 배부전 (200) is below its sender_total (500) goes negative;
    # no clamping, no warning.
    cc_list = ["S", "R"]
    pre_alloc = _pre_alloc([("E1", "C1", "S", 200.0)])
    cycle_df = pd.DataFrame({
        "차수": [1],
        "Sender CC": ["S"],
        "Receiver CC": ["R"],
        "%": [1.0],
    })
    sender_totals = {1: {("E1", "C1", "S"): 500.0}}

    files = build_by_cc(cc_list, pre_alloc, cycle_df, sender_totals)
    assert _row(files[1], "S", "E1", "C1")["1차후금액"] == pytest.approx(-300.0)
    assert _row(files[1], "R", "E1", "C1")["1차후금액"] == pytest.approx(500.0)


def test_by_cc_option_b_empty_cc_gets_blank_coa_row():
    # CC_Z never holds 배부전 and never receives anything. Option B guarantees it
    # one all-zero, blank-COA row so it is not dropped from the output.
    cc_list = ["S", "R", "Z"]
    pre_alloc = _pre_alloc([("E1", "C1", "S", 1000.0)])
    cycle_df = pd.DataFrame({
        "차수": [1],
        "Sender CC": ["S"],
        "Receiver CC": ["R"],
        "%": [1.0],
    })
    sender_totals = {1: {("E1", "C1", "S"): 1000.0}}

    file1 = build_by_cc(cc_list, pre_alloc, cycle_df, sender_totals)[1]

    z_rows = file1[file1["CC"] == "Z"]
    assert len(z_rows) == 1
    z = z_rows.iloc[0]
    assert z["전기COA"] == ""
    assert z["기존COA"] == ""
    assert z["배부전금액"] == pytest.approx(0.0)
    assert z["1차후금액"] == pytest.approx(0.0)
    assert z["배부합계"] == pytest.approx(0.0)
    # The guarantee does not perturb conservation.
    assert file1["배부합계"].sum() == pytest.approx(file1["배부전금액"].sum())
