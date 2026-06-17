"""Tests for src.allocation: pivot build, allocation loop, aggregation, decompose."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.allocation import (
    aggregate_received_by_cycle, alloc_col, build_pivot_matrix,
    decompose_sender_to_original_coa, decompose_to_original_coa,
    run_allocation_loop,
)


def _alloc_cols(df):
    return [c for c in df.columns if "배분금액" in c]


# SUCCESS cases


def test_build_pivot_matrix_shape(pipeline_outputs, loaded_inputs):
    df_5b = pipeline_outputs["df_5b"]
    pivot = pipeline_outputs["pivot"]
    n_ecoa = df_5b["전기COA"].nunique()
    n_cc = len(loaded_inputs["cc_list"])
    assert pivot.shape == (n_ecoa, n_cc)


def test_run_allocation_loop_senders_balance_to_zero(pipeline_outputs, loaded_inputs):
    # Each sender distributes 100% of its balance, so the loop never flags an
    # unbalanced sender. A sender that is *only* a sender (2001) ends at 0; a
    # sender that is also a later receiver (1001) legitimately ends non-zero,
    # so the zero-balance invariant is verified via the absence of the
    # "잔액이 0이 되지 않" warning plus the pure-sender 2001 balance.
    pivot = pipeline_outputs["pivot"]
    cycle_df = loaded_inputs["cycle_df"]
    with pytest.warns() as record:
        final_pivot, _, _ = run_allocation_loop(pivot, cycle_df)
    assert not any("0이 되지 않" in str(w.message) for w in record)
    assert final_pivot["2001"].sum() == pytest.approx(0.0)


def test_aggregate_received_matches_sent_per_cycle(pipeline_outputs, loaded_inputs):
    pivot = pipeline_outputs["pivot"]            # pre-loop initial balances
    cycle_df = loaded_inputs["cycle_df"]
    received_by_cycle = pipeline_outputs["received_by_cycle"]

    for cycle_num, cycle_rows in cycle_df.groupby("차수"):
        # Each sender's % sums to 1, so the amount sent in a cycle equals the
        # sum of that cycle's senders' initial balances.
        senders = cycle_rows["Sender CC"].unique()
        expected_sent = pivot[list(senders)].values.sum()
        received_sum = received_by_cycle[cycle_num]["Amounts"].sum()
        assert received_sum == pytest.approx(expected_sent)


def test_decompose_conserves_received_total(pipeline_outputs):
    decomposed = pipeline_outputs["decomposed"]
    received_by_cycle = pipeline_outputs["received_by_cycle"]

    # 비중 already verified to sum to 1.0 per 전기COA in test_prepare; here we
    # confirm the decomposition conserves the received total. (The output has no
    # single "allocated" column; it is split across the per-cycle 차배분금액
    # columns, which we sum.)
    received_total = sum(
        df["Amounts"].sum() for df in received_by_cycle.values() if not df.empty
    )
    decomposed_total = decomposed[_alloc_cols(decomposed)].values.sum()
    assert decomposed_total == pytest.approx(received_total)


def test_conservation_decomposed_equals_sender_common(pipeline_outputs, loaded_inputs):
    decomposed = pipeline_outputs["decomposed"]
    df_5b = pipeline_outputs["df_5b"]
    senders = set(loaded_inputs["cycle_df"]["Sender CC"])

    decomposed_total = decomposed[_alloc_cols(decomposed)].values.sum()
    sender_common = df_5b[df_5b["Cost Center"].isin(senders)]["Amounts"].sum()
    assert decomposed_total == pytest.approx(sender_common)


def test_decompose_sender_applies_ratio_and_restores_coa():
    # A single sender sends 1,000 of transfer COA E1, which decomposes back to
    # base COAs 100/200 by their 비중 (0.3 / 0.7).
    sender_delta_by_cycle = {
        1: pd.DataFrame(
            {"전기COA": ["E1"], "Sender CC": ["S1"], "Amounts": [1000.0]}
        )
    }
    df_ratio = pd.DataFrame(
        {"전기COA": ["E1", "E1"], "기존COA": ["100", "200"], "비중": [0.3, 0.7]}
    )
    out = decompose_sender_to_original_coa(sender_delta_by_cycle, df_ratio)

    assert list(out.columns) == ["차수", "전기COA", "기존COA", "Sender CC", "배분금액"]
    # Sorted by 차수 → 전기COA → 기존COA → Sender CC.
    assert list(out["기존COA"]) == ["100", "200"]
    amt = dict(zip(out["기존COA"], out["배분금액"]))
    assert amt["100"] == pytest.approx(300.0)
    assert amt["200"] == pytest.approx(700.0)


def test_decompose_sender_conserves_sent_total(pipeline_outputs):
    # Decomposition only splits each sent amount across base COAs, so the grand
    # total of 배분금액 must equal the total amount the senders pushed out.
    sender_delta_by_cycle = pipeline_outputs["sender_delta_by_cycle"]
    sender_decomposed = pipeline_outputs["sender_decomposed"]

    sent_total = sum(
        df["Amounts"].sum()
        for df in sender_delta_by_cycle.values()
        if not df.empty
    )
    assert sender_decomposed["배분금액"].sum() == pytest.approx(sent_total)


# WARNING cases


def test_run_allocation_loop_warns_on_unallocated_non_sender(loaded_inputs):
    # CC 1002 appears in the pivot with a non-zero balance but is not a sender.
    pivot = pd.DataFrame(
        {"1002": [1000.0], "1003": [0.0]},
        index=pd.Index(["E6200"], name="전기COA"),
    )
    cycle_df = pd.DataFrame(
        {"차수": [1], "Sender CC": ["1003"], "Receiver CC": ["1002"], "%": [1.0]}
    )
    with pytest.warns(UserWarning, match="1002"):
        run_allocation_loop(pivot, cycle_df)


def test_decompose_warns_when_alloc_total_nan():
    # A 전기COA whose ratio is NaN makes the allocated amount NaN, which fails the
    # finite/conservation check inside decompose_to_original_coa.
    df_ratio = pd.DataFrame(
        {"전기COA": ["X"], "기존COA": ["100"], "비중": [np.nan]}
    )
    received_by_cycle = {
        1: pd.DataFrame(
            {"전기COA": ["X"], "Receiver CC": ["1001"], "Amounts": [100.0]}
        )
    }
    with pytest.warns(UserWarning, match="총액 보존"):
        decompose_to_original_coa(received_by_cycle, df_ratio)


# FAILURE cases


def test_run_allocation_loop_unbalanced_when_pct_sum_below_one():
    # Sender distributes only 50%, leaving a residual -> unbalanced warning.
    pivot = pd.DataFrame(
        {"S": [1000.0], "R": [0.0]},
        index=pd.Index(["E6100"], name="전기COA"),
    )
    cycle_df = pd.DataFrame(
        {"차수": [1], "Sender CC": ["S"], "Receiver CC": ["R"], "%": [0.5]}
    )
    with pytest.warns(UserWarning, match="0원이 되지 않"):
        run_allocation_loop(pivot, cycle_df)


def test_run_allocation_loop_warning_names_sender_and_ratio():
    # The unbalanced warning must identify the cycle, sender, and ratio sum so
    # the user knows exactly which cycle.csv entry to fix.
    pivot = pd.DataFrame(
        {"S": [1000.0], "R": [0.0]},
        index=pd.Index(["E6100"], name="전기COA"),
    )
    cycle_df = pd.DataFrame(
        {"차수": [2], "Sender CC": ["S"], "Receiver CC": ["R"], "%": [0.5]}
    )
    with pytest.warns(UserWarning, match=r"2차 Sender S.*50\.0%"):
        run_allocation_loop(pivot, cycle_df)
