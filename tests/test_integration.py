"""End-to-end integration tests over the full pipeline (pipeline_outputs)."""
from __future__ import annotations

import warnings

import pytest

from src.allocation import (
    build_pivot_matrix, decompose_sender_to_original_coa, run_allocation_loop,
)
from src.output import build_result
from src.prepare import (
    aggregate_detail, aggregate_for_allocation, assign_transfer_coa,
    calculate_coa_ratio, separate_common_direct, validate_sender_coverage,
)


def test_end_to_end_conservation(pipeline_outputs, loaded_inputs):
    result = pipeline_outputs["result"]
    df_direct = pipeline_outputs["df_direct"]

    # The sender-keyed result holds only distributed common cost; its 배분금액
    # total equals the amount the senders pushed out (== receiver-side total).
    allocated = result["배분금액"].sum()
    direct_total = df_direct["Amounts"].sum()
    total_in = loaded_inputs["raw_coa_df"]["Amounts"].sum()

    # sample_data deliberately leaves CC 1002's common cost undistributed (it is
    # never a Sender). That residual is reported by validate_sender_coverage and
    # must be added back for the input/output identity to hold.
    residual = sum(
        amount for _cc, amount in
        validate_sender_coverage(pipeline_outputs["df_5b"], loaded_inputs["cycle_df"])
    )
    assert allocated + direct_total + residual == pytest.approx(total_in)


def test_result_row_count(pipeline_outputs):
    # Rows = the (차수, 전기COA, 기존COA, Sender CC) combinations the senders
    # actually distributed. Fixed expected count (independent oracle); update
    # when sample_data changes.
    assert len(pipeline_outputs["result"]) == 4


def test_only_expected_residual_warnings_on_happy_path(loaded_inputs):
    # Re-run the full pipeline while capturing warnings. sample_data is not
    # warning-free: CC 1002 holds common cost but is not a Sender, so exactly
    # one residual warning is expected. Assert no *other* warnings appear.
    coa_df = loaded_inputs["coa_df"]
    mapping_df = loaded_inputs["mapping_df"]
    cycle_df = loaded_inputs["cycle_df"]
    cc_list = loaded_inputs["cc_list"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        enriched = assign_transfer_coa(coa_df, mapping_df)
        df_common, _ = separate_common_direct(enriched)
        df_5a = aggregate_detail(df_common)
        df_5b = aggregate_for_allocation(df_5a)
        df_ratio = calculate_coa_ratio(df_5a)

        pivot = build_pivot_matrix(df_5b, cc_list)
        _, _, sender_delta_by_cycle = run_allocation_loop(pivot, cycle_df)
        sender_decomposed = decompose_sender_to_original_coa(
            sender_delta_by_cycle, df_ratio
        )
        build_result(sender_decomposed)

    residual_ccs = {cc for cc, _amt in validate_sender_coverage(df_5b, cycle_df)}
    unexpected = [
        str(w.message) for w in caught
        if not any(cc in str(w.message) for cc in residual_ccs)
    ]
    assert unexpected == [], f"Unexpected warnings on happy path: {unexpected}"
