"""Tests for src.prepare: transfer-COA assignment, splitting, ratios, validators."""
from __future__ import annotations

import pandas as pd
import pytest

from src.allocation import build_pivot_matrix, run_allocation_loop
from src.prepare import (
    assign_transfer_coa, calculate_coa_ratio, separate_common_direct,
    validate_cycle_cc, validate_sender_coverage,
)


# SUCCESS cases


def test_assign_transfer_coa(loaded_inputs):
    df = assign_transfer_coa(loaded_inputs["coa_df"], loaded_inputs["mapping_df"])
    # COAs present in the mapping get their 전기COA value.
    assert (df.loc[df["COA"] == "6100", "전기COA"] == "E6100").all()
    assert (df.loc[df["COA"] == "6200", "전기COA"] == "E6200").all()
    # COAs absent from the mapping (direct costs) are left null (filled with ""
    # only later, in separate_common_direct).
    assert df.loc[df["COA"] == "7100", "전기COA"].isna().all()


def test_separate_common_direct(loaded_inputs):
    df = assign_transfer_coa(loaded_inputs["coa_df"], loaded_inputs["mapping_df"])
    df_common, df_direct = separate_common_direct(df)

    assert (df_common["전기COA"].astype(str) != "").all()
    assert (df_direct["전기COA"].astype(str) == "").all()


def test_calculate_coa_ratio_sums_to_one(pipeline_outputs):
    ratio_sums = (
        pipeline_outputs["df_ratio"]
        .groupby("전기COA", observed=True)["비중"]
        .sum()
    )
    assert ratio_sums.to_numpy() == pytest.approx(1.0)


def test_validate_sender_coverage_passes_when_all_senders():
    # A common-cost CC that is also a registered sender produces no violation.
    df_5b = pd.DataFrame(
        {"전기COA": ["E6100"], "Cost Center": ["1001"], "Amounts": [100.0]}
    )
    cycle_df = pd.DataFrame(
        {"차수": [1], "Sender CC": ["1001"], "Receiver CC": ["1002"], "%": [1.0]}
    )
    assert validate_sender_coverage(df_5b, cycle_df) == []


def test_validate_sender_coverage_flags_unregistered(pipeline_outputs, loaded_inputs):
    # In sample_data, CC 1002 holds 1.5M of common cost (COA 6200) but never
    # appears as a Sender, so it is reported.
    violators = validate_sender_coverage(
        pipeline_outputs["df_5b"], loaded_inputs["cycle_df"]
    )
    assert violators == [("1002", 1500000.0)]


# WARNING cases


def test_run_allocation_loop_warns_for_non_sender_common_cost(
    pipeline_outputs, loaded_inputs
):
    # CC 1002 carries common cost but is not a Sender -> its balance is never
    # distributed, which the loop reports.
    pivot = pipeline_outputs["pivot"]
    with pytest.warns(UserWarning, match="1002"):
        run_allocation_loop(pivot, loaded_inputs["cycle_df"])


# FAILURE cases


def test_validate_cycle_cc_returns_unknown_ccs(loaded_inputs):
    cycle_df = pd.DataFrame({
        "차수": [1, 1],
        "Sender CC": ["1001", "9999"],
        "Receiver CC": ["8888", "1002"],
        "%": [0.5, 0.5],
    })
    unknown = validate_cycle_cc(cycle_df, loaded_inputs["cc_df"])
    assert unknown == ["8888", "9999"]
