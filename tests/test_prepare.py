"""Tests for src.prepare: transfer-COA assignment, enrichment."""
from __future__ import annotations

import pandas as pd

from src.data.loader import load_coa_amount
from src.core.prepare import (
    assign_transfer_coa,
    build_enriched,
    fill_missing_cycle_cc,
)


def _object_coa_df():
    """Small object-dtype coa_df, mirroring the stage fill runs at."""
    return pd.DataFrame({
        "COA": pd.Series(["6100", "7200"], dtype="object"),
        "Cost Center": pd.Series(["1001", "2002"], dtype="object"),
        "Amounts": pd.Series([5_000_000.0, 2_000_000.0], dtype="float64"),
    })


def _cycle_df(sender_ccs, receiver_ccs):
    return pd.DataFrame({
        "차수": [1] * len(sender_ccs),
        "Sender CC": list(sender_ccs),
        "Receiver CC": list(receiver_ccs),
        "%": [1.0] * len(sender_ccs),
    })


# fill_missing_cycle_cc cases


def test_fill_missing_cycle_cc_adds_missing():
    coa_df = _object_coa_df()
    # 1001 exists in master; 1003 (receiver) and 9999 (sender) do not.
    cycle_df = _cycle_df(["1001", "9999"], ["1003", "1001"])
    result = fill_missing_cycle_cc(coa_df, cycle_df)

    added = result[result["Cost Center"].isin(["1003", "9999"])]
    assert set(added["Cost Center"]) == {"1003", "9999"}
    assert (added["Amounts"] == 0).all()
    assert added["COA"].isna().all()


def test_fill_missing_cycle_cc_no_duplicates():
    coa_df = _object_coa_df()
    # All cycle CCs already exist in the master -> nothing added, frame unchanged.
    cycle_df = _cycle_df(["1001"], ["2002"])
    result = fill_missing_cycle_cc(coa_df, cycle_df)

    assert len(result) == len(coa_df)
    assert (result["Cost Center"] == "1001").sum() == 1


def test_fill_missing_cycle_cc_preserves_columns_and_dtypes():
    coa_df = _object_coa_df()
    cycle_df = _cycle_df(["7777"], ["8888"])
    result = fill_missing_cycle_cc(coa_df, cycle_df)

    assert list(result.columns) == list(coa_df.columns)
    assert result["COA"].dtype == "object"
    assert result["Cost Center"].dtype == "object"
    assert result["Amounts"].dtype == "float64"


# SUCCESS cases


def test_assign_transfer_coa(loaded_inputs):
    df = assign_transfer_coa(loaded_inputs["coa_df"], loaded_inputs["mapping_df"])
    # COAs present in the mapping get their 전기COA value.
    assert (df.loc[df["COA"] == "6100", "전기COA"] == "E6100").all()
    assert (df.loc[df["COA"] == "6200", "전기COA"] == "E6200").all()
    # COAs absent from the mapping (direct costs) are left null (filled with ""
    # only later, in build_enriched).
    assert df.loc[df["COA"] == "7100", "전기COA"].isna().all()


def test_build_enriched_columns_and_fill(loaded_inputs):
    enriched = build_enriched(loaded_inputs["coa_df"], loaded_inputs["mapping_df"])

    # The original COA is exposed as 기존COA; fixed column order.
    assert list(enriched.columns) == ["전기COA", "기존COA", "Cost Center", "Amounts"]

    # Mapped COAs carry their 전기COA; direct costs get an empty string, not NaN.
    mapped = enriched[enriched["기존COA"] == "6100"]
    assert (mapped["전기COA"].astype(str) == "E6100").all()
    direct = enriched[enriched["기존COA"] == "7100"]
    assert (direct["전기COA"].astype(str) == "").all()
    assert enriched["전기COA"].astype(str).ne("nan").all()


def test_build_enriched_preserves_amount_total(loaded_inputs):
    # Enrichment only relabels; the total amount is unchanged.
    enriched = build_enriched(loaded_inputs["coa_df"], loaded_inputs["mapping_df"])
    assert enriched["Amounts"].sum() == loaded_inputs["coa_df"]["Amounts"].sum()
