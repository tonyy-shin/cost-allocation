"""Tests for src.prepare: transfer-COA assignment, enrichment, CC validation."""
from __future__ import annotations

import pandas as pd

from src.prepare import assign_transfer_coa, build_enriched, validate_cycle_cc


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


# FAILURE cases


def test_validate_cycle_cc_returns_unknown_ccs(loaded_inputs):
    cycle_df = pd.DataFrame({
        "차수": [1, 1],
        "Sender CC": ["1001", "9999"],
        "Receiver CC": ["8888", "1002"],
        "%": [0.5, 0.5],
    })
    unknown = validate_cycle_cc(cycle_df, loaded_inputs["coa_df"])
    assert unknown == ["8888", "9999"]
