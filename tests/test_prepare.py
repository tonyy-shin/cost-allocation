"""Tests for src.prepare: override, transfer-COA assignment, enrichment, CC validation."""
from __future__ import annotations

import pandas as pd

from src.loader import load_coa_amount
from src.prepare import (
    apply_override,
    assign_transfer_coa,
    build_enriched,
    validate_cycle_cc,
)


# apply_override cases


def test_apply_override_replaces_matching_combos(sample_paths, override_df):
    # apply_override runs on object-dtype coa_df (before CategoricalDtype is applied).
    coa_df = load_coa_amount(sample_paths["coa_amount"])
    result = apply_override(coa_df, override_df)

    def amount(coa, cc):
        row = result[(result["COA"] == coa) & (result["Cost Center"] == cc)]
        return row["Amounts"].iloc[0]

    # Matched (COA, CC) combos take the override value (master had them at 0).
    assert amount("6100", "1001") == 5_000_000
    assert amount("6200", "2001") == 500_000


def test_apply_override_keeps_unmatched_rows(sample_paths, override_df):
    coa_df = load_coa_amount(sample_paths["coa_amount"])
    result = apply_override(coa_df, override_df)

    # 7200/2002 is absent from override_df, so its original value is preserved.
    row = result[(result["COA"] == "7200") & (result["Cost Center"] == "2002")]
    assert row["Amounts"].iloc[0] == 2_000_000


def test_apply_override_preserves_columns_and_dtypes(sample_paths, override_df):
    coa_df = load_coa_amount(sample_paths["coa_amount"])
    result = apply_override(coa_df, override_df)

    # Column order is unchanged; the key columns keep their (object) dtype so the
    # later Categorical cast is unaffected. Amounts is the canonical float64.
    assert list(result.columns) == list(coa_df.columns)
    assert result["COA"].dtype == coa_df["COA"].dtype
    assert result["Cost Center"].dtype == coa_df["Cost Center"].dtype
    assert result["Amounts"].dtype == "float64"


def test_apply_override_collapses_duplicate_combos():
    # A duplicated (COA, CC) in the master must be summed to one row before the
    # override merge, otherwise the override value would broadcast and inflate.
    coa_df = pd.DataFrame({
        "COA": ["6100", "6100", "7100"],
        "Cost Center": ["1001", "1001", "2002"],
        "Amounts": [1_000_000.0, 2_000_000.0, 500_000.0],
    })
    override_df = pd.DataFrame({
        "COA": ["6100"],
        "Cost Center": ["1001"],
        "Amounts": [9_000_000.0],
    })
    result = apply_override(coa_df, override_df)

    # The two 6100/1001 rows collapse to one, replaced by the single override value.
    matched = result[(result["COA"] == "6100") & (result["Cost Center"] == "1001")]
    assert len(matched) == 1
    assert matched["Amounts"].iloc[0] == 9_000_000.0
    # Unmatched row keeps its original value; no broadcast inflation.
    assert result["Amounts"].sum() == 9_000_000.0 + 500_000.0


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
