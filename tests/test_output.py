"""Tests for src.output: full-grid build_result and CSV writing."""
from __future__ import annotations

import pandas as pd
import pytest

from src.allocation import TOTAL_COL
from src.output import save_result


def _alloc_cols(df):
    return [c for c in df.columns if "배분금액" in c]


# SUCCESS cases


def test_build_result_full_grid_row_count(pipeline_outputs, loaded_inputs):
    # Full (기존COA x CC) grid: 4 base COAs x 6 CCs = 24 rows.
    result = pipeline_outputs["result"]
    n_coa = loaded_inputs["raw_coa_df"]["COA"].nunique()
    n_cc = loaded_inputs["cc_df"]["CC"].nunique()
    assert n_coa == 4 and n_cc == 6
    assert len(result) == 24


def test_direct_rows_have_zero_allocation(pipeline_outputs):
    result = pipeline_outputs["result"]
    # Direct-cost rows carry an empty 전기COA and must be zero in every cycle col.
    direct = result[result["전기COA"] == ""]
    assert len(direct) > 0
    assert (direct[_alloc_cols(result)] == 0).all().all()


def test_common_rows_total_equals_alloc_sum(pipeline_outputs):
    result = pipeline_outputs["result"]
    # For common-cost rows, 배부합계 is the sum of the per-cycle allocations.
    # (Direct rows instead carry their original Amounts in 배부합계 with zero
    # allocation columns, so the identity is asserted on common rows only.)
    common = result[result["전기COA"] != ""]
    assert len(common) > 0
    expected = common[_alloc_cols(result)].sum(axis=1)
    assert common[TOTAL_COL].to_numpy() == pytest.approx(expected.to_numpy())


def test_save_result_writes_utf8_sig_csv(pipeline_outputs, tmp_path):
    out_path = save_result(pipeline_outputs["result"], tmp_path)

    assert out_path.exists()
    assert out_path.name == "result.csv"
    # utf-8-sig writes a BOM prefix for Excel compatibility.
    assert out_path.read_bytes().startswith(b"\xef\xbb\xbf")
    # Round-trips back to the same shape with utf-8-sig decoding.
    reloaded = pd.read_csv(out_path, encoding="utf-8-sig")
    assert len(reloaded) == len(pipeline_outputs["result"])
    assert list(reloaded.columns) == list(pipeline_outputs["result"].columns)
