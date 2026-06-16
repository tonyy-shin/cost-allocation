"""Tests for src.loader: CSV readers, code normalization, and path guard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.loader import (
    _validate_local_path, load_cc, load_coa_amount, load_cycle, load_mapping,
    normalize_code_column,
)


def _is_blank(value) -> bool:
    """A normalized code is "blank" if it is empty or a missing value.

    Under pandas 3.0's `str` dtype, the loader's `<NA>` -> "" replacement leaves
    missing codes as NA rather than the empty string, so accept either form.
    """
    return pd.isna(value) or value == ""


# SUCCESS cases


def test_all_four_csvs_load(sample_paths):
    cc_df = load_cc(sample_paths["cc"])
    coa_df = load_coa_amount(sample_paths["coa_amount"])
    mapping_df = load_mapping(sample_paths["mapping"])
    cycle_df = load_cycle(sample_paths["cycle"])

    assert "CC" in cc_df.columns
    assert {"COA", "Cost Center", "Amounts"} <= set(coa_df.columns)
    assert {"전기COA", "기존COA"} <= set(mapping_df.columns)
    assert {"차수", "Sender CC", "Receiver CC", "%"} <= set(cycle_df.columns)
    assert len(cc_df) > 0


def test_normalize_strips_trailing_zero():
    out = normalize_code_column(pd.Series(["7832.0"]))
    assert out.iloc[0] == "7832"


def test_normalize_nan_to_empty():
    out = normalize_code_column(pd.Series([np.nan]))
    assert _is_blank(out.iloc[0])


def test_normalize_empty_to_empty():
    out = normalize_code_column(pd.Series([""]))
    assert _is_blank(out.iloc[0])


# WARNING cases


def test_normalize_warns_on_non_numeric_value():
    # "E6100" cannot be coerced to a number, so it is dropped and a warning fires.
    with pytest.warns(UserWarning, match="변환되지 않은 값"):
        out = normalize_code_column(pd.Series(["E6100"]))
    assert _is_blank(out.iloc[0])


# FAILURE cases


def test_load_cc_missing_column_raises(tmp_path):
    p = tmp_path / "cc.csv"
    p.write_text("X\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="필수 컬럼"):
        load_cc(p)


def test_load_coa_amount_missing_column_raises(tmp_path):
    p = tmp_path / "coa.csv"
    # Missing the "Amounts" column.
    p.write_text("COA,Cost Center\n6100,1001\n", encoding="utf-8")
    with pytest.raises(ValueError, match="필수 컬럼"):
        load_coa_amount(p)


def test_load_mapping_missing_column_raises(tmp_path):
    p = tmp_path / "mapping.csv"
    # Missing the "기존COA" column.
    p.write_text("전기COA\nE6100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="필수 컬럼"):
        load_mapping(p)


def test_load_cycle_missing_column_raises(tmp_path):
    p = tmp_path / "cycle.csv"
    # Missing the "%" column.
    p.write_text("차수,Sender CC,Receiver CC\n1,1001,1002\n", encoding="utf-8")
    with pytest.raises(ValueError, match="필수 컬럼"):
        load_cycle(p)


def test_validate_local_path_rejects_url():
    # Passed as a string: a Windows Path would collapse the "//" in "http://"
    # and slip past the guard, so the URL string is checked directly.
    with pytest.raises(ValueError, match="Remote paths"):
        _validate_local_path("http://example.com/cc.csv")
