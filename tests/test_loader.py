"""Tests for src.loader: CSV readers, code normalization, and path guard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.loader import (
    _validate_local_path, load_cc, load_coa_amount, load_cycle, load_mapping,
    normalize_code_column, parse_numeric_column, parse_percent_column,
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


# NUMERIC / PERCENT parsing
#
# Excel "Number" cells with a thousands separator export as "5,000,000" and
# "Percentage" cells export as "30%". parse_numeric_column / parse_percent_column
# coerce these back to numbers so the downstream arithmetic does not hit
# str-vs-str type errors.


def test_parse_numeric_strips_thousands_separator():
    out = parse_numeric_column(pd.Series(["5,000,000"]))
    assert out.iloc[0] == 5000000.0


def test_parse_numeric_plain_number_unchanged():
    out = parse_numeric_column(pd.Series(["3000000"]))
    assert out.iloc[0] == 3000000.0


def test_parse_numeric_warns_on_non_numeric_value():
    # "abc" cannot be coerced, so it becomes NaN and a warning fires.
    with pytest.warns(UserWarning, match="변환되지 않은 값"):
        out = parse_numeric_column(pd.Series(["abc"]), "coa_amount.csv")
    assert pd.isna(out.iloc[0])


def test_parse_percent_strips_percent_sign():
    out = parse_percent_column(pd.Series(["30%"]))
    assert out.iloc[0] == pytest.approx(0.3)


def test_parse_percent_decimal_unchanged():
    # No '%' sign: already a decimal ratio, kept as-is.
    out = parse_percent_column(pd.Series(["0.3"]))
    assert out.iloc[0] == pytest.approx(0.3)


def test_parse_percent_warns_on_non_numeric_value():
    with pytest.warns(UserWarning, match="변환되지 않은 값"):
        out = parse_percent_column(pd.Series(["abc"]), "cycle.csv")
    assert pd.isna(out.iloc[0])


def test_parse_percent_warns_on_value_above_one_without_sign():
    # "30" with no '%' is likely meant as 0.3; it is NOT auto-corrected, only
    # flagged, so the value stays 30.0.
    with pytest.warns(UserWarning, match="1을 초과하는 값"):
        out = parse_percent_column(pd.Series(["30"]), "cycle.csv")
    assert out.iloc[0] == pytest.approx(30.0)


# ENCODING cases
#
# Excel "CSV (comma delimited)" on Korean Windows saves files in EUC-KR, which
# corrupts Korean headers when read as UTF-8 and surfaces as a "필수 컬럼 누락"
# error. The loaders fall back to EUC-KR so these files load correctly.


def test_load_mapping_reads_euc_kr(tmp_path):
    p = tmp_path / "mapping.csv"
    # Korean headers (전기COA, 기존COA) saved as EUC-KR, like Excel's default CSV.
    p.write_text("전기COA,기존COA\nE6100,6100\n", encoding="euc-kr")
    df = load_mapping(p)
    assert {"전기COA", "기존COA"} <= set(df.columns)
    assert df["기존COA"].iloc[0] == "6100"


def test_load_cycle_reads_euc_kr(tmp_path):
    p = tmp_path / "cycle.csv"
    # Korean header 차수 saved as EUC-KR.
    p.write_text(
        "차수,Sender CC,Receiver CC,%\n1,1001,1002,0.3\n", encoding="euc-kr"
    )
    df = load_cycle(p)
    assert {"차수", "Sender CC", "Receiver CC", "%"} <= set(df.columns)
    assert df["Sender CC"].iloc[0] == "1001"


def test_load_mapping_still_reads_utf8_sig(tmp_path):
    # UTF-8 with BOM (Excel's "CSV UTF-8") must keep working unchanged.
    p = tmp_path / "mapping.csv"
    p.write_text("전기COA,기존COA\nE6100,6100\n", encoding="utf-8-sig")
    df = load_mapping(p)
    assert {"전기COA", "기존COA"} <= set(df.columns)


def test_load_mapping_unknown_encoding_raises(tmp_path):
    p = tmp_path / "mapping.csv"
    # 0xFF is an invalid lead byte for both UTF-8 and EUC-KR, so neither decoder
    # succeeds and the loader reports an encoding error rather than crashing.
    p.write_bytes(b"\xff\xff\xff\n")
    with pytest.raises(ValueError, match="인코딩"):
        load_mapping(p)


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
