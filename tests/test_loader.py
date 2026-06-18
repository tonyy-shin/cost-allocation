"""Tests for src.loader: CSV readers, code normalization, and path guard."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.loader import (
    _normalize_cycle_ratios, _validate_local_path,
    apply_category_dtypes, build_category_dtypes,
    load_coa_amount, load_cycle, load_mapping,
    normalize_code_column, parse_numeric_column, parse_percent_column,
)


def _is_blank(value) -> bool:
    """A normalized code is "blank" if it is empty or a missing value.

    Under pandas 3.0's `str` dtype, the loader's `<NA>` -> "" replacement leaves
    missing codes as NA rather than the empty string, so accept either form.
    """
    return pd.isna(value) or value == ""


# SUCCESS cases


def test_all_three_csvs_load(sample_paths):
    coa_df = load_coa_amount(sample_paths["coa_amount"])
    mapping_df = load_mapping(sample_paths["mapping"])
    cycle_df = load_cycle(sample_paths["cycle"])

    assert {"COA", "Cost Center", "Amounts"} <= set(coa_df.columns)
    assert {"전기COA", "기존COA"} <= set(mapping_df.columns)
    assert {"차수", "Sender CC", "Receiver CC", "%"} <= set(cycle_df.columns)
    # The CC list comes from the master's Cost Center column.
    assert coa_df["Cost Center"].nunique() > 0


def test_load_cycle_wide_to_long(tmp_path):
    # Wide format: 차수/Sender CC id columns + one column per Receiver CC. Blank
    # and 0 cells mean "no allocation" and must be dropped; the surviving rows
    # are sorted by (차수, Sender CC) and each sender's ratios sum to 1.0.
    p = tmp_path / "cycle.csv"
    p.write_text(
        "차수,Sender CC,1001,1002,1003,3001\n"
        "1,1001,,0.3,0.7,\n"
        "2,2001,0.5,0,,0.5\n",
        encoding="utf-8",
    )
    df = load_cycle(p)

    assert {"차수", "Sender CC", "Receiver CC", "%"} <= set(df.columns)
    # 2 senders × 2 non-empty receivers each = 4 rows; blanks and the explicit 0
    # are dropped.
    assert len(df) == 4
    # Sorted by (차수, Sender CC): 차수 1 group comes first.
    assert df["차수"].tolist() == [1, 1, 2, 2]
    # Cycle-1 sender 1001 allocates to 1002/1003 only (1001 and 3001 were blank).
    cyc1 = df[df["차수"] == 1].set_index("Receiver CC")["%"]
    assert set(cyc1.index) == {"1002", "1003"}
    assert cyc1["1002"] == pytest.approx(0.3)
    assert cyc1["1003"] == pytest.approx(0.7)
    # Cycle-2 sender 2001 allocates to 1001/3001 only (1002=0, 1003 blank dropped).
    cyc2 = df[df["차수"] == 2].set_index("Receiver CC")["%"]
    assert set(cyc2.index) == {"1001", "3001"}
    # Each sender's ratios sum to 1.0.
    assert cyc1.sum() == pytest.approx(1.0)
    assert cyc2.sum() == pytest.approx(1.0)


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


def test_parse_numeric_accounting_parentheses():
    out = parse_numeric_column(pd.Series(["(5,000,000)"]))
    assert out.iloc[0] == pytest.approx(-5000000.0)


def test_parse_numeric_unicode_minus():
    out = parse_numeric_column(pd.Series(["−5000000"]))
    assert out.iloc[0] == pytest.approx(-5000000.0)


def test_parse_numeric_trailing_minus():
    out = parse_numeric_column(pd.Series(["5000000-"]))
    assert out.iloc[0] == pytest.approx(-5000000.0)


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


# CYCLE RATIO VALIDATION
#
# _normalize_cycle_ratios inspects each (차수, Sender CC) group's '%' sum:
#   < 1e-9 from 1.0  → OK (no action)
#   1e-9..0.005 away → float precision, auto-normalize + warn
#   ≥ 0.005 away     → data error, collect all offenders, raise ValueError


def _cycle_df(rows) -> pd.DataFrame:
    """Build a minimal cycle DataFrame for _normalize_cycle_ratios tests."""
    return pd.DataFrame(rows, columns=["차수", "Sender CC", "Receiver CC", "%"])


def test_normalize_cycle_ratios_exact_sum_no_warning():
    df = _cycle_df([(1, "A", "B", 0.3), (1, "A", "C", 0.7)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = _normalize_cycle_ratios(df)
    assert result["%"].tolist() == pytest.approx([0.3, 0.7])


def test_normalize_cycle_ratios_float_precision_auto_normalizes():
    # 5e-9 is in [1e-9, 0.005): triggers auto-normalization with a UserWarning.
    eps = 5e-9
    df = _cycle_df([(1, "A", "B", 0.5), (1, "A", "C", 0.5 + eps)])
    with pytest.warns(UserWarning, match="자동 정규화"):
        result = _normalize_cycle_ratios(df)
    assert result["%"].sum() == pytest.approx(1.0)


def test_normalize_cycle_ratios_data_error_raises():
    # sum = 1.05 (diff 0.05 ≥ 0.005): data error, must raise.
    df = _cycle_df([(1, "A", "B", 0.6), (1, "A", "C", 0.45)])
    with pytest.raises(ValueError, match="직접 수정"):
        _normalize_cycle_ratios(df)


def test_normalize_cycle_ratios_multiple_errors_all_reported():
    # Two separate groups each with data errors: both must appear in the message.
    df = _cycle_df([
        (1, "A", "B", 1.05),
        (2, "X", "Y", 0.800),
    ])
    with pytest.raises(ValueError) as exc_info:
        _normalize_cycle_ratios(df)
    msg = str(exc_info.value)
    assert "차수=1" in msg and "A" in msg
    assert "차수=2" in msg and "X" in msg


def test_normalize_cycle_ratios_mixed_raises_without_normalizing():
    # Float-precision group + data-error group: ValueError is raised and the
    # float-precision group is never normalized (no '자동 정규화' warning emitted).
    eps = 5e-9
    df = _cycle_df([
        (1, "A", "B", 0.5),
        (1, "A", "C", 0.5 + eps),
        (2, "X", "Y", 1.05),
    ])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="직접 수정"):
            _normalize_cycle_ratios(df)
    assert not any("자동 정규화" in str(w.message) for w in caught)


def test_normalize_cycle_ratios_zero_sum_skipped():
    # A group whose ratios are all 0 has sum=0; it is skipped without error or warning.
    df = _cycle_df([(1, "A", "B", 0.0), (1, "A", "C", 0.0)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = _normalize_cycle_ratios(df)
    assert result["%"].tolist() == pytest.approx([0.0, 0.0])


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
    # Korean header 차수 saved as EUC-KR. Wide format: the single Receiver CC
    # column (1002) gets 100% so the ratio-sum check in _normalize_cycle_ratios
    # passes without error.
    p.write_text(
        "차수,Sender CC,1002\n1,1001,1.0\n", encoding="euc-kr"
    )
    df = load_cycle(p)
    assert {"차수", "Sender CC", "Receiver CC", "%"} <= set(df.columns)
    assert df["Sender CC"].iloc[0] == "1001"
    assert df["Receiver CC"].iloc[0] == "1002"


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


# CATEGORY cross-sheet mismatch cases
#
# Casting a code column to a CategoricalDtype whose categories come from another
# sheet drops codes that do not exist there to NaN. Since pandas 3.0 that cast
# also emits a deprecation (a future version will raise), so apply_category_dtypes
# masks unknowns first and reports exactly which codes were dropped.


def _category_frames(기존coa="6100"):
    coa_df = pd.DataFrame(
        {"COA": ["6100"], "Cost Center": ["1001"], "Amounts": [1.0]}
    )
    mapping_df = pd.DataFrame({"전기COA": ["E6100"], "기존COA": [기존coa]})
    return coa_df, mapping_df


def test_apply_category_silent_on_base_coa_absent_from_amount_sheet():
    # The mapping sheet legitimately lists base COAs absent from this period's
    # amount sheet, so 기존COA 7777 is masked to NaN WITHOUT any warning.
    coa_df, mapping_df = _category_frames(기존coa="7777")
    dtypes = build_category_dtypes(coa_df, mapping_df)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, mapping_out = apply_category_dtypes(
            coa_df, mapping_df, dtypes=dtypes
        )
    assert pd.isna(mapping_out["기존COA"].iloc[0])


def test_apply_category_no_future_warning_on_mismatch():
    # A base-COA cross-sheet mismatch must NOT emit the pandas deprecation
    # (which would raise in a future version); it is masked silently.
    coa_df, mapping_df = _category_frames(기존coa="7777")
    dtypes = build_category_dtypes(coa_df, mapping_df)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        apply_category_dtypes(coa_df, mapping_df, dtypes=dtypes)
    assert not any(issubclass(w.category, FutureWarning) for w in caught)


def test_apply_category_clean_inputs_emit_no_warning():
    # When every code matches across sheets, no warning of any kind fires.
    coa_df, mapping_df = _category_frames()
    dtypes = build_category_dtypes(coa_df, mapping_df)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        apply_category_dtypes(coa_df, mapping_df, dtypes=dtypes)


# FAILURE cases


def test_load_coa_amount_drops_blank_cost_center(tmp_path):
    p = tmp_path / "coa.csv"
    # The first row has a blank Cost Center; it must be dropped so it never
    # becomes an empty-CC row in the by_cc output.
    p.write_text(
        "COA,Cost Center,Amounts\n"
        "6100,,0\n"
        "6100,1001,500\n",
        encoding="utf-8",
    )
    df = load_coa_amount(p)
    assert "" not in set(df["Cost Center"])
    assert df["Cost Center"].tolist() == ["1001"]


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
    # Wide format requires the id columns 차수 and Sender CC; here 차수 is missing.
    p.write_text("Sender CC,1002\n1001,1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="필수 컬럼"):
        load_cycle(p)


def test_validate_local_path_rejects_url():
    # Passed as a string: a Windows Path would collapse the "//" in "http://"
    # and slip past the guard, so the URL string is checked directly.
    with pytest.raises(ValueError, match="Remote paths"):
        _validate_local_path("http://example.com/cc.csv")
