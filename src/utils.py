from __future__ import annotations

import os
import warnings
from pathlib import Path

import pandas as pd


# Parsing Utilities


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV trying UTF-8 first, then EUC-KR (CP949).

    Excel on Korean Windows saves "CSV (comma delimited)" files in EUC-KR
    rather than UTF-8, which corrupts Korean headers (e.g. 전기COA, 차수) and
    makes required-column checks fail. utf-8-sig is tried first so existing
    UTF-8/BOM files behave exactly as before; only on UnicodeDecodeError do we
    fall back to euc-kr.

    Parameters
    ----------
    path : Path
        Local CSV path. Extra keyword arguments (dtype, etc.) are forwarded
        to pandas.read_csv unchanged.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If the file decodes as neither UTF-8 nor EUC-KR.
    """
    for encoding in ("utf-8-sig", "euc-kr", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
    raise ValueError(
        f"{os.path.basename(path)} 인코딩을 인식할 수 없습니다 (UTF-8 또는 EUC-KR이 아님)."
    )


def normalize_code_column(series: pd.Series, filename: str = "") -> pd.Series:
    """Normalize code strings '7832.0' to '7832'.

    CSVs exported from Excel may store integer codes as '7832.0' strings.
    Converts via float -> int -> str.

    Notes
    -----
    If leading-zero preservation is ever required, this function must be
    revised before  float -> int conversion.

    Parameters
    ----------
    series : pd.Series
        Code column read with dtype=str.
    filename : str, optional
        Source CSV filename, used only to make the conversion warning more
        helpful. Defaults to "" when the caller does not have it.

    Returns
    -------
    pd.Series
        Normalized string code column.
    """
    numeric = pd.to_numeric(series, errors="coerce")  # str("100.0") to float(100.0)

    # Detect values that were non-empty before conversion but became NaN after.
    # These are silently dropped by the coerce, so warn the user about them.
    before = series.astype(str).str.strip()
    lost_mask = (
        before.notna()
        & (before != "")
        & (before != "nan")
        & numeric.isna()
    )
    lost_values = series[lost_mask].astype(str).unique().tolist()
    if lost_values:
        col_name = series.name if series.name is not None else "코드"
        file_hint = f"{filename} " if filename else ""
        warnings.warn(
            f"{file_hint}{col_name} 컬럼에서 코드로 변환되지 않은 값이 있습니다: "
            f"{lost_values} 해당 행은 매핑에서 제외됩니다."
        )

    return (
        numeric
        .astype("Int64")  # float to int
        .astype(str)  # int to str
        .fillna("")  # drop missing value as ""
    )


def _warn_lost_values(
    original: pd.Series,
    numeric: pd.Series,
    *,
    filename: str,
    suffix: str,
) -> None:
    """Warn about values that became NaN during numeric coercion.

    Mirrors the lost-value detection in normalize_code_column: a value that was
    non-empty before conversion but NaN afterwards was silently dropped by the
    coerce, so it is surfaced to the user. Genuinely empty or missing cells are
    excluded so they do not generate noise.

    Parameters
    ----------
    original : pd.Series
        The column as read, before numeric conversion.
    numeric : pd.Series
        The coerced numeric column.
    filename : str
        Source CSV filename for the warning message ("" when unknown).
    suffix : str
        Trailing sentence describing what happens to the unconverted values.
    """
    before = original.astype(str).str.strip()
    lost_mask = (
        before.notna()
        & (before != "")
        & (before != "nan")
        & numeric.isna()
    )
    lost_values = original[lost_mask].astype(str).unique().tolist()
    if lost_values:
        col_name = original.name if original.name is not None else "값"
        file_hint = f"{filename} " if filename else ""
        warnings.warn(
            f"{file_hint}{col_name} 컬럼에서 숫자로 변환되지 않은 값이 있습니다: "
            f"{lost_values} {suffix}"
        )


def parse_numeric_column(series: pd.Series, filename: str = "") -> pd.Series:
    """Parse Excel-exported numeric text ('5,000,000') into float64.

    Excel cells formatted as "Number" with a thousands separator are written to
    CSV as comma-grouped, quoted text (e.g. "5,000,000"), which pandas keeps as
    str rather than a number. Stripping the commas before pd.to_numeric lets
    these values survive the CSV round-trip. Unconvertible values become NaN and
    trigger a warning.

    Parameters
    ----------
    series : pd.Series
        Amount column, possibly read as str.
    filename : str, optional
        Source CSV filename, used only to make the conversion warning more
        helpful. Defaults to "" when the caller does not have it.

    Returns
    -------
    pd.Series
        float64 column. Unparseable entries are NaN.
    """
    cleaned = (
        series.astype(str)
        .str.strip()
        # Unicode minus (U+2212) → ASCII hyphen
        .str.replace("−", "-", regex=False)
        # Accounting parentheses: (5,000,000) → -5000000
        .str.replace(r"^\(([0-9,]+)\)$", r"-\1", regex=True)
        # Trailing minus: 5000000- → -5000000
        .str.replace(r"^([0-9,]+)-$", r"-\1", regex=True)
        # Remove thousands separators
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    numeric = pd.to_numeric(cleaned, errors="coerce")
    _warn_lost_values(
        series,
        numeric,
        filename=filename,
        suffix="해당 값은 NaN으로 처리됩니다.",
    )
    return numeric


def parse_percent_column(series: pd.Series, filename: str = "") -> pd.Series:
    """Parse the cycle '%' column into a decimal ratio (0.3 = 30%).

    Excel cells formatted as "Percentage" are written to CSV with a trailing '%'
    (e.g. "30%"), while values entered as plain decimals are written as-is
    ("0.3"). Entries ending in '%' have the sign stripped and are divided by 100;
    all other entries are treated as already-decimal ratios. A plain "30"
    (no '%') is intentionally left as 30.0 rather than guessed to be 0.3 — only a
    separate warning is emitted for values above 1.0. Unconvertible values become
    NaN and trigger a warning.

    Parameters
    ----------
    series : pd.Series
        '%' column, possibly read as str.
    filename : str, optional
        Source CSV filename, used only to make the warning messages more
        helpful. Defaults to "" when the caller does not have it.

    Returns
    -------
    pd.Series
        float64 decimal ratios. Unparseable entries are NaN.
    """
    s = series.astype(str).str.strip()
    has_pct = s.str.endswith("%")
    cleaned = s.str.rstrip("%").str.replace(",", "", regex=False).str.strip()
    numeric = pd.to_numeric(cleaned, errors="coerce")
    result = numeric.where(~has_pct, numeric / 100.0)

    _warn_lost_values(
        series,
        result,
        filename=filename,
        suffix="소수(0.3) 또는 백분율(30%) 형식인지 확인하세요.",
    )

    # A value above 1.0 with no '%' sign is likely a percentage typed as a whole
    # number (e.g. "30" meaning 0.3). It is not auto-corrected; only flagged.
    suspicious = (~has_pct) & result.notna() & (result > 1.0)
    if suspicious.any():
        bad = series[suspicious].astype(str).unique().tolist()
        col_name = series.name if series.name is not None else "%"
        file_hint = f"{filename} " if filename else ""
        warnings.warn(
            f"{file_hint}{col_name} 컬럼에 1을 초과하는 값이 있습니다: {bad} "
            f"소수(0.3) 형식인지 확인하세요."
        )

    return result
