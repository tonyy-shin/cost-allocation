from __future__ import annotations

import os
import warnings
from pathlib import Path

import pandas as pd
from pandas.api.types import CategoricalDtype


# Helper Functions


def _validate_local_path(path: Path) -> None:
    """Reject paths that start with a URL scheme(company policy).

    pd.read_csv accepts URLs as input, so this guard is called
    before each read_csv call to prevent remote fetches.

    Raises
    ------
    ValueError
        If the path starts with http://, https://, ftp://, or ftps://
    """
    url = ("http://", "https://", "ftp://", "ftps://")
    if str(path).startswith(url):
        raise ValueError(f"Remote paths are not allowed: {path}")


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


def _normalize_cycle_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and auto-normalize '%' sums for each (차수, Sender CC) group.

    Three tolerance tiers applied after parse_percent_column:

    - abs(sum - 1.0) < 1e-9         : OK — no action
    - 1e-9 ≤ abs(sum - 1.0) < 0.005 : float precision — auto-normalize + warn
    - abs(sum - 1.0) ≥ 0.005        : data error — all offending groups are
                                       collected before a single ValueError is raised

    Groups whose '%' sum is 0 or NaN are skipped without error.
    When any data-error group exists, no float-precision group is normalized
    (the ValueError is raised before any modification is made).

    Parameters
    ----------
    df : pd.DataFrame
        cycle.csv DataFrame with '%' already parsed to decimal ratios.

    Returns
    -------
    pd.DataFrame
        Copy of df with '%' normalized for float-precision groups.

    Raises
    ------
    ValueError
        If one or more (차수, Sender CC) groups deviate from 1.0 by ≥ 0.005.
        All offending groups are listed in a single message.
    """
    df = df.copy()
    errors: list[str] = []
    to_normalize: list[tuple] = []

    for (cycle_num, sender), group in df.groupby(["차수", "Sender CC"]):
        pct_sum = group["%"].sum()
        if pd.isna(pct_sum) or pct_sum == 0.0:
            continue
        diff = abs(pct_sum - 1.0)
        if diff < 1e-9:
            continue
        if diff < 0.005:
            to_normalize.append((group.index, pct_sum, cycle_num, sender))
        else:
            errors.append(
                f"cycle.csv 차수={cycle_num}, Sender CC={sender}: "
                f"비율 합이 {pct_sum:.6f}입니다. "
                f"합계가 1.0이 되도록 cycle.csv를 직접 수정해 주세요."
            )

    if errors:
        raise ValueError("\n".join(errors))

    for idx, pct_sum, cycle_num, sender in to_normalize:
        df.loc[idx, "%"] = df.loc[idx, "%"] / pct_sum
        warnings.warn(
            f"cycle.csv 차수={cycle_num}, Sender CC={sender}: "
            f"비율 합이 {pct_sum:.9f}이므로 자동 정규화했습니다."
        )

    return df


# Step 1: CSV readers


def load_coa_amount(path: Path) -> pd.DataFrame:
    """Read the COA·CC master amount CSV and return DataFrame.

    This sheet is also the source of the CC list: every Cost Center used by the
    pipeline is taken from its ``Cost Center`` unique values.

    Parameters
    ----------
    path : Path
        Path to coa_amount.csv.

    Returns
    -------
    pd.DataFrame
        Columns: COA (str), Cost Center (str), Amounts (float64)
    """
    _validate_local_path(path)

    df = _read_csv(
        path,
        dtype = {"COA": str, "Cost Center": str},
    )

    required = ["COA", "Cost Center", "Amounts"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)}에 필수 컬럼이 없습니다: {missing}")

    fname = os.path.basename(path)
    df["COA"] = normalize_code_column(df["COA"], fname)
    df["Cost Center"] = normalize_code_column(df["Cost Center"], fname)
    df["Amounts"] = parse_numeric_column(df["Amounts"], fname)
    return df


def load_pre_allocation(path: Path) -> dict[str, float]:
    """Read pre_allocation.csv and sum Amounts by Cost Center.

    The file shares coa_amount.csv's schema (COA, Cost Center, Amounts), so the
    same reader is reused. The COA column is ignored; only the CC-level total is
    needed, exclusively for the by_cc output's 배부전금액 column.

    Parameters
    ----------
    path : Path
        Path to pre_allocation.csv.

    Returns
    -------
    dict[str, float]
        Cost Center code -> summed Amounts.
    """
    df = load_coa_amount(path)  # reuses schema validation + numeric parsing
    return (
        df.groupby("Cost Center", observed=True)["Amounts"]
        .sum()
        .to_dict()
    )


def load_mapping(path: Path) -> pd.DataFrame:
    """Read the transfer COA mapping CSV and return a DataFrame.

    Parameters
    ----------
    path : Path
        Path to mapping.csv.

    Returns
    -------
    pd.DataFrame
        Columns: 전기COA (str), 기존COA (str)
    """
    _validate_local_path(path)

    df = _read_csv(
        path,
        dtype = {"전기COA": str, "기존COA": str},
    )

    required = ["전기COA", "기존COA"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)}에 필수 컬럼이 없습니다: {missing}")

    df["전기COA"] = df["전기COA"].fillna("").astype(str).str.strip()
    df["기존COA"] = normalize_code_column(df["기존COA"], os.path.basename(path))
    return df



def load_cycle(path: Path) -> pd.DataFrame:
    """Read the allocation cycle CSV (wide format) and return a long DataFrame.

    The cycle CSV is authored as a matrix: ``차수`` and ``Sender CC`` are id
    columns, every remaining column header is a Receiver CC code, and each cell
    holds the allocation ratio. This is more intuitive to maintain in Excel than
    one row per (Sender, Receiver) pair. The wide grid is melted back to the long
    layout the rest of the pipeline expects, so no downstream code changes.

    Empty relationships may be left blank (→ NaN) or as 0; both are dropped. The
    0/NaN drop also keeps Receiver CC free of phantom receivers that never
    actually receive an allocation, which validate_master_completeness relies on.

    Parameters
    ----------
    path : Path
        Path to cycle.csv (wide format).
    Returns
    -------
    pd.DataFrame
        Columns: 차수 (int64), Sender CC (str), Receiver CC (str), % (float64)
        The % column is normalized to decimal form via parse_percent_column
        (e.g. "30%" -> 0.3, "0.3" -> 0.3).
        Sender CC and Receiver CC have normalize_code_column applied.
    """
    _validate_local_path(path)

    df = _read_csv(
        path,
        dtype = {"Sender CC": str},
    )

    required = ["차수", "Sender CC"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)}에 필수 컬럼이 없습니다: {missing}")

    # Reshape wide → long: every non-id column header is a Receiver CC code.
    cycle_df = df.melt(
        id_vars = ["차수", "Sender CC"],
        var_name = "Receiver CC",
        value_name = "%",
    )

    # Drop empty relationships: 0 cells and blank (NaN) cells both mean "no
    # allocation". A plain `!= 0` filter keeps NaN, so notna() is required too.
    cycle_df = cycle_df[(cycle_df["%"] != 0) & cycle_df["%"].notna()]

    cycle_df = (
        cycle_df.sort_values(["차수", "Sender CC"]).reset_index(drop=True)
    )

    fname = os.path.basename(path)
    cycle_df["Sender CC"] = normalize_code_column(cycle_df["Sender CC"], fname)
    cycle_df["Receiver CC"] = normalize_code_column(cycle_df["Receiver CC"], fname)
    cycle_df["%"] = parse_percent_column(cycle_df["%"], fname)
    cycle_df = _normalize_cycle_ratios(cycle_df)
    return cycle_df



# Step 2-A: CategoricalDtype


def build_category_dtypes(
    coa_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> dict[str, CategoricalDtype]:
    """Build three shared CategoricalDtype objects for transfer COA, base COA, and CC.

    Categories are defined once from master data so that all DataFrames share
    identical dtypes. The CC categories are derived from the COA·CC master sheet
    (coa_df) itself, which is the single source of the CC list.

    Parameters
    ----------
    coa_df     : load_coa_amount result. COA column -> base COA range;
                 Cost Center column -> CC range.
    mapping_df : load_mapping result (전기COA column)

    Returns
    -------
    dict with keys:
        'cc'    : CategoricalDtype - derived from the master's Cost Center column
        'coa'   : CategoricalDtype - derived from COA amount sheet (base COA)
        'e_coa' : CategoricalDtype - derived from mapping sheet (transfer COA).
                  Includes "" (empty string) for direct-cost rows.
    """
    cc_cats = coa_df["Cost Center"].unique().tolist()
    coa_cats = coa_df["COA"].unique().tolist()
    e_coa_cats = [""] + mapping_df["전기COA"].unique().tolist()

    return {
        "cc": CategoricalDtype(categories = cc_cats),
        "coa": CategoricalDtype(categories = coa_cats),
        "e_coa": CategoricalDtype(categories = e_coa_cats),
    }


def _cast_to_category(
    series: pd.Series,
    dtype: CategoricalDtype,
    *,
    reference: str | None = None,
) -> pd.Series:
    """Cast a code column to a shared CategoricalDtype, optionally reporting unknowns.

    A value absent from the dtype's categories cannot be represented and becomes
    NaN. Since pandas 3.0 the Categorical constructor warns (and a future version
    will raise) when such non-null values are passed, so unknowns are masked to
    NaN before casting to avoid the deprecation.

    When ``reference`` is given, genuine unknowns — a code in one sheet that does
    not exist in the sheet defining the categories — are surfaced by name so the
    cross-sheet mismatch is visible. Empty and missing cells are excluded so they
    do not generate noise. When ``reference`` is None the masking is silent: the
    mismatch is expected (e.g. the mapping sheet legitimately lists base COAs that
    are absent from the current period's amount sheet) and only the deprecation
    needs avoiding.

    Parameters
    ----------
    series : pd.Series
        str-normalized code column.
    dtype : CategoricalDtype
        Shared target dtype whose categories define the valid codes.
    reference : str, optional
        Human-readable name of the sheet that defines the valid categories. When
        provided, out-of-category codes trigger a warning naming it. When None,
        unknowns are masked silently.

    Returns
    -------
    pd.Series
        Categorical column; unknown and empty values become NaN.
    """
    in_category = series.isin(dtype.categories)

    if reference is not None:
        before = series.astype(str).str.strip()
        unknown_mask = (
            ~in_category
            & before.notna()
            & (before != "")
            & (before != "nan")
        )
        unknown_values = series[unknown_mask].astype(str).unique().tolist()
        if unknown_values:
            col_name = series.name if series.name is not None else "코드"
            warnings.warn(
                f"{col_name} 컬럼에 {reference}에 없는 코드가 있습니다: "
                f"{unknown_values} 해당 행은 매핑에서 제외됩니다."
            )

    # Mask out-of-category values to NaN before casting so the categorical
    # constructor never receives a non-null value outside its categories.
    return series.where(in_category).astype(dtype)


def apply_category_dtypes(
    coa_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    *,
    dtypes: dict[str, CategoricalDtype],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply shared CategoricalDtype objects.

    The Cost Center column's categories are derived from the master itself, so
    every value is in-category and a plain cast is safe. A base COA in the
    mapping sheet that is absent from the amount sheet is expected (the mapping is
    a superset reference) and is masked silently via _cast_to_category.

    Parameters
    ----------
    coa_df, mapping_df : DataFrames with str-normalized code columns.
    dtypes : Return value of build_category_dtypes.

    Returns
    -------
    (coa_df, mapping_df) with code columns cast to CategoricalDtype.
    """
    coa_df = coa_df.assign(
        COA=coa_df["COA"].astype(dtypes["coa"]),
        **{"Cost Center": coa_df["Cost Center"].astype(dtypes["cc"])},
    )
    mapping_df = mapping_df.assign(
        전기COA=mapping_df["전기COA"].astype(dtypes["e_coa"]),
        기존COA=_cast_to_category(mapping_df["기존COA"], dtypes["coa"]),
    )
    return coa_df, mapping_df
