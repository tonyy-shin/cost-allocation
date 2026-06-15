from __future__ import annotations

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


def normalize_code_column(series: pd.Series) -> pd.Series:
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

    Returns
    -------
    pd.Series
        Normalized string code column.
    """
    return (
        pd.to_numeric(series, errors="coerce") # str("100.0") to float(100.0)
        .astype("Int64") # float to int
        .astype(str) # int to str
        .str.replace("<NA>","",regex = False) # drop missing value as ""
    )


# Step 1: CSV readers


def load_cc(path: Path) -> pd.DataFrame:
    """Read CC master CSV and return a DataFrame.

    Parameters
    ----------
    path : Path
        Path to cc.csv

    Returns
    -------
    pd.DataFrame
        Column: CC (str, normalized)
    """
    _validate_local_path(path)
    df = pd.read_csv(
        path, 
        dtype = {"CC": str}, 
        encoding = "utf-8-sig",
    )

    df["CC"] = normalize_code_column(df["CC"])
    return df


def load_coa_amount(path: Path) -> pd.DataFrame:
    """Read COA amount CSV and return DataFrame.

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

    df = pd.read_csv(
        path,
        dtype = {"COA": str, "Cost Center": str},
        encoding = "utf-8-sig"
    )

    df["COA"] = normalize_code_column(df["COA"])
    df["Cost Center"] = normalize_code_column(df["Cost Center"])
    return df


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

    df = pd.read_csv(
        path,
        dtype = {"전기COA": str, "기존COA": str},
        encoding = "utf-8-sig"
    )

    df["전기COA"] = normalize_code_column(df["전기COA"])
    df["기존COA"] = normalize_code_column(df["기존COA"])
    return df



def load_cycle(path: Path) -> pd.DataFrame:
    """Read the allocation cycle CSV and return a DataFrame.

    Parameters
    ----------
    path : Path
        Path to cycle.csv
    Returns
    -------
    pd.DataFrame
        Columns: 차수 (int64), Sender CC (str), Receiver CC (str), % (float64)
        The % column is kept as-is in decimal form (0.3 = 30%).
        Sender CC and Receiver CC have normalize_code_column applied.
    """
    _validate_local_path(path)

    df = pd.read_csv(
        path,
        dtype = {"Sender CC": str, "Receiver CC": str},
        encoding = "utf-8-sig"
    )

    df["Sender CC"] = normalize_code_column(df["Sender CC"])
    df["Receiver CC"] = normalize_code_column(df["Receiver CC"])
    return df



# Step 2-A: CategoricalDtype


def build_category_dtypes(
    cc_df: pd.DataFrame,
    coa_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> dict[str, CategoricalDtype]:
    """Build three shared CategoricalDtype objects for transfer COA, base COA, and CC.

    Categories are defined once from master data so that all DataFrames share
    identical dtypes.

    Parameters
    ----------
    cc_df      : load_cc result (CC column)
    coa_df     : load_coa_amount result (COA column -> base COA range)
    mapping_df : load_mapping result (전기COA column)

    Returns
    -------
    dict with keys:
        'cc'    : CategoricalDtype - derived from CC master
        'coa'   : CategoricalDtype - derived from COA amount sheet (base COA)
        'e_coa' : CategoricalDtype - derived from mapping sheet (transfer COA).
                  Includes "" (empty string) for direct-cost rows.
    """
    cc_cats = cc_df["CC"].unique().tolist()
    coa_cats = coa_df["COA"].unique().tolist()
    e_coa_cats = [""] + mapping_df["전기COA"].unique().tolist()

    return {
        "cc": CategoricalDtype(categories = cc_cats),
        "coa": CategoricalDtype(categories = coa_cats),
        "e_coa": CategoricalDtype(categories = e_coa_cats),
    }


def apply_category_dtypes(
    cc_df: pd.DataFrame,
    coa_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    *,
    dtypes: dict[str, CategoricalDtype],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply shared CategoricalDtype objects.

    Parameters
    ----------
    cc_df, coa_df, mapping_df : DataFrames with str-normalized code columns.
    dtypes : Return value of build_category_dtypes.

    Returns
    -------
    (cc_df, coa_df, mapping_df) with code columns cast to CategoricalDtype.
    """
    cc_df = cc_df.assign(CC=cc_df["CC"].astype(dtypes["cc"]))
    coa_df = coa_df.assign(
        COA=coa_df["COA"].astype(dtypes["coa"]),
        **{"Cost Center": coa_df["Cost Center"].astype(dtypes["cc"])},
    )
    mapping_df = mapping_df.assign(
        전기COA=mapping_df["전기COA"].astype(dtypes["e_coa"]),
        기존COA=mapping_df["기존COA"].astype(dtypes["coa"]),
    )
    return cc_df, coa_df, mapping_df


# Step 2-B: CC enrichment


def enrich_cc(coa_df: pd.DataFrame, cc_df: pd.DataFrame) -> pd.DataFrame:
    """Add missing CCs from the master to the COA amount DataFrame.

    CCs present in the master but absent from the amount sheet are appended
    so that all CCs are represented throughout the pipeline.
    Added rows have COA = '' (empty string) and Amounts = 0.

    Parameters
    ----------
    coa_df : COA amount DataFrame after apply_category_dtypes.
    cc_df  : CC master DataFrame after apply_category_dtypes.

    Returns
    -------
    pd.DataFrame
        COA amount DataFrame containing every CC in the master.
    """
    e_ccs = coa_df["Cost Center"].unique()
    m_ccs = cc_df.loc[~cc_df["CC"].isin(e_ccs), "CC"]

    if m_ccs.empty:
        return coa_df
    filler = pd.DataFrame({
        "COA": pd.Categorical([""] * len(m_ccs),
                            dtype=coa_df["COA"].dtype),
        "Cost Center": pd.Categorical(m_ccs.values,
                                    dtype=coa_df["Cost Center"].dtype),
        "Amounts": 0.0,
    })

    return pd.concat([coa_df, filler], ignore_index = True)