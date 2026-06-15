from __future__ import annotations

from pathlib import Path

import pandas as pd


# ── Steps 10–11: Build final result with full grid ─────────────────────────


def build_result(
    common_decomposed: pd.DataFrame,
    direct_df: pd.DataFrame,
    coa_df: pd.DataFrame,
    cc_df: pd.DataFrame,
    n_cycles: int,
) -> pd.DataFrame:
    """Combine common and direct costs, then expand to the full (base COA x CC) grid.

    Covers Steps 10 and 11.

    Common costs (common_decomposed):
        Per-cycle allocation columns are populated; 배부합계 = sum of cycle columns.
    Direct costs (direct_df):
        All allocation columns are 0; 배부합계 = original Amounts.

    Full grid (Step 11):
        COA range  : determined by coa_df (amount sheet only; mapping-only COAs excluded).
        CC range   : determined by cc_df (full CC master).
        Implementation: MultiIndex.from_product + reindex(fill_value=0).
        Assert (기존COA, CC) uniqueness before reindex to catch silent fill errors.

    Output column order:
        전기COA, 기존COA, 코스트센터, 1차배분금액, ..., n차배분금액, 배부합계

    Parameters
    ----------
    common_decomposed : decompose_to_original_coa result.
    direct_df         : df_direct from separate_common_direct.
    coa_df            : COA amount DataFrame before enrich_cc (defines the COA range).
    cc_df             : CC master DataFrame (defines the CC range).
    n_cycles          : Number of allocation cycles (determines allocation column count).

    Returns
    -------
    pd.DataFrame
        All columns as described above.
        Every (base COA x CC) combination is present; missing values filled with 0.
    """
    ...


# ── Step 12: Save to CSV ───────────────────────────────────────────────────


def save_result(result_df: pd.DataFrame, output_dir: Path) -> Path:
    """Write the final result DataFrame to a CSV file.

    Filename: result.csv, encoding: utf-8-sig (Excel-compatible Korean support).

    Parameters
    ----------
    result_df  : build_result output.
    output_dir : Directory where the file will be saved.

    Returns
    -------
    Path
        Full path of the saved file.
    """
    ...
