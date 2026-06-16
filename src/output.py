from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.allocation import TOTAL_COL, alloc_col, parse_alloc_col


# Steps 10–11: Build final result with full grid


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
    alloc_cols = [alloc_col(i) for i in range(1, n_cycles + 1)]
    numeric_cols = alloc_cols + [TOTAL_COL]

    # Common costs
    common = common_decomposed.copy()
    for col in alloc_cols:
        if col not in common.columns:
            common[col] = 0.0
    common[TOTAL_COL] = common[alloc_cols].sum(axis=1)
    common = common[["전기COA", "기존COA", "Cost Center"] + numeric_cols]

    # Direct costs
    direct = direct_df[["COA", "Cost Center", "Amounts", "전기COA"]].copy()
    direct = direct.rename(columns={"COA": "기존COA"})
    for col in alloc_cols:
        direct[col] = 0.0
    direct[TOTAL_COL] = direct["Amounts"]
    direct = direct[["전기COA", "기존COA", "Cost Center"] + numeric_cols]

    # Combine and group
    combined = pd.concat([common, direct], ignore_index=True)
    combined = (
        combined
        .groupby(["전기COA", "기존COA", "Cost Center"], observed=True)
        [numeric_cols]
        .sum()
        .reset_index()
    )

    # Derive 기존COA → 전기COA lookup before reindex loses the mapping
    coa_to_ecoa = (
        combined[["기존COA", "전기COA"]]
        .drop_duplicates("기존COA")
        .set_index("기존COA")["전기COA"]
        .astype(str)
    )

    # Full grid expansion
    coa_list = coa_df["COA"].unique().tolist()
    cc_list = cc_df["CC"].unique().tolist()
    full_index = pd.MultiIndex.from_product(
        [coa_list, cc_list], names=["기존COA", "Cost Center"]
    )

    assert not combined.duplicated(["기존COA", "Cost Center"]).any(), \
        "Duplicate (기존COA, Cost Center) pairs found before reindex"

    result = (
        combined
        .set_index(["기존COA", "Cost Center"])[numeric_cols]
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    # Restore 전기COA for all rows
    result["전기COA"] = result["기존COA"].map(coa_to_ecoa).fillna("")

    # Final column order and rename
    result = result.rename(columns={"Cost Center": "코스트센터"})
    return result[["전기COA", "기존COA", "코스트센터"] + numeric_cols]


# Step 12: Save to CSV


def save_result(result_df: pd.DataFrame, output_dir: Path) -> Path:
    """Write the final result DataFrame to a CSV file.

    Filename: result.csv, encoding: utf-8-sig (Excel compatible Korean support).

    Parameters
    ----------
    result_df  : build_result output.
    output_dir : Directory where the file will be saved.

    Returns
    -------
    Path
        Full path of the saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "result.csv"
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path