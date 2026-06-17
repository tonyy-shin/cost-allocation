from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.allocation import TOTAL_COL, alloc_col, parse_alloc_col


# Steps 10–11: Build final result over the computed (base COA, CC) pairs


def build_result(
    common_decomposed: pd.DataFrame,
    direct_df: pd.DataFrame,
    n_cycles: int,
) -> pd.DataFrame:
    """Combine common and direct costs into the final result grid.

    Covers Steps 10 and 11.

    Common costs (common_decomposed):
        Per-cycle allocation columns are populated; 배부합계 = sum of cycle columns.
    Direct costs (direct_df):
        All allocation columns are 0; 배부합계 = original Amounts.

    Grid (Step 11):
        Row range  : the (전기COA, 기존COA, Cost Center) groups that actually appear
                     in the computed result — the union of the allocated common-cost
                     rows and the direct-cost rows. The result is NOT reindexed onto
                     the COA·CC master, so master pairs with neither an allocation nor
                     a direct cost do not appear, and no computed row can be dropped.
        Implementation: concat(common, direct) then groupby the three key columns.

    Output column order:
        전기COA, 기존COA, 코스트센터, 1차배분금액, ..., n차배분금액, 배부합계

    Parameters
    ----------
    common_decomposed : decompose_to_original_coa result.
    direct_df         : df_direct from separate_common_direct.
    n_cycles          : Number of allocation cycles (determines allocation column count).

    Returns
    -------
    pd.DataFrame
        All columns as described above. One row per computed
        (전기COA, 기존COA, Cost Center) group; no master-only zero-filled rows.
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

    # Combine and group: the result grid is the computed (전기COA, 기존COA, CC)
    # groups themselves — no reindex onto the master, so nothing is added or dropped.
    combined = pd.concat([common, direct], ignore_index=True)
    combined = (
        combined
        .groupby(["전기COA", "기존COA", "Cost Center"], observed=True)
        [numeric_cols]
        .sum()
        .reset_index()
    )

    # Final column order and rename
    result = combined.rename(columns={"Cost Center": "코스트센터"})
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