from __future__ import annotations

import numpy as np
import pandas as pd


# Step 3: Assign transfer COA


def assign_transfer_coa(
    coa_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add transfer COA column by looking up each base COA in the mapping table.

    COAs not found in the mapping are left as NaN.

    Parameters
    ----------
    coa_df     : enrich_cc result. Columns: COA, Cost Center, Amounts
    mapping_df : Transfer COA mapping DataFrame. Columns: 전기COA, 기존COA

    Returns
    -------
    pd.DataFrame
        Columns: COA, Cost Center, Amounts, 전기COA
        Common costs have a transfer COA value, while direct costs have NaN.
    """
    df = coa_df.merge(
        mapping_df,
        left_on = "COA",
        right_on = "기존COA",
        how = "left"
    )
    return df.drop(columns = ["기존COA"])


# Step 4: Separate common and direct costs


def separate_common_direct(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into common costs and direct costs.

    Rows where 전기COA is NaN (direct costs) have their 전기COA filled with ""
    before splitting. The empty string prevents NaN keys from being silently
    dropped by groupby (which uses dropna=True by default).

    Parameters
    ----------
    df : assign_transfer_coa result.

    Returns
    -------
    (df_common, df_direct)
        df_common : Rows where 전기COA is not null.
        df_direct : Rows where 전기COA == "" (originally NaN).
    """
    dtype = df["전기COA"].dtype
    df = df.assign(전기COA = df["전기COA"].fillna("").astype(dtype))

    df_common = df[df["전기COA"] != ""].copy()
    df_direct = df[df["전기COA"] == ""].copy()
    return df_common, df_direct



# Step 5-A: Detail aggregation for ratio calculation


def aggregate_detail(df_common: pd.DataFrame) -> pd.DataFrame:
    """Sum common cost amounts by (transfer COA, base COA, Cost Center).

    This is the reference dataset for Step 6 base-COA ratio calculation.
    All groupby calls must use observed=True to avoid row explosion when
    CategoricalDtype columns contain unobserved category combinations.

    Parameters
    ----------
    df_common : df_common from separate_common_direct.

    Returns
    -------
    pd.DataFrame
        Columns: 전기COA, 기존COA, Cost Center, Amounts
        Summed over (transfer COA x base COA x CC).
    """
    return (
        df_common
        .groupby(["전기COA", "COA", "Cost Center"], observed = True)
        .sum()
        .reset_index()
        .rename(columns = {"COA": "기존COA"})
    )


# Step 5-B: Allocation aggregation for allocation calculation


def aggregate_for_allocation(df_5a: pd.DataFrame) -> pd.DataFrame:
    """Re-sum Step 5-A results by (transfer COA, Cost Center).

    Produces pivot input for Step 7. Base COA information is lost at this
    step, so aggregate_detail must always be called before this function.
    All groupby calls must use observed=True.

    Parameters
    ----------
    df_5a : aggregate_detail result.

    Returns
    -------
    pd.DataFrame
        Columns: 전기COA, Cost Center, Amounts
        Summed over (transfer COA x CC).
    """
    return (
        df_5a
        .groupby(["전기COA", "Cost Center"], observed = True)["Amounts"]
        .sum()
        .reset_index()
    )


# Step 6: Base COA ratio calculation


def calculate_coa_ratio(
    df_5a: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute each base COA's share within its (transfer COA, CC) group.

    Fallback order:
      1st : sum(transfer COA, CC) != 0  ->  Amount / sum(transfer COA, CC)
      2nd : sum(transfer COA) != 0      ->  sum(transfer COA, base COA) / sum(transfer COA)
      Equal: transfer COA total == 0    ->  1 / n_coa  (number of base COAs per transfer COA
                                            from mapping_df)

    Implementation must use np.select with vectorized transforms to avoid
    row-wise apply / iterrows. All groupby calls must use observed=True.

    Parameters
    ----------
    df_5a      : aggregate_detail result. Columns: 전기COA, 기존COA, Cost Center, Amounts
    mapping_df : Transfer COA mapping DataFrame. Columns: 전기COA, 기존COA

    Returns
    -------
    pd.DataFrame
        df_5a with an additional '비중' (ratio) column.
    """
    df = df_5a.copy()

    df["sum_ec_cc"] = (
        df.groupby(["전기COA", "Cost Center"], observed = True)
        ["Amounts"].transform("sum")
    )
    df["sum_ec"] = (
        df.groupby("전기COA", observed = True)
        ["Amounts"].transform("sum")
    )
    df["sum_ec_coa"] = (
        df.groupby(["전기COA", "기존COA"], observed=True)
        ["Amounts"].transform("sum")
    )

    n_coa_map = mapping_df.groupby("전기COA", observed=True)["기존COA"].count()
    df["n_coa"] = df["전기COA"].map(n_coa_map)

    conditions = [
        df["sum_ec_cc"] != 0,
        df["sum_ec"] != 0,
    ]
    choices = [
        df["Amounts"] / df["sum_ec_cc"],
        df["sum_ec_coa"] / df["sum_ec"],
    ]
    df["비중"] = np.select(conditions, choices, default = 1 / df["n_coa"])

    return df.drop(columns = ["sum_ec_cc", "sum_ec", "sum_ec_coa", "n_coa"])


# Input validation


def validate_cycle_cc(
    cycle_df: pd.DataFrame,
    cc_df: pd.DataFrame,
) -> list[str]:
    """Check that every Sender and Receiver CC in the cycle sheet exists in the CC master.

    Parameters
    ----------
    cycle_df : load_cycle result.
    cc_df    : CC master DataFrame.

    Returns
    -------
    list[str]
        CC codes not found in the master. Empty list means validation passed.
    """
    master = set(cc_df["CC"])
    cycle_ccs = pd.concat([
        cycle_df["Sender CC"],
        cycle_df["Receiver CC"],
    ]).unique()

    return sorted(cc for cc in cycle_ccs if cc not in master)
