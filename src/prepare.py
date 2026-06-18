from __future__ import annotations

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
    coa_df     : load_coa_amount result. Columns: COA, Cost Center, Amounts
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


# Step 4: Build the enriched frame shared by both outputs


def build_enriched(
    coa_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the enriched frame used by both the by_coa and by_cc outputs.

    Rows whose COA has a mapping entry receive the mapped 전기COA (common cost);
    rows without a mapping entry get an empty-string 전기COA (direct cost). The
    empty string keeps direct-cost rows out of the common-cost filter while still
    surviving any groupby (which drops NaN keys by default).

    Parameters
    ----------
    coa_df     : load_coa_amount result. Columns: COA, Cost Center, Amounts
    mapping_df : load_mapping result. Columns: 전기COA, 기존COA

    Returns
    -------
    pd.DataFrame
        Columns: 전기COA, 기존COA, Cost Center, Amounts.
        The original COA is exposed as 기존COA; 전기COA is "" for direct costs.
    """
    df = assign_transfer_coa(coa_df, mapping_df)  # COA, Cost Center, Amounts, 전기COA
    dtype = df["전기COA"].dtype
    df = df.assign(전기COA = df["전기COA"].fillna("").astype(dtype))
    return (
        df.rename(columns = {"COA": "기존COA"})
          [["전기COA", "기존COA", "Cost Center", "Amounts"]]
    )


# Input validation


def validate_cycle_cc(
    cycle_df: pd.DataFrame,
    coa_df: pd.DataFrame,
) -> list[str]:
    """Check that every Sender and Receiver CC in the cycle sheet exists in the master.

    The CC master is coa_amount.csv itself; valid CCs are its ``Cost Center``
    unique values. There is no separate cc.csv master file.

    Parameters
    ----------
    cycle_df : load_cycle result.
    coa_df   : load_coa_amount result (Cost Center column is the CC list).

    Returns
    -------
    list[str]
        CC codes not found in the master. Empty list means validation passed.
    """
    master = set(coa_df["Cost Center"])
    cycle_ccs = pd.concat([
        cycle_df["Sender CC"],
        cycle_df["Receiver CC"],
    ]).unique()

    return sorted(cc for cc in cycle_ccs if cc not in master)
