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
        ["Amounts"].sum()
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


def calculate_coa_ratio(df_5a: pd.DataFrame) -> pd.DataFrame:
    """Compute each base COA's share within its transfer COA.

    Per the allocation business rule, a received amount is decomposed back to
    base COA using the SENDER's transfer COA composition; the receiver CC's own
    amounts are irrelevant. The ratio therefore has no Cost Center dimension:

        비중 = sum(Amounts | 전기COA, 기존COA) / sum(Amounts | 전기COA)

    Fallback: when a transfer COA total is 0, its base COAs split evenly at
    1 / n, where n is the number of distinct base COAs observed for that
    transfer COA. The observed count (not the mapping-wide count) is used so the
    shares always sum to 1 and the total amount is conserved on decomposition.

    All groupby calls must use observed=True.

    Parameters
    ----------
    df_5a : aggregate_detail result. Columns: 전기COA, 기존COA, Cost Center, Amounts

    Returns
    -------
    pd.DataFrame
        Columns: 전기COA, 기존COA, 비중. One row per (transfer COA, base COA) pair.
    """
    koa = (
        df_5a.groupby(["전기COA", "기존COA"], observed = True)["Amounts"]
        .sum()
        .reset_index(name = "koa_total")
    )
    koa["ec_total"] = (
        koa.groupby("전기COA", observed = True)["koa_total"].transform("sum")
    )
    koa["n_coa"] = (
        koa.groupby("전기COA", observed = True)["기존COA"].transform("count")
    )

    with np.errstate(divide = "ignore", invalid = "ignore"):
        koa["비중"] = np.where(
            koa["ec_total"] != 0,
            koa["koa_total"] / koa["ec_total"],
            1.0 / koa["n_coa"],
        )

    return koa[["전기COA", "기존COA", "비중"]]


# Input validation


def validate_cycle_cc(
    cycle_df: pd.DataFrame,
    coa_df: pd.DataFrame,
) -> list[str]:
    """Check that every Sender and Receiver CC in the cycle sheet exists in the master.

    The CC master is the COA·CC amount sheet itself; valid CCs are its
    ``Cost Center`` values.

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


def validate_sender_coverage(
    df_5b: pd.DataFrame,
    cycle_df: pd.DataFrame,
) -> list[tuple[str, float]]:
    """Check that every CC holding common-cost balance appears as a Sender.

    A common-cost CC that never sends keeps its balance through every cycle.
    Because the final result is built only from received (allocated) amounts,
    that undistributed balance never reaches build_result and silently vanishes,
    breaking the conservation law 배부전액 합 == 배부합계. This function only
    detects such CCs; it does not correct them.

    Parameters
    ----------
    df_5b    : aggregate_for_allocation result. Columns: 전기COA, Cost Center, Amounts.
    cycle_df : load_cycle result.

    Returns
    -------
    list[tuple[str, float]]
        (CC, total common-cost amount) for every CC that holds a non-zero
        common-cost balance but never appears as a Sender. Sorted by CC.
        Empty list means every common-cost CC is covered.
    """
    cc_totals = df_5b.groupby("Cost Center", observed = True)["Amounts"].sum()
    senders = set(cycle_df["Sender CC"])
    violators = [
        (str(cc), float(amount))
        for cc, amount in cc_totals.items()
        if cc not in senders and abs(amount) > 1e-6
    ]
    return sorted(violators, key = lambda item: item[0])


def validate_master_completeness(
    coa_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    cycle_df: pd.DataFrame,
) -> list[tuple[str, str]]:
    """Check that every Receiver CC x common-cost COA pair exists in the master.

    Common-cost COAs are the base COAs listed in mapping_df["기존COA"]. The final
    result is assembled only from received (allocated) amounts, so if a Receiver
    CC has no master row for a common-cost COA, the amount allocated to that CC
    loses its place in decompose / build_result and silently disappears, breaking
    the conservation law 배부전액 합 == 배부합계. This is the receiver-side
    counterpart to validate_sender_coverage and must run before the pipeline.

    Direct-cost COAs (those absent from the mapping) are excluded automatically,
    since the common-cost COA set is derived solely from mapping_df["기존COA"].

    Invariant: the four code columns (coa_df["COA"], coa_df["Cost Center"],
    mapping_df["기존COA"], cycle_df["Receiver CC"]) are already normalized to
    float->int->str strings by the loader's normalize_code_column, so only "6100"
    appears and a "6100.0" vs "6100" mismatch cannot occur. The .astype(str)
    below is a guard for categorical-dtype inputs (test fixtures).

    Parameters
    ----------
    coa_df     : load_coa_amount result. (COA, Cost Center) pairs are the master.
    mapping_df : load_mapping result. 기존COA column defines common-cost COAs.
    cycle_df   : load_cycle result. Receiver CC column is the set of receivers.

    Returns
    -------
    list[tuple[str, str]]
        Missing (기존COA, Receiver CC) pairs, sorted. Empty means validation passed.
    """
    common_coas = [
        c for c in mapping_df["기존COA"].unique() if str(c) not in ("", "nan")
    ]
    receiver_ccs = [
        c for c in cycle_df["Receiver CC"].unique() if str(c) not in ("", "nan")
    ]
    master_pairs = set(
        zip(coa_df["COA"].astype(str), coa_df["Cost Center"].astype(str))
    )
    missing = [
        (str(coa), str(cc))
        for coa in common_coas
        for cc in receiver_ccs
        if (str(coa), str(cc)) not in master_pairs
    ]
    return sorted(missing)
