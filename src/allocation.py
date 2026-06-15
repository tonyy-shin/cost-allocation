from __future__ import annotations

import warnings

import numpy as np
import pandas as pd



# Step 7-1: Build pivot matrix


def build_pivot_matrix(
    df_5b: pd.DataFrame,
    cc_list: list[str],
) -> pd.DataFrame:
    """Create a (transfer COA x CC) pivot matrix.

    Rows = transfer COA, columns = Cost Center, values = Amounts.
    Missing CC columns are filled with 0 via reindex against the full CC list.

    Parameters
    ----------
    df_5b    : aggregate_for_allocation result.
    cc_list  : All CC codes from the CC master (reindex reference).

    Returns
    -------
    pd.DataFrame
        Index: 전기COA, columns: CC, values: Amounts (float64).
        Missing CCs = 0.
    """
    pivot = (
        df_5b
        .groupby(["전기COA", "Cost Center"], observed = True)
        ["Amounts"].sum()
        .unstack("Cost Center", fill_value = 0)
    )
    return pivot.reindex(columns = cc_list, fill_value = 0)


# Step 7-2: Sequential allocation loop


def run_allocation_loop(
    pivot: pd.DataFrame,
    cycle_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Run the sequential allocation in cycle order.

    Within the same cycle, all receivers of the same sender are calculated
    from the sender's balance before any deduction.
    The sender's balance is reduced by the distributed amount after each cycle.
    A sender with zero balance distributes zero and continues without error.

    After all cycles complete, a sender-balance-zero check is performed
    to check for errors.

    Parameters
    ----------
    pivot    : build_pivot_matrix result. Index = transfer COA, columns = CC.
    cycle_df : load_cycle result.

    Returns
    -------
        final_pivot    : Pivot state after allocations.
        delta_by_cycle : {cycle: DataFrame(전기COA, Receiver CC, Amounts)}
                         Amount received by each Receiver CC per cycle.
    """
    pivot = pivot.copy().astype(float)
    delta_by_cycle = {}
    sender_ccs = set(cycle_df["Sender CC"].unique())

    for cycle_num, cycle_rows in cycle_df.groupby("차수", sort = True):
        deltas = []
        for sender, sender_rows in cycle_rows.groupby("Sender CC", sort = False):
            if sender not in pivot.columns:
                continue
            sender_bal = pivot[sender].copy()
            total_sent = pd.Series(0.0, index = pivot.index)

            for receiver, pct in zip(sender_rows["Receiver CC"], sender_rows["%"]):
                received = sender_bal * pct
                if receiver in pivot.columns:
                    pivot[receiver] += received
                total_sent += received

                tmp = received.reset_index()
                tmp.columns = ["전기COA", "Amounts"]
                tmp["Receiver CC"] = receiver
                deltas.append(tmp)
            pivot[sender] -= total_sent
        delta_by_cycle[cycle_num] = (
            pd.concat(deltas, ignore_index = True) if deltas 
            else pd.DataFrame(columns = ["전기COA", "Receiver CC", "Amounts"])
        )
    
    valid_senders = [cc for cc in sender_ccs if cc in pivot.columns]
    if valid_senders and (pivot[valid_senders].abs() > 1e-6).any().any():
        warnings.warn("Sender CC balances did not reach 0 after allocation")

    return pivot, delta_by_cycle


# Step 8: Aggregate received amounts by cycle


def aggregate_received_by_cycle(
    delta_by_cycle: dict[int, pd.DataFrame],
) -> dict[int, pd.DataFrame]:
    """Group each cycle's received amounts by (transfer COA, Receiver CC).

    Parameters
    ----------
    delta_by_cycle : delta_by_cycle from run_allocation_loop.

    Returns
    -------
    dict[int, pd.DataFrame]
        {cycle: DataFrame(전기COA, Receiver CC, Amounts)}
        Amounts are summed over duplicate (transfer COA, Receiver CC) pairs.
    """
    result = {}

    for cycle_num, df in delta_by_cycle.items():
        if df.empty:
            result[cycle_num] = df.copy()
            continue
        result[cycle_num] = (
            df.groupby(["전기COA", "Receiver CC"], observed = True)
            ["Amounts"].sum()
            .reindex()
        )

    return result


# Step 9: Decompose back to base COA


def decompose_to_original_coa(
    received_by_cycle: dict[int, pd.DataFrame],
    df_ratio: pd.DataFrame,
) -> pd.DataFrame:
    """Apply base COA ratios to per-cycle received amounts to restore base COA granularity.

    A total-amount conservation check is performed after decomposition.
    Discrepancies exceeding a float tolerance emit a warning but do not halt execution.
    Amounts are kept as float without rounding.

    Parameters
    ----------
    received_by_cycle : aggregate_received_by_cycle result.
    df_ratio          : calculate_coa_ratio result (includes the '비중' ratio column).

    Returns
    -------
    pd.DataFrame
        Columns: 전기COA, 기존COA, Cost Center, 1차배분금액, 2차배분금액, ..., n차배분금액
        n equals the number of keys in received_by_cycle.
    """
    ec_koa = (
        df_ratio.groupby(["전기COA", "기존COA"], observed = "True")
        ["Amounts"].sum()
        .reset_index(name = "koa_total")
    )
    ec_total = (
        df_ratio.groupby("전기COA", observed = True)
        ["Amounts"].sum()
        .reset_index(name = "ec_total")
    )

    ec_koa = ec_koa.merge(ec_total, on = "전기COA", how = "left")
    n_koa = ec_koa.groupby("전기COA", observed = True)["기존COA"].transform("count")

    ec_koa["ratio"] = np.where(
        ec_koa["ec_total"] != 0,
        ec_koa["koa_total"] / ec_koa["ec_total"],
        1.0 / n_koa,
    )
    decomp = ec_koa[["전기COA", "기존COA", "ratio"]]

    all_rows = []
    for cycle_num, received_df in received_by_cycle.items():
        if received_df.empty:
            continue
        merged = received_df.merge(decomp, on = "전기COA", how = "left")
        merged["allocated"] = merged["Amounts"] * merged["ratio"]
        merged = merged.rename(columns = {"Receiver CC": "Cost Center"})
        merged["col"] = f"{cycle_num}차배분금액"
        all_rows.append(
            merged[["전기COA", "기존COA", "Cost Center", "col", "allocated"]]
        )

    if not all_rows:
        return pd.DataFrame(columns = ["전기COA", "기존COA", "Cost Center"])
    
    combined = pd.concat(all_rows, ignore_index = True)
    result = (
        combined
        .groupby(["전기COA", "기존COA", "Cost Center", "col"], observed = True)
        ["allocated"].sum()
        .unstack("col", fill_value = 0)
        .reset_index()
    )
    result.columns.name = None

    alloc_cols = sorted(
        [c for c in result.columns if "배분금액" in c],
        key=lambda c: int(c.replace("차배분금액", "")),
    )
    result = result[["전기COA", "기존COA", "Cost Center"] + alloc_cols]

    received_total = sum(
        df["Amounts"].sum() for df in received_by_cycle.values() if not df.empty
    )
    alloc_total = result[alloc_cols].values.sum()
    tol = 1e-6 * max(abs(received_total), 1.0)
    if abs(alloc_total - received_total) > total:
        warnings.warn(
            f"Decomposition conservation check failed: "
            f"received = {received_total: .4f}, allocated = {alloc_total: .4f}"
        )

    return result
