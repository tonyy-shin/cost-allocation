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
    # Per-CC common-cost balance before any allocation. A CC's own original
    # balance is only ever pushed out if that CC is a Sender; otherwise it
    # stays put and is dropped from the result (it is never "received"). The
    # initial balance is captured here so non-sender residuals can be reported
    # with the correct amount, without flagging receivers that legitimately
    # accumulate balance during the loop.
    initial_cc_balance = pivot.sum(axis=0)
    delta_by_cycle = {}
    unbalanced = False

    for cycle_num, cycle_rows in cycle_df.groupby("차수", sort=True):
        deltas = []
        for sender, sender_rows in cycle_rows.groupby("Sender CC", sort=False):
            if sender not in pivot.columns:
                continue
            sender_bal = pivot[sender].copy()
            total_sent = pd.Series(0.0, index=pivot.index)

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
            # Check residual the moment this sender finishes distributing,
            # before a later cycle can credit it back as a receiver.
            if (sender_bal - total_sent).abs().max() > 1e-6 * max(sender_bal.abs().max(), 1.0):
                unbalanced = True
        delta_by_cycle[cycle_num] = (
            pd.concat(deltas, ignore_index=True) if deltas
            else pd.DataFrame(columns=["전기COA", "Receiver CC", "Amounts"])
        )

    if unbalanced:
        warnings.warn("배부 후 sender CC 잔액이 0이 되지 않았습니다")

    # A CC carrying common cost that never sends keeps its original balance,
    # which then disappears from the result. Report each such CC by amount.
    senders = set(cycle_df["Sender CC"])
    for cc, amount in initial_cc_balance.items():
        if cc not in senders and abs(amount) > 1e-6:
            warnings.warn(
                f"CC {cc}의 공통비 잔액 {amount:,.0f}원이 배부되지 않았습니다. "
                f"cycle.csv에 Sender로 추가하거나 배부 비율 합계를 확인하세요."
            )

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
            .reset_index()
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
    df_ratio          : calculate_coa_ratio result. Columns: 전기COA, 기존COA, 비중.
                        The single source of truth for base COA shares; this
                        function applies it directly without recomputing ratios.

    Returns
    -------
    pd.DataFrame
        Columns: 전기COA, 기존COA, Cost Center, 1차배분금액, 2차배분금액, ..., n차배분금액
        n equals the number of keys in received_by_cycle.
    """
    all_rows = []
    for cycle_num, received_df in received_by_cycle.items():
        if received_df.empty:
            continue
        merged = received_df.merge(df_ratio, on = "전기COA", how = "left")
        merged["allocated"] = merged["Amounts"] * merged["비중"]
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
    total = 1e-6 * max(abs(received_total), 1.0)
    if not np.isfinite(alloc_total) or abs(alloc_total - received_total) > total:
        warnings.warn(
            f"분해 총액 보존 검증 실패: "
            f"수령액 = {received_total: .4f}, 배분액 = {alloc_total: .4f}"
        )


    return result
