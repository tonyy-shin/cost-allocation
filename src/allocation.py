from __future__ import annotations

from functools import reduce

import pandas as pd


def _amt_col(cycle: int) -> str:
    """Allocation-amount column name for a cycle (by_coa)."""
    return f"{cycle}차배부금액"


def _sum_col(cycle: int) -> str:
    """Allocation-total column name for a cycle (by_coa)."""
    return f"{cycle}차배부합계"


def _after_col(cycle: int) -> str:
    """Post-allocation balance column name for a cycle (by_cc)."""
    return f"{cycle}차후금액"


# by_coa: sender common-cost amounts keyed by (전기COA, 기존COA, Sender CC)


def build_by_coa(
    enriched: pd.DataFrame,
    cycle_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, dict[str, float]]]:
    """Build the by_coa result table and per-(cycle, sender) totals.

    For each cycle n, common-cost rows (전기COA != "") whose Cost Center is a
    Sender CC in cycle n contribute their Amounts to that cycle's 배부금액 column.
    Rows are keyed by (전기COA, 기존COA, Sender CC).

    The 배부합계 columns hold the *column-wide* scalar total of each cycle's
    배부금액, placed in row 0 only (blank elsewhere); a single empty column
    separates the per-row 배부금액 block from these summary values.

    The returned ``sender_totals`` is instead *sender-level* — each Sender CC's
    own total for the cycle — and drives the by_cc deduction/addition math. (The
    two differ when a cycle has multiple senders.)

    Parameters
    ----------
    enriched  : build_enriched result. Columns: 전기COA, 기존COA, Cost Center, Amounts.
    cycle_df  : load_cycle result.

    Returns
    -------
    (by_coa_df, sender_totals)
        by_coa_df     : Columns 전기COA, 기존COA, Sender CC, Receiver CC,
                        1..n차배부금액, "", 1..n차배부합계. Each per-row
                        n차배부금액 is the sender's cycle-n amount split by the
                        (Sender → Receiver) ratio from cycle.csv.
        sender_totals : {cycle: {Sender CC: total distributed in that cycle}}.
    """
    common = enriched[enriched["전기COA"].astype(str) != ""].copy()
    # Normalize keys to str for stable grouping/merging and clean CSV output.
    for col in ("전기COA", "기존COA", "Cost Center"):
        common[col] = common[col].astype(str)

    cycles = sorted(int(c) for c in cycle_df["차수"].unique())
    keys = ["전기COA", "기존COA", "Sender CC", "Receiver CC"]

    per_cycle: list[pd.DataFrame] = []
    sender_totals: dict[int, dict[str, float]] = {}
    for n in cycles:
        rows_n = cycle_df[cycle_df["차수"] == n]
        senders = set(rows_n["Sender CC"].astype(str))
        sub = common[common["Cost Center"].isin(senders)]
        grouped = (
            sub.groupby(["전기COA", "기존COA", "Cost Center"], observed=True)
            ["Amounts"].sum()
            .reset_index()
            .rename(columns={"Cost Center": "Sender CC", "Amounts": _amt_col(n)})
        )
        # sender_totals stays sender-level (pre-explode, ratio-free) so by_cc's
        # math is unchanged.
        sender_totals[n] = (
            grouped.groupby("Sender CC")[_amt_col(n)].sum().to_dict()
        )

        # Explode each sender amount into (Sender → Receiver) shares using the
        # cycle's ratios. Sender CC is forced to str so the merge keys line up
        # even if a Categorical dtype lingered through the groupby.
        grouped["Sender CC"] = grouped["Sender CC"].astype(str)
        pairs = rows_n[["Sender CC", "Receiver CC", "%"]].copy()
        pairs["Sender CC"] = pairs["Sender CC"].astype(str)
        pairs["Receiver CC"] = pairs["Receiver CC"].astype(str)
        exploded = grouped.merge(pairs, on="Sender CC", how="inner")
        exploded[_amt_col(n)] = exploded[_amt_col(n)] * exploded["%"]
        per_cycle.append(exploded[keys + [_amt_col(n)]])

    if per_cycle:
        result = reduce(
            lambda left, right: left.merge(right, on=keys, how="outer"), per_cycle
        )
    else:
        result = pd.DataFrame(columns=keys)

    amt_cols = [_amt_col(n) for n in cycles]
    for col in amt_cols:
        if col not in result.columns:
            result[col] = 0.0
    result[amt_cols] = result[amt_cols].fillna(0.0)
    result = result.sort_values(keys).reset_index(drop=True)

    # One empty separator column between the 배부금액 and 배부합계 blocks. The
    # 배부합계 columns mix a numeric scalar (row 0) with blanks, so they must be
    # object dtype — pandas would otherwise infer StringDtype and reject the float.
    result[""] = pd.Series([""] * len(result), index=result.index, dtype=object)

    # Each 배부합계 column holds a single column-wide scalar in row 0 only.
    for n in cycles:
        col = _sum_col(n)
        result[col] = pd.Series([""] * len(result), index=result.index, dtype=object)
        if len(result):
            result.iloc[0, result.columns.get_loc(col)] = float(
                result[_amt_col(n)].sum()
            )

    ordered = keys + amt_cols + [""] + [_sum_col(n) for n in cycles]
    return result[ordered], sender_totals


# by_cc: settled per-cycle CC balances (arrival-cycle partition)


def build_by_cc(
    cc_list: list[str],
    pre_alloc_cc: dict[str, float],
    cycle_df: pd.DataFrame,
    sender_totals: dict[int, dict[str, float]],
) -> dict[int, pd.DataFrame]:
    """Build one settled by_cc snapshot per cycle.

    Each CC's balance is tracked as labeled buckets (label 0 = 배부전금액;
    label k = money that arrived in cycle k). Per cycle k, in ascending order:

    - Receivers gain ``sender_total * ratio`` from each of their senders, labeled
      with cycle k.
    - Senders are drained by their ``sender_totals[k]`` amount, consuming received
      buckets first (oldest cycle first) and the original (label 0) bucket last.
      The amount comes from coa_amount and is independent of the pre_allocation
      balance, so label 0 may go negative — subtracted unconditionally, no warning.

    In file n, ``k차후금액`` (k < n) is bucket k of the state after cycle n, and
    ``n차후금액`` folds in the still-held original (label 0) balance — so static
    money (never sent/received) lands in the last column. ``배부합계`` is the
    row-wise sum of the 후금액 columns and equals the CC's final balance, giving
    ``배부전 total == 배부합계 total`` per file.

    Parameters
    ----------
    cc_list       : All Cost Center codes from coa_amount.csv (one row per CC).
    pre_alloc_cc  : load_pre_allocation result. {CC: 배부전금액} (0 if absent).
    cycle_df      : load_cycle result.
    sender_totals : build_by_coa's second return value.

    Returns
    -------
    dict[int, pd.DataFrame]
        {cycle n: DataFrame with columns
         CC, 배부전금액, 1차후금액 .. n차후금액, 배부합계}.
    """
    cycles = sorted(int(c) for c in cycle_df["차수"].unique())
    ccs = [str(cc) for cc in cc_list]
    pre = {str(k): float(v) for k, v in pre_alloc_cc.items()}

    bal: dict[str, dict[int, float]] = {cc: {0: pre.get(cc, 0.0)} for cc in ccs}
    snapshots: dict[int, dict[str, dict[int, float]]] = {}

    for k in cycles:
        rows = cycle_df[cycle_df["차수"] == k]
        senders = sender_totals.get(k, {})

        # Receivers: credit inflow, labeled with the current cycle.
        for sender, receiver, pct in zip(
            rows["Sender CC"].astype(str),
            rows["Receiver CC"].astype(str),
            rows["%"],
        ):
            total = float(senders.get(sender, 0.0))
            if receiver in bal:
                bal[receiver][k] = bal[receiver].get(k, 0.0) + total * float(pct)

        # Senders: drain received buckets first, then the original bucket.
        for sender in {str(s) for s in rows["Sender CC"]}:
            if sender not in bal:
                continue
            remaining = float(senders.get(sender, 0.0))
            for label in sorted(lbl for lbl in bal[sender] if lbl != 0):
                if remaining <= 0:
                    break
                take = min(bal[sender][label], remaining)
                bal[sender][label] -= take
                remaining -= take
            bal[sender][0] = bal[sender].get(0, 0.0) - remaining

        snapshots[k] = {cc: dict(labels) for cc, labels in bal.items()}

    files: dict[int, pd.DataFrame] = {}
    for n in cycles:
        snap = snapshots[n]
        after_cols = [_after_col(k) for k in cycles if k <= n]
        records = []
        for cc in ccs:
            labels = snap.get(cc, {})
            row: dict[str, object] = {"CC": cc, "배부전금액": pre.get(cc, 0.0)}
            for k in cycles:
                if k > n:
                    continue
                val = labels.get(k, 0.0)
                if k == n:
                    val += labels.get(0, 0.0)  # fold original/static into last col
                row[_after_col(k)] = val
            row["배부합계"] = sum(row[c] for c in after_cols)
            records.append(row)
        files[n] = pd.DataFrame.from_records(
            records, columns=["CC", "배부전금액"] + after_cols + ["배부합계"]
        )

    return files
