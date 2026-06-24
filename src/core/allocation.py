from __future__ import annotations

import warnings
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


# by_coa: sender common-cost amounts keyed by (전기COA, 기존COA, Sender CC, Receiver CC)


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
        sender_totals : {cycle: {(전기COA, 기존COA, Sender CC): total distributed
                        in that cycle}}. Keyed by COA pair so by_cc can carry the
                        money's COA identity through the bucket simulation.
    """
    # Surface unmapped COAs (no 전기COA mapping) that sit on a sender CC: their
    # money is excluded from allocation by the common-cost filter below and would
    # otherwise remain as a silent residual balance on the sender.
    senders = set(cycle_df["Sender CC"].astype(str))
    excluded = enriched[
        (enriched["전기COA"].astype(str) == "")
        & enriched["기존COA"].astype(str).str.strip().ne("")
        & enriched["기존COA"].astype(str).str.lower().ne("nan")
        & enriched["Cost Center"].isin(senders)
        & (enriched["Amounts"] != 0)
    ]
    if not excluded.empty:
        lines = []
        for (coa, cc), grp in excluded.groupby(
            ["기존COA", "Cost Center"], observed=True
        ):
            amount = int(grp["Amounts"].sum())
            lines.append(f"  - COA {coa} / Sender CC {cc}: {amount:,}")
        warnings.warn(
            "전기COA 매핑이 없어 배부에서 제외된 항목이 있습니다. "
            "해당 금액은 배부되지 않고 잔액으로 남습니다.\n"
            + "\n".join(lines)
        )

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
        # sender_totals stays pre-explode and ratio-free, but keyed by the full
        # (전기COA, 기존COA, Sender CC) pair so by_cc can preserve each amount's
        # COA identity. `grouped` is already unique per that triple.
        sender_totals[n] = {
            (str(e), str(b), str(s)): float(amt)
            for e, b, s, amt in zip(
                grouped["전기COA"],
                grouped["기존COA"],
                grouped["Sender CC"],
                grouped[_amt_col(n)],
            )
        }

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
    enriched: pd.DataFrame,
    cycle_df: pd.DataFrame,
    sender_totals: dict[int, dict[tuple[str, str, str], float]],
) -> dict[int, pd.DataFrame]:
    """Build one per-cycle by_cc balance snapshot, keyed by (전기COA, 기존COA, CC).

    coa_amount.csv (via ``enriched``) is the single source of truth for every
    CC's settled balance. Two reference frames drive the simulation:

    - 원본 (original): the per-(전기COA, 기존COA, CC) ``enriched`` amounts, never
      modified. It defines both each CC's settled balance and the 배부전금액 base.
    - 복사본 (copy): a copy of 원본 with every CC zeroed except the *1차 sender*
      CCs (the senders of the minimum 차수), which keep their 원본 amount. Each
      cycle drains each sender's ``sender_totals[k]`` amount from 복사본; the per-
      cycle total drained is validated against the 원본 amount sitting at that
      cycle's receiver CCs (a warning, not an error, on mismatch).

    The displayed balance state starts from 복사본's layout. Per cycle k, in
    ascending order, every row whose CC sends in cycle k drops to 0 and every row
    whose CC receives in cycle k is (re)filled from its 원본 value. ``k차후금액``
    is the snapshot of this state after cycle k, so in any file a sender's row
    reads 0 in its own cycle and a receiver's row reads its 원본 amount.

    ``배부전금액`` is 0 for 1차 sender CCs and the 원본 amount for every other CC.
    ``배부합계`` is the final resting balance — the last 후금액 column of the file.

    Every CC in ``cc_list`` is guaranteed at least one row: a CC absent from
    ``enriched`` gets a single all-zero row with 전기COA="" and 기존COA="" so it is
    never dropped from the output.

    Parameters
    ----------
    cc_list       : All Cost Center codes from coa_amount.csv.
    enriched      : build_enriched result (the 원본). Columns:
                    전기COA, 기존COA, Cost Center, Amounts.
    cycle_df      : load_cycle result.
    sender_totals : build_by_coa's second return value, keyed by
                    (전기COA, 기존COA, Sender CC).

    Returns
    -------
    dict[int, pd.DataFrame]
        {cycle n: DataFrame with columns
         전기COA, 기존COA, CC, 배부전금액, 1차후금액 .. n차후금액, 배부합계}.
    """
    cycles = sorted(int(c) for c in cycle_df["차수"].unique())
    ccs = [str(cc) for cc in cc_list]

    # 원본: settled amount per (전기COA, 기존COA, CC). Never modified. Code columns
    # are str-normalized with NaN -> "" so cycle-only CCs (COA=NaN from
    # fill_missing_cycle_cc) sort cleanly and surface as a blank-COA row.
    def _codes(col: str) -> pd.Series:
        s = enriched[col]
        return s.astype(object).where(s.notna(), "").astype(str)

    원본: dict[tuple[str, str, str], float] = {}
    for e, b, cc, amt in zip(
        _codes("전기COA"),
        _codes("기존COA"),
        _codes("Cost Center"),
        enriched["Amounts"],
    ):
        key = (e, b, cc)
        원본[key] = 원본.get(key, 0.0) + float(amt)

    # 1차 sender CCs: the senders of the minimum 차수 in cycle_df.
    first_senders: set[str] = set()
    if cycles:
        first_cycle = cycles[0]
        first_senders = set(
            cycle_df.loc[cycle_df["차수"] == first_cycle, "Sender CC"].astype(str)
        )

    # 복사본: copy of 원본 with every CC zeroed except the 1차 senders.
    복사본 = {
        key: (amt if key[2] in first_senders else 0.0)
        for key, amt in 원본.items()
    }

    # Displayed balance state, snapshotted after each cycle to feed 후금액 columns.
    state = dict(복사본)
    snapshots: dict[int, dict[tuple[str, str, str], float]] = {}

    for k in cycles:
        rows = cycle_df[cycle_df["차수"] == k]
        senders_k = set(rows["Sender CC"].astype(str))
        receivers_k = set(rows["Receiver CC"].astype(str))

        # Deduct each sender's allocated amount from 복사본 and tally the total.
        deducted = 0.0
        for (e, b, s), total in sender_totals.get(k, {}).items():
            복사본[(e, b, s)] = 복사본.get((e, b, s), 0.0) - float(total)
            deducted += float(total)

        # Validate: the amount drained this cycle should equal the 원본 amount
        # sitting at this cycle's receiver CCs. Warn (do not raise) on mismatch.
        expected = sum(v for (e, b, cc), v in 원본.items() if cc in receivers_k)
        if abs(deducted - expected) > 1e-6:
            warnings.warn(
                f"{k}차 배부 검증 실패: 복사본에서 차감된 금액({deducted:,.0f})이 "
                f"원본 수신 CC 합계({expected:,.0f})와 일치하지 않습니다."
            )

        # Display: sender rows drop to 0, receiver rows take their 원본 value.
        for key in state:
            if key[2] in senders_k:
                state[key] = 0.0
        for key, amt in 원본.items():
            if key[2] in receivers_k:
                state[key] = amt
        snapshots[k] = dict(state)

    files: dict[int, pd.DataFrame] = {}
    for n in cycles:
        after_cols = [_after_col(k) for k in cycles if k <= n]
        records = []
        covered_ccs: set[str] = set()
        for key in sorted(원본):
            e, b, cc = key
            covered_ccs.add(cc)
            row: dict[str, object] = {
                "전기COA": e,
                "기존COA": b,
                "CC": cc,
                "배부전금액": 0.0 if cc in first_senders else 원본[key],
            }
            for k in cycles:
                if k > n:
                    continue
                row[_after_col(k)] = snapshots[k].get(key, 0.0)
            row["배부합계"] = row[_after_col(n)]
            records.append(row)

        # Option B: guarantee every CC at least one row. A CC absent from 원본
        # gets a single all-zero, blank-COA row.
        for cc in ccs:
            if cc in covered_ccs:
                continue
            row = {"전기COA": "", "기존COA": "", "CC": cc, "배부전금액": 0.0}
            for c in after_cols:
                row[c] = 0.0
            row["배부합계"] = 0.0
            records.append(row)

        df = pd.DataFrame.from_records(
            records,
            columns=["전기COA", "기존COA", "CC", "배부전금액"] + after_cols + ["배부합계"],
        )
        files[n] = df.sort_values(["전기COA", "기존COA", "CC"]).reset_index(drop=True)

    return files
