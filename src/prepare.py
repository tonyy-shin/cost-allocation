from __future__ import annotations

import pandas as pd


# Step 2-B: Override master amounts


def apply_override(coa_df: pd.DataFrame, override_df: pd.DataFrame) -> pd.DataFrame:
    """Override coa_df Amounts with override_df values, matched on (COA, Cost Center).

    coa_df is first collapsed to one row per (COA, Cost Center) via groupby-sum:
    a duplicated combo in the raw master would otherwise broadcast the override
    value across every duplicate row and inflate the total. No other columns
    exist yet (전기COA is assigned later) and build_by_coa re-aggregates on the
    same keys downstream, so the collapse is lossless.

    Applied before CategoricalDtype harmonization so both merge keys are plain
    str (object); a Categorical-vs-object key mismatch would silently fail to match.

    Combos present in both sheets take the override Amounts; combos only in coa_df
    keep their original value. Combos only in override_df do not occur (confirmed),
    so they are not validated.

    Parameters
    ----------
    coa_df      : load_coa_amount result. Columns: COA, Cost Center, Amounts.
    override_df : load_override_amount result. Same schema as coa_df.

    Returns
    -------
    pd.DataFrame
        coa_df with Amounts overridden where a matching (COA, Cost Center) exists.
        Same column order and key-column dtypes as the input; Amounts is float64
        (the canonical money type) regardless of whether the merge introduced NaN.
    """
    cols = list(coa_df.columns)
    coa_df = (
        coa_df.groupby(["COA", "Cost Center"], as_index=False, observed=True)["Amounts"]
        .sum()
    )
    merged = coa_df.merge(
        override_df[["COA", "Cost Center", "Amounts"]],
        on=["COA", "Cost Center"],
        how="left",
        suffixes=("", "_override"),
    )
    merged["Amounts"] = (
        merged["Amounts_override"].combine_first(merged["Amounts"]).astype("float64")
    )
    return merged[cols]


# Step 2-C: Fill cycle-only Cost Centers


def fill_missing_cycle_cc(coa_df: pd.DataFrame, cycle_df: pd.DataFrame) -> pd.DataFrame:
    """Add zero-amount rows for cycle CCs absent from the master.

    A CC that appears in cycle.csv (as sender or receiver) but has no row in
    coa_amount.csv would be dropped from cc_list and never appear in by_cc, so
    money it receives as a receiver is lost. Each missing CC is inserted as
    Cost Center=<cc>, Amounts=0, COA=NaN. COA=NaN keeps the row out of by_coa
    (common cost = 전기COA != "") while still contributing the CC to the by_cc list.

    Run before CategoricalDtype harmonization so keys are still plain str; the
    new CC then joins the CC category set built downstream.

    Parameters
    ----------
    coa_df   : load_coa_amount result (post-override). Columns: COA, Cost Center, Amounts.
    cycle_df : load_cycle result. Sender CC / Receiver CC columns are scanned.

    Returns
    -------
    pd.DataFrame
        coa_df plus one zero-amount, NaN-COA row per cycle CC missing from the
        master. Same column order and dtypes (keys object str, Amounts float64).
    """
    cols = list(coa_df.columns)
    cycle_ccs = (
        pd.concat([cycle_df["Sender CC"], cycle_df["Receiver CC"]])
        .astype(str).unique()
    )
    existing = set(coa_df["Cost Center"].astype(str))
    missing = [cc for cc in cycle_ccs if cc not in existing]
    if not missing:
        return coa_df

    extra = pd.DataFrame({
        "COA": pd.Series([pd.NA] * len(missing), dtype="object"),
        "Cost Center": pd.Series(missing, dtype="object"),
        "Amounts": pd.Series([0.0] * len(missing), dtype="float64"),
    })[cols]
    result = pd.concat([coa_df, extra], ignore_index=True)
    return result[cols]


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
