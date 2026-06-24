from __future__ import annotations

from pathlib import Path

import pandas as pd


# Save the two-branch result tree (배부금액 / 잔액)


def append_total_row(df: pd.DataFrame) -> pd.DataFrame:
    """Append a display-only totals row to a by_cc snapshot.

    Every numeric column — 배부전금액 and each per-cycle 후금액 snapshot — is summed
    and rounded to an integer; CC is labelled "합계" and the 전기COA/기존COA key
    columns are left blank. Individual rows are untouched and a new frame is
    returned (the input is not mutated). Applied only when writing the CSV, so the
    in-memory by_cc frames keep their (전기COA, 기존COA, CC) rows.

    Parameters
    ----------
    df : One by_cc per-cycle snapshot. Columns 전기COA, 기존COA, CC, 배부전금액 and
         one or more n차후금액 columns.

    Returns
    -------
    pd.DataFrame
        df with a totals row concatenated at the bottom.
    """
    total = {col: "" for col in df.columns}
    total["CC"] = "합계"
    for col in df.columns:
        if col in ("전기COA", "기존COA", "CC"):
            continue
        total[col] = int(round(df[col].sum(), 0))
    total_row = pd.DataFrame([total], columns=df.columns)
    return pd.concat([df, total_row], ignore_index=True)


def save_results(
    by_coa_df: pd.DataFrame,
    by_cc_files: dict[int, pd.DataFrame],
    output_dir: Path,
) -> Path:
    """Write the by_coa and by_cc outputs under the output directory.

    Layout:
        <output_dir>/배부금액/result.csv
        <output_dir>/잔액/{n}차배부후.csv   (one per cycle)

    All files use utf-8-sig encoding for Excel-compatible Korean.

    Parameters
    ----------
    by_coa_df   : build_by_coa result (single table).
    by_cc_files : build_by_cc result. {cycle n: per-cycle snapshot DataFrame}.
    output_dir  : Directory under which the 배부금액/ and 잔액/ folders are created.

    Returns
    -------
    Path
        The output_dir root.
    """
    output_dir = Path(output_dir)
    by_coa_dir = output_dir / "배부금액"
    by_cc_dir = output_dir / "잔액"
    by_coa_dir.mkdir(parents=True, exist_ok=True)
    by_cc_dir.mkdir(parents=True, exist_ok=True)

    by_coa_df.to_csv(by_coa_dir / "result.csv", index=False, encoding="utf-8-sig")
    for n, df in by_cc_files.items():
        out_df = append_total_row(df)
        out_df.to_csv(by_cc_dir / f"{n}차배부후.csv", index=False, encoding="utf-8-sig")

    return output_dir
