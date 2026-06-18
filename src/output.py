from __future__ import annotations

from pathlib import Path

import pandas as pd


# Save the two-branch result tree (by_coa / by_cc)


def save_results(
    by_coa_df: pd.DataFrame,
    by_cc_files: dict[int, pd.DataFrame],
    output_dir: Path,
) -> Path:
    """Write the by_coa and by_cc outputs under the output directory.

    Layout:
        <output_dir>/by_coa/result.csv
        <output_dir>/by_cc/{n}차배부후.csv   (one per cycle)

    All files use utf-8-sig encoding for Excel-compatible Korean.

    Parameters
    ----------
    by_coa_df   : build_by_coa result (single table).
    by_cc_files : build_by_cc result. {cycle n: per-cycle snapshot DataFrame}.
    output_dir  : Directory under which the by_coa/ and by_cc/ folders are created.

    Returns
    -------
    Path
        The output_dir root.
    """
    output_dir = Path(output_dir)
    by_coa_dir = output_dir / "by_coa"
    by_cc_dir = output_dir / "by_cc"
    by_coa_dir.mkdir(parents=True, exist_ok=True)
    by_cc_dir.mkdir(parents=True, exist_ok=True)

    by_coa_df.to_csv(by_coa_dir / "result.csv", index=False, encoding="utf-8-sig")
    for n, df in by_cc_files.items():
        df.to_csv(by_cc_dir / f"{n}차배부후.csv", index=False, encoding="utf-8-sig")

    return output_dir
