from __future__ import annotations

from pathlib import Path

import pandas as pd


# Build the final result grid (Sender CC keyed, long format)


def build_result(sender_decomposed: pd.DataFrame) -> pd.DataFrame:
    """Build the final result grid from the sender-side decomposition.

    The result is keyed by Sender CC (who distributed the cost) in long format,
    one row per (차수, 전기COA, 기존COA, Sender CC). This is simply the
    decompose_sender_to_original_coa output with a fixed column order; the heavy
    lifting (ratio decomposition, sorting) happens upstream.

    Parameters
    ----------
    sender_decomposed : decompose_sender_to_original_coa result.

    Returns
    -------
    pd.DataFrame
        Columns: 차수, 전기COA, 기존COA, Sender CC, 배분금액.
    """
    cols = ["차수", "전기COA", "기존COA", "Sender CC", "배분금액"]
    return sender_decomposed[cols].reset_index(drop=True)


# Save to CSV


def save_result(result_df: pd.DataFrame, output_dir: Path) -> Path:
    """Write the final result DataFrame to a CSV file.

    Filename: result.csv, encoding: utf-8-sig (Excel compatible Korean support).

    Parameters
    ----------
    result_df  : build_result output.
    output_dir : Directory where the file will be saved.

    Returns
    -------
    Path
        Full path of the saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "result.csv"
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path