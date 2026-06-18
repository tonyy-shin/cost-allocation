"""Tests for src.output: the by_coa / by_cc directory tree and CSV writing."""
from __future__ import annotations

import pandas as pd

from src.output import save_results


def test_save_results_writes_directory_tree(pipeline_outputs, tmp_path):
    out = save_results(
        pipeline_outputs["by_coa_df"],
        pipeline_outputs["by_cc_files"],
        tmp_path,
    )

    assert out == tmp_path
    by_coa = tmp_path / "by_coa" / "result.csv"
    assert by_coa.exists()
    # One by_cc file per cycle, named {n}차배부후.csv.
    for n in pipeline_outputs["by_cc_files"]:
        assert (tmp_path / "by_cc" / f"{n}차배부후.csv").exists()


def test_save_results_uses_utf8_sig(pipeline_outputs, tmp_path):
    save_results(
        pipeline_outputs["by_coa_df"],
        pipeline_outputs["by_cc_files"],
        tmp_path,
    )
    # utf-8-sig writes a BOM prefix for Excel compatibility.
    by_coa = tmp_path / "by_coa" / "result.csv"
    assert by_coa.read_bytes().startswith(b"\xef\xbb\xbf")


def test_save_results_by_cc_round_trips(pipeline_outputs, tmp_path):
    save_results(
        pipeline_outputs["by_coa_df"],
        pipeline_outputs["by_cc_files"],
        tmp_path,
    )
    file1 = pd.read_csv(tmp_path / "by_cc" / "1차배부후.csv", encoding="utf-8-sig")
    expected = pipeline_outputs["by_cc_files"][1]
    assert list(file1.columns) == list(expected.columns)
    assert len(file1) == len(expected)
