"""End-to-end integration tests over the full pipeline (pipeline_outputs)."""
from __future__ import annotations

import warnings

import pytest

from src.allocation import build_by_cc, build_by_coa
from src.output import save_results
from src.prepare import build_enriched


def test_end_to_end_writes_expected_tree(pipeline_outputs, tmp_path):
    out = save_results(
        pipeline_outputs["by_coa_df"],
        pipeline_outputs["by_cc_files"],
        tmp_path,
    )
    assert (out / "by_coa" / "result.csv").exists()
    assert (out / "by_cc" / "1차배부후.csv").exists()
    assert (out / "by_cc" / "2차배부후.csv").exists()


def test_end_to_end_conservation_per_file(pipeline_outputs):
    # Allocation only moves money between CCs, so each file's post-allocation
    # total equals its pre-allocation total.
    for df in pipeline_outputs["by_cc_files"].values():
        assert df["배부합계"].sum() == pytest.approx(df["배부전금액"].sum())


def test_end_to_end_hand_checked_cycle_flow(pipeline_outputs):
    # Cycle 1: sender 1001 distributes 7,000,000 (0.3 → 1002, 0.7 → 1003).
    file1 = pipeline_outputs["by_cc_files"][1]

    def after1(cc):
        return file1.loc[file1["CC"] == cc, "1차후금액"].iloc[0]

    assert after1("1002") == pytest.approx(500_000.0 + 7_000_000.0 * 0.3)
    assert after1("1003") == pytest.approx(300_000.0 + 7_000_000.0 * 0.7)

    # Cycle 2: sender 2001 distributes 3,500,000 (0.5 → 1001, 0.5 → 3001).
    file2 = pipeline_outputs["by_cc_files"][2]

    def after2(cc):
        return file2.loc[file2["CC"] == cc, "2차후금액"].iloc[0]

    # 3001 starts at 0 (absent from pre_allocation) and receives half.
    assert after2("3001") == pytest.approx(3_500_000.0 * 0.5)


def test_no_unexpected_warnings_on_happy_path(loaded_inputs):
    # The build stage (enrichment + both output builders) is warning-free for
    # well-formed sample data.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        enriched = build_enriched(
            loaded_inputs["coa_df"], loaded_inputs["mapping_df"]
        )
        by_coa_df, sender_totals = build_by_coa(
            enriched, loaded_inputs["cycle_df"]
        )
        build_by_cc(
            loaded_inputs["cc_list"],
            loaded_inputs["pre_alloc_cc"],
            loaded_inputs["cycle_df"],
            sender_totals,
        )

    assert [str(w.message) for w in caught] == []
