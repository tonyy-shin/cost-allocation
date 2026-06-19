"""End-to-end integration tests over the full pipeline (pipeline_outputs)."""
from __future__ import annotations

import warnings

import pytest

from src.core.allocation import build_by_cc, build_by_coa
from src.data.output import save_results
from src.core.prepare import build_enriched


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
    # Cycle 1: sender 1001 distributes 7,000,000 (0.3 → 1002, 0.7 → 1003). A CC
    # now spans multiple (전기COA, 기존COA) rows, so sum across them per CC.
    file1 = pipeline_outputs["by_cc_files"][1]

    def after1(cc):
        return file1.loc[file1["CC"] == cc, "1차후금액"].sum()

    assert after1("1002") == pytest.approx(500_000.0 + 7_000_000.0 * 0.3)
    assert after1("1003") == pytest.approx(300_000.0 + 7_000_000.0 * 0.7)

    # Cycle 2: sender 2001 distributes 3,500,000 (0.5 → 1001, 0.5 → 3001).
    file2 = pipeline_outputs["by_cc_files"][2]

    def after2(cc):
        return file2.loc[file2["CC"] == cc, "2차후금액"].sum()

    # 3001 starts at 0 (absent from pre_allocation) and receives half.
    assert after2("3001") == pytest.approx(3_500_000.0 * 0.5)


def test_by_coa_amounts_nonzero(pipeline_outputs):
    # The master (coa_amount.csv) supplies the real amounts for the cycle
    # sender/receiver combos, so by_coa's per-cycle totals must be the real
    # (non-zero) values, not 0.
    by_coa = pipeline_outputs["by_coa_df"]

    # 배부합계 columns carry a single column-wide scalar in row 0 only.
    assert by_coa["1차배부합계"].iloc[0] == pytest.approx(7_000_000.0)
    assert by_coa["2차배부합계"].iloc[0] == pytest.approx(3_500_000.0)
    assert (by_coa["1차배부금액"].sum() + by_coa["2차배부금액"].sum()) != 0


def test_cycle_only_cc_appears_in_by_cc(pipeline_outputs):
    # 4001 is a receiver in cycle.csv but absent from coa_amount.csv. It must be
    # filled in (배부전금액 0) and still receive its cycle-3 allocation (7,000,000),
    # now spread across the COA pairs it received (E6100/6100 + E6200/6200).
    file3 = pipeline_outputs["by_cc_files"][3]
    rows = file3[file3["CC"] == "4001"]
    assert len(rows) >= 1
    assert rows["배부전금액"].sum() == pytest.approx(0.0)
    assert rows["3차후금액"].sum() == pytest.approx(7_000_000.0)
    assert rows["배부합계"].sum() == pytest.approx(7_000_000.0)


def test_cycle_only_cc_absent_from_by_coa(pipeline_outputs):
    # COA is NaN for the filled CC, so it is a non-common row and never sends a
    # common cost: it must not appear as a Sender CC in by_coa. It may appear as a
    # Receiver CC (cycle 3 routes 1001 → 4001), which is expected.
    by_coa = pipeline_outputs["by_coa_df"]
    assert "4001" not in set(by_coa["Sender CC"].astype(str))


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
            loaded_inputs["pre_alloc_enriched"],
            loaded_inputs["cycle_df"],
            sender_totals,
        )

    assert [str(w.message) for w in caught] == []
