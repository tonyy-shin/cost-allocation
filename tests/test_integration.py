"""End-to-end integration tests over the full pipeline (pipeline_outputs).

The sample data is internally consistent with the single-source-of-truth model:
coa_amount.csv is the sole balance source, and each cycle's drained total equals
the 원본 amount at its receiver CCs, so build_by_cc emits no per-cycle validation
warning. The sample flow is

    cycle 1: 1001 -> 1002 (0.3), 1003 (0.7)
    cycle 2: 2001 -> 3001 (1.0)
    cycle 3: 1002 -> 1004 (1.0)        # 1002 forwards what it received in cycle 1

so 1002 is a pass-through. The only warning over the sample data is the unmapped
direct cost 7100 sitting on sender CCs 1001/2001 (from build_by_coa).
"""
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
    assert (out / "배부금액" / "result.csv").exists()
    assert (out / "잔액" / "1차배부후.csv").exists()
    assert (out / "잔액" / "2차배부후.csv").exists()


def test_end_to_end_first_sender_balance_is_zero(pipeline_outputs):
    # 1001 is the sole 1차 sender, so every one of its rows reports 배부전금액 = 0
    # in every file; non-1차 CCs keep their 원본 amount (1003 holds 6100=3,500,000
    # + 6200=1,400,000).
    for df in pipeline_outputs["by_cc_files"].values():
        assert (df.loc[df["CC"] == "1001", "배부전금액"] == 0.0).all()
        assert df.loc[df["CC"] == "1003", "배부전금액"].sum() == pytest.approx(
            3_500_000.0 + 1_400_000.0
        )


def test_end_to_end_receiver_rows_match_원본(pipeline_outputs):
    # After a cycle, each receiver row is filled from its 원본 (coa_amount) value,
    # so a receiver's 후금액 equals the sum of its 원본 amounts.
    file1 = pipeline_outputs["by_cc_files"][1]

    def after1(cc):
        return file1.loc[file1["CC"] == cc, "1차후금액"].sum()

    # Cycle 1 receivers 1002 / 1003 take their full 원본 balances.
    assert after1("1002") == pytest.approx(1_500_000.0 + 600_000.0)
    assert after1("1003") == pytest.approx(3_500_000.0 + 1_400_000.0)

    # Cycle 2 routes 2001 -> 3001, which takes its 원본 value (6100=3M + 6200=0.5M).
    file2 = pipeline_outputs["by_cc_files"][2]
    after2_3001 = file2.loc[file2["CC"] == "3001", "2차후금액"].sum()
    assert after2_3001 == pytest.approx(3_000_000.0 + 500_000.0)

    # Cycle 3 is the pass-through: 1002 forwards its money to 1004, which takes its
    # 원본 value in the last column.
    file3 = pipeline_outputs["by_cc_files"][3]
    after3_1004 = file3.loc[file3["CC"] == "1004", "3차후금액"].sum()
    assert after3_1004 == pytest.approx(1_500_000.0 + 600_000.0)


def test_end_to_end_sender_rows_zero_in_their_cycle(pipeline_outputs):
    # A sender's rows drop to 0 in its own cycle's 후금액 column.
    file1 = pipeline_outputs["by_cc_files"][1]
    assert file1.loc[file1["CC"] == "1001", "1차후금액"].sum() == pytest.approx(0.0)
    file2 = pipeline_outputs["by_cc_files"][2]
    assert file2.loc[file2["CC"] == "2001", "2차후금액"].sum() == pytest.approx(0.0)
    # The pass-through CC 1002 holds its money after cycle 1, then forwards it all
    # in cycle 3, so its 3차후금액 returns to 0.
    file3 = pipeline_outputs["by_cc_files"][3]
    assert file3.loc[file3["CC"] == "1002", "3차후금액"].sum() == pytest.approx(0.0)


def test_by_coa_amounts_nonzero(pipeline_outputs):
    # The master (coa_amount.csv) supplies the real amounts for the cycle
    # sender/receiver combos, so by_coa's per-cycle totals must be the real
    # (non-zero) values, not 0.
    by_coa = pipeline_outputs["by_coa_df"]

    # 배부합계 columns carry a single column-wide scalar in row 0 only.
    assert by_coa["1차배부합계"].iloc[0] == pytest.approx(7_000_000.0)
    assert by_coa["2차배부합계"].iloc[0] == pytest.approx(3_500_000.0)
    assert (by_coa["1차배부금액"].sum() + by_coa["2차배부금액"].sum()) != 0


def test_no_validation_warning_on_consistent_sample(loaded_inputs):
    # The sample data is internally consistent, so no per-cycle validation warning
    # fires. The only warning is the unmapped-COA notice (7100 on sender CCs with
    # no 전기COA mapping), emitted by build_by_coa.
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
            enriched,
            loaded_inputs["cycle_df"],
            sender_totals,
        )

    messages = [str(w.message) for w in caught]

    # No per-cycle validation mismatch.
    assert not any("배부 검증 실패" in m for m in messages)

    # Exactly the expected unmapped-COA notice.
    unmapped = [m for m in messages if m.startswith("전기COA 매핑이 없어")]
    assert len(unmapped) == 1
    assert "COA 7100 / Sender CC 1001: 10,000,000" in unmapped[0]
    assert "COA 7100 / Sender CC 2001: 6,000,000" in unmapped[0]
