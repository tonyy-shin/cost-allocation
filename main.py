from __future__ import annotations
import sys
from tkinter import messagebox
from src.ui import prompt_file_paths, show_completion
import warnings

import pandas as pd

from src.allocation import (
    aggregate_received_by_cycle,
    build_pivot_matrix,
    decompose_to_original_coa,
    run_allocation_loop,
)
from src.loader import (
    apply_category_dtypes,
    build_category_dtypes,
    enrich_cc,
    load_cc,
    load_coa_amount,
    load_cycle,
    load_mapping,
)
from src.output import build_result, save_result
from src.prepare import (
    aggregate_detail,
    aggregate_for_allocation,
    assign_transfer_coa,
    calculate_coa_ratio,
    separate_common_direct,
    validate_cycle_cc,
    validate_sender_coverage,
)


class PipelineAborted(Exception):
    """Raised when the user cancels the run. Treated as a normal exit."""



def _load_inputs(
    paths: dict,
    notes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Steps 1–2: load files, validate cycle CCs, apply dtypes, enrich CC.

    Returns (cc_df, raw_coa_df, coa_df, mapping_df, cycle_df).
    raw_coa_df is the pre-enrich snapshot passed to build_result for COA range.
    Appends a note to `notes` if the user continues past unknown CCs.
    Raises PipelineAborted if the user declines to continue.
    """
    validation_errors: list[str] = []

    cc_df = coa_df = mapping_df = cycle_df = None
    try:
        cc_df = load_cc(paths["cc"])
    except ValueError as e:
        validation_errors.append(str(e))
    try:
        coa_df = load_coa_amount(paths["coa_amount"])
    except ValueError as e:
        validation_errors.append(str(e))
    try:
        mapping_df = load_mapping(paths["mapping"])
    except ValueError as e:
        validation_errors.append(str(e))
    try:
        cycle_df = load_cycle(paths["cycle"])
    except ValueError as e:
        validation_errors.append(str(e))

    if validation_errors:
        for msg in validation_errors:
            notes.append(msg)
        raise PipelineAborted("입력 파일 검증에 실패했습니다.")

    unknown_ccs = validate_cycle_cc(cycle_df, cc_df)
    if unknown_ccs:
        msg = (
            f"다음 CC가 마스터에서 발견되지 않았습니다:\n"
            f"{', '.join(unknown_ccs)}\n\n"
            f"계속 진행하시겠습니까?"
        )
        if not messagebox.askyesno("알 수 없는 CC 경고", msg):
            raise PipelineAborted("실행이 중단되었습니다: cycle 시트에 알 수 없는 CC가 있습니다.")
        notes.append(
            "cycle 시트에서 알 수 없는 CC 발견 (계속 진행): "
            + ", ".join(unknown_ccs)
        )


    dtypes = build_category_dtypes(cc_df, coa_df, mapping_df)
    cc_df, coa_df, mapping_df = apply_category_dtypes(
        cc_df, coa_df, mapping_df, dtypes=dtypes
    )

    raw_coa_df = coa_df
    coa_df = enrich_cc(coa_df, cc_df)
    return cc_df, raw_coa_df, coa_df, mapping_df, cycle_df




def _prepare_costs(
    coa_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    cycle_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Steps 3–6: separate common/direct, aggregate, compute base COA ratios.

    Returns (df_direct, df_5b, df_ratio).
    """
    enriched = assign_transfer_coa(coa_df, mapping_df)
    df_common, df_direct = separate_common_direct(enriched)

    df_5a = aggregate_detail(df_common)
    df_5b = aggregate_for_allocation(df_5a)
    df_ratio = calculate_coa_ratio(df_5a)

    return df_direct, df_5b, df_ratio



def _run_allocation(
    df_5b: pd.DataFrame,
    df_ratio: pd.DataFrame,
    cc_list: list[str],
    cycle_df: pd.DataFrame,
) -> pd.DataFrame:
    """Steps 7–9: build pivot, run sequential allocation, decompose to base COA.

    Returns common_decomposed DataFrame.
    """
    pivot = build_pivot_matrix(df_5b, cc_list)
    _, delta_by_cycle = run_allocation_loop(pivot, cycle_df)
    received_by_cycle = aggregate_received_by_cycle(delta_by_cycle)
    
    return decompose_to_original_coa(received_by_cycle, df_ratio)



def main() -> None:
    """Entry point. Runs UI -> load -> allocate -> save in sequence.

    The whole pipeline runs inside a warnings.catch_warnings(record=True)
    block so that warnings emitted by warnings.warn (sender-balance and
    conservation checks) are collected instead of printed. Manual notes
    (e.g. unknown-CC continue) are merged in. The outcome is reported via
    a completion dialog: success / warning / failure.
    """
    paths = prompt_file_paths()
    if paths is None:
        sys.exit(0)  # user closed the file-selection window: silent cancel

    notes: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            cc_df, raw_coa_df, coa_df, mapping_df, cycle_df = _load_inputs(
                paths, notes
            )
            df_direct, df_5b, df_ratio = _prepare_costs(
                coa_df, mapping_df, cycle_df
            )

            cc_list = cc_df["CC"].unique().tolist()
            common_decomposed = _run_allocation(df_5b, df_ratio, cc_list, cycle_df)

            n_cycles = cycle_df["차수"].nunique()
            result = build_result(
                common_decomposed, df_direct, raw_coa_df, cc_df, n_cycles
            )
            out_path = save_result(result, paths["output_dir"])

        messages = notes + [str(w.message) for w in caught]

    except PipelineAborted as exc:
        show_completion("failure", error=str(exc), warnings=notes)
        sys.exit(0)  # deliberate user cancel: not an error exit code
    except Exception as exc:
        show_completion("failure", error=f"{type(exc).__name__}: {exc}", warnings=notes)
        sys.exit(1)

    if messages:
        show_completion("warning", out_path=out_path, warnings=messages)
    else:
        show_completion("success", out_path=out_path)


    


if __name__ == "__main__":
    main()
