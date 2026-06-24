from __future__ import annotations
import sys
from src.ui.ui import prompt_file_paths, show_completion
import warnings

import pandas as pd

from src.core.allocation import build_by_coa, build_by_cc
from src.data.loader import (
    apply_category_dtypes,
    build_category_dtypes,
    load_coa_amount,
    load_cycle,
    load_mapping,
    load_pre_allocation,
)
from src.data.output import save_results
from src.core.prepare import (
    build_enriched,
    fill_missing_cycle_cc,
)


class PipelineAborted(Exception):
    """Raised when the user cancels the run. Treated as a normal exit."""



def _load_inputs(
    paths: dict,
    notes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Steps 1–2: load files, apply fills, apply dtypes.

    Returns (coa_df, mapping_df, cycle_df, pre_alloc_enriched). The CC list is
    taken from coa_df["Cost Center"]; there is no separate CC master file. The
    pre_allocation amounts are summed by (COA, Cost Center) and enriched with the
    transfer COA (build_enriched) so the by_cc output's 배부전금액 column can be
    split by (전기COA, 기존COA). The enrichment uses the str-typed mapping_df
    *before* CategoricalDtype harmonization so the COA join does not hit a
    category/object dtype mismatch (and a pre_allocation COA absent from the
    period's amount sheet is not silently masked to NaN).
    Raises PipelineAborted if any input file fails validation.
    """
    validation_errors: list[str] = []

    coa_df = mapping_df = cycle_df = pre_alloc_df = None
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
    try:
        pre_alloc_df = load_pre_allocation(paths["pre_allocation"])
    except ValueError as e:
        validation_errors.append(str(e))

    if validation_errors:
        for msg in validation_errors:
            notes.append(msg)
        raise PipelineAborted("입력 파일 검증에 실패했습니다.")

    # Enrich pre_allocation while mapping_df is still str-typed (before the
    # category cast below) so the COA join in build_enriched stays object/object.
    pre_alloc_enriched = build_enriched(pre_alloc_df, mapping_df)

    # Insert zero-amount rows for cycle CCs missing from the master so they still
    # appear in by_cc and receive their allocations. A cycle CC absent from the
    # master is expected (no validation), and this runs before dtype harmonization.
    coa_df = fill_missing_cycle_cc(coa_df, cycle_df)

    dtypes = build_category_dtypes(coa_df, mapping_df)
    coa_df, mapping_df = apply_category_dtypes(coa_df, mapping_df, dtypes=dtypes)

    return coa_df, mapping_df, cycle_df, pre_alloc_enriched



def main() -> None:
    """Entry point. Runs UI -> load -> build outputs -> save in sequence.

    The whole pipeline runs inside a warnings.catch_warnings(record=True)
    block so that warnings emitted by warnings.warn (loader data-quality
    checks) are collected instead of printed. Manual notes (e.g. unknown-CC
    continue) are merged in. The outcome is reported via a completion dialog:
    success / warning / failure.
    """
    paths = prompt_file_paths()
    if paths is None:
        sys.exit(0)  # user closed the file-selection window: silent cancel

    notes: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            coa_df, mapping_df, cycle_df, pre_alloc_enriched = _load_inputs(
                paths, notes
            )

            cc_list = coa_df["Cost Center"].astype(str).unique().tolist()
            enriched = build_enriched(coa_df, mapping_df)

            by_coa_df, sender_totals = build_by_coa(enriched, cycle_df)
            by_cc_files = build_by_cc(
                cc_list, pre_alloc_enriched, cycle_df, sender_totals
            )

            out_path = save_results(by_coa_df, by_cc_files, paths["output_dir"])

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
