from __future__ import annotations
import sys
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
)
from src.ui import prompt_file_paths


def _load_inputs(
    paths: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Steps 1–2: load files, validate cycle CCs, apply dtypes, enrich CC.

    Returns (cc_df, raw_coa_df, coa_df, mapping_df, cycle_df).
    raw_coa_df is the pre-enrich snapshot passed to build_result for COA range.
    Exits with code 1 if user declines to continue after CC validation failure.
    """
    cc_df = load_cc(paths["cc"])
    coa_df = load_coa_amount(paths["coa_amount"])
    mapping_df = load_mapping(paths["mapping"])
    cycle_df = load_cycle(paths["cycle"])

    unknown_ccs = validate_cycle_cc(cycle_df, cc_df)
    if unknown_ccs:
        print(f"[Warning] CCs not found in master: {unknown_ccs}")
        ans = input("Continue anyway? [y/N]: ").strip().lower()
        if ans != "y":
            print("Aborted.")
            sys.exit(1)

    dtypes = build_category_dtypes(cc_df, coa_df, mapping_df)
    cc_df, coa_df, mapping_df = apply_category_dtypes(
        cc_df, coa_df, mapping_df, dtypes = dtypes
    )

    raw_coa_df = coa_df
    coa_df = enrich_cc(coa_df, cc_df)
    return cc_df, raw_coa_df, coa_df, mapping_df, cycle_df



def _prepare_costs(
    coa_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Steps 3–6: separate common/direct, aggregate, compute base COA ratios.

    Returns (df_direct, df_5b, df_ratio).
    """
    enriched = assign_transfer_coa(coa_df, mapping_df)
    df_common, df_direct = separate_common_direct(enriched)

    df_5a = aggregate_detail(df_common)
    df_5b = aggregate_for_allocation(df_5a)
    df_ratio = calculate_coa_ratio(df_5a, mapping_df)

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

    Flow:
        1. Collect five paths (four inputs + output directory) via prompt_file_paths.
        2. Load four CSVs, build and apply CategoricalDtypes, enrich CC.
        3. Validate cycle CCs against the master.
           - Unknown CCs found: ask the user whether to skip those rows or abort.
        4. Run the allocation pipeline (allocation module, Steps 3-9).
        5. Build the final result and save to CSV (output module, Steps 10-12).
        6. Print a completion message including the saved file path.
    """
    paths = prompt_file_paths()
    if paths is None:
        sys.exit(0)

    cc_df, raw_coa_df, coa_df, mapping_df, cycle_df = _load_inputs(paths)
    df_direct, df_5b, df_ratio = _prepare_costs(coa_df, mapping_df)

    cc_list = cc_df["CC"].unique().tolist()
    common_decomposed = _run_allocation(df_5b, df_ratio, cc_list, cycle_df)

    n_cycles = cycle_df["차수"].nunique()
    result = build_result(
        common_decomposed, df_direct, raw_coa_df, cc_df, n_cycles
    )

    out_path = save_result(result, paths["output_dir"])
    print(f"Done. Result saved to: {out_path}")

    


if __name__ == "__main__":
    warnings.simplefilter("always")
    main()
