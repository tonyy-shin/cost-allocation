"""Shared pytest fixtures for the cost-allocation pipeline tests.

The fixtures mirror the wiring in test_run.py so individual test modules can
request just the stage they care about (loaded inputs, or the full pipeline
output) without re-deriving the plumbing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the project root importable so `import src...` works regardless of the
# directory pytest is invoked from. conftest.py is imported before the sibling
# test modules, so this runs before any `from src...` import is evaluated.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loader import (  # noqa: E402
    apply_category_dtypes, build_category_dtypes, enrich_cc,
    load_cc, load_coa_amount, load_cycle, load_mapping,
)
from src.prepare import (  # noqa: E402
    aggregate_detail, aggregate_for_allocation,
    assign_transfer_coa, calculate_coa_ratio, separate_common_direct,
)
from src.allocation import (  # noqa: E402
    aggregate_received_by_cycle, build_pivot_matrix,
    decompose_to_original_coa, run_allocation_loop,
)
from src.output import build_result  # noqa: E402


@pytest.fixture
def sample_paths() -> dict[str, Path]:
    """TEST_PATHS dict pointing at the sample_data CSVs (cf. test_run.py 18-24)."""
    base = PROJECT_ROOT / "sample_data"
    return {
        "cc":         base / "cc.csv",
        "coa_amount": base / "coa_amount.csv",
        "mapping":    base / "mapping.csv",
        "cycle":      base / "cycle.csv",
        "output_dir": base / "output",
    }


@pytest.fixture
def loaded_inputs(sample_paths) -> dict:
    """Run loaders + dtype harmonization + CC enrichment (test_run.py steps 1-2).

    Returns
    -------
    dict with keys:
        cc_df, coa_df (enriched), mapping_df, cycle_df,
        raw_coa_df (before enrich_cc), cat_dtypes
    """
    cc_df = load_cc(sample_paths["cc"])
    coa_df = load_coa_amount(sample_paths["coa_amount"])
    mapping_df = load_mapping(sample_paths["mapping"])
    cycle_df = load_cycle(sample_paths["cycle"])

    cat_dtypes = build_category_dtypes(cc_df, coa_df, mapping_df)
    cc_df, coa_df, mapping_df = apply_category_dtypes(
        cc_df, coa_df, mapping_df, dtypes=cat_dtypes
    )
    raw_coa_df = coa_df
    coa_df = enrich_cc(coa_df, cc_df)

    return {
        "cc_df": cc_df,
        "coa_df": coa_df,
        "mapping_df": mapping_df,
        "cycle_df": cycle_df,
        "raw_coa_df": raw_coa_df,
        "cat_dtypes": cat_dtypes,
    }


@pytest.fixture
def pipeline_outputs(loaded_inputs) -> dict:
    """Run the full pipeline through build_result (test_run.py steps 3-12).

    Returns
    -------
    dict with keys:
        df_direct, df_5a, df_5b, df_ratio, pivot, delta_by_cycle,
        received_by_cycle, decomposed, result
    """
    cc_df = loaded_inputs["cc_df"]
    coa_df = loaded_inputs["coa_df"]
    mapping_df = loaded_inputs["mapping_df"]
    cycle_df = loaded_inputs["cycle_df"]
    raw_coa_df = loaded_inputs["raw_coa_df"]

    enriched = assign_transfer_coa(coa_df, mapping_df)
    df_common, df_direct = separate_common_direct(enriched)
    df_5a = aggregate_detail(df_common)
    df_5b = aggregate_for_allocation(df_5a)
    df_ratio = calculate_coa_ratio(df_5a)

    cc_list = cc_df["CC"].unique().tolist()
    pivot = build_pivot_matrix(df_5b, cc_list)
    _, delta_by_cycle = run_allocation_loop(pivot, cycle_df)
    received_by_cycle = aggregate_received_by_cycle(delta_by_cycle)
    decomposed = decompose_to_original_coa(received_by_cycle, df_ratio)

    n_cycles = cycle_df["차수"].nunique()
    result = build_result(decomposed, df_direct, raw_coa_df, cc_df, n_cycles)

    return {
        "df_direct": df_direct,
        "df_5a": df_5a,
        "df_5b": df_5b,
        "df_ratio": df_ratio,
        "pivot": pivot,
        "delta_by_cycle": delta_by_cycle,
        "received_by_cycle": received_by_cycle,
        "decomposed": decomposed,
        "result": result,
    }
