"""Shared pytest fixtures for the cost-allocation pipeline tests.

The fixtures mirror the wiring in main.py so individual test modules can request
just the stage they care about (loaded inputs, or the full pipeline output)
without re-deriving the plumbing.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

# Make the project root importable so `import src...` works regardless of the
# directory pytest is invoked from. conftest.py is imported before the sibling
# test modules, so this runs before any `from src...` import is evaluated.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import (  # noqa: E402
    apply_category_dtypes, build_category_dtypes,
    load_coa_amount, load_cycle, load_mapping,
)
from src.core.prepare import (  # noqa: E402
    build_enriched, fill_missing_cycle_cc,
)
from src.core.allocation import build_by_coa, build_by_cc  # noqa: E402


@pytest.fixture
def sample_paths() -> dict[str, Path]:
    """Paths to the sample_data CSVs (cf. main._load_inputs)."""
    base = PROJECT_ROOT / "sample_data"
    return {
        "coa_amount":      base / "coa_amount.csv",
        "mapping":         base / "mapping.csv",
        "cycle":           base / "cycle.csv",
        "output_dir":      base / "output",
    }


@pytest.fixture
def loaded_inputs(sample_paths) -> dict:
    """Run loaders + dtype harmonization (main._load_inputs).

    The CC list is derived from the COA·CC master's Cost Center column; there is
    no separate CC master file. coa_amount.csv is the single source of truth for
    both the allocation amounts and the by_cc settled balances.

    Returns
    -------
    dict with keys:
        coa_df, mapping_df, cycle_df, raw_coa_df, cc_list, cat_dtypes
    """
    coa_df = load_coa_amount(sample_paths["coa_amount"])
    mapping_df = load_mapping(sample_paths["mapping"])
    cycle_df = load_cycle(sample_paths["cycle"])

    # Add cycle-only CCs before dtype harmonization (cf. main._load_inputs).
    coa_df = fill_missing_cycle_cc(coa_df, cycle_df)

    cat_dtypes = build_category_dtypes(coa_df, mapping_df)
    coa_df, mapping_df = apply_category_dtypes(
        coa_df, mapping_df, dtypes=cat_dtypes
    )
    raw_coa_df = coa_df
    cc_list = coa_df["Cost Center"].astype(str).unique().tolist()

    return {
        "coa_df": coa_df,
        "mapping_df": mapping_df,
        "cycle_df": cycle_df,
        "raw_coa_df": raw_coa_df,
        "cc_list": cc_list,
        "cat_dtypes": cat_dtypes,
    }


@pytest.fixture
def pipeline_outputs(loaded_inputs) -> dict:
    """Run the full pipeline through both output builders (main.main).

    The sample data is internally consistent with the single-source-of-truth
    model: each cycle's drained total equals the 원본 amount at its receiver CCs,
    so build_by_cc emits no per-cycle validation warning. (build_by_coa still
    flags the unmapped direct cost 7100 on the sender CCs — a separate notice.)

    Returns
    -------
    dict with keys:
        enriched, by_coa_df, sender_totals, by_cc_files
    """
    coa_df = loaded_inputs["coa_df"]
    mapping_df = loaded_inputs["mapping_df"]
    cycle_df = loaded_inputs["cycle_df"]
    cc_list = loaded_inputs["cc_list"]

    enriched = build_enriched(coa_df, mapping_df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        by_coa_df, sender_totals = build_by_coa(enriched, cycle_df)
        by_cc_files = build_by_cc(cc_list, enriched, cycle_df, sender_totals)

    return {
        "enriched": enriched,
        "by_coa_df": by_coa_df,
        "sender_totals": sender_totals,
        "by_cc_files": by_cc_files,
    }
