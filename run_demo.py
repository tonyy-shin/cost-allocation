from pathlib import Path
import warnings

from src.data.loader import (
    apply_category_dtypes, build_category_dtypes,
    load_coa_amount, load_cycle, load_mapping, load_pre_allocation,
)
from src.core.prepare import build_enriched
from src.core.allocation import build_by_coa, build_by_cc
from src.data.output import save_results

TEST_PATHS = {
    "coa_amount":     Path("sample_data/coa_amount.csv"),
    "mapping":        Path("sample_data/mapping.csv"),
    "cycle":          Path("sample_data/cycle.csv"),
    "pre_allocation": Path("sample_data/pre_allocation.csv"),
    "output_dir":     Path("sample_data/output"),
}

warnings.simplefilter("always")

# Load
coa_df       = load_coa_amount(TEST_PATHS["coa_amount"])
mapping_df   = load_mapping(TEST_PATHS["mapping"])
cycle_df     = load_cycle(TEST_PATHS["cycle"])
pre_alloc_df = load_pre_allocation(TEST_PATHS["pre_allocation"])

# Enrich pre_allocation while mapping_df is still str-typed (cf. main._load_inputs).
pre_alloc_enriched = build_enriched(pre_alloc_df, mapping_df)

dtypes = build_category_dtypes(coa_df, mapping_df)
coa_df, mapping_df = apply_category_dtypes(coa_df, mapping_df, dtypes=dtypes)

# Build outputs
cc_list = coa_df["Cost Center"].astype(str).unique().tolist()
enriched = build_enriched(coa_df, mapping_df)
by_coa_df, sender_totals = build_by_coa(enriched, cycle_df)
by_cc_files = build_by_cc(cc_list, pre_alloc_enriched, cycle_df, sender_totals)

# Save
out_path = save_results(by_coa_df, by_cc_files, TEST_PATHS["output_dir"])

print("=== 배부금액/result.csv ===")
print(by_coa_df.to_string(index=False))
for n, df in by_cc_files.items():
    print(f"\n=== 잔액/{n}차배부후.csv ===")
    print(df.to_string(index=False))
print(f"\nSaved under: {out_path}")
