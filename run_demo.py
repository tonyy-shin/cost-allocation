from pathlib import Path
import warnings

from src.loader import (
    apply_category_dtypes, build_category_dtypes,
    load_coa_amount, load_cycle, load_mapping,
)
from src.prepare import (
    aggregate_detail, aggregate_for_allocation,
    assign_transfer_coa, calculate_coa_ratio, separate_common_direct,
)
from src.allocation import (
    aggregate_received_by_cycle, build_pivot_matrix,
    decompose_to_original_coa, run_allocation_loop,
)
from src.output import build_result, save_result

TEST_PATHS = {
    "coa_amount": Path("sample_data/coa_amount.csv"),
    "mapping":    Path("sample_data/mapping.csv"),
    "cycle":      Path("sample_data/cycle.csv"),
    "output_dir": Path("sample_data/output"),
}

warnings.simplefilter("always")

# Steps 1-2
coa_df     = load_coa_amount(TEST_PATHS["coa_amount"])
mapping_df = load_mapping(TEST_PATHS["mapping"])
cycle_df   = load_cycle(TEST_PATHS["cycle"])

dtypes = build_category_dtypes(coa_df, mapping_df)
coa_df, mapping_df = apply_category_dtypes(coa_df, mapping_df, dtypes=dtypes)
raw_coa_df = coa_df

# Steps 3-6
enriched = assign_transfer_coa(coa_df, mapping_df)
df_common, df_direct = separate_common_direct(enriched)
df_5a  = aggregate_detail(df_common)
df_5b  = aggregate_for_allocation(df_5a)
df_ratio = calculate_coa_ratio(df_5a)

# Steps 7-9
cc_list = coa_df["Cost Center"].unique().tolist()
pivot = build_pivot_matrix(df_5b, cc_list)
_, delta_by_cycle = run_allocation_loop(pivot, cycle_df)
received_by_cycle = aggregate_received_by_cycle(delta_by_cycle)
common_decomposed = decompose_to_original_coa(received_by_cycle, df_ratio)

# Steps 10-12
n_cycles = cycle_df["차수"].nunique()
result = build_result(common_decomposed, df_direct, raw_coa_df, n_cycles)
out_path = save_result(result, TEST_PATHS["output_dir"])

print(result.to_string())
print(f"\nSaved: {out_path}")
