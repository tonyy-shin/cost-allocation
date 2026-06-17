"""Large-scale scenario generator + pipeline benchmark + accuracy verification.

Generates sample_data/large_test_*.csv per the requested scale, runs the full
allocation pipeline with per-stage timing, and verifies correctness
(conservation, grid shape, sender residuals). Standalone; mirrors run_demo.py.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.loader import (
    apply_category_dtypes, build_category_dtypes,
    load_coa_amount, load_cycle, load_mapping,
)
from src.prepare import (
    aggregate_detail, aggregate_for_allocation,
    assign_transfer_coa, calculate_coa_ratio, separate_common_direct,
    validate_sender_coverage,
)
from src.allocation import (
    aggregate_received_by_cycle, build_pivot_matrix,
    decompose_to_original_coa, run_allocation_loop,
)
from src.output import TOTAL_COL, build_result, save_result

import random

SD = Path("sample_data")


# --------------------------------------------------------------------------
# 1. Data generation (exactly as specified)
# --------------------------------------------------------------------------
def generate() -> dict:
    random.seed(42)
    np.random.seed(42)

    cc_list = [f"{i:03d}" for i in range(1, 201)]

    coa_list = [str(1001 + i) for i in range(500)]
    common_coas = coa_list[:400]
    direct_coas = coa_list[400:]
    mapping = pd.DataFrame({
        "전기COA": [f"E{c}" for c in common_coas],
        "기존COA": common_coas,
    })
    mapping.to_csv(SD / "large_test_mapping.csv", index=False)

    # The COA·CC master is now the sole CC source and must enumerate every valid
    # (COA, CC) pair, so the common-cost section is generated as a full grid
    # (every common COA × every CC). This guarantees that every (base COA,
    # receiver CC) pair an allocation can produce already exists in the master,
    # so build_result never drops a received amount.
    rows = []
    for coa in common_coas:
        for cc in cc_list:
            rows.append({"COA": coa, "Cost Center": cc,
                         "Amounts": round(random.uniform(10000, 5000000), 0)})
    for coa in direct_coas:
        for cc in random.sample(cc_list, random.randint(1, 4)):
            rows.append({"COA": coa, "Cost Center": cc,
                         "Amounts": round(random.uniform(10000, 1000000), 0)})
    coa_amount_df = pd.DataFrame(rows)
    coa_amount_df.to_csv(SD / "large_test_coa_amount.csv", index=False)

    cycle_rows = []
    used_senders = set()
    for cycle_num in range(1, 6):
        senders = random.sample(cc_list, 10)
        used_senders.update(senders)
        for sender in senders:
            receivers = random.sample([c for c in cc_list if c != sender], 5)
            pcts = np.random.dirichlet(np.ones(5))
            for rec, pct in zip(receivers, pcts):
                cycle_rows.append({
                    "차수": cycle_num,
                    "Sender CC": sender,
                    "Receiver CC": rec,
                    "%": round(pct, 6),
                })
    cycle_df = pd.DataFrame(cycle_rows)
    cycle_df.to_csv(SD / "large_test_cycle.csv", index=False)

    return {
        "n_cc": len(cc_list),
        "n_coa": len(coa_list),
        "n_common": len(common_coas),
        "n_direct": len(direct_coas),
        "n_coa_amount_rows": len(coa_amount_df),
        "n_cycle_rows": len(cycle_df),
        "n_distinct_senders": len(used_senders),
    }


# --------------------------------------------------------------------------
# 2. Pipeline run with timing
# --------------------------------------------------------------------------
PATHS = {
    "coa_amount": SD / "large_test_coa_amount.csv",
    "mapping":    SD / "large_test_mapping.csv",
    "cycle":      SD / "large_test_cycle.csv",
    "output_dir": SD / "large_test_output",
}


def main() -> None:
    gen_t0 = time.perf_counter()
    summary = generate()
    gen_t = time.perf_counter() - gen_t0

    timings: dict[str, float] = {}
    warns: list[str] = []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        # --- Stage A: loading ---
        t0 = time.perf_counter()
        coa_df = load_coa_amount(PATHS["coa_amount"])
        mapping_df = load_mapping(PATHS["mapping"])
        cycle_df = load_cycle(PATHS["cycle"])
        dtypes = build_category_dtypes(coa_df, mapping_df)
        coa_df, mapping_df = apply_category_dtypes(
            coa_df, mapping_df, dtypes=dtypes
        )
        raw_coa_df = coa_df
        timings["1. 로딩/전처리"] = time.perf_counter() - t0

        # --- Stage B: prepare (separate, aggregate, ratio) ---
        t0 = time.perf_counter()
        enriched = assign_transfer_coa(coa_df, mapping_df)
        df_common, df_direct = separate_common_direct(enriched)
        df_5a = aggregate_detail(df_common)
        df_5b = aggregate_for_allocation(df_5a)
        df_ratio = calculate_coa_ratio(df_5a)
        coverage = validate_sender_coverage(df_5b, cycle_df)
        timings["2. 비용 준비/비율"] = time.perf_counter() - t0

        # --- Stage C: allocation calculation ---
        t0 = time.perf_counter()
        cc_list = coa_df["Cost Center"].unique().tolist()
        pivot = build_pivot_matrix(df_5b, cc_list)
        final_pivot, delta_by_cycle = run_allocation_loop(pivot, cycle_df)
        received_by_cycle = aggregate_received_by_cycle(delta_by_cycle)
        common_decomposed = decompose_to_original_coa(received_by_cycle, df_ratio)
        timings["3. 배부 계산"] = time.perf_counter() - t0

        # --- Stage D: result assembly ---
        t0 = time.perf_counter()
        n_cycles = cycle_df["차수"].nunique()
        result = build_result(common_decomposed, df_direct, raw_coa_df, n_cycles)
        timings["4. 결과 조립"] = time.perf_counter() - t0

        # --- Stage E: save ---
        t0 = time.perf_counter()
        out_path = save_result(result, PATHS["output_dir"])
        timings["5. 저장"] = time.perf_counter() - t0

    warns = [str(w.message) for w in caught]

    # ----------------------------------------------------------------------
    # 3. Accuracy verification
    # ----------------------------------------------------------------------
    # Pre-allocation common-cost total (all common cost in the system)
    common_total_in = df_common["Amounts"].sum()
    direct_total_in = df_direct["Amounts"].sum()

    # Common cost that is held by CCs that DO appear as senders (distributable)
    senders = set(cycle_df["Sender CC"])
    df_5b_cc = df_5b.groupby("Cost Center", observed=True)["Amounts"].sum()
    common_in_senders = df_5b_cc[df_5b_cc.index.isin(senders)].sum()
    common_in_nonsenders = df_5b_cc[~df_5b_cc.index.isin(senders)].sum()

    # Allocated common total = received total across cycles
    received_total = sum(
        d["Amounts"].sum() for d in received_by_cycle.values() if not d.empty
    )

    # Physical conservation: the pivot only moves money between CCs, so the
    # grand total must be invariant from start to finish.
    # run_allocation_loop copies its input, so `pivot` is the untouched initial.
    pivot_initial_total = float(pivot.sum().sum())
    final_pivot_total = float(final_pivot.sum().sum())

    # result.csv totals
    total_sum = result[TOTAL_COL].sum()
    # split common vs direct rows in result by 전기COA != ""
    res_common_sum = result.loc[result["전기COA"] != "", TOTAL_COL].sum()
    res_direct_sum = result.loc[result["전기COA"] == "", TOTAL_COL].sum()

    # Sender residual after loop: final pivot column for each sender CC
    sender_residuals = {
        cc: float(final_pivot[cc].sum())
        for cc in senders if cc in final_pivot.columns
    }
    max_sender_resid = max((abs(v) for v in sender_residuals.values()), default=0.0)

    n_rows = len(result)
    out_size = out_path.stat().st_size

    # ----------------------------------------------------------------------
    # 4. Report
    # ----------------------------------------------------------------------
    print("=" * 70)
    print("■ 1. 생성 데이터 규모")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k:25s}: {v:,}")
    print(f"  {'데이터 생성 소요시간':25s}: {gen_t:.3f} s")

    print()
    print("=" * 70)
    print("■ 2. 단계별 소요 시간")
    print("=" * 70)
    total_pipe = sum(timings.values())
    for k, v in timings.items():
        print(f"  {k:20s}: {v*1000:9.1f} ms  ({v/total_pipe*100:5.1f}%)")
    print(f"  {'─'*20}   {'─'*9}")
    print(f"  {'합계':20s}: {total_pipe*1000:9.1f} ms")

    print()
    print("=" * 70)
    print("■ 3. 결과 규모")
    print("=" * 70)
    expected_pairs = raw_coa_df.drop_duplicates(["COA", "Cost Center"]).shape[0]
    print(f"  result.csv 행 수      : {n_rows:,}  (기대 마스터 실재 쌍 {expected_pairs:,})")
    print(f"  result.csv 컬럼       : {list(result.columns)}")
    print(f"  파일 크기             : {out_size:,} bytes ({out_size/1024/1024:.2f} MB)")

    print()
    print("=" * 70)
    print("■ 4. 정확성 검증")
    print("=" * 70)
    tol = 1e-4
    print(f"  배부 전 공통비 총액            : {common_total_in:20,.4f}")
    print(f"    └ Sender 보유분(배부가능)    : {common_in_senders:20,.4f}")
    print(f"    └ 비-Sender 보유분(미배부)   : {common_in_nonsenders:20,.4f}")
    print(f"  배부된 공통비(수령액 합)        : {received_total:20,.4f}")
    print(f"  result 공통비 배부합계          : {res_common_sum:20,.4f}")
    print(f"  result 직접비 배부합계          : {res_direct_sum:20,.4f}")
    print(f"  직접비 총액(입력)               : {direct_total_in:20,.4f}")
    print(f"  result 전체 배부합계            : {total_sum:20,.4f}")
    print()

    # Check 1a: full conservation (all common in == allocated)?
    diff_full = abs(common_total_in - received_total)
    ok_full = diff_full <= tol
    print(f"  [검증1-전체] 공통비총액 == 배부공통비 : diff={diff_full:,.4f} "
          f"-> {'PASS' if ok_full else 'FAIL'}")
    # Check 1b: distributable conservation (sender-held common == allocated)?
    diff_dist = abs(common_in_senders - received_total)
    ok_dist = diff_dist <= tol
    print(f"  [검증1-배부가능] Sender보유공통비 == 배부공통비 : diff={diff_dist:,.4f} "
          f"-> {'PASS' if ok_dist else 'FAIL'}")
    # Check 1c: result common total == received total (decompose conservation)
    diff_dec = abs(res_common_sum - received_total)
    print(f"  [검증1-분해] result공통비 == 배부공통비 : diff={diff_dec:,.4f} "
          f"-> {'PASS' if diff_dec <= tol else 'FAIL'}")
    # Check 1e: physical conservation in the pivot (money only moves, never lost)
    diff_phys = abs(pivot_initial_total - final_pivot_total)
    print(f"  [검증1-물리] pivot초기총액={pivot_initial_total:,.2f} == "
          f"최종총액={final_pivot_total:,.2f} : diff={diff_phys:,.6f} "
          f"-> {'PASS' if diff_phys <= 1e-3 else 'FAIL'}")
    # Final common balance held by non-sender CCs at end (the lost-to-result part)
    final_nonsender = float(
        final_pivot.loc[:, [c for c in final_pivot.columns if c not in senders]].sum().sum()
    )
    print(f"  └ 최종 비-Sender 보유 공통비(result서 누락): {final_nonsender:,.2f}")
    print(f"  └ result공통비 + 최종비Sender보유 = "
          f"{res_common_sum + final_nonsender:,.2f} (vs 초기 {pivot_initial_total:,.2f})")
    # Check 1d: direct preserved
    diff_dir = abs(res_direct_sum - direct_total_in)
    print(f"  [검증1-직접비] result직접비 == 직접비총액 : diff={diff_dir:,.4f} "
          f"-> {'PASS' if diff_dir <= tol else 'FAIL'}")

    # Check 2: grid shape — rows equal the master's actual (COA, CC) pairs.
    ok_grid = (n_rows == expected_pairs)
    n_unique = result.drop_duplicates(["기존COA", "코스트센터"]).shape[0]
    print(f"  [검증2-그리드] 행수==마스터쌍({expected_pairs:,}) : {n_rows:,} -> "
          f"{'PASS' if ok_grid else 'FAIL'}  (unique 조합 {n_unique:,})")

    # Check 3: sender residual -> 0
    ok_sender = max_sender_resid <= 1e-4
    print(f"  [검증3-Sender잔액] max|잔액|={max_sender_resid:.6e} -> "
          f"{'PASS' if ok_sender else 'FAIL'}")

    print()
    print("=" * 70)
    print("■ 경고 발생")
    print("=" * 70)
    print(f"  총 경고 수: {len(warns)}")
    # categorize
    cats: dict[str, int] = {}
    for w in warns:
        if "자동 정규화" in w:
            cats["정규화 경고"] = cats.get("정규화 경고", 0) + 1
        elif "배부되지 않았습니다" in w:
            cats["미배부 잔액 경고"] = cats.get("미배부 잔액 경고", 0) + 1
        elif "잔액이 0원이 되지 않" in w:
            cats["Sender 잔액 불일치"] = cats.get("Sender 잔액 불일치", 0) + 1
        elif "보존 검증 실패" in w:
            cats["보존 검증 실패"] = cats.get("보존 검증 실패", 0) + 1
        else:
            cats["기타"] = cats.get("기타", 0) + 1
    for k, v in cats.items():
        print(f"    - {k}: {v}건")
    print(f"  validate_sender_coverage 위반 CC 수: {len(coverage)}")
    if coverage:
        head = coverage[:5]
        print(f"    예시: {head}")
    # show up to 3 sample warning texts per category
    print("  경고 샘플(최대 4건):")
    for w in warns[:4]:
        print(f"    · {w[:120]}")

    print()
    print("=" * 70)
    print("■ 5. result.csv 상위 20행")
    print("=" * 70)
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", lambda x: f"{x:,.2f}"):
        print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
