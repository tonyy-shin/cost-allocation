"""Tests for src.output: sender-keyed long-format build_result and CSV writing."""
from __future__ import annotations

import pandas as pd
import pytest

from src.output import build_result, save_result


RESULT_COLS = ["차수", "전기COA", "기존COA", "Sender CC", "배분금액"]


# SUCCESS cases


def test_build_result_column_layout(pipeline_outputs):
    # The result is the sender-side long format: one row per
    # (차수, 전기COA, 기존COA, Sender CC) with its 배분금액.
    result = pipeline_outputs["result"]
    assert list(result.columns) == RESULT_COLS


def test_build_result_sorted(pipeline_outputs):
    # Rows are sorted by 차수 → 전기COA → 기존COA → Sender CC.
    result = pipeline_outputs["result"]
    sort_keys = ["차수", "전기COA", "기존COA", "Sender CC"]
    expected = result.sort_values(sort_keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(result, expected)


def test_build_result_passes_through_sender_decomposed(pipeline_outputs):
    # build_result is a thin column-ordering wrapper over the sender decomposition.
    result = pipeline_outputs["result"]
    sender_decomposed = pipeline_outputs["sender_decomposed"]
    assert len(result) == len(sender_decomposed)
    assert result["배분금액"].sum() == pytest.approx(
        sender_decomposed["배분금액"].sum()
    )


def test_save_result_writes_utf8_sig_csv(pipeline_outputs, tmp_path):
    out_path = save_result(pipeline_outputs["result"], tmp_path)

    assert out_path.exists()
    assert out_path.name == "result.csv"
    # utf-8-sig writes a BOM prefix for Excel compatibility.
    assert out_path.read_bytes().startswith(b"\xef\xbb\xbf")
    # Round-trips back to the same shape with utf-8-sig decoding.
    reloaded = pd.read_csv(out_path, encoding="utf-8-sig")
    assert len(reloaded) == len(pipeline_outputs["result"])
    assert list(reloaded.columns) == list(pipeline_outputs["result"].columns)
