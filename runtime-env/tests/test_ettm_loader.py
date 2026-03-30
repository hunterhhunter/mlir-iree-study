"""
ETTmLoader 단위 테스트

모든 단위 테스트는 mock CSV만 사용하며 외부 의존성이 없습니다.
통합 테스트 (실제 ETTm1.csv 필요) 는 -m integration 마커로 분리합니다.
"""

import math
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.core.model_spec import Model_Spec, Task
from src.dataloader import ETTmLoader, create_dataloader


# ── 공통 픽스처 ────────────────────────────────────────────────────────────────

COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
N_ROWS = 1000
CONTEXT = 128
PRED    = 32
STRIDE  = 32
VAL_RATIO = 0.2


@pytest.fixture
def csv_file(tmp_path):
    """행 수 N_ROWS의 합성 ETTm1 CSV를 tmp_path에 생성합니다."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        rng.standard_normal((N_ROWS, len(COLS))).astype(np.float32),
        columns=COLS,
    )
    df.insert(0, "date", pd.date_range("2016-07-01", periods=N_ROWS, freq="15min"))
    path = tmp_path / "ETTm1.csv"
    df.to_csv(path, index=False)
    return str(path)


def make_loader(csv_file, **overrides) -> ETTmLoader:
    spec = Model_Spec(
        name="patchtst-fm-r1",
        task=Task.TIME_SERIES_FORECASTING,
        input_shapes={
            "past_values":        (1, CONTEXT, 1),
            "past_observed_mask": (1, CONTEXT, 1),
        },
        input_dtype={
            "past_values":        "float32",
            "past_observed_mask": "bool",
        },
        output_shapes={"prediction_outputs": (1, PRED, 1)},
    )
    defaults = dict(
        csv_path=csv_file,
        context_length=CONTEXT,
        prediction_length=PRED,
        stride=STRIDE,
        val_ratio=VAL_RATIO,
        target_cols=["OT"],
    )
    defaults.update(overrides)
    return ETTmLoader(spec, **defaults)


# ── 테스트 케이스 ───────────────────────────────────────────────────────────────

def test_split_boundary(csv_file):
    """val 시작 인덱스가 len(df) * (1 - val_ratio - test_ratio) 와 일치해야 합니다."""
    loader = make_loader(csv_file)
    expected_start = int(N_ROWS * (1.0 - VAL_RATIO - loader.test_ratio))
    assert loader._split_start == expected_start


def test_window_count(csv_file):
    """총 샘플 수 == floor((val_len - context - pred) / stride) + 1."""
    loader = make_loader(csv_file)
    val_len = loader._split_end - loader._split_start
    expected = math.floor((val_len - CONTEXT - PRED) / STRIDE) + 1
    assert loader._window_count == expected


def test_output_shapes(csv_file):
    """past_values shape=(CONTEXT,1), future_values shape=(PRED,1) 검증."""
    loader = make_loader(csv_file)
    sample = loader.load_by_index(0)

    assert sample["input"]["past_values"].shape        == (CONTEXT, 1)
    assert sample["input"]["past_observed_mask"].shape == (CONTEXT, 1)
    assert sample["label"]["future_values"].shape      == (PRED, 1)


def test_output_shapes_multivariate(csv_file):
    """멀티바리에이트(C=7) 시 shape 검증."""
    loader = make_loader(csv_file, target_cols=None)
    sample = loader.load_by_index(0)
    C = 7

    assert sample["input"]["past_values"].shape        == (CONTEXT, C)
    assert sample["input"]["past_observed_mask"].shape == (CONTEXT, C)
    assert sample["label"]["future_values"].shape      == (PRED, C)


def test_revin(csv_file):
    """정규화 후 past_values.mean ≈ 0, 역정규화 후 원본 future_values 복원 검증."""
    loader = make_loader(csv_file)
    sample = loader.load_by_index(0)

    pv = sample["input"]["past_values"]   # (T, C)
    assert abs(pv.mean()) < 0.1, "RevIN 정규화 후 mean이 0에 가까워야 합니다."

    # 역정규화 검증: future_values는 norm_stats가 없는 원본 스케일
    # past 윈도우의 역정규화 확인 (mean/std 로 복원)
    mean = sample["label"]["norm_stats"]["mean"]
    std  = sample["label"]["norm_stats"]["std"]
    restored = pv * std + mean             # (T, C)

    # 복원된 값이 원본 CSV 데이터와 같아야 함
    abs_start = loader._split_start + 0 * STRIDE
    original  = loader._data[abs_start : abs_start + CONTEXT]  # (T, C)
    np.testing.assert_allclose(restored, original, rtol=1e-5, atol=1e-5)


def test_load_by_index_no_side_effect(csv_file):
    """load_by_index 호출 전후 current_idx가 불변이어야 합니다."""
    loader = make_loader(csv_file)
    before = loader._current_idx
    loader.load_by_index(0)
    after = loader._current_idx
    assert before == after


def test_cache_roundtrip(csv_file, tmp_path):
    """NPZ 캐시 저장 후 재로드 시 배열이 동일해야 합니다."""
    cache_dir = str(tmp_path / "cache")
    loader = make_loader(csv_file, cache_dir=cache_dir)

    original = loader.load_by_index(0)

    # 캐시에서 재로드
    loader2 = make_loader(csv_file, cache_dir=cache_dir)
    cached = loader2.load_by_index(0)

    np.testing.assert_array_equal(
        original["input"]["past_values"],
        cached["input"]["past_values"],
    )
    np.testing.assert_array_equal(
        original["label"]["future_values"],
        cached["label"]["future_values"],
    )
    np.testing.assert_array_equal(
        original["label"]["norm_stats"]["mean"],
        cached["label"]["norm_stats"]["mean"],
    )


def test_factory_routing(csv_file):
    """create_dataloader(patchtst_spec) → ETTmLoader 인스턴스 반환 검증."""
    spec = Model_Spec(
        name="patchtst-fm-r1",
        task=Task.TIME_SERIES_FORECASTING,
        input_shapes={"past_values": (1, CONTEXT, 1), "past_observed_mask": (1, CONTEXT, 1)},
        input_dtype={"past_values": "float32", "past_observed_mask": "bool"},
        output_shapes={"prediction_outputs": (1, PRED, 1)},
    )
    loader = create_dataloader(
        spec,
        csv_path=csv_file,
        context_length=CONTEXT,
        prediction_length=PRED,
        stride=STRIDE,
        val_ratio=VAL_RATIO,
        target_cols=["OT"],
    )
    assert isinstance(loader, ETTmLoader)


# ── 통합 테스트 (실제 ETTm1.csv 필요) ─────────────────────────────────────────

@pytest.mark.integration
def test_integration_real_ettm():
    """실제 ETTm1.csv (69680 rows) 로 윈도우 수를 검증합니다."""
    csv_path = "datasets/ettm/ETTm1.csv"
    if not os.path.exists(csv_path):
        pytest.skip("ETTm1.csv 없음 — wget -P datasets/ettm/ https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv")

    spec = Model_Spec(
        name="patchtst-fm-r1",
        task=Task.TIME_SERIES_FORECASTING,
        input_shapes={"past_values": (1, 512, 1), "past_observed_mask": (1, 512, 1)},
        input_dtype={"past_values": "float32", "past_observed_mask": "bool"},
        output_shapes={"prediction_outputs": (1, 96, 1)},
    )
    loader = create_dataloader(
        spec,
        csv_path=csv_path,
        context_length=512,
        prediction_length=96,
        stride=96,
        val_ratio=0.2,
        target_cols=["OT"],
    )
    meta = loader.get_metadata()
    # val 13936 rows, context=512, pred=96, stride=96 → 약 138
    assert meta["window_count"] >= 130
    sample = loader.load_by_index(0)
    assert sample["input"]["past_values"].shape == (512, 1)
