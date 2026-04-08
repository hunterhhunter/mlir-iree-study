"""
# Regression: ISSUE-001 — evaluate() 빈 배치에서 nan + RuntimeWarning 반환
# Found by /qa on 2026-03-29
# Report: .gstack/qa-reports/qa-report-llama-evaluator-2026-03-29.md

# Regression: ISSUE-002 — logits/labels 불일치 시 generic IndexError
# Found by /qa on 2026-03-29
# Report: .gstack/qa-reports/qa-report-llama-evaluator-2026-03-29.md

# Regression: ISSUE-003 — MLPerfResNet50Preprocess crop 크기 < 이미지 크기 시 무음 실패
# Found by /ship on 2026-04-06

# Regression: ISSUE-004 — LlamaPreprocessor 설정 변경 시 stale 캐시 재사용
# Found by /ship on 2026-04-06
"""

import os
import sys
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.inference_result import InferenceResult
from evaluators.llama_evaluator import LlamaEvaluator


def _make_evaluator():
    mock_tok = MagicMock()
    mock_tok.decode.return_value = "Paris"
    with patch("evaluators.llama_evaluator.AutoTokenizer") as mock_cls:
        mock_cls.from_pretrained.return_value = mock_tok
        evaluator = LlamaEvaluator(tokenizer_path="mock")
    return evaluator


def test_empty_batch_returns_zero_not_nan():
    """
    ISSUE-001: num_samples == 0일 때 nan + RuntimeWarning이 아닌 0.0을 반환해야 함.
    이전 동작: np.mean([]) → nan, RuntimeWarning 4개.
    수정 후: {"Exact Match": 0.0, "F1 Score": 0.0, "num_samples": 0}
    """
    evaluator = _make_evaluator()
    result = InferenceResult(
        outputs={"logits": np.zeros((0, 16, 100), dtype=np.float32)},
        labels=[],
        timing_records=[],
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metrics = evaluator.evaluate(result)

    # nan이 아닌 0.0
    assert metrics["Exact Match"] == 0.0, f"Expected 0.0, got {metrics['Exact Match']}"
    assert metrics["F1 Score"] == 0.0, f"Expected 0.0, got {metrics['F1 Score']}"
    assert metrics["num_samples"] == 0

    # RuntimeWarning 없음
    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(runtime_warnings) == 0, f"RuntimeWarning 발생: {[str(w.message) for w in runtime_warnings]}"


def test_labels_batch_mismatch_raises_value_error():
    """
    ISSUE-002: logits 배치 크기와 labels 길이가 다를 때 generic IndexError가 아닌
    구체적인 메시지를 포함한 ValueError를 발생시켜야 함.
    이전 동작: IndexError: list index out of range
    수정 후: ValueError: logits 배치 크기(2)와 labels 길이(1)가 일치하지 않습니다.
    """
    evaluator = _make_evaluator()
    result = InferenceResult(
        outputs={"logits": np.zeros((2, 16, 100), dtype=np.float32)},
        labels=[{"answers": [{"text": "Paris"}], "is_impossible": False}],  # 1개, logits는 2개
        timing_records=[10.0],
    )

    with pytest.raises(ValueError, match="logits 배치 크기"):
        evaluator.evaluate(result)


def test_resnet50_preprocess_crop_size_validation():
    """
    ISSUE-003: short_side resize 후 이미지가 crop 크기보다 작을 때 ValueError를 발생시켜야 함.
    이전 동작: 음수 crop 좌표로 PIL이 잘못된 결과를 반환 (무음 실패).
    수정 후: 명시적 ValueError 발생.
    """
    from preprocessor.strategies import MLPerfResNet50Preprocess
    from PIL import Image
    import numpy as np

    strategy = MLPerfResNet50Preprocess()
    # 정사각형 10x10 이미지 → short_side=256으로 resize하면 256x256이 되므로
    # target_hw를 (300, 300)으로 설정하면 crop 불가
    tiny_img = Image.new("RGB", (10, 10), color=(128, 128, 128))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    with pytest.raises(ValueError, match="smaller than crop size"):
        strategy(tiny_img, target_hw=(300, 300), mean=mean, std=std)


def test_llama_preprocessor_cache_path_differs_on_config_change():
    """
    ISSUE-004: tokenizer_path 또는 max_length가 달라지면 다른 캐시 경로를 반환해야 함.
    이전 동작: qa_id만으로 경로를 생성 → 설정 변경 후에도 stale 캐시가 재사용됨.
    수정 후: cfg_hash (md5[:8])가 경로에 포함되어 설정별로 격리됨.
    """
    from preprocessor.llama_preprocessor import LlamaPreprocessor
    from unittest.mock import MagicMock, patch

    def _make_preprocessor(tok_path, max_length):
        mock_tok = MagicMock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "<eos>"
        with patch("transformers.AutoTokenizer") as mock_cls:
            mock_cls.from_pretrained.return_value = mock_tok
            return LlamaPreprocessor(tokenizer_path=tok_path, max_length=max_length)

    qa_id = "squad_q001"
    cache_dir = "/tmp/cache"

    pp1 = _make_preprocessor("models/llama-3.1-8b", 4096)
    pp2 = _make_preprocessor("models/llama-3.2-3b", 4096)
    pp3 = _make_preprocessor("models/llama-3.1-8b", 2048)

    path1 = pp1.get_cache_path(cache_dir, qa_id)
    path2 = pp2.get_cache_path(cache_dir, qa_id)
    path3 = pp3.get_cache_path(cache_dir, qa_id)

    assert path1 != path2, "tokenizer_path가 다르면 캐시 경로도 달라야 함"
    assert path1 != path3, "max_length가 다르면 캐시 경로도 달라야 함"
    assert path2 != path3, "tokenizer + max_length 조합이 다르면 캐시 경로도 달라야 함"
    # 동일 설정은 동일 경로
    pp1_dup = _make_preprocessor("models/llama-3.1-8b", 4096)
    assert pp1.get_cache_path(cache_dir, qa_id) == pp1_dup.get_cache_path(cache_dir, qa_id)
