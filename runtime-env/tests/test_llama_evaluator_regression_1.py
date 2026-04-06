"""
# Regression: ISSUE-001 — evaluate() 빈 배치에서 nan + RuntimeWarning 반환
# Found by /qa on 2026-03-29
# Report: .gstack/qa-reports/qa-report-llama-evaluator-2026-03-29.md

# Regression: ISSUE-002 — logits/labels 불일치 시 generic IndexError
# Found by /qa on 2026-03-29
# Report: .gstack/qa-reports/qa-report-llama-evaluator-2026-03-29.md
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
    with patch("src.evaluators.llama_evaluator.AutoTokenizer") as mock_cls:
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
