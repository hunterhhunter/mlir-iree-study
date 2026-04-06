import os
import sys

# 테스트 실행 환경 경로 인식(Path) 결함 핫픽스 (sys.path 주입)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from core.inference_result import InferenceResult

# TDD RED 단계이므로, 해당 로드를 실패하거나, 인스턴스화 시 TypeError가 예상됨 
from evaluators.bert_classification_evaluator import BertClassificationEvaluator


def test_bert_evaluator_valid_accuracy():
    """
    [TDD] 정상적인 모델 추론 (Logits) 결과를 바탕으로
    Scikit-Learn/Torch 없이 순수 Numpy만의 다차원 Argmax 연산이 
    정확하게 Accuracy(%)를 도출하는지 깐깐하게 검증.
    """
    evaluator = BertClassificationEvaluator()

    # GLUE SST-2 형식: 2개의 Logit [부정(0), 긍정(1)]
    # N = 4 (샘플 4개)
    mock_logits = np.array([
        [1.5, -0.5],  # Argmax = 0 (부정) -> 정답 (0)
        [-1.2, 2.5],  # Argmax = 1 (긍정) -> 정답 (1)
        [-0.5, 0.5],  # Argmax = 1 (긍정) -> 오답 (0이어야 함)
        [3.0, -1.0]   # Argmax = 0 (부정) -> 정답 (0)
    ])
    
    # 3개 맞고 1개 틀림 -> 75.0% 예측치
    mock_labels = np.array([0, 1, 0, 0])

    # Runner 측이 전달해 줄 DTO 규격 그대로 포장
    mock_result = InferenceResult(
        outputs={"logits": mock_logits},
        timing_records=[10.0, 12.0, 11.0, 9.0],  # 평가기는 시간을 신경 쓰지 않음
        labels=mock_labels
    )
    
    # 채점 (Evaluate)
    metrics = evaluator.evaluate(mock_result)

    # Dictionary 형태 언패킹 검증
    assert "accuracy" in metrics, "퍼센트 메트릭 반환 누락"
    assert "total_samples" in metrics, "샘플 총 개수 반환 누락"
    
    assert metrics["accuracy"] == 75.0
    assert metrics["total_samples"] == 4


def test_bert_evaluator_unexpected_shape_handling():
    """
    [TDD] 예기치 못하게 텐서 차원이 1차원(스칼라)이거나 비어있을 때
    Evaluator가 ZeroDivisionError나 Shape Mismatch를 어떻게 방어하는지 검증.
    """
    evaluator = BertClassificationEvaluator()
    
    # 빈 배열 엣지 케이스 투척!
    empty_result = InferenceResult(
        outputs={"logits": np.array([])},
        timing_records=[],
        labels=np.array([])
    )
    
    # 방어 로직 검증: 크래시 없이 0점 처리 혹은 빈 사전 리턴
    metrics = evaluator.evaluate(empty_result)
    assert metrics["total_samples"] == 0
    assert metrics.get("accuracy", 0.0) == 0.0
