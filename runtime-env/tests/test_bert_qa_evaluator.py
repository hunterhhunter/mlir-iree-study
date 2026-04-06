import pytest
import numpy as np
from evaluators.bert_qa_evaluator import BertQAEvaluator
from core.inference_result import InferenceResult

class TestBertQAEvaluator:
    @pytest.fixture
    def evaluator(self):
        """평가기 인스턴스 생성 픽스처"""
        return BertQAEvaluator()

    def test_perfect_match(self, evaluator):
        """[만점 케이스] 모델이 정확히 시작과 끝을 예측했을 때 EM 100.0, F1 100.0 반환 검증"""
        predictions = {
            "start_logits": np.array([[0, 10, 0, 0, 0]]),
            "end_logits": np.array([[0, 0, 0, 10, 0]])
        }
        # List[Dict] 형태로 Production BenchmarkRunner 엔진의 _collate_batch 반환값을 모사
        labels = [{
            "start_positions": np.array([1]),
            "end_positions": np.array([3])
        }]
        
        result_dto = InferenceResult(outputs=predictions, labels=labels, timing_records=[10.0])
        results = evaluator.evaluate(result_dto)
        
        assert results["exact_match"] == 100.0
        assert results["f1"] == 100.0

    def test_partial_overlap(self, evaluator):
        """[부분 일치 케이스] 모델 예측과 정답이 일부만 겹칠 때 정확한 정밀도/재현율(F1스코어) 계산 검증"""
        start_logits = np.zeros((1, 20))
        end_logits = np.zeros((1, 20))
        start_logits[0, 7] = 10
        end_logits[0, 12] = 10
        
        predictions = {
            "start_logits": start_logits,
            "end_logits": end_logits
        }
        labels = [{
            "start_positions": np.array([5]),
            "end_positions": np.array([10])
        }]
        
        result_dto = InferenceResult(outputs=predictions, labels=labels, timing_records=[10.0])
        results = evaluator.evaluate(result_dto)
        
        assert results["exact_match"] == 0.0
        assert np.isclose(results["f1"], 66.66666666666666)

    def test_edge_case_reverse_prediction(self, evaluator):
        """[역전 예측 방어 케이스] 시작 인덱스가 끝 인덱스보다 클 때, ZeroDivisionError 방지 및 0점 처리 검증"""
        start_logits = np.zeros((1, 20))
        end_logits = np.zeros((1, 20))
        start_logits[0, 10] = 10
        end_logits[0, 5] = 10
        
        predictions = {
            "start_logits": start_logits,
            "end_logits": end_logits
        }
        labels = [{
            "start_positions": np.array([5]),
            "end_positions": np.array([10])
        }]
        
        result_dto = InferenceResult(outputs=predictions, labels=labels, timing_records=[10.0])
        results = evaluator.evaluate(result_dto)
        
        assert results["exact_match"] == 0.0
        assert results["f1"] == 0.0

    def test_completely_wrong(self, evaluator):
        """[완전 오답 케이스] 정답 영역과 예측 토큰 스팬이 전혀 겹치지 않을 때 정밀도 및 재현율 0점 산출 논리 검증"""
        start_logits = np.zeros((1, 10))
        end_logits = np.zeros((1, 10))
        start_logits[0, 5] = 10
        end_logits[0, 7] = 10
        
        predictions = {
            "start_logits": start_logits,
            "end_logits": end_logits
        }
        labels = [{
            "start_positions": np.array([0]),
            "end_positions": np.array([2])
        }]
        
        result_dto = InferenceResult(outputs=predictions, labels=labels, timing_records=[10.0])
        results = evaluator.evaluate(result_dto)
        
        assert results["exact_match"] == 0.0
        assert results["f1"] == 0.0
