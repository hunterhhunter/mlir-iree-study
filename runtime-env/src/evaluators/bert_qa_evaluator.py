import numpy as np
from typing import Dict, Any, List, Tuple
from .base import Evaluator
from ..core.model_spec import Model_Spec
from ..core.inference_result import InferenceResult

class BertQAEvaluator(Evaluator):
    """
    SQuAD 방식을 따르는 추출형 질의응답(Extractive QA) 성능 평가 모듈.
    OOM 방지를 위해 파이썬 for-loop를 완전 배제하고, DRY 원칙이 적용된 순수 Numpy 벡터화 로직만을 구동합니다.
    """
    def __init__(self, **eval_options):
        pass

    def evaluate(self, result: InferenceResult) -> Dict[str, Any]:
        """추론 결과(InferenceResult) DTO를 받아 EM과 F1 스코어를 채점합니다."""
        
        # 1. 모델 확률 텐서 역전파(Argmax)
        pred_starts = np.argmax(result.outputs["start_logits"], axis=-1)
        pred_ends = np.argmax(result.outputs["end_logits"], axis=-1)
        
        # 2. 정답지(Labels) 평탄화 파싱 분리 위임
        true_starts, true_ends = self._parse_flattened_labels(result.labels)

        # 3. 채점 엔진 구동
        metrics = self._calculate_qa_metrics(pred_starts, pred_ends, true_starts, true_ends)
        metrics["total_samples"] = len(true_starts)
        
        # 4. Latency 지표 통합
        metrics.update(self._calculate_latency_metrics(result.timing_records))

        return metrics

    def _parse_flattened_labels(self, labels: List[Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """단일 책임 원칙(SRP): 오프라인 베이킹 규격인 List[Dict] 형태의 라벨 덩어리들을 Numpy Flat 배열로 분리 추출합니다."""
        if not (isinstance(labels, list) and len(labels) > 0 and isinstance(labels[0], dict)):
            raise ValueError("[BertQAEvaluator] BenchmarkRunner 규격(List[Dict])에 맞지 않는 정답 형식입니다.")

        true_starts = np.concatenate([batch["start_positions"] for batch in labels])
        true_ends = np.concatenate([batch["end_positions"] for batch in labels])
        return true_starts, true_ends

    def _compute_lengths(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """DRY 규칙: 시작/끝 인덱스 쌍의 유효 토큰 길이를 산출합니다. (역전/음수 확률은 수학적으로 0으로 수렴시킴)"""
        return np.maximum(0, ends - starts + 1)

    def _calculate_qa_metrics(self, pred_starts: np.ndarray, pred_ends: np.ndarray, 
                              true_starts: np.ndarray, true_ends: np.ndarray) -> Dict[str, float]:
        """순수 Numpy 수학 공식을 통해 Exact Match와 F1 스코어를 대규모 벡터 행렬 연산합니다."""
        if len(true_starts) == 0:
            return {"exact_match": 0.0, "f1": 0.0}

        # [1] Exact Match (EM): 시작과 끝 좌표가 한치의 오차도 없는 경우
        em_array = (pred_starts == true_starts) & (pred_ends == true_ends)
        exact_match = float(np.mean(em_array) * 100.0)

        # [2] F1 스코어: 예측 길이, 정답 길이, 교집합(Intersection) 길이를 헬퍼로 뽑아서 더러운 마스킹 로직 절멸
        pred_lengths = self._compute_lengths(pred_starts, pred_ends)
        true_lengths = self._compute_lengths(true_starts, true_ends)

        inter_starts = np.maximum(pred_starts, true_starts)
        inter_ends = np.minimum(pred_ends, true_ends)
        
        # 교집합 자체가 성립하지 않거나 예측이 역전된 경우, 위 _compute_lengths 공식에 의해 자연스럽게 수학적 0 배열 반환됨
        num_same = self._compute_lengths(inter_starts, inter_ends)
        
        # 정밀도, 재현율 (분모 Zero-division 방어용 1e-9 더미 삽입)
        precision = num_same / np.maximum(pred_lengths, 1e-9)
        recall = num_same / np.maximum(true_lengths, 1e-9)
        
        # F1 계산: 2PR / (P + R)
        pr_sum = precision + recall
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_array = np.where(pr_sum > 0, 2 * (precision * recall) / pr_sum, 0.0)

        f1_score = float(np.mean(f1_array) * 100.0)

        return {"exact_match": exact_match, "f1": f1_score}

    def _calculate_latency_metrics(self, timing_records: List[float]) -> Dict[str, float]:
        """지연 시간 및 처리율(Throughtput)을 분할 계산하는 헬퍼 모듈"""
        if not timing_records:
            return {}
            
        avg_lat = float(np.mean(timing_records))
        p99_lat = float(np.percentile(timing_records, 99))
        samples_per_sec = (1000.0 / avg_lat) if avg_lat > 0 else 0.0
        
        return {
            "Average Latency (ms)": avg_lat,
            "P99 Latency (ms)": p99_lat,
            "Samples/s": samples_per_sec
        }

    def is_applicable(self, device_spec: Dict[str, Any], model_spec: Model_Spec) -> bool:
        task_name = str(getattr(model_spec, "task", ""))
        return "QUESTION_ANSWERING" in task_name

    def get_metric_names(self) -> List[str]:
        return ["exact_match", "f1", "total_samples", "Average Latency (ms)", "P99 Latency (ms)", "Samples/s"]
