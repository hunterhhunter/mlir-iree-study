import numpy as np
from typing import Dict, Any, List

from .base import Evaluator
from src.core.inference_result import InferenceResult
from src.core.model_spec import Model_Spec, Task

class BertClassificationEvaluator(Evaluator):
    """
    BERT 기반 텍스트 분류(SST-2 등) 텐서 추론 결과(Logits)의 정답률(Accuracy)을 산출합니다.
    Scikit-Learn이나 PyTorch 없이 순수 Numpy만의 O(1) 행렬 연산으로 채점을 수행합니다.
    """
    
    def __init__(self, **eval_options):
        self.eval_options = eval_options
        
    def get_metric_names(self) -> List[str]:
        return [
            "accuracy", "total_samples",
            "Average Latency (ms)", "P99 Latency (ms)", "Samples/s"
        ]
        
    def is_applicable(self, device_spec: Dict[str, Any], model_spec: Model_Spec) -> bool:
        return getattr(model_spec, "task", None) == Task.NLP_CLASSIFICATION
        
    def evaluate(self, result: InferenceResult) -> Dict[str, Any]:
        """추론 결과(Logits)를 받아 내부 프라이빗 객체(Helper)들에게 각각 연산을 위임하여 채점함."""
        metrics = {}
        
        # 1. 텐서 추출 및 차원 안전 장치 적용
        logits = self._extract_logits(result.outputs.get("logits", np.array([])))
        # 2. 정답지(labels) 1D 병합 위임 처리 (SOLID)
        raw_labels = result.labels
        labels = []
        if isinstance(raw_labels, list):
            for batch in raw_labels:
                if isinstance(batch, (list, np.ndarray)):
                    labels.extend(batch)
                else:
                    labels.append(batch)
        else:
            labels = raw_labels
            
        labels = np.array(labels)
        
        # 2. 정확도(Accuracy) 채점 위임
        accuracy_metrics = self._calculate_accuracy_metrics(logits, labels)
        metrics.update(accuracy_metrics)
        
        # 3. 레이턴시 및 처리량 통계 연산 위임
        latency_metrics = self._calculate_latency_metrics(result.timing_records)
        metrics.update(latency_metrics)
        
        return metrics

    def _extract_logits(self, logits: np.ndarray) -> np.ndarray:
        """예기치 못한 1차원 배열(단일 배치)을 2차원으로 복구(Reshape)하는 헬퍼 함수."""
        if logits.ndim == 1 and logits.size > 0:
            logits = np.expand_dims(logits, axis=0)
        return logits

    def _calculate_accuracy_metrics(self, logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """순수 Numpy를 사용하여 다차원 행렬 채점을 수행하는 핵심 헬퍼 함수."""
        total_samples = labels.size
        
        # 빈 배열 방어(ZeroDivisionError 크래시 방지)
        if total_samples == 0 or logits.size == 0:
            return {"accuracy": 0.0, "total_samples": 0}
            
        # Numpy 가속(Vectorized)을 통한 1 ms 이내 일괄 채점
        predictions = np.argmax(logits, axis=-1)
        correct_count = np.sum(predictions == labels)
        accuracy = (correct_count / total_samples) * 100.0
        
        return {
            "accuracy": float(accuracy),
            "total_samples": int(total_samples)
        }

    def _calculate_latency_metrics(self, timing_records: List[float]) -> Dict[str, float]:
        """MLPerf NLP 표준: 실행 소요 시간(Latency) 및 초당 처리 샘플 수(Samples/s) 기반 지표 계산."""
        if not timing_records:
            return {}
            
        avg_latency = float(np.mean(timing_records))
        p99_latency = float(np.percentile(timing_records, 99))
        samples_per_sec = 1000.0 / avg_latency if avg_latency > 0 else 0.0
        
        return {
            "Average Latency (ms)": avg_latency,
            "P99 Latency (ms)": p99_latency,
            "Samples/s": samples_per_sec
        }
