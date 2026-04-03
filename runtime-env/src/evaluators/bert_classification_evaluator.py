import numpy as np
from typing import Dict, Any, List

from .base import Evaluator
from src.core.inference_result import InferenceResult
from src.core.model_spec import Model_Spec, Task

class BertClassificationEvaluator(Evaluator):
    """
    BERT 기반 텍스트 분류(SST-2 등) 텐서 추론 결과(Logits)의 정답률(Accuracy)을 산출합니다.

    스트리밍 평가를 지원합니다.
    add_batch()에서 logits → argmax(경량 정수)로 즉시 변환하여 정답 카운트만 누산하고,
    원본 logits 텐서는 즉시 폐기합니다. 배치당 저장량: 스칼라 2개(correct, total)만 증가.
    """

    def __init__(self, **eval_options):
        self.eval_options = eval_options
        self._reset()

    # ------------------------------------------------------------------
    # 내부 상태 초기화
    # ------------------------------------------------------------------

    def _reset(self):
        """누산 상태를 초기화합니다."""
        self._correct: int = 0
        self._total: int = 0
        self._timing_records: List[float] = []

    # ------------------------------------------------------------------
    # 스트리밍 인터페이스
    # ------------------------------------------------------------------

    def add_batch(self, outputs: Dict[str, np.ndarray], labels: Any, timing_ms: float) -> None:
        """
        배치의 logits에서 argmax만 계산하여 정답 수를 누산하고 원본 텐서는 즉시 폐기합니다.
        배치당 추가 메모리: 정수 2개(correct_delta, batch_size) — O(1).
        """
        logits = self._extract_logits(outputs.get("logits", np.array([])))
        flat_labels = self._flatten_labels(labels)

        if logits.size > 0 and len(flat_labels) > 0:
            labels_arr = np.array(flat_labels)
            predictions = np.argmax(logits, axis=-1)
            self._correct += int(np.sum(predictions == labels_arr))
            self._total += int(labels_arr.size)

        self._timing_records.append(timing_ms)
        # logits 변수가 스코프를 벗어나면 GC 대상이 됩니다.

    def compute(self) -> Dict[str, Any]:
        """누산된 카운트로 최종 정확도 메트릭을 계산합니다."""
        metrics = {}
        if self._total == 0:
            metrics.update({"accuracy": 0.0, "total_samples": 0})
        else:
            accuracy = (self._correct / self._total) * 100.0
            metrics.update({"accuracy": float(accuracy), "total_samples": int(self._total)})

        metrics.update(self._calculate_latency_metrics(self._timing_records))
        return metrics

    # ------------------------------------------------------------------
    # 배치 호환 인터페이스 (단위 테스트 및 레거시 지원)
    # ------------------------------------------------------------------

    def evaluate(self, result: InferenceResult) -> Dict[str, Any]:
        """InferenceResult 전체를 받아 스트리밍 내부 로직으로 채점합니다."""
        self._reset()

        logits = self._extract_logits(result.outputs.get("logits", np.array([])))
        flat_labels = self._flatten_labels(result.labels)

        if logits.size > 0 and len(flat_labels) > 0:
            labels_arr = np.array(flat_labels)
            predictions = np.argmax(logits, axis=-1)
            self._correct = int(np.sum(predictions == labels_arr))
            self._total = int(labels_arr.size)

        self._timing_records = list(result.timing_records)
        return self.compute()

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _flatten_labels(self, raw_labels: Any) -> List:
        """배치 레이블을 1D 리스트로 평탄화합니다."""
        labels = []
        if isinstance(raw_labels, (list, np.ndarray)):
            for item in raw_labels:
                if isinstance(item, (list, np.ndarray)):
                    labels.extend(item)
                else:
                    labels.append(item)
        else:
            labels = [raw_labels]
        return labels

    def _extract_logits(self, logits: np.ndarray) -> np.ndarray:
        """예기치 못한 1차원 배열(단일 배치)을 2차원으로 복구(Reshape)하는 헬퍼 함수."""
        if logits.ndim == 1 and logits.size > 0:
            logits = np.expand_dims(logits, axis=0)
        return logits

    def _calculate_latency_metrics(self, timing_records: List[float]) -> Dict[str, float]:
        """MLPerf NLP 표준: 레이턴시 및 초당 처리 샘플 수(Samples/s) 기반 지표 계산."""
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

    def get_metric_names(self) -> List[str]:
        return [
            "accuracy", "total_samples",
            "Average Latency (ms)", "P99 Latency (ms)", "Samples/s"
        ]

    def is_applicable(self, device_spec: Dict[str, Any], model_spec: Model_Spec) -> bool:
        return getattr(model_spec, "task", None) == Task.NLP_CLASSIFICATION
