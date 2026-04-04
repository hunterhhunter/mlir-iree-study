import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import precision_recall_fscore_support

from .base import Evaluator
from core.model_spec import Model_Spec
from core.inference_result import InferenceResult

class ImageClassificationEvaluator(Evaluator):
    """
    이미지 분류(Image Classification) 성능 평가 모듈.

    스트리밍 평가를 지원합니다.
    add_batch()에서 logits → top-k 예측 인덱스(경량)로 즉시 변환하여 누산하고,
    원본 logits 텐서는 즉시 폐기합니다.
    """
    def __init__(self, **eval_options):
        self.top_k = eval_options.get("top_k", (1, 5))
        self._reset()

    # ------------------------------------------------------------------
    # 내부 상태 초기화
    # ------------------------------------------------------------------

    def _reset(self):
        """누산 상태를 초기화합니다. evaluate() 진입 시 자동으로 호출됩니다."""
        self._top_k_preds_list: List[np.ndarray] = []  # List of (B, max_k) — 경량 인덱스만 보관
        self._labels_flat: List[int] = []
        self._timing_records: List[float] = []

    # ------------------------------------------------------------------
    # 스트리밍 인터페이스
    # ------------------------------------------------------------------

    def add_batch(self, outputs: Dict[str, np.ndarray], labels: Any, timing_ms: float) -> None:
        """
        배치의 logits에서 top-k 예측 인덱스만 추출하고 원본 텐서는 즉시 폐기합니다.
        ImageNet-1K 기준 배치당 저장량: (B × max_k) 정수 vs (B × 1000) float32 → ~200배 절약.
        """
        logits_key = list(outputs.keys())[0]
        logits = outputs[logits_key]

        # 구글 MobileNetV2 등 1001-class 모델 처리
        if hasattr(logits, "shape") and len(logits.shape) >= 2 and logits.shape[-1] == 1001:
            logits = logits[..., 1:]

        max_k = max(self.top_k)
        sorted_indices = np.argsort(-logits, axis=1)
        top_k_preds = sorted_indices[:, :max_k]  # (B, max_k) — 경량 정수 배열

        self._top_k_preds_list.append(top_k_preds)
        self._labels_flat.extend(self._flatten_labels(labels))
        self._timing_records.append(timing_ms)
        # logits 변수가 스코프를 벗어나면 GC 대상이 됩니다.

    def compute(self) -> Dict[str, Any]:
        """누산된 경량 통계로 최종 메트릭을 계산합니다."""
        if not self._top_k_preds_list:
            return {"Total Samples": 0}

        all_top_k_preds = np.concatenate(self._top_k_preds_list, axis=0)  # (N, max_k)
        labels = np.array(self._labels_flat)  # (N,)

        metrics = {"Total Samples": int(labels.shape[0])}

        top_k_metrics, top_k_preds = self._calculate_top_k_accuracy(all_top_k_preds, labels)
        metrics.update(top_k_metrics)

        clf_metrics = self._calculate_classification_metrics(top_k_preds, labels)
        metrics.update(clf_metrics)

        latency_metrics = self._calculate_latency_metrics(self._timing_records)
        metrics.update(latency_metrics)

        return metrics

    # ------------------------------------------------------------------
    # 배치 호환 인터페이스 (단위 테스트 및 레거시 지원)
    # ------------------------------------------------------------------

    def evaluate(self, result: InferenceResult) -> Dict[str, Any]:
        """InferenceResult 전체를 받아 스트리밍 내부 로직으로 채점합니다."""
        self._reset()

        logits_key = list(result.outputs.keys())[0]
        logits = result.outputs[logits_key]

        if hasattr(logits, "shape") and len(logits.shape) >= 2 and logits.shape[-1] == 1001:
            logits = logits[..., 1:]

        max_k = max(self.top_k)
        sorted_indices = np.argsort(-logits, axis=1)
        top_k_preds = sorted_indices[:, :max_k]

        self._top_k_preds_list.append(top_k_preds)
        self._labels_flat.extend(self._flatten_labels(result.labels))
        self._timing_records = list(result.timing_records)

        return self.compute()

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _flatten_labels(self, raw_labels: Any) -> List[int]:
        """배치 레이블을 1D 정수 리스트로 평탄화합니다."""
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

    def _calculate_top_k_accuracy(
        self, top_k_preds: np.ndarray, labels: np.ndarray
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """누산된 top-k 예측 인덱스로 Top-K 정답률을 계산합니다."""
        batch_size = labels.shape[0]
        correct_counts = {k: 0 for k in self.top_k}
        for i in range(batch_size):
            target = labels[i]
            for k in self.top_k:
                if target in top_k_preds[i, :k]:
                    correct_counts[k] += 1

        metrics = {
            f"Top-{k} Accuracy": (correct_counts[k] / batch_size) * 100
            for k in self.top_k
        }
        return metrics, top_k_preds

    def _calculate_classification_metrics(
        self, top_k_preds: np.ndarray, labels: np.ndarray
    ) -> Dict[str, float]:
        """Top-1 예측으로 Precision, Recall, F1-Score를 계산합니다."""
        top1_preds = top_k_preds[:, 0]
        p, r, f1, _ = precision_recall_fscore_support(
            labels,
            top1_preds,
            average='macro',
            zero_division=0
        )
        return {
            "Precision (Macro)": p * 100,
            "Recall (Macro)": r * 100,
            "F1-Score (Macro)": f1 * 100
        }

    def _calculate_latency_metrics(self, timing_records: List[float]) -> Dict[str, float]:
        """레이턴시 통계를 계산합니다."""
        if not timing_records:
            return {}
        return {
            "Average Latency (ms)": float(np.mean(timing_records)),
            "P99 Latency (ms)": float(np.percentile(timing_records, 99))
        }

    def is_applicable(self, device_spec: Dict[str, Any], model_spec: Model_Spec) -> bool:
        task_name = str(getattr(model_spec, "task", ""))
        return "IMAGE_CLASSIFICATION" in task_name

    def get_metric_names(self) -> List[str]:
        metrics = [f"Top-{k} Accuracy" for k in self.top_k]
        metrics.extend([
            "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)",
            "Average Latency (ms)", "P99 Latency (ms)", "Total Samples"
        ])
        return metrics
