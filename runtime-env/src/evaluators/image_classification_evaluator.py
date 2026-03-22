import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import precision_recall_fscore_support

from .base import Evaluator
from ..core.model_spec import Model_Spec
from ..core.inference_result import InferenceResult

class ImageClassificationEvaluator(Evaluator):
    """
    이미지 분류(Image Classification) 성능 평가 모듈.
    """
    def __init__(self, **eval_options):
        # 딕셔너리에서 설정값을 추출 (기본값 설정)
        self.top_k = eval_options.get("top_k", (1, 5))

    def evaluate(self, result: InferenceResult) -> Dict[str, Any]:
        """추론 결과를 받아 각 프라이빗 메서드로 위임하여 채점함."""
        metrics = {}
        
        # 1. 런타임 결과물(outputs)에서 예측값 추출
        logits_key = list(result.outputs.keys())[0]
        logits = result.outputs[logits_key]
        
        # 2. 정답지(labels) 타입 검증 및 치환
        labels = result.labels
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
            
        metrics["Total Samples"] = labels.shape[0]

        # 3. 각 지표 계산 역할을 프라이빗 헬퍼 메서드들에게 위임
        top_k_metrics, top_k_preds = self._calculate_top_k_accuracy(logits, labels)
        metrics.update(top_k_metrics)

        clf_metrics = self._calculate_classification_metrics(top_k_preds, labels)
        metrics.update(clf_metrics)

        latency_metrics = self._calculate_latency_metrics(result.timing_records)
        metrics.update(latency_metrics)
            
        return metrics

    def _calculate_top_k_accuracy(self, logits: np.ndarray, labels: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
        """내부 헬퍼 함수: Top-K 정답률만 전담하여 계산함."""
        batch_size = labels.shape[0]
        max_k = max(self.top_k)
        
        # 내림차순 정렬 인덱스를 추출함 (Batch, Num_Classes)
        sorted_indices = np.argsort(-logits, axis=1)
        top_k_preds = sorted_indices[:, :max_k]
        
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

    def _calculate_classification_metrics(self, top_k_preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """내부 헬퍼 함수: Precision, Recall, F1-Score만 전담하여 계산함."""
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
        """내부 헬퍼 함수: 실행 소요 시간(Latency)만 전담하여 계산함."""
        if not timing_records:
            return {}
            
        return {
            "Average Latency (ms)": float(np.mean(timing_records)),
            "P99 Latency (ms)": float(np.percentile(timing_records, 99))
        }

    def is_applicable(self, device_spec: Dict[str, Any], model_spec: Model_Spec) -> bool:
        """이 평가기가 주어진 모델과 하드웨어를 채점할 수 있는지 검사함."""
        task_name = str(getattr(model_spec, "task", ""))
        return "IMAGE_CLASSIFICATION" in task_name

    def get_metric_names(self) -> List[str]:
        """해당 모듈에서 반환 가능한 지표 이름의 목록을 반환함."""
        metrics = [f"Top-{k} Accuracy" for k in self.top_k]
        metrics.extend([
            "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)", 
            "Average Latency (ms)", "P99 Latency (ms)", "Total Samples"
        ])
        return metrics
