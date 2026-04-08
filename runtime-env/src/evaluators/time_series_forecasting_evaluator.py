"""
TimeSeriesForecastingEvaluator — PatchTST-FM-R1 및 시계열 예측 모델용 평가기

스트리밍 평가를 지원합니다.
add_batch()에서 (N, pred_len, C) 예측 텐서를 받아 RevIN 역정규화 후
MAE·MSE를 즉시 누산하고, 원본 예측 텐서는 즉시 폐기합니다.
배치당 추가 메모리: float 2개(mae_delta, mse_delta) — O(1).
"""

from typing import Any, Dict, List, Optional

import numpy as np

from .base import Evaluator
from core.inference_result import InferenceResult


class TimeSeriesForecastingEvaluator(Evaluator):
    """
    시계열 예측 평가기.

    Parameters (kwargs)
    -------------------
    output_key : str | None
        InferenceResult.outputs 에서 예측 텐서를 가져올 키.
        None(기본값)이면 outputs 딕셔너리의 첫 번째 키를 자동 사용합니다.
    """

    def __init__(self,
                 output_key: Optional[str] = None,
                 train_global_mean: Optional[np.ndarray] = None,
                 train_global_std: Optional[np.ndarray] = None,
                 dataloader=None,
                 **kwargs):
        self._output_key = output_key
        if train_global_mean is not None:
            self.train_global_mean = train_global_mean
            self.train_global_std  = train_global_std if train_global_std is not None else 1.0
        elif dataloader is not None and hasattr(dataloader, "get_train_stats"):
            self.train_global_mean, self.train_global_std = dataloader.get_train_stats()
        else:
            self.train_global_mean = 0.0
            self.train_global_std  = 1.0
        self._reset()

    # ------------------------------------------------------------------
    # 내부 상태 초기화
    # ------------------------------------------------------------------

    def _reset(self):
        """누산 상태를 초기화합니다."""
        self._mae_sum: float = 0.0
        self._mse_sum: float = 0.0
        self._n: int = 0
        self._timing_records: List[float] = []

    # ------------------------------------------------------------------
    # 스트리밍 인터페이스
    # ------------------------------------------------------------------

    def add_batch(self, outputs: Dict[str, np.ndarray], labels: Any, timing_ms: float) -> None:
        """
        배치의 예측 텐서를 역정규화하여 MAE·MSE를 즉시 누산하고 원본 텐서는 즉시 폐기합니다.
        배치당 추가 메모리: float 2개(mae_delta, mse_delta) — O(1).
        """
        key = self._output_key or next(iter(outputs))
        preds_norm = outputs[key]  # (B, pred_len, C)

        flat_labels = self._flatten_labels(labels)
        self._accumulate(preds_norm, flat_labels)
        self._timing_records.append(timing_ms)
        # preds_norm 변수가 스코프를 벗어나면 GC 대상이 됩니다.

    def compute(self) -> Dict[str, Any]:
        """누산된 MAE·MSE 합산으로 최종 메트릭을 계산합니다."""
        if self._n == 0:
            raise ValueError("TimeSeriesForecastingEvaluator: 평가할 샘플이 없습니다.")

        mae  = self._mae_sum / self._n
        mse  = self._mse_sum / self._n
        rmse = float(np.sqrt(mse))

        avg_lat, p99_lat = self._compute_latency_metrics(self._timing_records)

        return {
            "MAE":                  round(mae,     6),
            "MSE":                  round(mse,     6),
            "RMSE":                 round(rmse,    6),
            "Average Latency (ms)": round(avg_lat, 3),
            "P99 Latency (ms)":     round(p99_lat, 3),
            "Total Samples":        self._n,
        }

    # ------------------------------------------------------------------
    # 배치 호환 인터페이스 (단위 테스트 및 레거시 지원)
    # ------------------------------------------------------------------

    def evaluate(self, result: InferenceResult) -> Dict[str, Any]:
        """InferenceResult 전체를 받아 스트리밍 내부 로직으로 채점합니다."""
        self._reset()

        key = self._output_key or next(iter(result.outputs))
        preds_norm = result.outputs[key]  # (N, pred_len, C)

        flat_labels = self._flatten_labels(result.labels)

        if len(flat_labels) == 0:
            raise ValueError("TimeSeriesForecastingEvaluator: 레이블이 비어 있습니다.")
        if len(flat_labels) != len(preds_norm):
            raise ValueError(
                f"TimeSeriesForecastingEvaluator: 예측 샘플 수({len(preds_norm)})와 "
                f"레이블 수({len(flat_labels)})가 일치하지 않습니다."
            )

        self._accumulate(preds_norm, flat_labels)
        self._timing_records = list(result.timing_records)
        return self.compute()

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _flatten_labels(self, labels: Any) -> List[Dict]:
        """레이블을 1D 딕셔너리 리스트로 평탄화합니다. 중첩 배치 형식을 지원합니다."""
        if isinstance(labels, list):
            if labels and isinstance(labels[0], list):
                # 중첩 배치: [[label_dict, ...], [...]] → [label_dict, ...]
                return [lbl for batch in labels for lbl in batch]
            return labels
        return [labels]

    def _accumulate(self, preds_norm: np.ndarray, flat_labels: List[Dict]) -> None:
        """역정규화 후 MAE·MSE를 내부 누산기에 더합니다."""
        global_mean = self.train_global_mean
        global_std  = self.train_global_std

        for i, label in enumerate(flat_labels):
            mean = label["norm_stats"]["mean"]  # (C,)
            std  = label["norm_stats"]["std"]   # (C,)

            pred_raw = preds_norm[i] * std + mean   # (pred_len, C), 원본 스케일
            gt_raw   = label["future_values"]       # (pred_len, C), 원본 스케일

            pred_norm_global = (pred_raw - global_mean) / global_std
            gt_norm_global   = (gt_raw   - global_mean) / global_std

            diff = pred_norm_global - gt_norm_global
            self._mae_sum += float(np.mean(np.abs(diff)))
            self._mse_sum += float(np.mean(diff ** 2))
            self._n += 1

    def _compute_latency_metrics(self, timing_records: List[float]):
        if not timing_records:
            return 0.0, 0.0
        lat = np.array(timing_records, dtype=np.float64)
        return float(np.mean(lat)), float(np.percentile(lat, 99))

    def is_applicable(self, device_spec: Any, model_spec: Any) -> bool:
        from core.model_spec import Task
        return model_spec.task == Task.TIME_SERIES_FORECASTING

    def get_metric_names(self) -> List[str]:
        return [
            "MAE",
            "MSE",
            "RMSE",
            "Average Latency (ms)",
            "P99 Latency (ms)",
            "Total Samples",
        ]
