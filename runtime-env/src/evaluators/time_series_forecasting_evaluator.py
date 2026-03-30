"""
TimeSeriesForecastingEvaluator — PatchTST-FM-R1 및 시계열 예측 모델용 평가기

ETTmLoader가 제공하는 원본 스케일 ground truth와
모델이 출력하는 정규화 공간 예측값을 RevIN 역정규화 후 비교하여
MAE / MSE / RMSE 및 레이턴시 통계를 산출합니다.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from .base import Evaluator
from ..core.inference_result import InferenceResult


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
                 **kwargs):
        self._output_key = output_key
        self.train_global_mean = train_global_mean if train_global_mean is not None else 0.0
        self.train_global_std = train_global_std if train_global_std is not None else 1.0

    # ------------------------------------------------------------------
    # Evaluator 추상 메서드 구현
    # ------------------------------------------------------------------

    def evaluate(self, result: InferenceResult) -> Dict[str, Any]:
        """
        Args:
            result: BenchmarkRunner가 전달하는 추론 결과 DTO.
                - result.outputs["<key>"]: shape (N, pred_len, C), 정규화 공간 예측값
                - result.labels: List[List[Dict]]
                    각 Dict = {"future_values": (pred_len, C) raw, "norm_stats": {"mean":(C,), "std":(C,)}}
                - result.timing_records: List[float] (ms)

        Returns:
            Dict with keys: MAE, MSE, RMSE,
                            Average Latency (ms), P99 Latency (ms), Total Samples
        """
        # 1. 예측 텐서 추출
        key = self._output_key or next(iter(result.outputs))
        preds_norm = result.outputs[key]  # (N, pred_len, C)

        # 2. 레이블 평탄화: [[label_dict, ...], ...] → [label_dict, ...]
        all_labels: List[Dict] = [lbl for batch in result.labels for lbl in batch]

        if len(all_labels) == 0:
            raise ValueError("TimeSeriesForecastingEvaluator: 레이블이 비어 있습니다.")
        if len(all_labels) != len(preds_norm):
            raise ValueError(
                f"TimeSeriesForecastingEvaluator: 예측 샘플 수({len(preds_norm)})와 "
                f"레이블 수({len(all_labels)})가 일치하지 않습니다."
            )

        # 3. Global 통계량 기반 Normalized Domain 역투영 후 MAE/MSE 누산
        mae_sum = 0.0
        mse_sum = 0.0
        
        global_mean = self.train_global_mean
        global_std = self.train_global_std

        for i, label in enumerate(all_labels):
            mean = label["norm_stats"]["mean"]  # (C,)
            std  = label["norm_stats"]["std"]   # (C,)
            
            # 1. 모델이 이미 정규화/역정규화 한 값과 관계없이 Raw 값을 획득
            # (DataLoader가 normalize=False 시 std=1, mean=0을 반환하므로 항등 연산으로 동작)
            pred_raw = preds_norm[i] * std + mean   # (pred_len, C), 원본 스케일
            gt_raw   = label["future_values"]       # (pred_len, C), 원본 스케일

            # 2. Evaluation Space를 Global Normalized Domain으로 투영
            pred_norm_global = (pred_raw - global_mean) / global_std
            gt_norm_global   = (gt_raw - global_mean) / global_std
            
            diff = pred_norm_global - gt_norm_global
            mae_sum += float(np.mean(np.abs(diff)))
            mse_sum += float(np.mean(diff ** 2))

        n = len(all_labels)
        mae  = mae_sum / n
        mse  = mse_sum / n
        rmse = float(np.sqrt(mse))

        # 4. 레이턴시 통계
        avg_lat, p99_lat = self._compute_latency_metrics(result.timing_records)

        return {
            "MAE":                  round(mae,     6),
            "MSE":                  round(mse,     6),
            "RMSE":                 round(rmse,    6),
            "Average Latency (ms)": round(avg_lat, 3),
            "P99 Latency (ms)":     round(p99_lat, 3),
            "Total Samples":        n,
        }

    def is_applicable(self, device_spec: Any, model_spec: Any) -> bool:
        from ..core.model_spec import Task
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

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _compute_latency_metrics(self, timing_records: List[float]):
        if not timing_records:
            return 0.0, 0.0
        lat = np.array(timing_records, dtype=np.float64)
        return float(np.mean(lat)), float(np.percentile(lat, 99))
