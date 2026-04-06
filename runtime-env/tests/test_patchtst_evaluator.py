"""
TimeSeriesForecastingEvaluator 단위 테스트

합성 InferenceResult를 사용하므로 실제 모델/데이터 없이 실행 가능합니다.

실행 방법:
    python -m pytest tests/test_patchtst_evaluator.py -v   # pytest 권장
    uv run tests/test_patchtst_evaluator.py                 # 직접 실행
"""

import sys
import os

# uv run / 직접 실행 시 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from core.inference_result import InferenceResult
from core.model_spec import Model_Spec, Task
from evaluators import create_evaluator
from evaluators.time_series_forecasting_evaluator import TimeSeriesForecastingEvaluator


# ------------------------------------------------------------------
# 합성 데이터 헬퍼
# ------------------------------------------------------------------

def _make_result(
    pred_norm: np.ndarray,  # (N, pred_len, C)
    gt_list: list,          # List of (pred_len, C) arrays
    mean_list: list,        # List of (C,) arrays
    std_list: list,         # List of (C,) arrays
    timing: list = None,
    batch_size: int = 1,
) -> InferenceResult:
    """합성 InferenceResult를 생성합니다."""
    n = len(gt_list)
    labels_flat = [
        {
            "future_values": gt_list[i],
            "norm_stats": {"mean": mean_list[i], "std": std_list[i]},
        }
        for i in range(n)
    ]
    # BenchmarkRunner 방식: 배치 리스트 형태로 포장
    labels = [labels_flat[i:i+batch_size] for i in range(0, n, batch_size)]

    return InferenceResult(
        outputs={"output": pred_norm},
        timing_records=timing or [1.0] * n,
        labels=labels,
    )


# ------------------------------------------------------------------
# 테스트
# ------------------------------------------------------------------

class TestTimeSeriesForecastingEvaluator:
    def test_perfect_prediction(self):
        """역정규화 후 pred == gt이면 MAE=MSE=RMSE=0."""
        pred_len, C = 96, 3
        mean = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        std  = np.array([0.5, 1.0, 2.0], dtype=np.float32)
        gt   = np.random.rand(pred_len, C).astype(np.float32)
        # 정규화된 예측 = (gt - mean) / std  → 역정규화하면 정확히 gt
        norm_pred = ((gt - mean) / std)[np.newaxis]  # (1, pred_len, C)

        result = _make_result(norm_pred, [gt], [mean], [std])
        ev = TimeSeriesForecastingEvaluator()
        metrics = ev.evaluate(result)

        assert metrics["MAE"]  < 1e-5
        assert metrics["MSE"]  < 1e-5
        assert metrics["RMSE"] < 1e-5

    def test_denormalization_applied(self):
        """역정규화가 실제로 적용되는지 수치로 검증."""
        pred_len, C = 4, 1
        mean = np.array([10.0], dtype=np.float32)
        std  = np.array([2.0],  dtype=np.float32)
        gt   = np.array([[12.0], [14.0], [10.0], [8.0]], dtype=np.float32)  # (4, 1)
        # 모델이 0으로만 예측 (정규화 공간)
        norm_pred = np.zeros((1, pred_len, C), dtype=np.float32)
        # 역정규화 후 pred = 0 * 2 + 10 = 10 → diff = [2, 4, 0, 2]
        expected_mae = (2.0 + 4.0 + 0.0 + 2.0) / 4  # 2.0

        result = _make_result(norm_pred, [gt], [mean], [std])
        ev = TimeSeriesForecastingEvaluator()
        metrics = ev.evaluate(result)

        assert abs(metrics["MAE"] - expected_mae) < 1e-5

    def test_metric_names(self):
        ev = TimeSeriesForecastingEvaluator()
        names = ev.get_metric_names()
        assert "MAE" in names
        assert "MSE" in names
        assert "RMSE" in names
        assert "Average Latency (ms)" in names
        assert "P99 Latency (ms)" in names
        assert "Total Samples" in names

    def test_is_applicable(self):
        ev = TimeSeriesForecastingEvaluator()

        ts_spec = Model_Spec(
            name="patchtst-fm-r1",
            task=Task.TIME_SERIES_FORECASTING,
            input_shapes={"past_values": (1, 512, 7)},
            input_dtype={"past_values": "float32"},
            output_shapes={"output": (1, 96, 7)},
        )
        clf_spec = Model_Spec(
            name="resnet50",
            task=Task.IMAGE_CLASSIFICATION,
            input_shapes={"input": (1, 3, 224, 224)},
            input_dtype={"input": "float32"},
            output_shapes={"output": (1, 1000)},
        )

        assert ev.is_applicable(None, ts_spec)  is True
        assert ev.is_applicable(None, clf_spec) is False

    def test_multi_batch_labels(self):
        """다중 배치 레이블이 올바르게 평탄화되는지 확인."""
        pred_len, C = 8, 2
        n_samples = 4
        mean = np.zeros(C, dtype=np.float32)
        std  = np.ones(C, dtype=np.float32)
        gt   = np.zeros((pred_len, C), dtype=np.float32)
        # 예측도 0 → MAE=0
        norm_preds = np.zeros((n_samples, pred_len, C), dtype=np.float32)

        result = _make_result(
            norm_preds,
            [gt] * n_samples,
            [mean] * n_samples,
            [std] * n_samples,
            batch_size=2,  # 2개씩 2배치로 구성
        )
        ev = TimeSeriesForecastingEvaluator()
        metrics = ev.evaluate(result)

        assert metrics["Total Samples"] == n_samples
        assert metrics["MAE"] < 1e-5

    def test_output_key_auto_detection(self):
        """output_key=None일 때 outputs 첫 번째 키를 자동 탐지하는지 확인."""
        pred_len, C = 4, 1
        mean = np.zeros(C, dtype=np.float32)
        std  = np.ones(C, dtype=np.float32)
        gt   = np.zeros((pred_len, C), dtype=np.float32)
        norm_pred = np.zeros((1, pred_len, C), dtype=np.float32)

        # 키를 "forecast"로 변경해도 자동 탐지되어야 함
        labels_flat = [{"future_values": gt, "norm_stats": {"mean": mean, "std": std}}]
        result = InferenceResult(
            outputs={"forecast": norm_pred},
            timing_records=[1.0],
            labels=[labels_flat],
        )
        ev = TimeSeriesForecastingEvaluator()  # output_key=None
        metrics = ev.evaluate(result)
        assert metrics["MAE"] < 1e-5

    def test_explicit_output_key(self):
        """output_key를 명시적으로 지정할 때도 동작하는지 확인."""
        pred_len, C = 4, 1
        mean = np.zeros(C, dtype=np.float32)
        std  = np.ones(C, dtype=np.float32)
        gt   = np.zeros((pred_len, C), dtype=np.float32)
        norm_pred = np.zeros((1, pred_len, C), dtype=np.float32)

        labels_flat = [{"future_values": gt, "norm_stats": {"mean": mean, "std": std}}]
        result = InferenceResult(
            outputs={"last_hidden_state": norm_pred},
            timing_records=[1.0],
            labels=[labels_flat],
        )
        ev = TimeSeriesForecastingEvaluator(output_key="last_hidden_state")
        metrics = ev.evaluate(result)
        assert metrics["MAE"] < 1e-5

    def test_latency_metrics(self):
        """레이턴시 통계(mean, p99)가 올바르게 계산되는지 확인."""
        pred_len, C = 4, 1
        mean = np.zeros(C, dtype=np.float32)
        std  = np.ones(C, dtype=np.float32)
        gt   = np.zeros((pred_len, C), dtype=np.float32)
        timing = [1.0, 2.0, 3.0, 100.0]  # P99는 100에 가까워야 함
        norm_preds = np.zeros((4, pred_len, C), dtype=np.float32)

        result = _make_result(
            norm_preds, [gt] * 4, [mean] * 4, [std] * 4, timing=timing
        )
        ev = TimeSeriesForecastingEvaluator()
        metrics = ev.evaluate(result)

        assert abs(metrics["Average Latency (ms)"] - 26.5) < 1e-3
        assert metrics["P99 Latency (ms)"] > 90.0  # 100에 근접

    def test_factory_routing(self):
        """create_evaluator()가 TIME_SERIES_FORECASTING을 올바르게 라우팅하는지 확인."""
        spec = Model_Spec(
            name="patchtst-fm-r1",
            task=Task.TIME_SERIES_FORECASTING,
            input_shapes={"past_values": (1, 512, 7)},
            input_dtype={"past_values": "float32"},
            output_shapes={"output": (1, 96, 7)},
        )
        ev = create_evaluator(spec)
        assert isinstance(ev, TimeSeriesForecastingEvaluator)

    def test_rmse_equals_sqrt_mse(self):
        """RMSE = sqrt(MSE) 관계 검증."""
        pred_len, C = 10, 2
        mean = np.zeros(C, dtype=np.float32)
        std  = np.ones(C, dtype=np.float32)
        gt   = np.random.rand(pred_len, C).astype(np.float32)
        # 모델 예측: 상수 0.5
        norm_pred = np.full((1, pred_len, C), 0.5, dtype=np.float32)

        result = _make_result(norm_pred, [gt], [mean], [std])
        ev = TimeSeriesForecastingEvaluator()
        metrics = ev.evaluate(result)

        assert abs(metrics["RMSE"] - np.sqrt(metrics["MSE"])) < 1e-6


if __name__ == "__main__":
    # uv run tests/test_patchtst_evaluator.py 로 직접 실행할 때 pytest를 통해 수행
    raise SystemExit(pytest.main([__file__, "-v"]))
