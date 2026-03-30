"""
granite-timeseries-patchtst E2E 벤치마크 테스트

ETTm1 데이터셋을 사용하여 granite-timeseries-patchtst ONNX 모델의
예측 성능(MAE, MSE, RMSE, Latency)을 측정합니다.

사전 준비:
    1. 모델 다운로드:
       python models/download_model_from_huggingface.py \
           --name ibm-granite/granite-timeseries-patchtst --output models

    2. ONNX export:
       uv run models/export_onnx_patchtst.py \
           --model models/ibm-granite_granite-timeseries-patchtst/ \
           --output models/ibm-granite_granite-timeseries-patchtst-ONNX/model.onnx \
           --context-length 512 --channels 7 --prediction-length 96

실행:
    uv run tests/test_patchtst_e2e.py
    python -m pytest tests/test_patchtst_e2e.py -v -s
"""

import os
import sys
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.core.model_spec import Model_Spec, Task
from src.core.compiled_model import CompiledModel
from src.core.benchmarkrunner import BenchmarkRunner
from src.dataloader import create_dataloader
from src.runtimes import OnnxRuntime
from src.evaluators import create_evaluator


# ------------------------------------------------------------------
# 경로 & 하이퍼파라미터 설정
# ------------------------------------------------------------------

ONNX_PATH   = os.path.join(
    project_root, "models",
    "ibm-granite_granite-timeseries-patchtst-ONNX", "model.onnx"
)
DATASET_PATH = os.path.join(project_root, "datasets", "etth1", "ETTh1.csv")
CACHE_DIR    = os.path.join(project_root, "datasets", "etth1", ".cache_npz")

CONTEXT_LENGTH    = 512
PREDICTION_LENGTH = 96
NUM_CHANNELS      = 7          # ETTh1 전체 피처
WARMUP_RUNS       = 1
BATCH_SIZE        = 1

# ETTh1 표준 벤치마크 split (논문 기준: 12/4/4개월, 1h 간격)
# train: [0, 8640), val: [8640, 11520), test: [11520, 14400)
ETTh1_SPLIT_BOUNDARIES = (8640, 11520)


# ------------------------------------------------------------------
# 메인
# ------------------------------------------------------------------

def main():
    print("=" * 60)
    print(" granite-timeseries-patchtst E2E Benchmark (ETTh1) ")
    print("=" * 60)

    # 1. 아티팩트 존재 확인
    if not os.path.isfile(ONNX_PATH):
        print(f"\n[!] ONNX 파일을 찾을 수 없습니다: {ONNX_PATH}")
        print("    아래 명령으로 먼저 export하세요:\n")
        print(
            "    uv run models/export_onnx_patchtst.py \\\n"
            "        --model models/ibm-granite_granite-timeseries-patchtst/ \\\n"
            "        --output models/ibm-granite_granite-timeseries-patchtst-ONNX/model.onnx \\\n"
            "        --context-length 512 --channels 7 --prediction-length 96\n"
        )
        sys.exit(1)

    if not os.path.isfile(DATASET_PATH):
        print(f"\n[!] 데이터셋을 찾을 수 없습니다: {DATASET_PATH}")
        sys.exit(1)

    print(f"[*] ONNX 경로     : {ONNX_PATH}")
    print(f"[*] 데이터셋 경로 : {DATASET_PATH}")

    # 2. Model_Spec 생성
    spec = Model_Spec(
        name="granite-timeseries-patchtst",
        task=Task.TIME_SERIES_FORECASTING,
        input_shapes={
            "past_values":        (1, CONTEXT_LENGTH, NUM_CHANNELS),
            "past_observed_mask": (1, CONTEXT_LENGTH, NUM_CHANNELS),
        },
        input_dtype={
            "past_values":        "float32",
            "past_observed_mask": "float32",
        },
        output_shapes={"predictions": (1, PREDICTION_LENGTH, NUM_CHANNELS)},
        model_paths={"onnx": ONNX_PATH},
    )

    # 3. CompiledModel 래핑
    compiled_model = CompiledModel(
        spec=spec,
        backend_name="onnxruntime",
        artifact_path=Path(ONNX_PATH),
    )

    # 4. ETTmLoader 초기화
    print("\n[*] ETTmLoader 초기화 중...")
    loader = create_dataloader(
        spec,
        csv_path=DATASET_PATH,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        target_cols=None,               # 전체 7채널
        split="test",                   # 논문과 동일한 test split
        split_boundaries=ETTh1_SPLIT_BOUNDARIES,  # 표준 절대 인덱스 (8640, 11520)
        normalize=True,                # 모델 내부 scaling(std) 사용 → raw 데이터 전달
        cache_dir=CACHE_DIR,
    )
    meta = loader.get_metadata()
    print(f"[*] test split 범위     : {meta['split_start']} ~ {meta['split_end']}")
    print(f"[*] test 분할 윈도우 수 : {meta['window_count']}")
    print(f"[*] context_length      : {meta['context_length']}")
    print(f"[*] prediction_length   : {meta['prediction_length']}")
    print(f"[*] stride              : {meta['stride']}")
    print(f"[*] 피처 컬럼           : {meta['feature_cols']}")

    # 5. OnnxRuntime 초기화
    print("\n[*] OnnxRuntime (cuda) 초기화 중...")
    runtime = OnnxRuntime(device="cuda")
    runtime.load(compiled_model)

    # 6. 논문 비교용 정규화(MSE/MAE)를 위한 Train Set 통계량 계산
    import pandas as pd
    _df = pd.read_csv(DATASET_PATH)
    _cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    _train_data = _df[_cols].values[:ETTh1_SPLIT_BOUNDARIES[0]].astype(float)
    train_global_mean = _train_data.mean(axis=0)
    train_global_std = _train_data.std(axis=0)

    print("\n[*] Evaluator (Global Normalization 적용) 초기화 중...")
    evaluator = create_evaluator(
        spec,
        train_global_mean=train_global_mean,
        train_global_std=train_global_std
    )

    # 7. BenchmarkRunner 구동
    print(f"\n[*] BenchmarkRunner 실행 (warmup={WARMUP_RUNS}, batch_size={BATCH_SIZE})...")
    runner = BenchmarkRunner(
        dataloader=loader,
        runtime=runtime,
        evaluator=evaluator,
    )
    results = runner.run(warmup_runs=WARMUP_RUNS, batch_size=BATCH_SIZE)

    # 8. (완료) 결과 집계 완료

    # 9. 결과 출력
    print("\n" + "=" * 50)
    print(" Final Metrics — granite-timeseries-patchtst ")
    print(" Dataset: ETTh1 (test split, zero-shot)      ")
    print("=" * 50)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:<26}: {v:.4f}")
        else:
            print(f"  {k:<26}: {v}")
    print(f"  {'Normalized MSE (paper)':26}: {results['MSE']:.4f}")
    print("=" * 50)

    runtime.unload()


if __name__ == "__main__":
    main()
