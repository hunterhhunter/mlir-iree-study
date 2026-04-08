"""
Llama 3.2 3B vLLM End-to-End 벤치마크 테스트

SQuAD 2.0 데이터셋을 사용하여 Llama 3.2 3B 모델의
추론 성능(EM, F1, Latency)을 vLLM 백엔드로 측정합니다.
GPU와 CPU 모드를 모두 지원합니다.

사전 준비:
    1. HuggingFace 로그인: huggingface-cli login
    2. 모델 다운로드 (HuggingFace 가중치 — ONNX 불필요):
       python models/download_hf_model.py \
           --name meta-llama/Llama-3.2-3B-Instruct \
           --output models
    3. vLLM 설치:
       pip install vllm

실행 (GPU):
    python tests/test_vllm_e2e.py
    python tests/test_vllm_e2e.py --device cuda

실행 (CPU):
    python tests/test_vllm_e2e.py --device cpu

실행 (GPU → CPU 순서로 연속 실행):
    python tests/test_vllm_e2e.py --device both
"""

import argparse
import os
import sys
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from core.model_spec import Model_Spec, Task
from core.compiled_model import CompiledModel
from core.benchmarkrunner import BenchmarkRunner
from dataloader import LlamaLoader
from runtimes import VllmRuntime
from evaluators import LlamaEvaluator


# ------------------------------------------------------------------
# 경로 설정
# ------------------------------------------------------------------

# meta-llama/Llama-3.2-3B-Instruct HuggingFace 가중치 디렉토리 후보
MODEL_DIR_CANDIDATES = [
    os.path.join(project_root, "models", "meta-llama_Llama-3.2-3B-Instruct"),
    os.path.join(project_root, "models", "Llama-3.2-3B-Instruct"),
    os.path.join(project_root, "models", "meta-llama_Llama-3.2-3B"),
    os.path.join(project_root, "models", "Llama-3.2-3B"),
]

DATASET_PATH = os.path.join(project_root, "datasets", "squad2")
CACHE_DIR = os.path.join(DATASET_PATH, ".cache_npz_vllm")

# 추론 설정
WARMUP_RUNS = 1
BATCH_SIZE = 1
MAX_SAMPLES = 1000   # 0으로 설정하면 전체 데이터셋 (11873개) 처리
MAX_NEW_TOKENS = 128
MAX_MODEL_LEN = 512  # vLLM 컨텍스트 윈도우 (GPU 메모리 절약)


# ------------------------------------------------------------------
# 헬퍼
# ------------------------------------------------------------------

def _resolve_model_dir() -> str:
    """다운로드된 HuggingFace 모델 디렉토리를 반환합니다. 없으면 빈 문자열."""
    for d in MODEL_DIR_CANDIDATES:
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "config.json")):
            return d
    return ""


def _build_model_spec(model_dir: str) -> Model_Spec:
    """vLLM용 Model_Spec을 직접 생성합니다 (ONNX 파싱 불필요)."""
    return Model_Spec(
        name="llama-3.2-3b",
        task=Task.NLP_GENERATION,
        input_shapes={
            "input_ids":      (1, MAX_MODEL_LEN),
            "attention_mask": (1, MAX_MODEL_LEN),
        },
        input_dtype={
            "input_ids":      "int64",
            "attention_mask": "int64",
        },
        output_shapes={"generated_ids": (1, MAX_NEW_TOKENS)},
        model_paths={"hf": model_dir},
    )


# ------------------------------------------------------------------
# 단일 디바이스 벤치마크
# ------------------------------------------------------------------

def run_benchmark(device: str, model_dir: str) -> dict:
    """
    지정된 디바이스(cuda / cpu)로 vLLM 벤치마크를 실행합니다.

    Returns:
        최종 메트릭 딕셔너리 (BenchmarkRunner.run() 반환값)
    """
    print("\n" + "=" * 60)
    print(f" Llama 3.2 3B vLLM Benchmark — device: {device.upper()} ")
    print("=" * 60)

    # 1. ModelSpec — vLLM은 ONNX 없이 HF 가중치 디렉토리를 직접 사용
    print("[*] ModelSpec 생성 중...")
    llama_spec = _build_model_spec(model_dir)

    # 2. CompiledModel — artifact_path = HF 가중치 디렉토리
    compiled_model = CompiledModel(
        spec=llama_spec,
        backend_name="vllm",
        artifact_path=Path(model_dir),
    )

    # 3. LlamaLoader 초기화
    print("[*] LlamaLoader 초기화 중...")
    loader = LlamaLoader(
        llama_spec,
        dataset_path=DATASET_PATH,
        tokenizer_path=model_dir,
        cache_dir=CACHE_DIR,
        max_length=MAX_MODEL_LEN,
    )
    meta = loader.get_metadata()
    print(f"[*] 전체 샘플 수     : {meta['total_samples']}")
    print(f"[*] 답변 가능 샘플   : {meta['answerable_samples']}")
    print(f"[*] 최대 시퀀스 길이 : {meta['max_length']}")

    if MAX_SAMPLES > 0:
        loader.samples = loader.samples[:MAX_SAMPLES]
        loader.total_samples = len(loader.samples)
        print(f"[!] 빠른 측정을 위해 {MAX_SAMPLES}개 샘플로 제한합니다.")

    # 4. VllmRuntime 초기화
    print(f"\n[*] VllmRuntime ({device.upper()}) 초기화 중...")
    runtime_kwargs = dict(
        device=device,
        max_model_len=MAX_MODEL_LEN,
        dtype="float16" if device == "cuda" else "float32",
    )
    if device == "cuda":
        runtime_kwargs["gpu_memory_utilization"] = 0.85
        runtime_kwargs["tensor_parallel_size"] = 1

    runtime = VllmRuntime(**runtime_kwargs)
    runtime.load(compiled_model)

    # 5. LlamaEvaluator 초기화
    evaluator = LlamaEvaluator(tokenizer_path=model_dir)

    # 6. BenchmarkRunner 구동
    print(f"\n[*] BenchmarkRunner 실행 (warmup={WARMUP_RUNS}, batch_size={BATCH_SIZE})...")
    runner = BenchmarkRunner(
        dataloader=loader,
        runtime=runtime,
        evaluator=evaluator,
    )
    results = runner.run(warmup_runs=WARMUP_RUNS, batch_size=BATCH_SIZE)

    # 7. 결과 출력
    print("\n" + "=" * 40)
    print(f" Final Metrics — vLLM [{device.upper()}] ")
    print("=" * 40)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("=" * 40)

    runtime.unload()
    return results


# ------------------------------------------------------------------
# 메인
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Llama 3.2 3B vLLM E2E 벤치마크"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "both"],
        default="cuda",
        help="실행 디바이스 (cuda / cpu / both). 기본값: cuda",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 모델 경로 확인
    model_dir = _resolve_model_dir()
    if not model_dir:
        print("\n[!] Llama 3.2 3B HuggingFace 가중치를 찾을 수 없습니다.")
        print("    다음 명령으로 모델을 먼저 다운로드하세요:\n")
        print(
            "    python models/download_hf_model.py \\\n"
            "        --name meta-llama/Llama-3.2-3B-Instruct \\\n"
            "        --output models\n"
        )
        print(f"    탐색 경로: {MODEL_DIR_CANDIDATES}")
        sys.exit(1)

    print(f"[*] 모델 디렉토리  : {model_dir}")
    print(f"[*] 데이터셋 경로  : {DATASET_PATH}")

    devices = ["cuda", "cpu"] if args.device == "both" else [args.device]
    all_results = {}

    for device in devices:
        results = run_benchmark(device=device, model_dir=model_dir)
        all_results[device] = results

    # both 모드: GPU vs CPU 나란히 비교
    if args.device == "both" and len(all_results) == 2:
        print("\n" + "=" * 60)
        print(" GPU vs CPU 비교 요약 ")
        print("=" * 60)
        gpu_r = all_results["cuda"]
        cpu_r = all_results["cpu"]
        key_metrics = [
            "Exact Match", "F1 Score",
            "Average Latency (ms)", "P99 Latency (ms)",
            "TTFT Mean (ms)", "TPOT Mean (ms)",
            "Throughput (tokens/s)",
        ]
        print(f"  {'Metric':<30} {'GPU':>12} {'CPU':>12}")
        print("  " + "-" * 56)
        for k in key_metrics:
            g = gpu_r.get(k, "N/A")
            c = cpu_r.get(k, "N/A")
            g_str = f"{g:.4f}" if isinstance(g, float) else str(g)
            c_str = f"{c:.4f}" if isinstance(c, float) else str(c)
            print(f"  {k:<30} {g_str:>12} {c_str:>12}")
        print("=" * 60)


if __name__ == "__main__":
    main()
