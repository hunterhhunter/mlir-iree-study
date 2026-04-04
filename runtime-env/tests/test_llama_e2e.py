"""
Llama 3.1 8B End-to-End 벤치마크 테스트

SQuAD 2.0 데이터셋을 사용하여 Llama 3.1 8B ONNX 모델의
추론 성능(EM, F1, Latency)을 측정합니다.

사전 준비:
    1. HuggingFace 로그인: huggingface-cli login
    2. 모델 다운로드:
       python models/download_hf_model.py \
           --name onnx-community/Llama-3.2-3B-Instruct-ONNX \
           --format onnx --output models

실행:
    python tests/test_llama_e2e.py
"""

import os
import sys
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.core.model_profiles import create_model_spec
from src.core.compiled_model import CompiledModel
from src.core.benchmarkrunner import BenchmarkRunner
from src.dataloader import LlamaLoader
from src.runtimes import OnnxRuntime
from src.evaluators import LlamaEvaluator


# ------------------------------------------------------------------
# 경로 설정
# ------------------------------------------------------------------

# 방법 A: onnx-community 변환 버전 직접 다운로드
MODEL_DIR_A = os.path.join(
    project_root, "models", "onnx-community_Llama-3.1-8B-Instruct-ONNX"
)
# 방법 B: pt/safetensors 다운로드 후 export_onnx_hf.py 로 변환한 경우
MODEL_DIR_B = os.path.join(
    project_root, "models", "meta-llama_Llama-3.1-8B-ONNX-int8"
)

# 방법 C: Llama-3.2-3B-ONNX
MODEL_DIR_C = os.path.join(
    project_root, "models", "meta-llama_Llama-3.2-3B-ONNX"
)

DATASET_PATH = os.path.join(project_root, "datasets", "squad2")
CACHE_DIR = os.path.join(DATASET_PATH, ".cache_npz")

# 추론 설정
WARMUP_RUNS = 1
BATCH_SIZE = 1
MAX_SAMPLES = 500  # 0으로 설정하면 전체 데이터셋 (11873개) 처리


# ------------------------------------------------------------------
# 헬퍼
# ------------------------------------------------------------------

def _find_onnx_path(model_dir: str) -> str:
    """
    모델 디렉토리에서 ONNX 아티팩트를 찾습니다.
    optimum 버전에 따라 파일명이 다를 수 있으므로 순서대로 탐색합니다.
    """
    candidates = [
        "model.onnx",
        "decoder_model_merged.onnx",
        "decoder_model.onnx",
    ]
    for name in candidates:
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            return path
    return ""


def _resolve_model_dir() -> str:
    """다운로드된 모델 디렉토리를 반환합니다. 없으면 빈 문자열."""
    for d in (MODEL_DIR_C, MODEL_DIR_A, MODEL_DIR_B):
        if os.path.isdir(d) and _find_onnx_path(d):
            return d
    return ""


# ------------------------------------------------------------------
# 메인
# ------------------------------------------------------------------

def main():
    print("=" * 60)
    print(" Llama 3.1 8B E2E Benchmark (SQuAD 2.0) ")
    print("=" * 60)

    # 1. 모델 경로 확인
    model_dir = _resolve_model_dir()
    if not model_dir:
        print("\n[!] Llama 3.1 8B ONNX 모델을 찾을 수 없습니다.")
        print("    다음 명령으로 모델을 먼저 다운로드하세요:\n")
        print(
            "    python models/download_hf_model.py \\\n"
            "        --name onnx-community/Llama-3.1-8B-Instruct-ONNX \\\n"
            "        --format onnx --output models\n"
        )
        sys.exit(1)

    onnx_path = _find_onnx_path(model_dir)
    print(f"[*] 모델 디렉토리 : {model_dir}")
    print(f"[*] ONNX 경로     : {onnx_path}")
    print(f"[*] 데이터셋 경로 : {DATASET_PATH}")

    # 2. ModelSpec 생성 (model_profiles 레지스트리 사용)
    print(f"\n[*] ModelSpec 생성 중...")
    llama_spec = create_model_spec("llama-3.1-8b", onnx_path)

    # 3. CompiledModel 래핑
    compiled_model = CompiledModel(
        spec=llama_spec,
        backend_name="onnxruntime",
        artifact_path=Path(onnx_path),
    )

    # 4. LlamaLoader 초기화
    print(f"\n[*] LlamaLoader 초기화 중...")
    loader = LlamaLoader(
        llama_spec,
        dataset_path=DATASET_PATH,
        tokenizer_path=model_dir,
        cache_dir=CACHE_DIR,
        max_length=512,
    )
    meta = loader.get_metadata()
    print(f"[*] 전체 샘플 수      : {meta['total_samples']}")
    print(f"[*] 답변 가능 샘플    : {meta['answerable_samples']}")
    print(f"[*] 유효 답변 샘플    : {meta['answerable_samples']}")
    print(f"[*] 최대 시퀀스 길이  : {meta['max_length']}")

    # 측정 데이터 개수 제한 로직
    if MAX_SAMPLES > 0:
        loader.samples = loader.samples[:MAX_SAMPLES]
        loader.total_samples = len(loader.samples)
        print(f"\n[!] 빠른 측정을 위해 {MAX_SAMPLES}개의 샘플로 추론을 제한합니다.")

    # 5. OnnxRuntime 초기화
    print(f"\n[*] OnnxRuntime (CUDA) 초기화 중...")
    runtime = OnnxRuntime(device="cuda")
    runtime.load(compiled_model)

    # 6. LlamaEvaluator 초기화
    evaluator = LlamaEvaluator(tokenizer_path=model_dir)

    # 7. BenchmarkRunner 구동
    print(f"\n[*] BenchmarkRunner 실행 (warmup={WARMUP_RUNS}, batch_size={BATCH_SIZE})...")
    runner = BenchmarkRunner(
        dataloader=loader,
        runtime=runtime,
        evaluator=evaluator,
    )
    results = runner.run(warmup_runs=WARMUP_RUNS, batch_size=BATCH_SIZE)

    # 8. 결과 출력
    print("\n" + "=" * 40)
    print(f" Final Metrics (LLAMA-3.1-8B-int8) ")
    print("=" * 40)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("=" * 40)

    runtime.unload()


if __name__ == "__main__":
    main()
