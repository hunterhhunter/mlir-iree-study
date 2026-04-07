import os
import sys
import argparse
import subprocess
from pathlib import Path

# 프로젝트 루트 경로 추가 (sys.path)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from core.model_spec import Model_Spec, Task
from core.model_profiles import create_model_spec
from core.compiled_model import CompiledModel
from core.benchmarkrunner import BenchmarkRunner

# 구체화된 컴포넌트 임포트 (Facade Pattern 적용)
from dataloader import create_dataloader
from evaluators import create_evaluator
from runtimes import create_runtime
# from src.runtimes.iree_rt import IREERuntime  # 향후 IREE 백엔드 추가 시 주석 해제

def run_auto_prepare(profile: dict, args: argparse.Namespace):
    """
    Zero-Config 벤치마크를 위해 누락된 리소스를 감지하고 백그라운드 준비 스크립트를 자동 실행합니다.
    """
    model_path = args.model_path if args.backend == "vllm" else args.onnx
    dataset_path = args.dataset

    if "prepare_model_script" in profile and profile["prepare_model_script"]:
        if not model_path or not os.path.exists(model_path):
            script = profile["prepare_model_script"]
            print(f"[*] 모델 리소스 누락 감지. 자동 준비 스크립트 실행: {script}")
            subprocess.run([sys.executable, script], check=True)
            
    if "prepare_dataset_script" in profile and profile["prepare_dataset_script"]:
        if not dataset_path or not os.path.exists(dataset_path):
            script = profile["prepare_dataset_script"]
            print(f"[*] 데이터셋 리소스 누락 감지. 자동 준비 스크립트 실행: {script}")
            subprocess.run([sys.executable, script], check=True)


def main():
    parser = argparse.ArgumentParser(description="Unified BenchmarkRunner CLI Orchestrator")
    parser.add_argument("--model", type=str, required=True, help="모델 이름 (예: resnet50, llama-3.2-3b)")
    parser.add_argument("--onnx", type=str, default=None, help="ONNX 파일의 절대 또는 상대 경로 (onnxruntime 백엔드 필수)")
    parser.add_argument("--model-path", type=str, default=None, help="HuggingFace 모델 디렉토리 경로 (vLLM 백엔드 필수)")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="HuggingFace 토크나이저 디렉토리 경로 (NLP 모델 필수)")
    parser.add_argument("--dataset", type=str, default=None, help="평가용 데이터셋 최상위 디렉토리 또는 CSV 파일 경로")
    parser.add_argument("--image-dir", type=str, default="", help="(옵션) 데이터셋 내 이미지 하위 폴더 경로")
    parser.add_argument("--label-dir", type=str, default="", help="(옵션) 데이터셋 내 라벨 하위 폴더 경로")
    parser.add_argument("--layout", type=str, default="NCHW", choices=["NCHW", "NHWC"], help="모델 텐서 레이아웃 (기본: NCHW)")
    parser.add_argument("--backend", type=str, default="onnxruntime", choices=["onnxruntime", "iree", "vllm"], help="추론을 실행할 백엔드 (기본: onnxruntime)")
    parser.add_argument("--device", type=str, default="cpu", help="추론 장치 (예: cpu, cuda, 기본: cpu)")
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="추론 배치 사이즈 (기본: 1)")
    parser.add_argument("--warmup", "-w", type=int, default=2, help="웜업 횟수 (기본: 2)")
    parser.add_argument("--max-steps", type=int, default=None, help="시간이 지루할 때 쓸 강제 종료 리미트 (옵션)")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="LLM 생성 최대 토큰 수 (기본: 256)")
    parser.add_argument("--max-model-len", type=int, default=None, help="vLLM 최대 컨텍스트 길이 (기본: 모델 기본값, 메모리 부족 시 줄이세요)")
    parser.add_argument("--enforce-eager", action="store_true", default=None, help="vLLM CUDA 그래프 캡처 비활성화 (메모리 부족 시 사용)")
    parser.add_argument("--debug", action="store_true", help="샘플별 예측/정답/점수 로그 출력 (기본: 비활성)")
    
    args = parser.parse_args()
    
    # [설계 개선] CLI 인자(--task)에 의존하지 않고, 레지스트리(SUPPORTED_PROFILES)에서 태스크를 자동 추론 (DRY 원칙)
    from core.model_profiles import SUPPORTED_PROFILES
    profile = SUPPORTED_PROFILES.get(args.model)
    if not profile:
        print(f"[Error] '{args.model}' 프로필을 찾을 수 없습니다. model_profiles.py에 등록되었는지 확인하세요.")
        sys.exit(1)
        
        
    # 누락된 인자(default) 주입 (Zero-Config)
    if args.onnx is None and "default_model_path" in profile:
        args.onnx = profile["default_model_path"]
    if args.model_path is None and "default_model_path" in profile:
        args.model_path = profile["default_model_path"]
    if args.dataset is None and "default_dataset_path" in profile:
        args.dataset = profile["default_dataset_path"]
        
    # 토크나이저 경로 자동 추론 (NLP 태스크용)
    if args.tokenizer_path is None:
        if args.backend == "vllm" and args.model_path:
            args.tokenizer_path = args.model_path
        elif args.onnx:
            # ONNX 파일 경로면 부모 디렉토리를 토크나이저 경로로 간주
            args.tokenizer_path = os.path.dirname(args.onnx) if args.onnx.endswith(".onnx") else args.onnx
            
    # 리소스 누락 시 백그라운드 준비 스크립트 실행 (Auto-Prepare)
    run_auto_prepare(profile, args)
    
    # 백엔드별 필수 인자 검증
    if args.backend == "vllm":
        if not args.model_path:
            print("[Error] vllm 백엔드에는 --model-path가 필요합니다.")
            sys.exit(1)
    else:
        if not args.onnx:
            print("[Error] onnxruntime/iree 백엔드에는 --onnx가 필요합니다.")
            sys.exit(1)
        if not os.path.exists(args.onnx):
            print(f"[Error] 모델 파일을 찾을 수 없습니다: {args.onnx}")
            sys.exit(1)
        # 디렉토리가 넘어온 경우 model.onnx 자동 탐색 (HuggingFace 다운로드 폴더 구조 대응)
        if os.path.isdir(args.onnx):
            candidate = os.path.join(args.onnx, "model.onnx")
            if os.path.exists(candidate):
                print(f"[Info] --onnx에 디렉토리가 지정되었습니다. {candidate} 를 사용합니다.")
                args.onnx = candidate
            else:
                print(f"[Error] 디렉토리 {args.onnx} 에서 model.onnx를 찾을 수 없습니다.")
                sys.exit(1)
    
    task_enum = profile["task"]

    # 백엔드-태스크 호환성 검증: vllm은 NLP_GENERATION 전용
    if args.backend == "vllm" and task_enum != Task.NLP_GENERATION:
        print(f"[Error] vllm 백엔드는 NLP_GENERATION 태스크만 지원합니다. "
              f"모델 '{args.model}'의 태스크는 {task_enum.name}입니다. "
              f"onnxruntime 백엔드를 사용하세요: --backend onnxruntime")
        sys.exit(1)

    print("\n" + "="*60)
    print(f" BenchmarkRunner CLI ")
    print(f"   Model: {args.model} | Task: {task_enum.name} | Layout: {args.layout}")
    print(f"   Backend: {args.backend} | Device: {args.device}")
    print("="*60)
    
    # 0. DataLoader 공통 인터페이스 규약 및 CoC 해소 (Resolver)
    from utils.dataset_resolver import resolve_dataset_paths
    image_dir, label_path = resolve_dataset_paths(task_enum, args.dataset, args.image_dir, args.label_dir)
    
    loader_kwargs = {}
    if image_dir:
        loader_kwargs["image_dir"] = image_dir
    if label_path:
        loader_kwargs["label_path"] = label_path
    
    # 1. Spec & Artifact 생성
    artifact_path = Path(args.model_path) if args.backend == "vllm" else Path(args.onnx)
    try:
        spec = create_model_spec(args.model, str(artifact_path), task=task_enum)
    except Exception as e:
        print(f"[Error] 스펙 파싱 실패: {e}")
        sys.exit(1)

    compiled_model = CompiledModel(spec=spec, backend_name=args.backend, artifact_path=artifact_path)
    
    # 2. 컴포넌트(주입 객체) 조립
    print(f"[Factory] Assembling components for {task_enum.name}...")
    # NLP_GENERATION: tokenizer_path 전달, TIME_SERIES_FORECASTING: csv_path로 dataset 직접 전달
    if task_enum == Task.NLP_GENERATION and args.tokenizer_path:
        loader_kwargs["tokenizer_path"] = args.tokenizer_path
    if task_enum == Task.TIME_SERIES_FORECASTING:
        loader_kwargs["csv_path"] = args.dataset
        # csv_path 옆 .cache_npz 폴더를 캐시 디렉토리로 자동 지정
        csv_dir = os.path.dirname(os.path.abspath(args.dataset))
        loader_kwargs["cache_dir"] = os.path.join(csv_dir, ".cache_npz")
    elif task_enum in (Task.IMAGE_CLASSIFICATION, Task.OBJECT_DETECTION):
        # 이미지 데이터셋 디렉토리 옆에 .cache_npz 자동 지정
        loader_kwargs["cache_dir"] = os.path.join(os.path.abspath(args.dataset), ".cache_npz")
    elif task_enum == Task.NLP_GENERATION:
        # val.json 옆 .cache_npz 자동 지정
        loader_kwargs["cache_dir"] = os.path.join(
            os.path.dirname(os.path.abspath(args.dataset)), ".cache_npz"
        )

    loader = create_dataloader(
        model_spec=spec,
        dataset_path=args.dataset,
        layout=args.layout,
        **loader_kwargs
    )
    
    # 런타임 팩토리 로직
    try:
        runtime_kwargs = {}
        if args.max_model_len is not None:
            runtime_kwargs["max_model_len"] = args.max_model_len
        elif "default_max_model_len" in profile:
            runtime_kwargs["max_model_len"] = profile["default_max_model_len"]
        if args.enforce_eager:
            runtime_kwargs["enforce_eager"] = True
        elif "default_enforce_eager" in profile:
            runtime_kwargs["enforce_eager"] = profile["default_enforce_eager"]
        runtime = create_runtime(args.backend, device=args.device, **runtime_kwargs)
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)
        
    runtime.load(compiled_model)
    
    # 평가기 팩토리 로직
    evaluator_kwargs = {}
    if task_enum == Task.NLP_GENERATION and args.tokenizer_path:
        evaluator_kwargs["tokenizer_path"] = args.tokenizer_path
    if args.debug:
        evaluator_kwargs["debug"] = True
    if task_enum == Task.TIME_SERIES_FORECASTING:
        evaluator_kwargs["dataloader"] = loader
    evaluator = create_evaluator(spec, top_k=(1, 5), **evaluator_kwargs)
    
    # 3. 오케스트레이터 구동
    runner = BenchmarkRunner(
        dataloader=loader, runtime=runtime, evaluator=evaluator,
        max_new_tokens=args.max_new_tokens
    )
    results = runner.run(warmup_runs=args.warmup, batch_size=args.batch_size, max_steps=args.max_steps)
    
    # 4. 최종 결과 리포팅
    print("\n" + "="*40)
    print(f" Final Metrics ({args.model.upper()}) ")
    print("="*40)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("="*40)
    
    runtime.unload()

if __name__ == "__main__":
    main()
