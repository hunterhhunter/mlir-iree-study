import os
import sys
import argparse
from pathlib import Path

# 프로젝트 루트 경로 추가 (sys.path)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.core.model_spec import Model_Spec, Task
from src.core.model_profiles import create_model_spec
from src.core.compiled_model import CompiledModel
from src.core.benchmarkrunner import BenchmarkRunner

# 구체화된 컴포넌트 임포트 (Facade Pattern 적용)
from src.dataloader import create_dataloader
from src.evaluators import create_evaluator
from src.runtimes import create_runtime
# from src.runtimes.iree_rt import IREERuntime  # 향후 IREE 백엔드 추가 시 주석 해제


def main():
    parser = argparse.ArgumentParser(description="Unified BenchmarkRunner CLI Orchestrator")
    parser.add_argument("--model", type=str, required=True, help="모델 이름 (예: resnet50, yolov5m)")
    parser.add_argument("--onnx", type=str, required=True, help="ONNX 파일의 절대 또는 상대 경로")
    parser.add_argument("--dataset", type=str, required=True, help="평가용 데이터셋 최상위 디렉토리 (예: datasets/imagenet_1k 또는 datasets/coco128)")
    parser.add_argument("--image-dir", type=str, default="", help="(옵션) 데이터셋 내 이미지 하위 폴더 경로")
    parser.add_argument("--label-dir", type=str, default="", help="(옵션) 데이터셋 내 라벨 하위 폴더 경로")
    parser.add_argument("--task", type=str, default="classification", choices=["classification", "detection", "nlp_classification", "nlp_generation"], help="평가할 태스크 유형 (기본: classification)")
    parser.add_argument("--layout", type=str, default="NCHW", choices=["NCHW", "NHWC"], help="모델 텐서 레이아웃 (기본: NCHW)")
    parser.add_argument("--backend", type=str, default="onnxruntime", choices=["onnxruntime", "iree"], help="추론을 실행할 백엔드 (기본: onnxruntime)")
    parser.add_argument("--device", type=str, default="cpu", help="추론 장치 (예: cpu, cuda, 기본: cpu)")
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="추론 배치 사이즈 (기본: 1)")
    parser.add_argument("--warmup", "-w", type=int, default=2, help="웜업 횟수 (기본: 2)")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print(f" BenchmarkRunner CLI - Project: Antigravity ")
    print(f"   Model: {args.model} | Task: {args.task.upper()} | Layout: {args.layout}")
    print(f"   Backend: {args.backend} | Device: {args.device}")
    print("="*60)
    
    if not os.path.exists(args.onnx):
        print(f"[Error] 모델 파일을 찾을 수 없습니다: {args.onnx}")
        sys.exit(1)
        
    TASK_MAP = {
        "classification": Task.IMAGE_CLASSIFICATION,
        "detection": Task.OBJECT_DETECTION,
        "nlp_classification": Task.NLP_CLASSIFICATION,
        "nlp_generation": Task.NLP_GENERATION
    }
    task_enum = TASK_MAP.get(args.task, Task.IMAGE_CLASSIFICATION)
    
    # 0. DataLoader 공통 인터페이스 규약 및 CoC 해소 (Resolver)
    from src.utils.dataset_resolver import resolve_dataset_paths
    image_dir, label_path = resolve_dataset_paths(task_enum, args.dataset, args.image_dir, args.label_dir)
    
    loader_kwargs = {}
    if image_dir:
        loader_kwargs["image_dir"] = image_dir
    if label_path:
        loader_kwargs["label_path"] = label_path
    
    # 1. Spec & Artifact 생성
    try:
        spec = create_model_spec(args.model, args.onnx, task=task_enum)
    except Exception as e:
        print(f"[Error] ONNX 스펙 파싱 실패: {e}")
        sys.exit(1)
        
    compiled_model = CompiledModel(spec=spec, backend_name=args.backend, artifact_path=Path(args.onnx))
    
    # 2. 컴포넌트(주입 객체) 조립
    print(f"[Factory] Assembling components for {args.task}...")
    loader = create_dataloader(
        model_spec=spec,
        dataset_path=args.dataset,
        layout=args.layout,
        **loader_kwargs
    )
    
    # 런타임 팩토리 로직
    try:
        runtime = create_runtime(args.backend, device=args.device)
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)
        
    runtime.load(compiled_model)
    
    # 평가기 팩토리 로직
    evaluator = create_evaluator(spec, top_k=(1, 5))
    
    # 3. 오케스트레이터 구동
    runner = BenchmarkRunner(dataloader=loader, runtime=runtime, evaluator=evaluator)
    results = runner.run(warmup_runs=args.warmup, batch_size=args.batch_size)
    
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
