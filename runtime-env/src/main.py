import os
import sys
import argparse
import onnx
from pathlib import Path

# 프로젝트 루트 경로 추가 (sys.path)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.core.model_spec import Model_Spec, Task
from src.core.compiled_model import CompiledModel
from src.core.benchmarkrunner import BenchmarkRunner

# 구체화된 컴포넌트 임포트
from src.dataloader.image_classification_loader import ImageClassificationLoader
# 사용자가 resnet50_evaluator.py를 image_classification_loader.py로 리네임하였음
from src.evaluators.image_classification_loader import ResNet50Evaluator  
from src.runtimes.onnx_rt import OnnxRuntime
# from src.runtimes.iree_rt import IREERuntime  # 향후 IREE 백엔드 추가 시 주석 해제

def _parse_onnx_io_names(onnx_path):
    """지정된 ONNX 모델을 로드하여 Input/Output 텐서 이름을 자동 추출합니다."""
    model = onnx.load(onnx_path)
    input_name = model.graph.input[0].name
    output_name = model.graph.output[0].name
    return input_name, output_name

def create_model_spec(model_name: str, onnx_path: str, task: Task = Task.IMAGE_CLASSIFICATION):
    """ONNX 바이너리를 분석하여 동적으로 모델 명세서(Spec)를 생성하는 Factory 함수"""
    input_n, output_n = _parse_onnx_io_names(onnx_path)
    print(f"[Factory] Detected ONNX I/O -> Input: '{input_n}', Output: '{output_n}'")
    
    # 기본 형태는 (1, 3, 224, 224)로 잡되 향후 고도화 가능
    spec = Model_Spec(
        name=model_name,
        task=task,
        input_shapes={input_n: (1, 3, 224, 224)},
        input_dtype={input_n: "float32"},
        output_shapes={output_n: (1, 1000)},  # Evaluator가 1001 예외를 0~999 범용으로 처리함
        model_paths={"onnx": onnx_path}
    )
    return spec

def main():
    parser = argparse.ArgumentParser(description="Unified BenchmarkRunner CLI Orchestrator")
    parser.add_argument("--model", type=str, required=True, help="모델 이름 (예: resnet50, mobilenet_v2)")
    parser.add_argument("--onnx", type=str, required=True, help="ONNX 파일의 절대 또는 상대 경로")
    parser.add_argument("--dataset", type=str, required=True, help="평가용 데이터셋 최상위 디렉토리 (예: datasets/imagenet_1k)")
    parser.add_argument("--backend", type=str, default="onnxruntime", choices=["onnxruntime", "iree"], help="추론을 실행할 백엔드 (기본: onnxruntime)")
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="추론 배치 사이즈 (기본: 1)")
    parser.add_argument("--warmup", "-w", type=int, default=2, help="웜업 횟수 (기본: 2)")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print(f" BenchmarkRunner CLI - Project: Antigravity ")
    print(f"   Model: {args.model} | Backend: {args.backend}")
    print("="*60)
    
    if not os.path.exists(args.onnx):
        print(f"[Error] 모델 파일을 찾을 수 없습니다: {args.onnx}")
        sys.exit(1)
        
    dataset_val_dir = os.path.join(args.dataset, "val")
    dataset_label_file = os.path.join(args.dataset, "val_labels.txt")
    
    if not os.path.exists(dataset_val_dir) or not os.path.exists(dataset_label_file):
        print(f"[Warn] 지정된 데이터셋 경로에 'val' 폴더 또는 'val_labels.txt'가 없습니다: {args.dataset}")
    
    # 1. Spec & Artifact 생성
    try:
        spec = create_model_spec(args.model, args.onnx)
    except Exception as e:
        print(f"[Error] ONNX 스펙 파싱 실패: {e}")
        sys.exit(1)
        
    compiled_model = CompiledModel(spec=spec, backend_name=args.backend, artifact_path=Path(args.onnx))
    
    # 2. 컴포넌트(주입 객체) 조립
    print(f"[Factory] Assembling components...")
    loader = ImageClassificationLoader(
        model_spec=spec,
        dataset_path=args.dataset,
        image_dir=dataset_val_dir,
        label_file=dataset_label_file
    )
    
    # 런타임 팩토리 로직 분기
    if args.backend.lower() == "onnxruntime":
        runtime = OnnxRuntime(device="cpu")
    else:
        print(f"[Error] '{args.backend}' 백엔드는 현재 CLI 연동이 준비 중입니다.")
        sys.exit(1)
        
    runtime.load(compiled_model)
    
    evaluator = ResNet50Evaluator(top_k=(1, 5))
    
    # 3. 오케스트레이터 구동
    runner = BenchmarkRunner(dataloader=loader, runtime=runtime, evaluator=evaluator)
    results = runner.run(warmup_runs=args.warmup, batch_size=args.batch_size)
    
    # 4. 최종 결과 리포팅
    print("\n" + "="*40)
    print(f" Final Metrics ({args.model}) ")
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
