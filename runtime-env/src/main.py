import argparse
import sys
import os
import time
from tqdm import tqdm

# 프로젝트 루트를 경로에 추가 (패키지 임포트 보장)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.runtimes.iree_rt import IREERuntime
from src.dataloader import get_dataloader
from src.evaluator import get_evaluator

# 지원하는 런타임 레지스트리
RUNTIMES = {
    "iree": IREERuntime,
}

def main():
    parser = argparse.ArgumentParser(description="Multi-Model & Multi-Dataset Precision Evaluation Framework")
    
    # 1. 실행 환경 설정
    parser.add_argument("--runtime", type=str, choices=RUNTIMES.keys(), default="iree", help="Choose execution runtime")
    parser.add_argument("--model", type=str, required=True, help="Input model path (e.g., models/mobilenetv2-10.onnx)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="Target acceleration device")
    parser.add_argument("--batch_size", type=int, default=1, help="Simultaneous inference count (batch dimension for model input)")
    parser.add_argument("--data_size", type=int, default=None, help="Total number of samples to evaluate (evaluates all if not set)")
    
    # 2. 데이터셋 설정
    parser.add_argument("--dataset", type=str, default="ILSVRC_imagenet-1k", help="Hugging Face dataset name")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split (validation/test)")
    parser.add_argument("--task", type=str, default="classification", help="Evaluation task type")
    
    # 3. 모델 사양 (ImageProcessor 로딩용)
    parser.add_argument("--model_id", type=str, default="google/mobilenet_v2_1.0_224", help="Hugging Face model ID for preprocessing")
    
    # 4. 실행 모드
    parser.add_argument("--eval", action="store_true", help="Enable precision evaluation loop")
    parser.add_argument("--profile", action="store_true", help="Enable hardware profiling/benchmarking")
    
    args = parser.parse_args()

    # [Phase 1: Runtime Setup & Compilation]
    print(f"\n🚀 Initializing Runtime: {args.runtime} (Device: {args.device})")
    runtime_cls = RUNTIMES.get(args.runtime)
    rt = runtime_cls(args.model, device=args.device)
    
    print(f"[*] Starting model compilation (Batch size: {args.batch_size})...")
    rt.compile()

    # [Phase 2: Execution]
    if args.eval:
        print(f"\n--- [Evaluation Mode: {args.dataset} ({args.split})] ---")
        
        # 신규 고도화된 데이터 로더 팩토리 사용
        try:
            loader = get_dataloader(
                dataset_name=args.dataset,
                split=args.split,
                model_id=args.model_id,
                batch_size=args.batch_size,
                data_size=args.data_size,  # 추가된 파라미터 연동
                task=args.task
            )
        except Exception as e:
            print(f"[!] Dataloader initialization failed: {e}")
            sys.exit(1)
        
        # 범용 에밸루에이터 팩토리 사용
        evaluator = get_evaluator(task=args.task, top_k=(1, 5))
        
        # IREE 런타임 초기화 (컨텍스트 로드)
        rt.initialize_runtime()
        
        start_time = time.time()
        print(f"[*] Inference loop started (Total samples: {loader.get_total_samples()})...")
        
        # 3. Inference Loop
        for pixel_values, labels, filenames in tqdm(loader, desc="Inferencing", unit="batch"):
            # IREE 추론 실행 (NumPy input -> NumPy output)
            logits = rt.execute(pixel_values)
            
            # 성능 지표 업데이트
            evaluator.update(logits, labels)
            
        total_time = time.time() - start_time
        
        # 4. Result Reporting
        evaluator.report()
        
        results = evaluator.compute()
        total_samples = int(results.get('Total Samples', 0))
        fps = total_samples / total_time if total_time > 0 else 0
        
        print(f"  Total Duration    : {total_time:.2f}s")
        print(f"  Average Latency   : {(total_time / total_samples * 1000):.2f}ms/sample" if total_samples > 0 else "  Average Latency   : N/A")
        print(f"  Throughput (FPS)  : {fps:.2f}")
        print("═"*50 + "\n")

    elif args.profile:
        print(f"\n--- [Profiling Mode] ---")
        rt.profile()
    
    else:
        print("[!] No mode selected. Use --eval for precision testing.")

if __name__ == "__main__":
    main()
