import argparse
import sys
import os
import time
from tqdm import tqdm

# src 경로 추가 (런타임 임포트용)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from runtimes.iree_rt import IREERuntime
from data_loader import CustomDataLoader
from evaluators.MobileNet_evaluator import MobileNetEvaluator

# 런타임 매핑
RUNTIMES = {
    "iree": IREERuntime,
}

def main():
    parser = argparse.ArgumentParser(description="MobileNetV2 Precision Evaluation Framework")
    parser.add_argument("--runtime", type=str, choices=RUNTIMES.keys(), default="iree", help="Choose runtime")
    parser.add_argument("--model", type=str, required=True, help="Input model path (e.g., models/mobilenetv2-10.onnx)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="Target device")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--eval", action="store_true", help="Enable precision evaluation on ImageNet-1K")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    
    args = parser.parse_args()

    # 1. 런타임 초기화 및 컴파일
    runtime_cls = RUNTIMES.get(args.runtime)
    rt = runtime_cls(args.model, device=args.device)
    
    print(f"\n--- [Phase 1: Compilation] ---")
    rt.compile()

    # 2. 평가 모드 (ImageNet-1K 전체 루프 실행)
    if args.eval:
        print(f"\n--- [Phase 2: Precision Evaluation] ---")
        loader = CustomDataLoader(batch_size=args.batch_size)
        evaluator = MobileNetEvaluator(top_k=(1, 5))
        
        # 런타임 컨텍스트 사전 초기화 (루프 밖에서 1회)
        rt.initialize_runtime()
        
        start_time = time.time()
        for batch_idx, (pixel_values, labels, filenames) in enumerate(tqdm(loader, desc="Evaluating")):
            # 추론 실행 (컨텍스트 재사용으로 매우 빠름)
            logits = rt.execute(pixel_values)
            
            # 정확도 업데이트
            evaluator.update(logits, labels)
            
        total_time = time.time() - start_time
        
        # 최종 리포트 출력
        evaluator.report()
        print(f"  Total Duration    : {total_time:.2f}s")
        print(f"  Average FPS       : {evaluator.total_samples / total_time:.2f}")
        print("═"*45 + "\n")

    # 3. 단일 추론/프로파일링 모드 (기존 기능 유지)
    elif args.profile:
        print(f"\n--- [Phase 2: Profiling] ---")
        rt.profile()
    
    else:
        print("[!] No mode selected. Use --eval for precision testing or --profile for benchmarking.")

if __name__ == "__main__":
    main()
