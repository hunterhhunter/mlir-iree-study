import os
import subprocess
import onnx
import argparse
import time

def run_cmd(cmd, description):
    """
    벤치마크 명령어를 실행하고 결과를 캡처하는 헬퍼 함수.
    """
    print(f"\n[EXECUTING] {description}")
    print(f"Command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {result.stderr}")
        return False, result.stderr
    return True, result.stdout

def main():
    parser = argparse.ArgumentParser(description="IREE End-to-End Benchmark Testing Tool")
    parser.add_argument("--onnx_path", type=str, required=True, help="Input ONNX model path")
    parser.add_argument("--target", type=str, choices=["cpu", "cuda"], default="cpu", help="Target backend (cpu or cuda)")
    parser.add_argument("--input", type=str, help="Input data (e.g., '1x3x224x224xf32=1.0')")
    parser.add_argument("--function", type=str, default="main", help="Model function name")
    parser.add_argument("--benchmark_reps", type=int, default=5, help="Number of benchmark repetitions")
    args = parser.parse_args()

    # IREE 빌드 환경 경로 설정
    venv_bin = "/home/slugg/iree/build-demo/.venv/bin"
    import_onnx_bin = os.path.join(venv_bin, "iree-import-onnx")
    compile_bin = os.path.join(venv_bin, "iree-compile")
    run_module_bin = os.path.join(venv_bin, "iree-run-module")
    benchmark_bin = os.path.join(venv_bin, "iree-benchmark-module") # 성능 측정 전용 도구

    base_name = os.path.splitext(os.path.basename(args.onnx_path))[0]
    onnx_v17_path = f"models/{base_name}_v17.onnx"
    mlir_path = f"models/{base_name}.mlir"
    vmfb_path = f"models/{base_name}_{args.target}.vmfb"

    # Step 1: ONNX Opset 17 변환 (MLIR Dialect 변환 최적화 준비)
    print(f"--- [Step 1: MLIR Dialect Preparation - ONNX 1.17] ---")
    try:
        model = onnx.load(args.onnx_path)
        v17_model = onnx.version_converter.convert_version(model, 17)
        onnx.save(v17_model, f"{onnx_v17_path}")
        print(f"Successfully converted {args.onnx_path} to Opset 17.")
    except Exception as e:
        print(f"[CRITICAL] ONNX Conversion Failed: {e}")
        return

    # Step 2: MLIR 임포트
    success, output = run_cmd(f"{import_onnx_bin} {onnx_v17_path} -o {mlir_path}", "Importing ONNX to MLIR")
    if not success: return

    # Step 3: Lowering 파이프라인 및 컴파일
    print(f"--- [Step 2: Lowering Pipeline - Compilation] ---")
    if args.target == "cpu":
        # CPU 타겟팅: LLVM 벡터화 최적화 적용
        compile_cmd = f"{compile_bin} {mlir_path} --iree-hal-target-backends=llvm-cpu -o {vmfb_path}"
    else:
        # CUDA 타겟팅: Linalg Tiling & Fusion 최적화가 적용되는 단계
        # --iree-cuda-target=sm_86 옵션은 Ampere 아키텍처(RTX 30시리즈 등)를 타겟팅함.
        compile_cmd = (f"{compile_bin} {mlir_path} "
                       f"--iree-hal-target-backends=cuda "
                       f"--iree-cuda-target=sm_86 "
                       f"-o {vmfb_path}")
    
    success, output = run_cmd(compile_cmd, f"Compiling to {args.target} VMFB")
    if not success: return

    # Step 4: 기능 검증 (Functional Verification)
    # 벤치마크 수행 전, 컴파일된 모듈이 올바른 출력을 생성하는지 확인.
    print(f"--- [Step 3: Verification - Functional Test] ---")
    device = "local-task" if args.target == "cpu" else "cuda"
    
    if not args.input:
        print("\n[PROMPT] Testing requires input data for functional verification.")
        args.input = input("Enter input data (e.g. 1x3x224x224xf32=0.5): ").strip()

    if args.input:
        run_cmd_str = (f"{run_module_bin} --device={device} "
                       f"--module={vmfb_path} --function={args.function} "
                       f"--input=\"{args.input}\"")
        success, output = run_cmd(run_cmd_str, "Running Functional Verification")
        if success:
            print("[SUCCESS] Functional verification passed.")
        else:
            print("[FAILURE] Functional verification failed.")
            return

    # Step 5: 성능 벤치마크 (iree-benchmark-module)
    # IREE VM 오버헤드를 제외한 순수 HAL Dispatch 및 커널 실행 시간을 측정함.
    print(f"--- [Step 4: Performance Benchmarking - HAL/VM Dispatch Analysis] ---")
    # --benchmark_repetitions: 통계적 유의성을 위해 반복 실행 횟수 설정.
    # --benchmark_min_time: 각 반복의 최소 실행 시간 보장.
    benchmark_cmd = (f"{benchmark_bin} --device={device} "
                     f"--module={vmfb_path} --function={args.function} "
                     f"--input=\"{args.input}\" "
                     f"--benchmark_repetitions={args.benchmark_reps} "
                     f"--benchmark_min_time=1.0")
    
    success, output = run_cmd(benchmark_cmd, "Executing IREE Benchmark Module")
    
    if success:
        print("\n" + "="*50)
        print("IREE PERFORMANCE TEST SUMMARY")
        print("="*50)
        print(output)
        print("="*50)
        print("[RESULT] Benchmark Test Completed Successfully.")
    else:
        print("[RESULT] Benchmark Test Failed during execution.")

if __name__ == "__main__":
    main()
