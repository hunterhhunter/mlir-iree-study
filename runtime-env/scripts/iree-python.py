import os
import argparse
import time
import numpy as np
import onnx
import subprocess
import iree.compiler.tools as ireec
import iree.runtime as ireert

def parse_input(input_str):
    """Parses input string like '1x3x224x224xf32=0.5' into a numpy array."""
    try:
        shape_dtype, val = input_str.split('=')
        parts = shape_dtype.split('x')
        dtype_str = parts[-1]
        shape = [int(p) for p in parts[:-1]]
        
        mapping = {'f32': np.float32, 'f16': np.float16, 'i32': np.int32, 'i64': np.int64}
        dtype = mapping.get(dtype_str, np.float32)
        
        return np.full(shape, float(val), dtype=dtype)
    except Exception as e:
        print(f"[ERROR] Failed to parse input string: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="IREE Python Native Pipeline")
    parser.add_argument("--onnx_path", type=str, required=True, help="Input ONNX model path")
    parser.add_argument("--target", type=str, choices=["cpu", "cuda"], default="cpu", help="Target backend")
    parser.add_argument("--input", type=str, help="Input data (e.g., '1x3x224x224xf32=0.5')")
    parser.add_argument("--function", type=str, default="main", help="Model function name")
    parser.add_argument("--benchmark_reps", type=int, default=5, help="Number of benchmark repetitions")
    args = parser.parse_args()

    venv_bin = "/home/slugg/iree/build-demo/.venv/bin"
    import_onnx_bin = os.path.join(venv_bin, "iree-import-onnx")
    
    base_name = os.path.splitext(os.path.basename(args.onnx_path))[0]
    onnx_v17_path = f"{base_name}_v17.onnx"
    mlir_path = f"{base_name}.mlir"
    vmfb_path = f"{base_name}_{args.target}.vmfb"

    # Step 1: ONNX Opset 17 변환 (Lowering Pattern 오류 방지)
    print(f"\n--- [Step 1: MLIR Dialect Preparation] ---")
    model = onnx.load(args.onnx_path)
    v17_model = onnx.version_converter.convert_version(model, 17)
    onnx.save(v17_model, onnx_v17_path)
    print(f"ONNX Opset 17 converted: {onnx_v17_path}")

    # Step 2: Import to MLIR
    print(f"\n--- [Step 2: Import to MLIR Dialect] ---")
    subprocess.run([import_onnx_bin, onnx_v17_path, "-o", mlir_path], check=True)
    print(f"MLIR generated: {mlir_path}")

    # Step 3: Lowering Pipeline & Compilation
    print(f"\n--- [Step 3: Lowering Pipeline - Compilation] ---")
    target_backend = "cuda" if args.target == "cuda" else "llvm-cpu"
    extra_args = ["--iree-cuda-target=sm_86"] if args.target == "cuda" else []
    
    # 이 단계에서 Linalg Tiling & Fusion 등 핵심 최적화 Pass가 실행됨
    start_time = time.time()
    ireec.compile_file(
        mlir_path,
        output_file=vmfb_path,
        target_backends=[target_backend],
        extra_args=extra_args
    )
    print(f"Compilation to {args.target} VMFB completed in {time.time() - start_time:.2f}s")

    # Step 4: Runtime Execution (IREE HAL/VM Dispatch)
    print(f"\n--- [Step 4: Runtime/Compiler Boundary - Execution] ---")
    driver_name = "cuda" if args.target == "cuda" else "local-task"
    
    try:
        device_uri = f"{driver_name}://0" if driver_name == "cuda" else driver_name
        config = ireert.Config(device_uri)
        print(f"Successfully initialized HAL Device: {device_uri}")
    except Exception as e:
        print(f"[CRITICAL] Device Creation Failed: {e}")
        # 시스템에 등록된 모든 HAL 드라이버와 장치를 쿼리
        print("Available IREE Drivers:", ireert.query_available_drivers())
        # 환경 변수 확인 권고
        print("Hint: Check if 'CUDA_VISIBLE_DEVICES' is set and 'nvidia-smi' is working.")
        raise

    
    with open(vmfb_path, "rb") as f:
        vmfb_content = f.read()
    
    # VM Module 로드 및 HAL 장치 바인딩
    vm_module = ireert.VmModule.from_flatbuffer(config.vm_instance, vmfb_content)
    context = ireert.SystemContext(config=config)
    context.add_vm_module(vm_module)
    
    # 입력 데이터 준비
    if not args.input:
        args.input = input("Enter input data (e.g. 1x3x224x224xf32=0.5): ").strip()
    
    input_data = parse_input(args.input)
    if input_data is None: return

    # Function Discovery (Module Name은 보통 'module')
    bound_module = context.modules.module
    func = getattr(bound_module, args.function)

    # Inference (Warm-up)
    print("Performing warm-up run...")
    result = func(input_data)
    print("Warm-up complete.")

    # Step 5: Performance Benchmarking
    print(f"\n--- [Step 5: Performance Benchmarking (Reps: {args.benchmark_reps})] ---")
    latencies = []
    for i in range(args.benchmark_reps):
        start = time.perf_counter()
        _ = func(input_data)
        latencies.append(time.perf_counter() - start)
    
    avg_latency = sum(latencies) / len(latencies) * 1000
    print(f"\n[BENCHMARK RESULT]")
    print(f"Average Latency: {avg_latency:.4f} ms")
    print(f"Min Latency: {min(latencies)*1000:.4f} ms")
    print(f"Max Latency: {max(latencies)*1000:.4f} ms")

if __name__ == "__main__":
    main()
