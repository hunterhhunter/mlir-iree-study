import os
import time
import numpy as np
import onnx
import subprocess
import iree.compiler.tools as ireec
import json
import argparse
import csv
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────
MODELS = {
    "resnet50": {
        "path": "models/Kalray_resnet50/resnet50-v1-7s.onnx",
        "inputs": {"data": (1, 3, 224, 224, np.float32)},
        "entry_point": "mxnet_converted_model"
    },
    "yolov5m": {
        "path": "models/yolov5m/yolov5mu.onnx",
        "inputs": {"images": (1, 3, 640, 640, np.float32)},
        "entry_point": "main_graph"
    },
    "bert": {
        "path": "models/google-bert_bert-base-uncased/model.onnx",
        "inputs": {
            "input_ids": (1, 128, np.int64),
            "attention_mask": (1, 128, np.int64),
            "token_type_ids": (1, 128, np.int64)
        },
        "entry_point": "torch_jit"
    }
}

TARGETS = ["cpu", "cuda"]
OPT_LEVELS = ["0", "1", "2", "3"] # O0, O1, O2, O3
BENCHMARK_REPS = 10
VENV_BIN = os.path.join(os.getcwd(), ".venv/bin")
IMPORT_ONNX_BIN = os.path.join(VENV_BIN, "iree-import-onnx")
BENCHMARK_BIN = os.path.join(VENV_BIN, "iree-benchmark-module")

def get_paths(model_name, target, opt_level):
    model_cfg = MODELS[model_name]
    onnx_path = model_cfg["path"]
    base_name = os.path.splitext(os.path.basename(onnx_path))[0]
    
    onnx_v17_path = f"{base_name}_v17.onnx"
    mlir_path = f"{base_name}_v17.mlir"
    vmfb_path = f"{base_name}_{target}_O{opt_level}.vmfb"
    
    return onnx_path, onnx_v17_path, mlir_path, vmfb_path

def compile_model(model_name, target, opt_level):
    onnx_path, onnx_v17_path, mlir_path, vmfb_path = get_paths(model_name, target, opt_level)
    
    print(f"\n🛠️ Preparing {model_name.upper()} for {target.upper()} (O{opt_level})...")

    # Step 1: ONNX Opset 17 Conversion
    current_onnx_path = onnx_path
    if not os.path.exists(onnx_v17_path):
        try:
            print(f"--- [Step 1: ONNX Opset 17 Conversion] ---")
            original_model = onnx.load(onnx_path)
            converted_model = onnx.version_converter.convert_version(original_model, 17)
            onnx.save(converted_model, onnx_v17_path)
            print(f"Saved: {onnx_v17_path}")
            current_onnx_path = onnx_v17_path
        except Exception as e:
            print(f"⚠️ ONNX Version Conversion Failed: {e}. Using original.")

    # Step 2: Import to MLIR
    if not os.path.exists(mlir_path):
        print(f"--- [Step 2: Import to MLIR Dialect] ---")
        subprocess.run([IMPORT_ONNX_BIN, current_onnx_path, "-o", mlir_path], check=True)
        print(f"MLIR generated: {mlir_path}")

    # Step 3: Compile to VMFB
    print(f"--- [Step 3: Compilation to VMFB] ---")
    target_backend = "cuda" if target == "cuda" else "llvm-cpu"
    extra_args = [f"--iree-opt-level=O{opt_level}"] 
    if target == "cuda":
        extra_args.append("--iree-cuda-target=sm_86")
    else:
        extra_args.append("--iree-llvmcpu-target-cpu=host")
        extra_args.append("--iree-llvmcpu-stack-allocation-limit=262144")
    
    start_time = time.time()
    ireec.compile_file(
        mlir_path,
        output_file=vmfb_path,
        target_backends=[target_backend],
        extra_args=extra_args
    )
    print(f"✅ Compiled in {time.time() - start_time:.2f}s -> {vmfb_path}")

def dtype_to_iree(dtype):
    if dtype == np.float32: return "f32"
    if dtype == np.float64: return "f64"
    if dtype == np.int32: return "i32"
    if dtype == np.int64: return "i64"
    return str(dtype)

def run_benchmark(model_name, target, opt_level):
    model_cfg = MODELS[model_name]
    _, _, _, vmfb_path = get_paths(model_name, target, opt_level)
    
    if not os.path.exists(vmfb_path):
        print(f"❌ VMFB file not found: {vmfb_path}")
        return None

    print(f"\n🚀 Running CLI Benchmark: {model_name.upper()} | {target.upper()} | O{opt_level}")

    device = "cuda" if target == "cuda" else "local-task"
    
    # Input String 생성 (예: 1x3x224x224xf32)
    input_args = []
    for name, input_info in model_cfg["inputs"].items():
        *shape, dtype = input_info
        shape_str = "x".join(map(str, shape))
        type_str = dtype_to_iree(dtype)
        input_args.append(f"--input={shape_str}x{type_str}")

    cmd = [
        BENCHMARK_BIN,
        f"--device={device}",
        f"--module={vmfb_path}",
        f"--function={model_cfg['entry_point']}",
        *input_args,
        f"--benchmark_repetitions={BENCHMARK_REPS}",
        "--benchmark_format=json"
    ]

    # WSL2 CUDA 지원을 위한 환경 변수
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"/usr/lib/wsl/lib:{env.get('LD_LIBRARY_PATH', '')}"

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
        bench_data = json.loads(result.stdout)
        
        # 결과 파싱 (real_time_mean, real_time_median 등)
        # Google Benchmark JSON 구조에서 "real_time" 통계를 찾음
        mean_latency = 0
        median_latency = 0
        qps = 0
        
        for run in bench_data.get("benchmarks", []):
            if "real_time_mean" in run["name"]:
                mean_latency = run["real_time"]
                # items_per_second가 있으면 QPS로 사용
                qps = run.get("items_per_second", 0)
            elif "real_time_median" in run["name"]:
                median_latency = run["real_time"]

        # 만약 qps가 0이면 계산 (mean_latency는 ms 단위)
        if qps == 0 and mean_latency > 0:
            qps = 1000.0 / mean_latency

        print(f"  [RESULT] Mean: {mean_latency:.2f}ms | Median: {median_latency:.2f}ms | QPS: {qps:.1f}")

        return {
            "model": model_name,
            "framework": "IREE",
            "target": target,
            "opt_level": f"O{opt_level}",
            "latency_mean_ms": f"{mean_latency:.2f}",
            "latency_p50_ms": f"{median_latency:.2f}",
            "latency_p90_ms": "N/A", # CLI 기본 출력에는 p90 없음
            "latency_p99_ms": "N/A",
            "throughput_qps": f"{qps:.1f}",
            "cpu_mem_rss_delta_mb": "0.00", # CLI 모드에선 정확한 측정이 어려워 0으로 설정
            "cpu_mem_heap_peak_mb": "0.00",
            "gpu_mem_vram_reserved_mb": "0.00"
        }
    except Exception as e:
        print(f"❌ CLI Benchmark Failed: {e}")
        if hasattr(e, 'stderr'): print(e.stderr)
        return None

def save_to_csv(results, filename="benchmark_results.csv"):
    if not results: return
    
    # 기존 파일이 있으면 로드해서 합치기 (onnx, pytorch 데이터 보존을 위해)
    existing_data = []
    if os.path.exists(filename):
        with open(filename, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # IREE 데이터는 새로 측정할 것이므로 제외하고 로드
                if row["framework"] != "IREE":
                    existing_data.append(row)
    
    all_data = existing_data + results
    keys = all_data[0].keys()
    
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_data)
    print(f"\n📊 Results updated in {filename} (IREE rows replaced).")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["compile", "run", "all"], default="all")
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all")
    parser.add_argument("--target", choices=["cpu", "cuda", "all"], default="all")
    args = parser.parse_args()

    selected_models = list(MODELS.keys()) if args.model == "all" else [args.model]
    selected_targets = TARGETS if args.target == "all" else [args.target]
    all_results = []

    if args.mode in ["compile", "all"]:
        for model in selected_models:
            for target in selected_targets:
                for opt in OPT_LEVELS:
                    compile_model(model, target, opt)

    if args.mode in ["run", "all"]:
        for model in selected_models:
            for target in selected_targets:
                for opt in OPT_LEVELS:
                    res = run_benchmark(model, target, opt)
                    if res: all_results.append(res)
        
        save_to_csv(all_results)

if __name__ == "__main__":
    main()
