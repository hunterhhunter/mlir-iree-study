import onnxruntime as ort
import numpy as np
import time, tracemalloc, psutil, os
import torch

# ── Device & Session 설정 ──────────────────────────────────
model_path = "models/Kalray_resnet50/resnet50-v1-7s.onnx"

# GPU(CUDA) 우선 순위 설정
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider',
]

print(f"Loading ResNet50 model on GPU: {model_path}")
session = ort.InferenceSession(model_path, providers=providers)

# ── 입력 생성 ──────────────────────────────────────────────
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape # [1, 3, 224, 224]
inputs = {input_name: np.random.randn(*input_shape).astype(np.float32)}

# ── Warm-up ────────────────────────────────────────────────
print("Warming up...")
for _ in range(50):
    session.run(None, inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# ── 메모리 측정 ────────────────────────────────────────────
process = psutil.Process(os.getpid())
tracemalloc.start()
mem_before = process.memory_info().rss / 1024**2

session.run(None, inputs)
if torch.cuda.is_available():
    torch.cuda.synchronize()

_, heap_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
mem_delta = process.memory_info().rss / 1024**2 - mem_before

# GPU 메모리 (torch 활용하여 측정 가능 시)
gpu_mem_used = None
if torch.cuda.is_available():
    torch.cuda.synchronize()
    gpu_mem_used = torch.cuda.memory_reserved(0) / 1024**2

# ── Latency 분포 측정 ──────────────────────────────────────
N = 1000
latencies = []

for _ in range(N):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    session.run(None, inputs)
    latencies.append((time.perf_counter() - t0) * 1000)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
latencies = np.array(latencies)
total_sec = latencies.sum() / 1000

# ── 결과 출력 ──────────────────────────────────────────────
print(f"\n===== ResNet50 ONNX GPU Benchmark =====")
print(f"[Model]: {model_path}")
print(f"[Latency]")
print(f"  Mean:        {latencies.mean():.2f} ms")
print(f"  Std:         {latencies.std():.2f} ms")
print(f"  P50:         {np.percentile(latencies, 50):.2f} ms")
print(f"  P90:         {np.percentile(latencies, 90):.2f} ms")
print(f"  P99:         {np.percentile(latencies, 99):.2f} ms")
print(f"[Throughput]")
print(f"  QPS:         {N / total_sec:.1f} req/s")
print(f"[Memory]")
print(f"  RSS Delta:   {mem_delta:.2f} MB")
print(f"  Heap Peak:   {heap_peak / 1024**2:.2f} MB")
if gpu_mem_used is not None:
    print(f"  VRAM Resv:   {gpu_mem_used:.2f} MB")

# ── CSV 저장 ──────────────────────────────────────────────
import csv
csv_file = "benchmark_results.csv"
p50 = np.percentile(latencies, 50)
p90 = np.percentile(latencies, 90)
p99 = np.percentile(latencies, 99)
qps = N / total_sec

result_data = {
    "model": "resnet",
    "framework": "onnx",
    "target": "gpu",
    "opt_level": "",
    "latency_mean_ms": f"{latencies.mean():.2f}",
    "latency_p50_ms": f"{p50:.2f}",
    "latency_p90_ms": f"{p90:.2f}",
    "latency_p99_ms": f"{p99:.2f}",
    "throughput_qps": f"{qps:.1f}",
    "cpu_mem_rss_delta_mb": f"{mem_delta:.2f}",
    "cpu_mem_heap_peak_mb": f"{heap_peak / 1024**2:.2f}",
    "gpu_mem_vram_reserved_mb": "0.00"
}

file_exists = os.path.isfile(csv_file)
with open(csv_file, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=result_data.keys())
    if not file_exists:
        writer.writeheader()
    writer.writerow(result_data)

print(f"📊 결과가 {csv_file}에 저장되었습니다.")
