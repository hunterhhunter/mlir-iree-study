import torch
import numpy as np
import time, tracemalloc, psutil, os
from tqdm import tqdm

# ── Device 설정 ──────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        try:
            test = torch.zeros(1).cuda()
            print(f"✅ CUDA GPU 사용: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        except Exception as e:
            print(f"⚠️  CUDA 로드 실패: {e}")
    print("⚠️  CPU fallback")
    return torch.device("cpu")

device = get_device()

# ── 모델 로드 ──────────────────────────────────────────────
print("Loading YOLOv5mu model...")
model_path = "models/yolov5m/yolov5mu.pt"

try:
    from ultralytics import YOLO
    # YOLOv8/v10 등 최신 라이브러리 형식을 지원하는 로더를 사용합니다.
    yolo_model = YOLO(model_path)
    model = yolo_model.model
except Exception as e:
    print(f"⚠️  Ultralytics loading failed: {e}")
    print("Attempting torch.hub fallback...")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

model = model.to(device)
model.eval()

# FP16 (GPU일 때만)
use_fp16 = device.type == "cuda"
if use_fp16:
    model = model.half()

# ── 입력 생성 ──────────────────────────────────────────────
batch_size = 1
input_shape = (batch_size, 3, 640, 640)
dummy_input = torch.randn(input_shape).to(device)

if use_fp16:
    dummy_input = dummy_input.half()

# ── Warm-up ────────────────────────────────────────────────
print("Warming up...")
with torch.no_grad():
    for _ in range(50):
        model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()

# ── 메모리 측정 ────────────────────────────────────────────
process = psutil.Process(os.getpid())
tracemalloc.start()
mem_before = process.memory_info().rss / 1024**2

with torch.no_grad():
    if device.type == "cuda":
        torch.cuda.synchronize()
    model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()

_, heap_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
mem_delta = process.memory_info().rss / 1024**2 - mem_before

# GPU 메모리
gpu_mem_used = None
if device.type == "cuda":
    torch.cuda.synchronize()
    gpu_mem_used = torch.cuda.memory_reserved(0) / 1024**2

# ── Latency 분포 측정 ──────────────────────────────────────
N = 500  # YOLO is heavier
latencies = []

print(f"Benchmarking ({N} reps)...")
with torch.no_grad():
    for _ in tqdm(range(N)):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(dummy_input)
        latencies.append((time.perf_counter() - t0) * 1000)
        if device.type == "cuda":
            torch.cuda.synchronize()
        

latencies = np.array(latencies)
total_sec = latencies.sum() / 1000

# ── 결과 출력 ──────────────────────────────────────────────
precision = "FP16" if use_fp16 else "FP32"
device_label = f"GPU (CUDA/{precision})" if device.type == "cuda" else "CPU (FP32)"

print(f"\n===== YOLOv5mu PyTorch Benchmark [{device_label}] =====")
print(f"[Model]: {model_path}")
print(f"[Latency]")
print(f"  Mean:        {latencies.mean():.2f} ms")
print(f"  Std:         {latencies.std():.2f} ms")
print(f"  P50:         {np.percentile(latencies, 50):.2f} ms")
print(f"  P90:         {np.percentile(latencies, 90):.2f} ms")
print(f"  P99:         {np.percentile(latencies, 99):.2f} ms")
print(f"[Throughput]")
print(f"  QPS:         {N / total_sec:.1f} req/s")
print(f"  Images/sec:  {N * batch_size / total_sec:.1f} img/s")
print(f"[Memory - CPU]")
print(f"  RSS Delta:   {mem_delta:.2f} MB")
print(f"  Heap Peak:   {heap_peak / 1024**2:.2f} MB")
if gpu_mem_used is not None:
    print(f"[Memory - GPU]")
    print(f"  VRAM Reserved: {gpu_mem_used:.2f} MB")

# ── CSV 저장 ──────────────────────────────────────────────
import csv
csv_file = "benchmark_results.csv"
p50 = np.percentile(latencies, 50)
p90 = np.percentile(latencies, 90)
p99 = np.percentile(latencies, 99)
qps = N / total_sec

result_data = {
    "model": "yolov5m",
    "framework": "pytorch",
    "target": "cuda",
    "opt_level": "",
    "latency_mean_ms": f"{latencies.mean():.2f}",
    "latency_p50_ms": f"{p50:.2f}",
    "latency_p90_ms": f"{p90:.2f}",
    "latency_p99_ms": f"{p99:.2f}",
    "throughput_qps": f"{qps:.1f}",
    "cpu_mem_rss_delta_mb": f"{mem_delta:.2f}",
    "cpu_mem_heap_peak_mb": f"{heap_peak / 1024**2:.2f}",
    "gpu_mem_vram_reserved_mb": f"{gpu_mem_used:.2f}" if gpu_mem_used is not None else "0.00"
}

file_exists = os.path.isfile(csv_file)
with open(csv_file, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=result_data.keys())
    if not file_exists:
        writer.writeheader()
    writer.writerow(result_data)

print(f"📊 결과가 {csv_file}에 저장되었습니다.")
