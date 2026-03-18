import torch
import numpy as np
import time, tracemalloc, psutil, os
from tqdm import tqdm

# ── Device 설정 ──────────────────────────────────────────
device = torch.device("cpu")
print(f"✅ 사용 장치: {device}")

# ── 모델 로드 ──────────────────────────────────────────────
print("Loading YOLOv5m model...")
# device 인자를 직접 전달하여 초기 로드부터 CPU로 고정합니다.
model = torch.hub.load('ultralytics/yolov5', 'yolov5mu', pretrained=True, device='cpu')
model.eval()

# ── 입력 생성 ──────────────────────────────────────────────
batch_size = 1
img_size = 640
img = torch.randn(batch_size, 3, img_size, img_size).to(device)

# ── Warm-up ────────────────────────────────────────────────
print("Warming up (50 reps)...")
with torch.no_grad():
    for _ in range(50):
        model(img)

# ── 메모리 및 Latency 측정 ────────────────────────────────────
process = psutil.Process(os.getpid())
tracemalloc.start()
mem_before = process.memory_info().rss / 1024**2

N = 500
latencies = []

print(f"Benchmarking ({N} reps)...")
with torch.no_grad():
    for _ in tqdm(range(N)):
        t0 = time.perf_counter()
        model(img)
        latencies.append((time.perf_counter() - t0) * 1000)

_, heap_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
mem_delta = (process.memory_info().rss / 1024**2) - mem_before

latencies = np.array(latencies)
total_sec = latencies.sum() / 1000

# ── 결과 출력 ──────────────────────────────────────────────
print(f"\n===== YOLOv5m PyTorch Benchmark [CPU] =====")
print(f"[Latency]")
print(f"  Mean:        {latencies.mean():.2f} ms")
print(f"  P50:         {np.percentile(latencies, 50):.2f} ms")
print(f"  P99:         {np.percentile(latencies, 99):.2f} ms")
print(f"[Throughput]")
print(f"  QPS:         {N / total_sec:.1f} req/s")
print(f"[Memory]")
print(f"  RSS Delta:   {mem_delta:.2f} MB")
print(f"  Heap Peak:   {heap_peak / 1024**2:.2f} MB")

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
    "target": "cpu",
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
