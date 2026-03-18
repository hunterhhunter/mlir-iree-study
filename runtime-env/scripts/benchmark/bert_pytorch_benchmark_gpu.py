import torch
import numpy as np
import time, tracemalloc, psutil, os
from transformers import BertModel
from tqdm import tqdm

# ── Device 설정 ──────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        try:
            # 실제 동작 가능한지 테스트
            test = torch.zeros(1).cuda()
            print(f"✅ CUDA GPU 사용: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        except Exception as e:
            print(f"⚠️  CUDA 로드 실패: {e}")
    print("⚠️  CPU fallback")
    return torch.device("cpu")

device = get_device()

# ── 모델 로드 ──────────────────────────────────────────────
print("Loading model...")
model = BertModel.from_pretrained("google-bert/bert-base-uncased")
model = model.to(device)
model.eval()

# FP16 (GPU일 때만)
use_fp16 = device.type == "cuda"
if use_fp16:
    model = model.half()

# ── 입력 생성 ──────────────────────────────────────────────
batch_size, seq_len = 1, 128

input_ids      = torch.randint(0, 30522, (batch_size, seq_len), dtype=torch.long).to(device)
attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long).to(device)
token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long).to(device)

# ── Warm-up ────────────────────────────────────────────────
print("Warming up (50 reps)...")
with torch.no_grad():
    for _ in range(50):
        model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if device.type == "cuda":
            torch.cuda.synchronize()

# ── 메모리 측정 ────────────────────────────────────────────
process = psutil.Process(os.getpid())
tracemalloc.start()
mem_before = process.memory_info().rss / 1024**2

with torch.no_grad():
    if device.type == "cuda":
        torch.cuda.synchronize()
    model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    if device.type == "cuda":
        torch.cuda.synchronize()

_, heap_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
mem_delta = process.memory_info().rss / 1024**2 - mem_before

# GPU 메모리
gpu_mem_used = 0
if device.type == "cuda":
    torch.cuda.synchronize()
    gpu_mem_used = torch.cuda.memory_reserved(0) / 1024**2

# ── Latency 분포 측정 ──────────────────────────────────────
N = 1000
latencies = []

print(f"Benchmarking ({N} reps)...")
with torch.no_grad():
    for _ in tqdm(range(N)):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        latencies.append((time.perf_counter() - t0) * 1000)
        if device.type == "cuda":
            torch.cuda.synchronize()
        

latencies = np.array(latencies)
total_sec = latencies.sum() / 1000

# ── 결과 출력 ──────────────────────────────────────────────
precision = "FP16" if use_fp16 else "FP32"
device_label = f"GPU (CUDA/{precision})" if device.type == "cuda" else "CPU (FP32)"

print(f"\n===== BERT PyTorch Benchmark [{device_label}] =====")
print(f"[Latency]")
print(f"  Mean:        {latencies.mean():.2f} ms")
print(f"  Std:         {latencies.std():.2f} ms")
print(f"  P50:         {np.percentile(latencies, 50):.2f} ms")
print(f"  P90:         {np.percentile(latencies, 90):.2f} ms")
print(f"  P99:         {np.percentile(latencies, 99):.2f} ms")
print(f"[Throughput]")
print(f"  QPS:         {N / total_sec:.1f} req/s")
print(f"  Tokens/sec:  {N * batch_size * seq_len / total_sec:.0f} tok/s")
print(f"[Memory - CPU]")
print(f"  RSS Delta:   {mem_delta:.2f} MB")
print(f"  Heap Peak:   {heap_peak / 1024**2:.2f} MB")
if gpu_mem_used > 0:
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
    "model": "bert",
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
    "gpu_mem_vram_reserved_mb": f"{gpu_mem_used:.2f}"
}

file_exists = os.path.isfile(csv_file)
with open(csv_file, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=result_data.keys())
    if not file_exists:
        writer.writeheader()
    writer.writerow(result_data)

print(f"📊 결과가 {csv_file}에 저장되었습니다.")
