"""
모든 ONNX 모델을 한 번에 실행하는 통합 벤치마크 스크립트.

모델 파일 또는 데이터셋이 없으면 해당 항목을 자동으로 건너뜁니다.

사용법:
    python tests/run_all_onnx_benchmarks.py
    python tests/run_all_onnx_benchmarks.py --device cuda
    python tests/run_all_onnx_benchmarks.py --warmup 5 --max-steps 50
    python tests/run_all_onnx_benchmarks.py --models resnet50 yolov5m llama-3.2-3b
"""

import subprocess
import sys
import argparse
import re
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─── 벤치마크 설정 레지스트리 ────────────────────────────────────────────────
# required_files: 실행 전 존재 여부를 확인할 파일/디렉토리 목록 (없으면 자동 스킵)
BENCHMARK_CONFIGS = [
    {
        "label": "ResNet50 (IMAGE_CLASSIFICATION)",
        "model": "resnet50",
        "onnx": "models/resnet-50/resnet50.onnx",
        "dataset": "datasets/imagenet_1k",
        "extra_args": [],
        "required_files": [
            "models/resnet-50/resnet50.onnx",
            "datasets/imagenet_1k/val",
        ],
        "skip_hint": "python datasets/download_imagenet_1k.py",
    },
    {
        "label": "YOLOv5m (OBJECT_DETECTION)",
        "model": "yolov5m",
        "onnx": "models/yolov5m/yolov5m.onnx",
        "dataset": "datasets/coco128",
        "extra_args": [],
        "required_files": [
            "models/yolov5m/yolov5m.onnx",
            "datasets/coco128/images/train2017",
            "datasets/coco128/labels/train2017",
        ],
        "skip_hint": "python models/download_yolov5m.py",
    },
    {
        "label": "BERT-base-uncased (NLP_CLASSIFICATION)",
        "model": "bert-base-uncased",
        "onnx": "models/google-bert_bert-base-uncased/model.onnx",
        "dataset": "datasets/sst2_numpy",
        "extra_args": [],
        "required_files": [
            "models/google-bert_bert-base-uncased/model.onnx",
            "datasets/sst2_numpy/input_ids.npy",
            "datasets/sst2_numpy/labels.npy",
        ],
        "skip_hint": "python datasets/tokenize_sst2_to_numpy.py  # SST-2 baked numpy 없음",
    },
    {
        "label": "BERT-base-uncased SQuAD (QUESTION_ANSWERING)",
        "model": "bert-base-uncased-squad-v1",
        "onnx": "models/bert-base-uncased-squad-v1/squad.onnx",
        "dataset": "datasets/squad_numpy",
        "extra_args": [],
        "required_files": [
            "models/bert-base-uncased-squad-v1/squad.onnx",
            "datasets/squad_numpy/input_ids.npy",
            "datasets/squad_numpy/start_positions.npy",
        ],
        "skip_hint": "python datasets/download_squad2.py && python datasets/prepare_squad_numpy.py",
    },
    {
        "label": "PatchTST-FM-R1 (TIME_SERIES_FORECASTING)",
        "model": "patchtst-fm-r1",
        "onnx": "models/ibm-research_patchtst-fm-r1-ONNX/model.onnx",
        "dataset": "datasets/ettm1/ETTm1.csv",
        "extra_args": [],
        "required_files": [
            "models/ibm-research_patchtst-fm-r1-ONNX/model.onnx",
            "datasets/ettm1/ETTm1.csv",
        ],
        "skip_hint": (
            "wget -P datasets/ettm1/ "
            "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv"
        ),
    },
    {
        "label": "Llama-3.2-3B (NLP_GENERATION)",
        "model": "llama-3.2-3b",
        "onnx": "models/meta-llama_Llama-3.2-3B-ONNX/model.onnx",
        "dataset": "datasets/squad2",
        "tokenizer_path": "models/meta-llama_Llama-3.2-3B",
        "extra_args": ["--max-new-tokens", "64"],
        "required_files": [
            "models/meta-llama_Llama-3.2-3B-ONNX/model.onnx",
            "datasets/squad2/val.json",
            "models/meta-llama_Llama-3.2-3B/tokenizer.json",
        ],
        "skip_hint": "python models/download_hf_model.py --model meta-llama/Llama-3.2-3B",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="모든 ONNX 모델 통합 벤치마크 실행기",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="추론 장치 (기본: cpu)",
    )
    parser.add_argument(
        "--warmup", "-w", type=int, default=2,
        help="워밍업 횟수 (기본: 2)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="모델당 최대 배치 스텝 수 (기본: 전체 실행)",
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=1,
        help="배치 크기 (기본: 1)",
    )
    parser.add_argument(
        "--models", nargs="*", metavar="MODEL",
        help="실행할 모델 이름 한정 (예: resnet50 yolov5m). 생략 시 전체 실행.",
    )
    return parser.parse_args()


def check_required_files(config: dict) -> list[str]:
    """누락된 required_files 목록을 반환. 빈 리스트면 모두 존재."""
    missing = []
    for rel_path in config["required_files"]:
        if not (PROJECT_ROOT / rel_path).exists():
            missing.append(rel_path)
    return missing


def build_cmd(config: dict, args) -> list[str]:
    """main.py 호출 커맨드 리스트를 구성."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "main.py"),
        "--model", config["model"],
        "--onnx", str(PROJECT_ROOT / config["onnx"]),
        "--dataset", str(PROJECT_ROOT / config["dataset"]),
        "--backend", "onnxruntime",
        "--device", args.device,
        "--warmup", str(args.warmup),
        "--batch-size", str(args.batch_size),
    ]
    if args.max_steps is not None:
        cmd += ["--max-steps", str(args.max_steps)]
    if config.get("tokenizer_path"):
        cmd += ["--tokenizer-path", str(PROJECT_ROOT / config["tokenizer_path"])]
    cmd += config.get("extra_args", [])
    return cmd


def extract_metrics(output: str) -> dict:
    """stdout에서 'Final Metrics' 블록 내 key: value 쌍을 파싱."""
    metrics = {}
    in_block = False
    for line in output.splitlines():
        if "Final Metrics" in line:
            in_block = True
            continue
        if in_block:
            if line.startswith("="):
                break
            m = re.match(r"\s+(.+?):\s+(.+)", line)
            if m:
                metrics[m.group(1).strip()] = m.group(2).strip()
    return metrics


def print_separator(char="─", width=72):
    print(char * width)


def run_benchmarks(configs: list[dict], args):
    results = []  # [{label, status, metrics, elapsed, error}]

    for cfg in configs:
        label = cfg["label"]
        print()
        print_separator("═")
        print(f"  {label}")
        print_separator("═")

        # ── 파일 존재 확인 ──────────────────────────────────────────────
        missing = check_required_files(cfg)
        if missing:
            print(f"  [SKIP] 필수 파일 누락:")
            for f in missing:
                print(f"           - {f}")
            print(f"  [SKIP] 준비 방법: {cfg['skip_hint']}")
            results.append({"label": label, "status": "SKIP", "metrics": {}, "elapsed": 0.0, "error": ""})
            continue

        # ── 실행 ────────────────────────────────────────────────────────
        cmd = build_cmd(cfg, args)
        print(f"  $ {' '.join(cmd[2:])}")  # python 경로 제외 출력
        print()

        t_start = time.perf_counter()
        proc = subprocess.run(
            cmd,
            capture_output=False,   # stdout/stderr를 터미널에 직접 출력
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        elapsed = time.perf_counter() - t_start

        if proc.returncode != 0:
            results.append({
                "label": label, "status": "FAIL",
                "metrics": {}, "elapsed": elapsed,
                "error": f"returncode={proc.returncode}",
            })
        else:
            results.append({
                "label": label, "status": "OK",
                "metrics": {},  # 메트릭은 터미널 출력으로 확인
                "elapsed": elapsed,
                "error": "",
            })

    return results


def print_summary(results: list[dict]):
    print()
    print_separator("═")
    print("  BENCHMARK SUMMARY")
    print_separator("═")
    fmt = "  {:<45}  {:>6}  {:>10}"
    print(fmt.format("Model", "Status", "Wall (s)"))
    print_separator()
    for r in results:
        wall = f"{r['elapsed']:.1f}s" if r["elapsed"] > 0 else "-"
        status = r["status"]
        marker = "✓" if status == "OK" else ("✗" if status == "FAIL" else "–")
        print(fmt.format(r["label"][:45], f"{marker} {status}", wall))
        if r["error"]:
            print(f"    └─ {r['error']}")
    print_separator()

    ok = sum(1 for r in results if r["status"] == "OK")
    skip = sum(1 for r in results if r["status"] == "SKIP")
    fail = sum(1 for r in results if r["status"] == "FAIL")
    print(f"  결과: {ok} 성공 / {skip} 스킵 / {fail} 실패  (총 {len(results)}개)")
    print_separator("═")


def main():
    args = parse_args()

    # 실행 대상 필터링
    configs = BENCHMARK_CONFIGS
    if args.models:
        configs = [c for c in BENCHMARK_CONFIGS if c["model"] in args.models]
        if not configs:
            print(f"[Error] 지정한 모델을 찾을 수 없습니다: {args.models}")
            sys.exit(1)

    print()
    print("=" * 72)
    print("  ONNX 통합 벤치마크 실행기")
    print(f"  device={args.device}  warmup={args.warmup}  "
          f"batch_size={args.batch_size}  max_steps={args.max_steps or '전체'}")
    print(f"  대상 모델: {len(configs)}개")
    print("=" * 72)

    results = run_benchmarks(configs, args)
    print_summary(results)

    # 실패가 있으면 비정상 종료
    if any(r["status"] == "FAIL" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
