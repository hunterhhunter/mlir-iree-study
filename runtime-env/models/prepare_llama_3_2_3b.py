"""
Download meta-llama/Llama-3.2-3B from Hugging Face Hub and optionally convert to ONNX.
Usage:
    python models/prepare_llama_3_2_3b.py
    python models/prepare_llama_3_2_3b.py --format onnx
    python models/prepare_llama_3_2_3b.py --output /path/to/models
    python models/prepare_llama_3_2_3b.py --convert-onnx
    python models/prepare_llama_3_2_3b.py --convert-onnx --dtype fp16
    python models/prepare_llama_3_2_3b.py --convert-onnx --no-post-process
"""
import sys
import os
import subprocess

sys.path.insert(0, os.path.dirname(__file__))
from prepare_hf_model import download_model_repository

import argparse

REPO_ID = "meta-llama/Llama-3.2-3B"
MODEL_NAME = REPO_ID.replace("/", "_")
TASK = "text-generation"


def convert_to_onnx(src_dir: str, output_root: str, dtype: str = None, no_post_process: bool = False):
    """
    optimum-cli를 사용하여 다운로드된 모델을 ONNX 포맷으로 변환합니다.
    변환 결과는 {output_root}/{MODEL_NAME}-ONNX 에 저장됩니다.
    """
    onnx_dir = os.path.join(output_root, f"{MODEL_NAME}-ONNX")
    os.makedirs(onnx_dir, exist_ok=True)

    cmd = [
        "optimum-cli", "export", "onnx",
        "--model", src_dir,
        "--task", TASK,
    ]
    if dtype:
        cmd += ["--dtype", dtype]
    if no_post_process:
        cmd.append("--no-post-process")
    cmd.append(onnx_dir)

    print(f"\n[*] Converting {REPO_ID} to ONNX format...")
    print(f"[*] Source : {src_dir}")
    print(f"[*] Output : {onnx_dir}")
    print(f"[*] Executing: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
        print(f"\n[+] ONNX export completed: {onnx_dir}")
    except subprocess.CalledProcessError as e:
        print(f"\n[!] ONNX export failed with exit code: {e.returncode}")
        sys.exit(1)

    # 변환 결과 검증 (inspect_onnx_model 활용)
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from inspect_onnx_model import print_io_info
        import onnx

        onnx_files = [f for f in os.listdir(onnx_dir) if f.endswith(".onnx")]
        if onnx_files:
            model_path = os.path.join(onnx_dir, onnx_files[0])
            print(f"\n[*] Inspecting converted model: {model_path}")
            model = onnx.load(model_path)
            print_io_info(model.graph.input, "Input")
            print_io_info(model.graph.output, "Output")
        else:
            print("[!] No .onnx file found in output directory for inspection.")
    except Exception as e:
        print(f"[!] Inspection skipped: {e}")

    return onnx_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Llama-3.2-3B from Hugging Face and optionally convert to ONNX")
    parser.add_argument("--format", type=str, default=None,
                        help="Specific model format to include (e.g., 'onnx', 'safetensors')")
    parser.add_argument("--output", type=str, default="models",
                        help="Output root directory (default: models)")
    parser.add_argument("--convert-onnx", action="store_true",
                        help="Download safetensors model and convert to ONNX using optimum-cli")
    parser.add_argument("--dtype", type=str, default=None,
                        help="ONNX export dtype (e.g., fp16, fp32). Only used with --convert-onnx")
    parser.add_argument("--no-post-process", action="store_true",
                        help="Skip post-process validation after ONNX export (use when RAM is limited)")
    args = parser.parse_args()

    if args.convert_onnx:
        # safetensors 다운로드 후 ONNX 변환
        src_dir = os.path.join(args.output, MODEL_NAME)
        download_model_repository(REPO_ID, "safetensors", args.output)
        convert_to_onnx(src_dir, args.output, dtype=args.dtype, no_post_process=args.no_post_process)
    else:
        download_model_repository(REPO_ID, args.format, args.output)
