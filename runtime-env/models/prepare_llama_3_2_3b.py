"""
Download meta-llama/Llama-3.2-3B from Hugging Face Hub.
Usage:
    python models/prepare_llama_3_2_3b.py
    python models/prepare_llama_3_2_3b.py --format onnx
    python models/prepare_llama_3_2_3b.py --output /path/to/models
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from prepare_hf_model import download_model_repository

import argparse

REPO_ID = "meta-llama/Llama-3.2-3B"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Llama-3.2-3B from Hugging Face")
    parser.add_argument("--format", type=str, default=None,
                        help="Specific model format to include (e.g., 'onnx', 'safetensors')")
    parser.add_argument("--output", type=str, default="models",
                        help="Output root directory (default: models)")
    args = parser.parse_args()

    download_model_repository(REPO_ID, args.format, args.output)
