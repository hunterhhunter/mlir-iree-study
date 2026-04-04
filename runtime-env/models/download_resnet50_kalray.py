"""
Download Kalray/resnet50 from Hugging Face Hub.
Usage:
    python models/download_resnet50_kalray.py
    python models/download_resnet50_kalray.py --format onnx
    python models/download_resnet50_kalray.py --output /path/to/models
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from download_hf_model import download_model_repository

import argparse

REPO_ID = "Kalray/resnet50"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Kalray/resnet50 from Hugging Face")
    parser.add_argument("--format", type=str, default=None,
                        help="Specific model format to include (e.g., 'onnx', 'safetensors')")
    parser.add_argument("--output", type=str, default="models",
                        help="Output root directory (default: models)")
    args = parser.parse_args()

    download_model_repository(REPO_ID, args.format, args.output)
