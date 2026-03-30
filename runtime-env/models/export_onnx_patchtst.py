"""
PatchTST ONNX Export Script
----------------------------
optimum-cli 없이 순수 transformers + torch.onnx.export로 PatchTST 계열 모델을 내보냅니다.

사용법:
    # 로컬 모델 디렉토리 (patchtst_fm config를 PatchTSTConfig로 변환하여 로드)
    uv run models/export_onnx_patchtst.py \
        --model models/ibm-research_patchtst-fm-r1/ \
        --output models/ibm-research_patchtst-fm-r1-ONNX/model.onnx \
        --context-length 512 --channels 7 --prediction-length 96

    # HuggingFace 모델 ID 직접 사용
    uv run models/export_onnx_patchtst.py \
        --model ibm-granite/granite-timeseries-patchtst \
        --output models/patchtst-ONNX/model.onnx
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn


# ──────────────────────────────────────────────
# 1. Forward Wrapper — torch.onnx.export는 positional args만 받으므로
# ──────────────────────────────────────────────

class PatchTSTWrapper(nn.Module):
    """past_values positional arg → model(past_values=...) 변환 래퍼."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, past_values):
        out = self.model(past_values=past_values)
        # PatchTSTForPrediction → prediction_outputs: (B, pred_len, C)
        return out.prediction_outputs


# ──────────────────────────────────────────────
# 2. 모델 로딩 (patchtst_fm config 자동 변환 포함)
# ──────────────────────────────────────────────

def _load_model(model_path: str, context_length: int, channels: int, prediction_length: int):
    from transformers import PatchTSTConfig, PatchTSTForPrediction

    local_config_path = os.path.join(model_path, "config.json")

    if os.path.isfile(local_config_path):
        with open(local_config_path) as f:
            raw = json.load(f)

        model_type = raw.get("model_type", "")
        print(f"[*] Detected model_type: '{model_type}'")

        if model_type == "patchtst_fm":
            # patchtst_fm 필드 → PatchTSTConfig 필드로 매핑
            print("[*] patchtst_fm 감지 — PatchTSTConfig로 변환하여 로드합니다.")
            patch_length = raw.get("d_patch", 16)
            cfg = PatchTSTConfig(
                num_input_channels=channels,
                context_length=context_length,
                patch_length=patch_length,
                patch_stride=patch_length,              # non-overlapping patches
                num_hidden_layers=raw.get("n_layer", 20),
                d_model=raw.get("d_model", 1024),
                num_attention_heads=raw.get("n_head", 16),
                prediction_length=prediction_length,
                pre_norm=raw.get("norm_first", True),
            )
            print("[*] 변환된 PatchTSTConfig:", cfg)
            model = PatchTSTForPrediction(cfg)
            # 가중치 로드 (safetensors 우선)
            weights_path = os.path.join(model_path, "model.safetensors")
            if os.path.isfile(weights_path):
                from safetensors.torch import load_file
                state_dict = load_file(weights_path)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                print(f"[*] 가중치 로드 완료 — missing: {len(missing)}, unexpected: {len(unexpected)}")
                if missing:
                    print(f"    [!] 누락된 키 (일부): {missing[:5]}")
                if unexpected:
                    print(f"    [!] 예상치 못한 키 (일부): {unexpected[:5]}")
            else:
                print("[!] safetensors 없음 — 무작위 초기화 모델로 진행합니다.")
            return model

    # 표준 patchtst 또는 HuggingFace Hub ID
    print(f"[*] PatchTSTForPrediction.from_pretrained('{model_path}') 시도...")
    model = PatchTSTForPrediction.from_pretrained(
        model_path,
        num_input_channels=channels,
        context_length=context_length,
        prediction_length=prediction_length,
        ignore_mismatched_sizes=True,
    )
    return model


# ──────────────────────────────────────────────
# 3. ONNX Export
# ──────────────────────────────────────────────

def export(model_path: str, output_path: str, context_length: int,
           channels: int, prediction_length: int, opset: int):

    print(f"\n[*] 모델 로드: {model_path}")
    model = _load_model(model_path, context_length, channels, prediction_length)
    model.eval()

    wrapped = PatchTSTWrapper(model)

    dummy = torch.randn(1, context_length, channels)

    # 출력 shape 확인
    with torch.no_grad():
        out = wrapped(dummy)
    print(f"[*] 더미 추론 출력 shape: {out.shape}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"[*] ONNX export → {output_path}  (opset={opset})")
    torch.onnx.export(
        wrapped,
        (dummy,),
        output_path,
        input_names=["past_values"],
        output_names=["predictions"],
        dynamic_axes={
            "past_values": {0: "batch_size"},
            "predictions": {0: "batch_size"},
        },
        opset_version=opset,
    )
    print(f"[+] Export 완료: {output_path}")

    # 간단한 검증
    import onnxruntime as ort
    import numpy as np
    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    dummy_np = np.random.randn(1, context_length, channels).astype(np.float32)
    result = sess.run(None, {"past_values": dummy_np})
    print(f"[+] ONNX Runtime 검증 완료 — 출력 shape: {result[0].shape}")


# ──────────────────────────────────────────────
# 4. CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PatchTST ONNX Export (optimum 없이)")
    parser.add_argument("--model",             required=True,  help="로컬 디렉토리 또는 HuggingFace 모델 ID")
    parser.add_argument("--output",            required=True,  help="출력 .onnx 파일 경로")
    parser.add_argument("--context-length",    type=int, default=512,  help="입력 시퀀스 길이 (기본 512)")
    parser.add_argument("--channels",          type=int, default=7,    help="입력 채널 수 (기본 7, ETTm1)")
    parser.add_argument("--prediction-length", type=int, default=96,   help="예측 호라이즌 (기본 96)")
    parser.add_argument("--opset",             type=int, default=17,   help="ONNX opset (기본 17)")
    args = parser.parse_args()

    export(
        model_path=args.model,
        output_path=args.output,
        context_length=args.context_length,
        channels=args.channels,
        prediction_length=args.prediction_length,
        opset=args.opset,
    )


if __name__ == "__main__":
    main()
