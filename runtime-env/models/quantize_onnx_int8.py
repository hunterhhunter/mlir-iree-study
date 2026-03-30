"""
ONNX 모델을 INT8 동적 양자화하는 스크립트.

사용법:
    python models/quantize_onnx_int8.py \
        --input  models/meta-llama_Llama-3.1-8B-ONNX-fp16/model.onnx \
        --output models/meta-llama_Llama-3.1-8B-ONNX-int8

외부 데이터 파일(model.onnx_data)이 있는 대형 모델도 자동으로 처리합니다.
결과 폴더에 model.onnx + model.onnx_data(int8 가중치) 가 생성됩니다.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="ONNX Dynamic INT8 Quantization")
    parser.add_argument("--input", type=str, required=True,
                        help="입력 ONNX 파일 경로 (예: models/xxx/model.onnx)")
    parser.add_argument("--output", type=str, required=True,
                        help="출력 디렉토리 (예: models/meta-llama_Llama-3.1-8B-ONNX-int8)")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_path.exists():
        print(f"[!] 입력 파일을 찾을 수 없습니다: {input_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.onnx"

    print(f"[*] 입력  : {input_path}")
    print(f"[*] 출력  : {output_path}")

    # 토크나이저 등 부속 파일 복사 (*.json, special_tokens_map.json 등)
    input_dir = input_path.parent
    for f in input_dir.iterdir():
        if f.suffix == ".json" and f.is_file():
            dst = output_dir / f.name
            shutil.copy2(f, dst)
            print(f"[*] 복사  : {f.name}")

    print("\n[*] INT8 동적 양자화 시작...")

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("[!] onnxruntime 패키지가 필요합니다: pip install onnxruntime")
        sys.exit(1)

    # 외부 데이터 파일 여부 확인
    has_external_data = (input_dir / (input_path.name + "_data")).exists() or \
                        any(f.suffix == "" and f.stem.startswith("model") for f in input_dir.iterdir())

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        # 외부 가중치 파일이 있으면 출력도 외부 파일로 저장
        use_external_data_format=has_external_data,
        extra_options={"MatMulConstBOnly": True},  # MatMul 가중치만 양자화 (LLM 성능 보존)
    )

    print(f"\n[+] 양자화 완료!")
    print(f"    출력 디렉토리: {output_dir}")
    for f in output_dir.iterdir():
        size_mb = f.stat().st_size / (1024 ** 2)
        print(f"    {f.name:40s} {size_mb:8.1f} MB")


if __name__ == "__main__":
    main()
