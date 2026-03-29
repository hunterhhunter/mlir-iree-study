"""
자연어 평가를 위한 순수 Numpy 데이터셋 범용 베이킹 스크립트.

특정 데이터셋(SST-2 등)에 국한되지 않고, 사내 보안 로컬 엑셀/CSV 데이터셋이나
HuggingFace 허브의 다양한 텍스트 데이터를 `--csv-file` 및 컬럼 매개변수(`--text-column`)로
유연하게 받아내어 Static Shape 방식의 `.npy` 배열 보관함으로 구워냄.
"""

import os
import argparse
import numpy as np

try:
    from transformers import AutoTokenizer
    from datasets import load_dataset
except ImportError:
    print("[Error] 현재 환경에 HuggingFace 라이브러리가 없습니다.")
    print("        실행 전 다음 명령어로 설치해주세요: pip install transformers datasets")
    import sys; 
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Universal Text Dataset to Pure Numpy Converter")
    parser.add_argument("--model-id", type=str, default="bert-base-uncased", help="HuggingFace 토크나이저 이름")
    parser.add_argument("--seq-len", type=int, default=128, help="고정 타일링을 위한 정적 패딩 길이")
    
    # [범용 데이터셋 지원을 위한 파라미터들]
    parser.add_argument("--csv-file", type=str, default="", help="로컬 CSV/Excel 데이터를 벤치마크할 경우 사용 (절대경로)")
    parser.add_argument("--dataset-name", type=str, default="glue", help="HF 데이터셋 종류 (기본: glue)")
    parser.add_argument("--dataset-config", type=str, default="sst2", help="HF 데이터셋 하위 설정 (기본: sst2)")
    parser.add_argument("--split", type=str, default=None, help="처리할 범주 (기본: HF=validation, CSV=train)")
    
    # [칼럼 이름 매핑]
    parser.add_argument("--text-column", type=str, default="sentence", help="데이터 안에서 문장이 담긴 열의 이름")
    parser.add_argument("--label-column", type=str, default="label", help="데이터 안에서 정답이 담긴 열의 이름")
    
    parser.add_argument("--output-dir", "-o", type=str, default=os.path.join(os.path.dirname(__file__), "../datasets/baked_numpy"), help="npy 파일 저장 경로")
    args = parser.parse_args()

    # 1. 원본 데이터셋 로드 (HuggingFace Hub OR 로컬 CSV)
    try:
        if args.csv_file:
            print(f"[*] 로컬 보안 데이터셋(CSV) 로딩 중: {args.csv_file}...")
            # 파라미터가 비어있다면 CSV는 기본적으로 'train'으로 묶이는 HF 규칙을 따름
            final_split = args.split if args.split else "train"
            dataset = load_dataset("csv", data_files=args.csv_file, split=final_split)
        else:
            # 파라미터가 비어있다면 HF 데이터셋은 'validation'을 기본 평가셋으로 간주
            final_split = args.split if args.split else "validation"
            print(f"[*] HuggingFace 허브 데이터셋 로딩 중: {args.dataset_name} ({args.dataset_config}), split={final_split}...")
            dataset = load_dataset(args.dataset_name, args.dataset_config, split=final_split)
    except Exception as e:
        print(f"[Error] 데이터셋 처리(다운로드 또는 CSV 스캔) 실패: {e}")
        return
    
    # 2. 토크나이저 로드 (영단어 -> 숫자 ID 변환기)
    print(f"[*] Тоkenizer 로딩 중: {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    print(f"[*] {len(dataset)}개 샘플 배치 토크나이징 (Rust 가속) 진행 중 (Length={args.seq_len})...")
    print(f"    - 타겟 텍스트 컬럼: '{args.text_column}' / 정답 라벨 컬럼: '{args.label_column}'")
    
    # 3. 토크나이징 및 고정 길이 강제 패딩 (배치 입력을 통한 병렬 가속 및 OOM 방지)
    def tokenize_function(examples):
        return tokenizer(
            examples[args.text_column],      # 자유롭게 지정한 텍스트 컬럼
            max_length=args.seq_len,
            padding="max_length",            # 빈칸(0)을 채워 Static Shape 통일
            truncation=True
        )

    try:
        # batched=True: 대용량 데이터 메모리 폭발(OOM) 방지를 위해 1,000개 청크 단위로 병렬 매핑
        encoded_dataset = dataset.map(
            tokenize_function, 
            batched=True,
            desc="Running tokenizer on dataset"
        )
    except KeyError:
        print(f"[Error] 데이터 안에 '{args.text_column}' 이라는 텍스트 열이 없습니다.")
        print(f"   -> 실제 존재하는 열 목록: {dataset.column_names}")
        return
    
    # 4. 결과 매핑 및 정답 라벨 추출
    # [OOM 방지 2단계]: Python List를 거치지 않고 HF의 Arrow 포맷에서 Numpy로 직행 (Zero-copy)
    try:
        encoded_dataset.set_format(type="numpy", columns=["input_ids", "attention_mask", args.label_column])
    except KeyError:
        print(f"[Error] 데이터 안에 '{args.label_column}' 이라는 정답 열이 없습니다.")
        print(f"   -> 실제 존재하는 열 목록: {dataset.column_names}")
        return
        
    np_input_ids = np.asarray(encoded_dataset["input_ids"], dtype=np.int64)
    np_attention_mask = np.asarray(encoded_dataset["attention_mask"], dtype=np.int64)
    np_labels = np.asarray(encoded_dataset[args.label_column], dtype=np.int64)

    # 5. 최종 산출물을 디스크에 영구 저장 (.npy)
    os.makedirs(args.output_dir, exist_ok=True)
    
    id_path = os.path.join(args.output_dir, "input_ids.npy")
    mask_path = os.path.join(args.output_dir, "attention_mask.npy")
    label_path = os.path.join(args.output_dir, "labels.npy")
    
    np.save(id_path, np_input_ids)
    np.save(mask_path, np_attention_mask)
    np.save(label_path, np_labels)

    print("\n[SUCCESS] 오프라인 데이터셋 추출(Baking) 완료! 🥐")
    print(f"  - Input IDs 배열 크기     : {np_input_ids.shape} -> '{id_path}' 저장됨")
    print(f"  - Attention Mask 배열 크기: {np_attention_mask.shape} -> '{mask_path}' 저장됨")
    print(f"  - Labels 배열 크기        : {np_labels.shape}     -> '{label_path}' 저장됨")
    print("\n[Guide] 이제 다음과 같이 메인 벤치마크 엔진에 연결하세요:")
    print(f"  python src/main.py \\")
    print(f"    --model {args.model_id} \\")
    print(f"    --onnx <ONNX_모델_경로> \\")
    print(f"    --dataset {args.output_dir} \\")
    print(f"    --task nlp_classification")

if __name__ == "__main__":
    main()
