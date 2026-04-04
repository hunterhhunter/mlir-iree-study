"""
BERT SQuAD 질의응답(QA) 평가를 위한 순수 Numpy 데이터셋 오프라인 굽기 스크립트.

HuggingFace Hub에서 SQuAD validation 세트를 다운로드 한 후,
`tokenizer`의 `return_offsets_mapping=True` 옵션을 활용하여
단어(Character) 단위의 정답지 위치(answer_start)를 
BERT 기반의 질문응답 타겟 모델(예: `csarron/bert-base-uncased-squad-v1`)이
이해할 수 있는 토큰(Token) 인덱스 위치(start_positions, end_positions)로 
수리적으로 변환(Mapping)한 뒤 디스크에 영구 저장(.npy)합니다. 
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
    import sys; sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="SQuAD Text Dataset to Tokenized Numpy Converter")
    parser.add_argument("--model-id", type=str, default="csarron/bert-base-uncased-squad-v1", help="QA용 사전학습된 토크나이저 이름 (NPU 엣지 벤치마킹용 bert-base)")
    parser.add_argument("--seq-len", type=int, default=384, help="Context가 길기 때문에 QA 모델은 보통 384 차원을 규격으로 사용함")
    parser.add_argument("--dataset-name", type=str, default="squad", help="HF 데이터셋 (기본: squad v1.1)")
    parser.add_argument("--split", type=str, default="validation", help="벤치마크 평가용 스플릿 (기본: validation)")
    
    # 출력 경로 기본값
    parser.add_argument("--output-dir", "-o", type=str, default=os.path.join(os.path.dirname(__file__), "../datasets/baked_numpy/squad_val"), help="npy 파일 저장 절대 경로")
    args = parser.parse_args()

    # 1. 원본 데이터 로드 (validation 셋만 부름 - 벤치마크 목적)
    print(f"[*] HuggingFace 허브 데이터셋 로딩 중: {args.dataset_name}, split={args.split}...")
    try:
        dataset = load_dataset(args.dataset_name, split=args.split)
    except Exception as e:
        print(f"[Error] 데이터셋 파싱 실패: {e}")
        return
        
    print(f"[*] Тоkenizer 로딩 중: {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # 파싱될 배열 수급용 임시 리스트
    all_input_ids = []
    all_attention_masks = []
    all_start_positions = []
    all_end_positions = []

    print(f"[*] {len(dataset)}개 샘플 Token-Offset Mapping 진행 중 (Length={args.seq_len})...")
    
    # 2. 토크나이징 및 Character -> Token Offset 스위칭 로직 (순수 파이썬 + HF Iterator)
    # batched=True 매핑은 offset 처리가 복잡하므로 명시적인 반복 순해석(Iteration)으로 안정성 확보
    for i, example in enumerate(dataset):
        # 질문(Question)과 지문(Context) 쌍
        question = example["question"]
        context = example["context"]
        
        # SQuAD Validation 셋은 복수의 정답(Human Annotators)을 가짐
        # 모델 비교 성능 측정을 위해 관례대로 '첫 번째' 정답만 타겟팅
        answers = example["answers"]
        
        # 정답이 아예 없는 예외 케이스(SQuAD v2) 방어. v1은 항상 정답이 있음.
        if len(answers["text"]) == 0:
            continue
            
        answer_text = answers["text"][0]
        answer_start_char = answers["answer_start"][0]
        answer_end_char = answer_start_char + len(answer_text)

        # 토크나이징 (question과 context를 "[CLS] Q [SEP] C [SEP]" 구조로 병합)
        tokenized = tokenizer(
            question,
            context,
            max_length=args.seq_len,
            padding="max_length",
            truncation="only_second", # 긴 지문(context)만 적당히 잘라냄
            return_offsets_mapping=True, # 여기가 핵심! 토큰이 텍스트의 몇 번째 글자에 해당하는지 담김
        )
        
        offset_mapping = tokenized["offset_mapping"]
        # 어느 부분이 질문이고 어느 부분이 지문인지 식별 (None=스페셜, 0=Question, 1=Context)
        sequence_ids = tokenized.sequence_ids()

        # Context의 토큰 시작/끝 인덱스를 파악 (SEP 등 스페셜 토큰 제외)
        context_start_token = 0
        while sequence_ids[context_start_token] != 1:
            context_start_token += 1
            if context_start_token >= len(sequence_ids):
                break # 에러 엣지 케이스
                
        context_end_token = len(sequence_ids) - 1
        while sequence_ids[context_end_token] != 1:
            context_end_token -= 1

        # 정답 글자가 지문(Context) 배열의 완전 바깥에 잘린(Truncation) 경우 정답 오프셋을 0으로 묵살
        if context_start_token >= len(sequence_ids) or \
           offset_mapping[context_start_token][0] > answer_start_char or \
           offset_mapping[context_end_token][1] < answer_end_char:
            target_start = 0
            target_end = 0
        else:
            # Token 인덱스를 추적하여 실제 `answer_start` 캐릭터 위치를 커버하는 토큰 번호를 찾음
            idx = context_start_token
            while idx <= context_end_token and offset_mapping[idx][0] <= answer_start_char:
                idx += 1
            target_start = idx - 1

            # Token 인덱스를 추적하여 `answer_end` 캐릭터 위치를 커버하는 토큰 번호를 찾음
            idx = context_end_token
            while idx >= context_start_token and offset_mapping[idx][1] >= answer_end_char:
                idx -= 1
            target_end = idx + 1

        all_input_ids.append(tokenized["input_ids"])
        all_attention_masks.append(tokenized["attention_mask"])
        all_start_positions.append(target_start)
        all_end_positions.append(target_end)
        
        if (i+1) % 2000 == 0:
            print(f"   ... Processed {i+1} / {len(dataset)} samples")

    # 3. 누적된 파이썬 리스트를 O(1) Numpy 벡터 통짜 데이터로 묶음
    np_input_ids = np.asarray(all_input_ids, dtype=np.int64)
    np_attention_mask = np.asarray(all_attention_masks, dtype=np.int64)
    np_start_positions = np.asarray(all_start_positions, dtype=np.int64)
    np_end_positions = np.asarray(all_end_positions, dtype=np.int64)

    # 4. 디스크 영구 저장
    os.makedirs(args.output_dir, exist_ok=True)
    
    id_path = os.path.join(args.output_dir, "input_ids.npy")
    mask_path = os.path.join(args.output_dir, "attention_mask.npy")
    start_path = os.path.join(args.output_dir, "start_positions.npy")
    end_path = os.path.join(args.output_dir, "end_positions.npy")
    
    np.save(id_path, np_input_ids)
    np.save(mask_path, np_attention_mask)
    np.save(start_path, np_start_positions)
    np.save(end_path, np_end_positions)

    print("\n[SUCCESS] SQuAD 오프라인 베이킹(.npy) 완료! 🥐")
    print(f"  - Input IDs      : {np_input_ids.shape} -> '{id_path}'")
    print(f"  - Attention Mask : {np_attention_mask.shape} -> '{mask_path}'")
    print(f"  - Target Starts  : {np_start_positions.shape} -> '{start_path}'")
    print(f"  - Target Ends    : {np_end_positions.shape} -> '{end_path}'")

if __name__ == "__main__":
    main()
