"""
BertQAPreprocessor — BERT SQuAD 질의응답(QA) 데이터셋 전처리기

데이터셋 단위(전체 분할)로 question+context를 토큰화하여 numpy 파일로 저장합니다.
numpy 파일이 이미 존재하면 전처리를 건너뜁니다.

이 전처리기는 샘플 단위가 아닌 데이터셋 전체 단위로 동작합니다.
BertQALoader가 초기화될 때 numpy 파일 존재 여부를 확인하고,
없으면 이 전처리기를 호출하여 생성합니다.
"""

import os
from typing import Optional

import numpy as np

from .base import BasePreprocessor


# 전처리 결과 파일명 규약 (BertQALoader와 동일해야 함)
_REQUIRED_FILES = (
    "input_ids.npy",
    "attention_mask.npy",
    "start_positions.npy",
    "end_positions.npy",
)


class BertQAPreprocessor(BasePreprocessor):
    """
    BERT SQuAD QA 모델용 데이터셋 전처리기.

    HuggingFace Hub SQuAD 데이터셋 또는 로컬 SQuAD JSON 파일을 받아
    Character 단위 정답 위치를 Token 인덱스로 변환(Offset Mapping)한 뒤,
    output_dir에 numpy 파일 4개를 저장합니다:
      - input_ids.npy          : shape (N, seq_len), dtype int64
      - attention_mask.npy     : shape (N, seq_len), dtype int64
      - start_positions.npy    : shape (N,), dtype int64
      - end_positions.npy      : shape (N,), dtype int64

    Args:
        tokenizer_path_or_id: HuggingFace 모델 ID 또는 로컬 토크나이저 경로.
                              예: "csarron/bert-base-uncased-squad-v1"
        seq_len:              정적 패딩 시퀀스 길이. 기본값 384 (QA 표준).
        dataset_name:         HuggingFace 데이터셋 이름. 기본값 "squad".
        split:                데이터셋 분할. 기본값 "validation".
    """

    def __init__(
        self,
        tokenizer_path_or_id: str = "csarron/bert-base-uncased-squad-v1",
        seq_len: int = 384,
        dataset_name: str = "squad",
        split: str = "validation",
    ):
        self.tokenizer_path_or_id = tokenizer_path_or_id
        self.seq_len              = seq_len
        self.dataset_name         = dataset_name
        self.split                = split

    # ------------------------------------------------------------------
    # 데이터셋 단위 전처리 인터페이스
    # ------------------------------------------------------------------

    def is_preprocessed(self, output_dir: str) -> bool:
        """
        output_dir에 필요한 numpy 파일 4개가 모두 존재하는지 확인합니다.

        Args:
            output_dir: numpy 파일 저장 경로.

        Returns:
            bool: 전처리 완료 여부.
        """
        return all(
            os.path.exists(os.path.join(output_dir, fname))
            for fname in _REQUIRED_FILES
        )

    def preprocess_dataset(
        self,
        output_dir: str,
        squad_json: Optional[str] = None,
    ) -> None:
        """
        SQuAD 데이터셋을 토큰화하여 output_dir에 numpy 파일을 저장합니다.

        Character 단위의 정답 위치(answer_start)를
        Token 인덱스(start_positions, end_positions)로 Offset Mapping을 통해 변환합니다.

        Args:
            output_dir:  numpy 파일 저장 경로.
            squad_json:  로컬 SQuAD val.json 파일 경로.
                         지정 시 HuggingFace Hub 대신 사용합니다.
        """
        try:
            from transformers import AutoTokenizer
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "BertQAPreprocessor는 transformers와 datasets 라이브러리가 필요합니다.\n"
                "pip install transformers datasets"
            )

        print(f"[BertQAPreprocessor] 토크나이저 로딩: {self.tokenizer_path_or_id}")
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path_or_id)

        # 데이터셋 로드
        if squad_json:
            print(f"[BertQAPreprocessor] 로컬 SQuAD JSON 로딩: {squad_json}")
            dataset = load_dataset("json", data_files=squad_json, split="train")
        else:
            print(f"[BertQAPreprocessor] HuggingFace Hub 로딩: {self.dataset_name} ({self.split})")
            dataset = load_dataset(self.dataset_name, split=self.split)

        print(f"[BertQAPreprocessor] {len(dataset)}개 샘플 Offset Mapping 중 (seq_len={self.seq_len})...")

        all_input_ids      = []
        all_attention_masks = []
        all_start_positions = []
        all_end_positions   = []

        for example in dataset:
            question          = example["question"]
            context           = example["context"]
            answers           = example["answers"]

            # SQuAD v2: 정답 없는 샘플(is_impossible) 건너뜀
            if len(answers["text"]) == 0:
                continue

            answer_text       = answers["text"][0]
            answer_start_char = answers["answer_start"][0]
            answer_end_char   = answer_start_char + len(answer_text)

            tokenized = tokenizer(
                question,
                context,
                max_length=self.seq_len,
                padding="max_length",
                truncation="only_second",
                return_offsets_mapping=True,
            )

            offset_mapping = tokenized["offset_mapping"]

            # Character 위치 → Token 인덱스 변환
            start_token = 0
            end_token   = 0
            for idx, (start_char, end_char) in enumerate(offset_mapping):
                if start_char <= answer_start_char < end_char:
                    start_token = idx
                if start_char < answer_end_char <= end_char:
                    end_token = idx
                    break

            all_input_ids.append(tokenized["input_ids"])
            all_attention_masks.append(tokenized["attention_mask"])
            all_start_positions.append(start_token)
            all_end_positions.append(end_token)

        np_input_ids       = np.array(all_input_ids,       dtype=np.int64)
        np_attention_masks = np.array(all_attention_masks, dtype=np.int64)
        np_start_positions = np.array(all_start_positions, dtype=np.int64)
        np_end_positions   = np.array(all_end_positions,   dtype=np.int64)

        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "input_ids.npy"),       np_input_ids)
        np.save(os.path.join(output_dir, "attention_mask.npy"),  np_attention_masks)
        np.save(os.path.join(output_dir, "start_positions.npy"), np_start_positions)
        np.save(os.path.join(output_dir, "end_positions.npy"),   np_end_positions)

        print(
            f"[BertQAPreprocessor] 완료.\n"
            f"  input_ids       : {np_input_ids.shape}\n"
            f"  attention_mask  : {np_attention_masks.shape}\n"
            f"  start_positions : {np_start_positions.shape}\n"
            f"  end_positions   : {np_end_positions.shape}\n"
            f"  저장 경로: {output_dir}"
        )

    def ensure_preprocessed(
        self,
        output_dir: str,
        squad_json: Optional[str] = None,
    ) -> None:
        """
        numpy 파일이 없을 때만 데이터셋 전처리를 실행합니다.

        Args:
            output_dir:  numpy 파일 저장 경로.
            squad_json:  로컬 SQuAD val.json 파일 경로 (선택).
        """
        if self.is_preprocessed(output_dir):
            print(f"[BertQAPreprocessor] 캐시 히트 — 전처리 건너뜀: {output_dir}")
            return
        print(f"[BertQAPreprocessor] numpy 파일 없음 — 전처리 시작: {output_dir}")
        self.preprocess_dataset(output_dir, squad_json=squad_json)

    # ------------------------------------------------------------------
    # BasePreprocessor 추상 메서드 구현 (데이터셋 전처리기에선 미사용)
    # ------------------------------------------------------------------

    def preprocess(self, raw_input) -> np.ndarray:
        """
        이 전처리기는 샘플 단위가 아닌 데이터셋 단위로 동작합니다.
        단일 샘플 전처리가 필요하면 ensure_preprocessed()를 사용하세요.
        """
        raise NotImplementedError(
            "BertQAPreprocessor는 데이터셋 단위 전처리기입니다. "
            "ensure_preprocessed(output_dir) 를 사용하세요."
        )
