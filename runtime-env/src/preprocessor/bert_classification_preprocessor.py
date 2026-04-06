"""
BertClassificationPreprocessor — BERT 텍스트 분류(SST-2 등) 데이터셋 전처리기

데이터셋 단위(전체 분할)로 텍스트를 토큰화하여 numpy 파일로 저장합니다.
numpy 파일이 이미 존재하면 전처리를 건너뜁니다.

이 전처리기는 샘플 단위가 아닌 데이터셋 전체 단위로 동작합니다.
BertClassificationLoader가 초기화될 때 numpy 파일 존재 여부를 확인하고,
없으면 이 전처리기를 호출하여 생성합니다.
"""

import os
from typing import Optional

import numpy as np

from .base import BasePreprocessor


# 전처리 결과 파일명 규약 (BertClassificationLoader와 동일해야 함)
_REQUIRED_FILES = ("input_ids.npy", "attention_mask.npy", "labels.npy")


class BertClassificationPreprocessor(BasePreprocessor):
    """
    BERT 텍스트 분류 모델용 데이터셋 전처리기.

    HuggingFace Hub 데이터셋(기본: glue/sst2) 또는 로컬 CSV 파일을 받아
    토크나이저로 인코딩한 뒤 output_dir에 numpy 파일 3개를 저장합니다:
      - input_ids.npy       : shape (N, seq_len), dtype int64
      - attention_mask.npy  : shape (N, seq_len), dtype int64
      - labels.npy          : shape (N,), dtype int64

    Args:
        tokenizer_path_or_id: HuggingFace 모델 ID 또는 로컬 토크나이저 경로.
                              예: "bert-base-uncased" 또는 "models/bert-base-uncased"
        seq_len:              정적 패딩 시퀀스 길이. 기본값 128.
        dataset_name:         HuggingFace 데이터셋 이름. 기본값 "glue".
        dataset_config:       HuggingFace 데이터셋 설정 이름. 기본값 "sst2".
        split:                데이터셋 분할. 기본값 "validation".
        text_column:          텍스트 컬럼 이름. 기본값 "sentence".
        label_column:         라벨 컬럼 이름. 기본값 "label".
    """

    def __init__(
        self,
        tokenizer_path_or_id: str = "bert-base-uncased",
        seq_len: int = 128,
        dataset_name: str = "glue",
        dataset_config: str = "sst2",
        split: str = "validation",
        text_column: str = "sentence",
        label_column: str = "label",
    ):
        self.tokenizer_path_or_id = tokenizer_path_or_id
        self.seq_len              = seq_len
        self.dataset_name         = dataset_name
        self.dataset_config       = dataset_config
        self.split                = split
        self.text_column          = text_column
        self.label_column         = label_column

    # ------------------------------------------------------------------
    # 데이터셋 단위 전처리 인터페이스
    # ------------------------------------------------------------------

    def is_preprocessed(self, output_dir: str) -> bool:
        """
        output_dir에 필요한 numpy 파일 3개가 모두 존재하는지 확인합니다.

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
        csv_file: Optional[str] = None,
    ) -> None:
        """
        데이터셋 전체를 토큰화하여 output_dir에 numpy 파일을 저장합니다.

        Args:
            output_dir: numpy 파일 저장 경로.
            csv_file:   로컬 CSV 파일 경로. 지정 시 HuggingFace Hub 대신 사용합니다.
        """
        try:
            from transformers import AutoTokenizer
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "BertClassificationPreprocessor는 transformers와 datasets 라이브러리가 필요합니다.\n"
                "pip install transformers datasets"
            )

        print(f"[BertClassificationPreprocessor] 토크나이저 로딩: {self.tokenizer_path_or_id}")
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path_or_id)

        # 데이터셋 로드
        if csv_file:
            print(f"[BertClassificationPreprocessor] 로컬 CSV 로딩: {csv_file}")
            dataset = load_dataset("csv", data_files=csv_file, split="train")
        else:
            print(
                f"[BertClassificationPreprocessor] HuggingFace Hub 로딩: "
                f"{self.dataset_name}/{self.dataset_config} ({self.split})"
            )
            dataset = load_dataset(
                self.dataset_name, self.dataset_config, split=self.split
            )

        print(f"[BertClassificationPreprocessor] {len(dataset)}개 샘플 토크나이징 중 (seq_len={self.seq_len})...")

        def tokenize_fn(examples):
            return tokenizer(
                examples[self.text_column],
                max_length=self.seq_len,
                padding="max_length",
                truncation=True,
            )

        encoded = dataset.map(tokenize_fn, batched=True, desc="Tokenizing")
        encoded.set_format(
            type="numpy",
            columns=["input_ids", "attention_mask", self.label_column],
        )

        np_input_ids      = np.asarray(encoded["input_ids"],          dtype=np.int64)
        np_attention_mask = np.asarray(encoded["attention_mask"],     dtype=np.int64)
        np_labels         = np.asarray(encoded[self.label_column],    dtype=np.int64)

        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "input_ids.npy"),     np_input_ids)
        np.save(os.path.join(output_dir, "attention_mask.npy"), np_attention_mask)
        np.save(os.path.join(output_dir, "labels.npy"),        np_labels)

        print(
            f"[BertClassificationPreprocessor] 완료.\n"
            f"  input_ids      : {np_input_ids.shape}\n"
            f"  attention_mask : {np_attention_mask.shape}\n"
            f"  labels         : {np_labels.shape}\n"
            f"  저장 경로: {output_dir}"
        )

    def ensure_preprocessed(
        self,
        output_dir: str,
        csv_file: Optional[str] = None,
    ) -> None:
        """
        numpy 파일이 없을 때만 데이터셋 전처리를 실행합니다.

        Args:
            output_dir: numpy 파일 저장 경로.
            csv_file:   로컬 CSV 파일 경로 (선택).
        """
        if self.is_preprocessed(output_dir):
            print(f"[BertClassificationPreprocessor] 캐시 히트 — 전처리 건너뜀: {output_dir}")
            return
        print(f"[BertClassificationPreprocessor] numpy 파일 없음 — 전처리 시작: {output_dir}")
        self.preprocess_dataset(output_dir, csv_file=csv_file)

    # ------------------------------------------------------------------
    # BasePreprocessor 추상 메서드 구현 (데이터셋 전처리기에선 미사용)
    # ------------------------------------------------------------------

    def preprocess(self, raw_input) -> np.ndarray:
        """
        이 전처리기는 샘플 단위가 아닌 데이터셋 단위로 동작합니다.
        단일 샘플 전처리가 필요하면 ensure_preprocessed()를 사용하세요.
        """
        raise NotImplementedError(
            "BertClassificationPreprocessor는 데이터셋 단위 전처리기입니다. "
            "ensure_preprocessed(output_dir) 를 사용하세요."
        )
