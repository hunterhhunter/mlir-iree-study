"""
LlamaPreprocessor — LLaMA 3.1 8B SQuAD QA용 전처리기

SQuAD 샘플(question + context)을 LLaMA 3.1 8B 입력 텐서로 토큰화합니다.
qa_id 기반의 .npz 파일로 샘플 단위 캐싱을 제공합니다.
"""

from typing import Any, Dict, Optional
from pathlib import Path

import numpy as np

from .base import BasePreprocessor
from .strategies import SQuADPreprocessStrategy


class LlamaPreprocessor(BasePreprocessor):
    """
    LLaMA 3.1 8B 모델용 SQuAD QA 전처리기.

    SQuADPreprocessStrategy를 내부적으로 사용하여 질문+컨텍스트를
    input_ids / attention_mask numpy 배열로 변환합니다.

    샘플 단위로 .npz 파일에 캐싱하여 반복 실행 시 토큰화 비용을 제거합니다.

    Args:
        tokenizer_path: HuggingFace 로컬 토크나이저 디렉토리 경로.
        max_length:     최대 시퀀스 길이. 기본값 4096.
    """

    def __init__(self, tokenizer_path: str, max_length: int = 4096):
        self._strategy = SQuADPreprocessStrategy(
            tokenizer_path=tokenizer_path,
            max_length=max_length,
        )

    # ------------------------------------------------------------------
    # BasePreprocessor 구현
    # ------------------------------------------------------------------

    def preprocess(self, raw_input: Any) -> Dict[str, np.ndarray]:
        """
        단일 SQuAD QA 샘플을 토큰화합니다.

        Args:
            raw_input: dict {'question': str, 'context': str}
                       또는 tuple (question: str, context: str).

        Returns:
            dict: {'input_ids': ndarray (1, L), 'attention_mask': ndarray (1, L)}
        """
        if isinstance(raw_input, dict):
            question = raw_input["question"]
            context  = raw_input["context"]
        else:
            question, context = raw_input
        return self._strategy.tokenize(question=question, context=context)

    # ------------------------------------------------------------------
    # .npz 캐시 전용 헬퍼
    # ------------------------------------------------------------------

    def load_or_tokenize(
        self, cache_path: Optional[str], sample: Dict
    ) -> Dict[str, np.ndarray]:
        """
        .npz 캐시가 존재하면 로드, 없으면 토큰화하여 저장합니다.

        Args:
            cache_path: .npz 캐시 파일 경로. None이면 캐시 없이 항상 토큰화합니다.
            sample:     {'question': str, 'context': str} 딕셔너리.

        Returns:
            dict: {'input_ids': ndarray, 'attention_mask': ndarray}
        """
        return self.load_or_preprocess_npz(cache_path, sample)

    def get_cache_path(self, cache_dir: Optional[str], qa_id: str) -> Optional[str]:
        """
        qa_id + 설정 fingerprint 기반으로 .npz 캐시 파일 경로를 생성합니다.

        max_length나 tokenizer_path가 달라지면 다른 캐시 파일을 사용하므로
        이전 설정의 캐시가 재사용되는 문제를 방지합니다.

        Args:
            cache_dir: 캐시 디렉토리 경로. None이면 None 반환.
            qa_id:     SQuAD QA 쌍의 고유 식별자.

        Returns:
            str 또는 None: 캐시 파일 경로.
        """
        if not cache_dir:
            return None
        import hashlib
        cfg_key = f"{self._strategy.tokenizer_path}:{self._strategy.max_length}"
        cfg_hash = hashlib.md5(cfg_key.encode()).hexdigest()[:8]
        return str(Path(cache_dir) / f"{qa_id}_{cfg_hash}.npz")

    # ------------------------------------------------------------------
    # 편의 속성 (LlamaLoader 메타데이터용)
    # ------------------------------------------------------------------

    @property
    def tokenizer(self):
        return self._strategy.tokenizer

    @property
    def max_length(self) -> int:
        return self._strategy.max_length
