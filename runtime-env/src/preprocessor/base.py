"""
BasePreprocessor — 전처리기 추상 기본 클래스

모든 모델별 전처리기가 상속하는 공통 인터페이스를 정의합니다.

핵심 원칙:
- numpy 캐시가 존재하면 전처리 없이 디스크에서 로드합니다.
- 캐시가 없으면 preprocess()를 실행한 뒤 numpy 파일로 저장합니다.
- 이 원칙은 이미지(샘플 단위 .npy)와 NLP(샘플 단위 .npz)에 공통으로 적용됩니다.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import os

import numpy as np


class BasePreprocessor(ABC):
    """
    전처리기 추상 기본 클래스.

    서브클래스는 preprocess() 메서드만 구현하면 됩니다.
    numpy 캐시 저장·로드 로직은 이 기본 클래스에서 제공합니다.
    """

    @abstractmethod
    def preprocess(self, raw_input: Any) -> Any:
        """
        원시(raw) 데이터를 모델 입력 형태로 변환합니다.

        Args:
            raw_input: 처리할 원시 입력 (이미지 경로, PIL.Image, 딕셔너리 등 서브클래스마다 다름)

        Returns:
            np.ndarray 또는 dict[str, np.ndarray]: 전처리된 텐서 데이터
        """
        pass

    def load_or_preprocess(self, cache_path: Optional[str], raw_input: Any) -> np.ndarray:
        """
        단일 .npy 파일 기반 캐시 로직 (이미지 분류, 객체 탐지 등 단순 배열 반환에 사용).

        - cache_path에 파일이 있으면 전처리 없이 np.load()로 반환합니다.
        - 없으면 preprocess()를 실행하고, 결과를 .npy로 저장한 뒤 반환합니다.
        - cache_path가 None이면 캐시 없이 항상 preprocess()를 실행합니다.

        Args:
            cache_path: .npy 캐시 파일의 절대 또는 상대 경로. None이면 캐시 비활성.
            raw_input:  preprocess()에 전달할 원시 입력 데이터.

        Returns:
            np.ndarray: 전처리된 배열.
        """
        if cache_path and os.path.exists(cache_path):
            return np.load(cache_path)

        result = self.preprocess(raw_input)

        if cache_path:
            cache_dir = os.path.dirname(os.path.abspath(cache_path))
            os.makedirs(cache_dir, exist_ok=True)
            np.save(cache_path, result)

        return result

    def load_or_preprocess_npz(
        self, cache_path: Optional[str], raw_input: Any
    ) -> dict:
        """
        .npz 파일 기반 캐시 로직 (LLaMA, ETTm 등 다중 배열 반환에 사용).

        - cache_path에 .npz 파일이 있으면 dict로 로드하여 반환합니다.
        - 없으면 preprocess()를 실행하고, 결과 dict를 .npz로 저장한 뒤 반환합니다.
        - cache_path가 None이면 캐시 없이 항상 preprocess()를 실행합니다.

        Args:
            cache_path: .npz 캐시 파일의 절대 또는 상대 경로. None이면 캐시 비활성.
            raw_input:  preprocess()에 전달할 원시 입력 데이터.

        Returns:
            dict: 전처리된 배열들의 딕셔너리.
        """
        if cache_path and os.path.exists(cache_path):
            loaded = np.load(cache_path, allow_pickle=False)
            return dict(loaded)

        result = self.preprocess(raw_input)

        if cache_path:
            cache_dir = os.path.dirname(os.path.abspath(cache_path))
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(cache_path, **result)

        return result
