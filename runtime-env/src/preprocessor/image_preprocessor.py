"""
ImagePreprocessor — 이미지 분류(Image Classification)용 전처리기

PIL 이미지를 전처리 전략(Strategy)을 통해 numpy 텐서로 변환합니다.
샘플 단위 .npy 파일로 캐싱하여 반복 실행 시 전처리 비용을 제거합니다.
"""

from typing import Any, Optional, Tuple
from pathlib import Path

import numpy as np
from PIL import Image

from .base import BasePreprocessor
from .strategies import (
    PreprocessStrategy,
    MLPerfResNet50Preprocess,
)


# 범용 ImageNet 정규화 상수 (기본 폴백)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ImagePreprocessor(BasePreprocessor):
    """
    이미지 분류 모델용 전처리기.

    PIL Image 또는 이미지 파일 경로를 받아 지정된 전처리 전략을 실행하고,
    결과를 .npy 파일로 캐싱합니다.

    Args:
        target_hw:  모델 입력 해상도 (H, W).
        mean:       채널별 정규화 평균 (shape [3]).
        std:        채널별 정규화 표준편차 (shape [3]).
        strategy:   PreprocessStrategy 구현체. 기본값은 MLPerfResNet50Preprocess.
    """

    def __init__(
        self,
        target_hw: Tuple[int, int],
        mean: np.ndarray = None,
        std: np.ndarray = None,
        strategy: Optional[PreprocessStrategy] = None,
    ):
        self.target_hw = target_hw
        self.mean = mean if mean is not None else IMAGENET_MEAN
        self.std  = std  if std  is not None else IMAGENET_STD
        self.strategy: PreprocessStrategy = strategy or MLPerfResNet50Preprocess()

    def preprocess(self, raw_input: Any) -> np.ndarray:
        """
        PIL Image 또는 이미지 파일 경로를 numpy 텐서로 변환합니다.

        Args:
            raw_input: PIL.Image 객체 또는 이미지 파일 경로 문자열.

        Returns:
            np.ndarray: shape (C, H, W), dtype float32.
        """
        img = Image.open(raw_input) if isinstance(raw_input, str) else raw_input
        return self.strategy(img, self.target_hw, self.mean, self.std)

    def get_cache_path(self, cache_dir: Optional[str], img_filename: str) -> Optional[str]:
        """
        이미지 파일명 기반으로 .npy 캐시 파일 경로를 생성합니다.

        Args:
            cache_dir:    캐시 디렉토리 경로. None이면 None 반환.
            img_filename: 이미지 파일명 (확장자 포함).

        Returns:
            str 또는 None: 캐시 파일 경로.
        """
        if not cache_dir:
            return None
        stem = Path(img_filename).stem
        return str(Path(cache_dir) / f"{stem}.npy")
