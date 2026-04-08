"""
ObjectDetectionPreprocessor — 객체 탐지(Object Detection)용 전처리기

YOLO 스타일의 이미지를 리사이즈·정규화하여 numpy 텐서로 변환합니다.
샘플 단위 .npy 파일로 캐싱하여 반복 실행 시 전처리 비용을 제거합니다.
"""

from typing import Any, Optional, Tuple
from pathlib import Path

import numpy as np
from PIL import Image

from .base import BasePreprocessor


class ObjectDetectionPreprocessor(BasePreprocessor):
    """
    객체 탐지 모델(YOLO 계열)용 전처리기.

    이미지를 비율 왜곡 방식으로 target_hw로 직접 리사이즈한 뒤,
    픽셀 값 스케일링과 채널별 정규화를 수행합니다.
    NCHW 또는 NHWC 레이아웃을 지원합니다.

    Args:
        target_hw: 모델 입력 해상도 (H, W). 기본값 (640, 640).
        mean:      채널별 정규화 평균 (shape [3]). 기본값 [0, 0, 0].
        std:       채널별 정규화 표준편차 (shape [3]). 기본값 [1, 1, 1].
        layout:    메모리 레이아웃 "NCHW" 또는 "NHWC". 기본값 "NCHW".
    """

    def __init__(
        self,
        target_hw: Tuple[int, int] = (640, 640),
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        layout: str = "NCHW",
    ):
        self.target_hw = target_hw
        self.mean   = mean   if mean   is not None else np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.std    = std    if std    is not None else np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.layout = layout.upper()

    def preprocess(self, raw_input: Any) -> np.ndarray:
        """
        이미지를 리사이즈하고 정규화하여 numpy 텐서를 반환합니다.

        Args:
            raw_input: PIL.Image 객체 또는 이미지 파일 경로 문자열.

        Returns:
            np.ndarray:
              NCHW → shape (C, H, W), dtype float32
              NHWC → shape (H, W, C), dtype float32
        """
        img = Image.open(raw_input) if isinstance(raw_input, str) else raw_input
        img = img.convert("RGB")
        img = img.resize((self.target_hw[1], self.target_hw[0]), Image.Resampling.BILINEAR)

        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = (img_array - self.mean) / self.std  # (H, W, C)

        if self.layout == "NCHW":
            img_array = np.transpose(img_array, (2, 0, 1))  # (C, H, W)

        return img_array

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
