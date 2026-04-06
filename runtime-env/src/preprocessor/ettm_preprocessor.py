"""
ETTmPreprocessor — 시계열(Time Series) 전처리기

ETTm1 슬라이딩 윈도우 배열에 RevIN 정규화를 적용합니다.
윈도우 단위 .npz 파일로 캐싱하여 반복 실행 시 전처리 비용을 제거합니다.
"""

from typing import Any, Dict, Optional
from pathlib import Path

import numpy as np

from .base import BasePreprocessor
from .strategies import TimeSeriesPreprocessStrategy


class ETTmPreprocessor(BasePreprocessor):
    """
    ETTm1 시계열 데이터용 전처리기 (PatchTST-FM-R1 전용).

    (context_length, C) 형태의 원본 윈도우 배열에 RevIN 정규화를 적용하여
    past_values / past_observed_mask / norm_stats를 반환합니다.

    윈도우 단위로 .npz 파일에 캐싱하여 반복 실행 시 전처리 비용을 제거합니다.

    Args:
        normalize: RevIN 정규화 적용 여부. False이면 raw 데이터를 그대로 반환합니다.
                   모델 내부에 자체 정규화가 있는 경우 False로 설정합니다.
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self._strategy = TimeSeriesPreprocessStrategy()

    # ------------------------------------------------------------------
    # BasePreprocessor 구현
    # ------------------------------------------------------------------

    def preprocess(self, raw_input: Any) -> Dict[str, np.ndarray]:
        """
        (context_length, C) 원본 윈도우를 정규화된 딕셔너리로 변환합니다.

        Args:
            raw_input: np.ndarray, shape (context_length, C), dtype float32.

        Returns:
            dict:
                'past_values'        : (T, C) float32, RevIN 정규화 후
                'past_observed_mask' : (T, C) bool
                'norm_stats'         : {'mean': (C,), 'std': (C,)}
        """
        window: np.ndarray = raw_input

        if self.normalize:
            return self._strategy(window)
        else:
            C = window.shape[1]
            return {
                "past_values":        window.astype(np.float32),
                "past_observed_mask": np.ones_like(window, dtype=bool),
                "norm_stats": {
                    "mean": np.zeros(C, dtype=np.float32),
                    "std":  np.ones(C,  dtype=np.float32),
                },
            }

    # ------------------------------------------------------------------
    # .npz 캐시 전용 헬퍼
    # ------------------------------------------------------------------

    def load_or_preprocess_window(
        self, cache_path: Optional[str], window: np.ndarray
    ) -> Dict[str, Any]:
        """
        .npz 캐시가 존재하면 로드, 없으면 전처리 후 저장합니다.

        ETTmLoader._get_window() 의 캐시 로직을 이 전처리기로 위임하기 위한 헬퍼입니다.

        Args:
            cache_path: .npz 캐시 파일 경로. None이면 캐시 없이 항상 전처리합니다.
            window:     (context_length, C) 원본 윈도우 배열.

        Returns:
            dict:
                'past_values'        : (T, C) float32
                'past_observed_mask' : (T, C) bool
                'norm_stats'         : {'mean': (C,), 'std': (C,)}
        """
        if cache_path and Path(cache_path).exists():
            npz = np.load(cache_path, allow_pickle=False)
            return {
                "past_values":        npz["past_values"],
                "past_observed_mask": npz["past_observed_mask"],
                "norm_stats": {
                    "mean": npz["norm_mean"],
                    "std":  npz["norm_std"],
                },
            }

        result = self.preprocess(window)

        if cache_path:
            cache_dir = Path(cache_path).parent
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache_path,
                past_values        = result["past_values"],
                past_observed_mask = result["past_observed_mask"],
                norm_mean          = result["norm_stats"]["mean"],
                norm_std           = result["norm_stats"]["std"],
            )

        return result

    def get_cache_path(
        self,
        cache_dir: Optional[str],
        split: str,
        context_length: int,
        prediction_length: int,
        stride: int,
        window_idx: int,
    ) -> Optional[str]:
        """
        윈도우 인덱스 기반으로 .npz 캐시 파일 경로를 생성합니다.

        Args:
            cache_dir:         캐시 루트 디렉토리. None이면 None 반환.
            split:             데이터 분할 ("val", "test", "train").
            context_length:    입력 윈도우 길이.
            prediction_length: 예측 호라이즌.
            stride:            윈도우 이동 간격.
            window_idx:        윈도우 순번 (0-based).

        Returns:
            str 또는 None: 캐시 파일 경로.
        """
        if not cache_dir:
            return None
        subdir = Path(cache_dir) / f"{split}_{context_length}_{prediction_length}_{stride}"
        return str(subdir / f"{window_idx:06d}.npz")
