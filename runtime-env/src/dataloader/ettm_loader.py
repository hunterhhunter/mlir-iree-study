"""
ETTmLoader — ETTm1 시계열 데이터셋 → PatchTST-FM-R1 입력 텐서 변환 로더

슬라이딩 윈도우 방식으로 val 분할에서 (context_length, prediction_length) 쌍을 추출하고,
RevIN 정규화를 적용한 뒤 모델 입력 포맷으로 반환합니다.

출력 딕셔너리 포맷:
    {
        "input": {
            "past_values":        np.ndarray,  # (1, context_length, C) float32
            "past_observed_mask": np.ndarray,  # (1, context_length, C) bool
        },
        "label": {
            "future_values": np.ndarray,       # (prediction_length, C) float32, 원본 스케일
            "norm_stats": {
                "mean": np.ndarray,            # (C,) 역정규화용
                "std":  np.ndarray,            # (C,)
            },
        },
        "window_idx": int,
    }
"""

import math
import os
from typing import Any, Dict, List, Optional

import numpy as np

from .base import DataLoader
from .preprocess_strategies import TimeSeriesPreprocessStrategy
from ..core.model_spec import Model_Spec


# ETTm1 데이터 컬럼 (date 제외)
_ETTM_FEATURE_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


class ETTmLoader(DataLoader):
    """
    ETTm1 CSV를 슬라이딩 윈도우로 순회하며 PatchTST-FM-R1 입력 텐서를 공급합니다.

    Parameters (kwargs)
    -------------------
    csv_path         : str       ETTm1.csv 경로 (필수)
    context_length   : int       입력 윈도우 길이 (기본 512)
    prediction_length: int       예측 호라이즌 (기본 96)
    target_cols      : list|None 사용할 피처 컬럼 리스트. None이면 전체 7개
    split            : str       "train" | "val" | "test" (기본 "val")
    val_ratio        : float     val 분할 비율 (기본 0.2)
    test_ratio       : float     test 분할 비율 (기본 0.2). split="test" 시 사용
    stride           : int|None  윈도우 이동 간격. None이면 prediction_length
    cache_dir        : str|None  NPZ 캐시 디렉토리. None이면 캐시 비활성
    normalize        : bool      RevIN 정규화 적용 여부 (기본 True).
                                 False이면 raw 데이터를 그대로 반환하고
                                 norm_stats는 항등 변환(mean=0, std=1)으로 설정됨.
                                 모델 내부에 자체 정규화가 있는 경우(scaling="std") 사용.
    split_boundaries : tuple|None (train_end, val_end) 절대 행 인덱스.
                                 지정 시 val_ratio/test_ratio 무시.
                                 예: ETTh1 표준 = (8640, 11520)
    """

    def __init__(self, model_spec: Model_Spec, **kwargs):
        self.model_spec = model_spec

        # --- 파라미터 ---
        csv_path = kwargs.get("csv_path")
        if csv_path is None:
            raise ValueError("ETTmLoader: 'csv_path' kwargs가 필요합니다.")

        self.context_length    = int(kwargs.get("context_length",    512))
        self.prediction_length = int(kwargs.get("prediction_length",  96))
        self.val_ratio         = float(kwargs.get("val_ratio",        0.2))
        self.test_ratio        = float(kwargs.get("test_ratio",       0.2))
        self.split             = kwargs.get("split", "val")
        self.normalize         = bool(kwargs.get("normalize",         True))
        self.split_boundaries  = kwargs.get("split_boundaries",       None)
        self.cache_dir         = kwargs.get("cache_dir", None)

        target_cols = kwargs.get("target_cols", None)
        self.feature_cols: List[str] = _ETTM_FEATURE_COLS if target_cols is None else list(target_cols)

        stride = kwargs.get("stride", None)
        self.stride = int(stride) if stride is not None else self.prediction_length

        # --- 데이터 로드 & 분할 ---
        self._data = self._load_csv(csv_path)          # (N, C) float32
        self._split_start, self._split_end = self._compute_split_bounds()

        # 윈도우 총 수
        usable = self._split_end - self._split_start
        self._window_count = max(
            0,
            math.floor((usable - self.context_length - self.prediction_length) / self.stride) + 1,
        )

        self._preprocess = TimeSeriesPreprocessStrategy()
        self._current_idx = 0

    # ------------------------------------------------------------------
    # DataLoader 추상 메서드 구현
    # ------------------------------------------------------------------

    def load_single(self) -> Dict[str, Any]:
        """현재 인덱스에서 샘플 하나를 반환하고 인덱스를 한 칸 전진합니다."""
        if self._current_idx >= self._window_count:
            raise StopIteration("ETTmLoader: 모든 윈도우를 소진했습니다.")
        sample = self._get_window(self._current_idx)
        self._current_idx += 1
        return sample

    def load_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """batch_size 만큼 순차적으로 샘플을 반환합니다."""
        batch = []
        for _ in range(batch_size):
            if self._current_idx >= self._window_count:
                break
            batch.append(self._get_window(self._current_idx))
            self._current_idx += 1
        return batch

    def load_by_index(self, index: int) -> Dict[str, Any]:
        """인덱스를 변경하지 않고 특정 윈도우를 반환합니다."""
        if not (0 <= index < self._window_count):
            raise IndexError(f"ETTmLoader: index {index} out of range [0, {self._window_count})")
        return self._get_window(index)

    def get_labels(self) -> np.ndarray:
        """전체 val 구간의 타겟 시계열을 원본 스케일로 반환합니다."""
        return self._data[self._split_start:self._split_end]

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "split":             self.split,
            "split_start":       self._split_start,
            "split_end":         self._split_end,
            "window_count":      self._window_count,
            "context_length":    self.context_length,
            "prediction_length": self.prediction_length,
            "stride":            self.stride,
            "feature_cols":      self.feature_cols,
            "num_channels":      len(self.feature_cols),
        }

    def preprocess(self, raw_input: np.ndarray) -> np.ndarray:
        """(T, C) 원본 윈도우 → RevIN 정규화된 past_values (1, T, C)."""
        result = self._preprocess(raw_input)
        return result["past_values"]

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _load_csv(self, csv_path: str) -> np.ndarray:
        """CSV를 읽어 feature_cols 컬럼만 추출한 (N, C) float32 배열을 반환합니다."""
        import pandas as pd

        df = pd.read_csv(csv_path)
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"ETTmLoader: CSV에 없는 컬럼: {missing}")
        return df[self.feature_cols].values.astype(np.float32)

    def _compute_split_bounds(self):
        """split 문자열에 따라 (start, end) 행 인덱스를 계산합니다."""
        n = len(self._data)
        if self.split_boundaries is not None:
            # 절대 인덱스 지정 (예: ETTh1 표준 벤치마크 split_boundaries=(8640, 11520))
            train_end, val_end = int(self.split_boundaries[0]), int(self.split_boundaries[1])
            # Val 구간 길이와 동일하게 Test 구간 할당
            test_end = val_end + (val_end - train_end)
            
            if self.split == "train":
                return 0, train_end
            elif self.split == "val":
                return train_end, val_end
            elif self.split == "test":
                return val_end, min(test_end, n)
        else:
            test_start = int(n * (1.0 - self.test_ratio))
            val_start  = int(n * (1.0 - self.val_ratio - self.test_ratio))
            if self.split == "test":
                return test_start, n
            elif self.split == "val":
                return val_start, test_start
            elif self.split == "train":
                return 0, val_start
        raise ValueError(f"ETTmLoader: split='{self.split}' 은 'train', 'val', 'test' 중 하나여야 합니다.")

    def _get_window(self, window_idx: int) -> Dict[str, Any]:
        """캐시 또는 계산을 통해 window_idx번째 샘플을 반환합니다."""
        cached = self._load_from_cache(window_idx)
        if cached is not None:
            return cached

        abs_start = self._split_start + window_idx * self.stride
        past_raw   = self._data[abs_start : abs_start + self.context_length]               # (T, C)
        future_raw = self._data[abs_start + self.context_length :
                                abs_start + self.context_length + self.prediction_length]  # (H, C)

        if self.normalize:
            preprocessed = self._preprocess(past_raw)
            past_values        = preprocessed["past_values"]
            past_observed_mask = preprocessed["past_observed_mask"]
            norm_stats         = preprocessed["norm_stats"]
        else:
            # normalize=False: raw 데이터 그대로 반환. 모델 내부 scaling(예: scaling="std") 사용 시.
            # norm_stats를 항등 변환(mean=0, std=1)으로 설정해 평가기 역정규화가 no-op이 되게 함.
            C = past_raw.shape[1]
            past_values        = past_raw.astype(np.float32)
            past_observed_mask = np.ones_like(past_values, dtype=bool)
            norm_stats         = {
                "mean": np.zeros(C, dtype=np.float32),
                "std":  np.ones(C,  dtype=np.float32),
            }

        sample = {
            "input": {
                "past_values":        past_values,
                "past_observed_mask": past_observed_mask,
            },
            "label": {
                "future_values": future_raw.astype(np.float32),
                "norm_stats":    norm_stats,
            },
            "window_idx": window_idx,
        }

        self._save_to_cache(window_idx, sample)
        return sample

    # ------------------------------------------------------------------
    # 캐시 (NPZ)
    # ------------------------------------------------------------------

    def _cache_path(self, window_idx: int) -> Optional[str]:
        if self.cache_dir is None:
            return None
        subdir = os.path.join(
            self.cache_dir,
            f"{self.split}_{self.context_length}_{self.prediction_length}_{self.stride}",
        )
        return os.path.join(subdir, f"{window_idx:06d}.npz")

    def _load_from_cache(self, window_idx: int) -> Optional[Dict[str, Any]]:
        path = self._cache_path(window_idx)
        if path is None or not os.path.exists(path):
            return None
        npz = np.load(path, allow_pickle=False)
        return {
            "input": {
                "past_values":        npz["past_values"],
                "past_observed_mask": npz["past_observed_mask"],
            },
            "label": {
                "future_values": npz["future_values"],
                "norm_stats": {
                    "mean": npz["norm_mean"],
                    "std":  npz["norm_std"],
                },
            },
            "window_idx": int(npz["window_idx"]),
        }

    def _save_to_cache(self, window_idx: int, sample: Dict[str, Any]) -> None:
        path = self._cache_path(window_idx)
        if path is None:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(
            path,
            past_values        = sample["input"]["past_values"],
            past_observed_mask = sample["input"]["past_observed_mask"],
            future_values      = sample["label"]["future_values"],
            norm_mean          = sample["label"]["norm_stats"]["mean"],
            norm_std           = sample["label"]["norm_stats"]["std"],
            window_idx         = np.array(window_idx),
        )
