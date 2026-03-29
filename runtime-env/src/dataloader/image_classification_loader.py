"""
Image Classification DataLoader

- 로컬 데이터 폴더에서 이미지를 읽고 전처리하는 순수 공급 프레임워크입니다.
- `Model_Spec` 객체를 활용하여 모델 입력 형태(shape)를 추출합니다.
- 전처리 로직은 PreprocessStrategy 객체로 완전히 분리됩니다.
- MLPerf의 .npy 디스크 캐싱 패턴을 채택하여 반복 실행 시 전처리 비용을 제거합니다.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from PIL import Image

from .base import DataLoader
from .preprocess_strategies import (
    PreprocessStrategy,
    MLPerfResNet50Preprocess,
    DirectResizePreprocess,
)
from ..core.model_spec import Model_Spec

# 범용 ImageNet 정규화 상수 (기본 폴백 용도)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class ImageClassificationLoader(DataLoader):
    def __init__(self, model_spec: Model_Spec, **kwargs):
        """
        초기화 메서드.

        Args:
            model_spec (Model_Spec): 코어 스펙 규격 인스턴스.
            **kwargs:
                - dataset_path (str)            : 데이터셋 루트 디렉토리
                - image_dir (str)               : 이미지 파일 디렉토리 (dataset_path/images 폴백)
                - label_file (str)              : 레이블 파일 경로 (json or txt)
                - mean (tuple/list)             : 커스텀 정규화 평균값
                - std  (tuple/list)             : 커스텀 정규화 표준편차
                - preprocess_strategy           : PreprocessStrategy 구현 인스턴스
                                                  (기본값: MLPerfResNet50Preprocess)
                - cache_dir (str)               : .npy 캐시 디렉토리 (None이면 캐싱 비활성)
        """
        self.model_spec = model_spec

        # 1. 경로 설정
        self.base_path  = kwargs.get("dataset_path", "./data/dummy_dataset")
        self.image_dir  = kwargs.get("image_dir",  os.path.join(self.base_path, "images"))
        self.label_file = kwargs.get("label_file", os.path.join(self.base_path, "labels.json"))

        # 2. 이미지 파일 목록 & 레이블 맵 구성
        self.image_files: List[str] = []
        self.labels_map: Dict[str, int] = {}

        if os.path.exists(self.image_dir):
            self.image_files = sorted(os.listdir(self.image_dir))

        if os.path.exists(self.label_file):
            if self.label_file.endswith(".json"):
                with open(self.label_file, "r") as f:
                    self.labels_map = json.load(f)
            else:
                # "filename.jpg 123" 형식 TXT
                with open(self.label_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            self.labels_map[parts[0]] = int(parts[1])

        self.total_samples = len(self.image_files)
        self.current_idx   = 0

        # 3. 입력 형상(H, W) 파싱
        input_shape_tuple = next(iter(model_spec.input_shapes.values()))

        # 배치(보통 1)나 채널(보통 3)이 아닌, 4를 초과하는 큰 숫자들을 해상도(H, W)로 인식
        spatial_dims = [dim for dim in input_shape_tuple if dim is not None and dim > 4]

        if len(spatial_dims) >= 2:
            self.target_hw = (spatial_dims[0], spatial_dims[1])
        else:
            self.target_hw = (224, 224)  # 폴백

        # 4. 정규화 상수 결정 (kwargs > preprocessor_config.json > ImageNet 기본값)
        config_mean, config_std = self._try_load_preprocessor_config()

        # 사용자가 kwargs로 지정 > config.json 파싱 > 기본값 순으로 결정
        if "mean" in kwargs:
            self.mean = np.array(kwargs["mean"], dtype=np.float32)
        elif config_mean is not None:
            self.mean = np.array(config_mean, dtype=np.float32)
        else:
            self.mean = np.array(IMAGENET_MEAN, dtype=np.float32)

        if "std" in kwargs:
            self.std = np.array(kwargs["std"], dtype=np.float32)
        elif config_std is not None:
            self.std = np.array(config_std, dtype=np.float32)
        else:
            self.std = np.array(IMAGENET_STD, dtype=np.float32)

        self.layout = kwargs.get("layout", "NCHW").upper()

        # 5. 전처리 전략 주입 (기본값: MLPerfResNet50Preprocess)
        self.preprocess_strategy: PreprocessStrategy = kwargs.get(
            "preprocess_strategy", MLPerfResNet50Preprocess()
        )

        # 6. .npy 디스크 캐시 설정 (MLPerf 캐싱 패턴 채택)
        self.cache_dir: Optional[str] = kwargs.get("cache_dir", None)
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"[DataLoader] Preprocessing cache enabled: {self.cache_dir}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _try_load_preprocessor_config(self):
        """모델 경로 인근의 preprocessor_config.json을 탐색하여 mean/std를 반환."""
        if "onnx" not in self.model_spec.model_paths:
            return None, None
        onnx_path   = self.model_spec.model_paths["onnx"]
        config_path = os.path.join(os.path.dirname(onnx_path), "preprocessor_config.json")
        if not os.path.exists(config_path):
            return None, None
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            print(f"[DataLoader] Loaded preprocessing config from {config_path}")
            return cfg.get("image_mean"), cfg.get("image_std")
        except Exception as e:
            print(f"[DataLoader] Failed to parse preprocessor_config.json: {e}")
            return None, None

    def _get_cache_path(self, img_filename: str) -> Optional[str]:
        """이미지 파일명 기반으로 .npy 캐시 파일 경로를 반환. 캐시 비활성 시 None."""
        if not self.cache_dir:
            return None
        stem = Path(img_filename).stem
        return os.path.join(self.cache_dir, stem + ".npy")

    def _load_or_preprocess(self, img_path: str, img_filename: str) -> np.ndarray:
        """
        .npy 캐시가 있으면 로드, 없으면 전처리 전략을 실행 후 캐시에 저장합니다.
        (MLPerf Imagenet.process() 패턴 채택)
        """
        cache_path = self._get_cache_path(img_filename)

        # 캐시 히트
        if cache_path and os.path.exists(cache_path):
            return np.load(cache_path)

        # 전처리 실행
        img    = Image.open(img_path)
        tensor = self.preprocess_strategy(img, self.target_hw, self.mean, self.std)

        # 캐시 저장
        if cache_path:
            np.save(cache_path, tensor)

        return tensor

    # ------------------------------------------------------------------
    # DataLoader ABC 구현
    # ------------------------------------------------------------------

    def load_single(self) -> Dict[str, Any]:
        if self.current_idx >= self.total_samples:
            raise StopIteration("모든 샘플이 소진되었습니다.")

        img_filename   = self.image_files[self.current_idx]
        self.current_idx += 1

        img_path = os.path.join(self.image_dir, img_filename)
        label    = self.labels_map.get(img_filename, -1)
        tensor   = self._load_or_preprocess(img_path, img_filename)

        return {"input": tensor, "label": label, "img_path": img_path}

    def load_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(self.load_single())
            except StopIteration:
                break
        return batch

    def load_by_index(self, index: int) -> Dict[str, Any]:
        """
        인덱스 기반 직접 접근 — LoadGen QSL 콜백 및 랜덤 접근 지원용.
        current_idx 상태를 변경하지 않습니다.
        """
        if index < 0 or index >= self.total_samples:
            raise IndexError(
                f"index {index} is out of range [0, {self.total_samples})"
            )

        img_filename = self.image_files[index]
        img_path     = os.path.join(self.image_dir, img_filename)
        label        = self.labels_map.get(img_filename, -1)
        tensor       = self._load_or_preprocess(img_path, img_filename)

        return {"input": tensor, "label": label, "img_path": img_path}

    def get_labels(self) -> Any:
        return self.labels_map

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "total_samples":        self.total_samples,
            "dataset_path":         self.base_path,
            "target_hw":            self.target_hw,
            "mean":                 self.mean.tolist(),
            "std":                  self.std.tolist(),
            "preprocess_strategy":  type(self.preprocess_strategy).__name__,
            "cache_dir":            self.cache_dir,
        }

    def preprocess(self, raw_input: Any) -> np.ndarray:
        """
        단일 raw 입력을 현재 전략으로 전처리합니다.
        파일 경로(str) 또는 PIL.Image 객체를 받습니다.
        """
        img = Image.open(raw_input) if isinstance(raw_input, str) else raw_input
        return self.preprocess_strategy(img, self.target_hw, self.mean, self.std)
