"""
Image Classification DataLoader

- 로컬 데이터 폴더에서 이미지를 읽고 전처리하는 순수 공급 프레임워크입니다.
- `Model_Spec` 객체를 활용하여 모델 입력 형태(shape)를 추출합니다.
- 영상 처리 특화 속성인(mean, std)는 `model_spec.name`을 통해 유추하거나
  kwargs를 통해 커스텀 주입받습니다.
"""

import os
import json
from typing import Dict, List, Any
import numpy as np
from PIL import Image
from .base import DataLoader
from ..core.model_spec import Model_Spec

# 범용 ImageNet 정규화 상수 (기본 폴백 용도)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class ImageClassificationLoader(DataLoader):
    def __init__(self, model_spec: Model_Spec, **kwargs):
        """
        초기화 메서드. 
        
        Args:
            model_spec (Model_Spec): 코어 스펙 규격 인스턴스.
            **kwargs:
                - 'dataset_path' (str): 데이터셋 로컬 루트 디렉토리
                - 'mean' (tuple/list): 커스텀 정규화 평균값 (ex: [0.5, 0.5, 0.5])
                - 'std' (tuple/list): 커스텀 정규화 표준편차값
        """
        self.model_spec = model_spec
        
        # 1. 경로 설정 (kwargs로 주입받거나 디폴트 사용)
        self.base_path = kwargs.get("dataset_path", "./data/dummy_dataset")
        # 파일 구조가 다른 경우(예: ImageNet 1k 스크립트) kwargs로 개별 주입 수용
        self.image_dir = kwargs.get("image_dir", os.path.join(self.base_path, "images"))
        self.label_file = kwargs.get("label_file", os.path.join(self.base_path, "labels.json"))
        
        # 대상 폴더에서 파일 읽기
        self.image_files: List[str] = []
        self.labels_map = {}
        if os.path.exists(self.image_dir):
            self.image_files = sorted(os.listdir(self.image_dir))
            if os.path.exists(self.label_file):
                if self.label_file.endswith(".json"):
                    with open(self.label_file, "r") as f:
                        self.labels_map = json.load(f)
                else:
                    # TXT 등 한줄 형태 (예: image_0000.jpg 123)
                    with open(self.label_file, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                self.labels_map[parts[0]] = int(parts[1])
                
        self.total_samples = len(self.image_files)
        self.current_idx = 0
        
        # 2. 형상(Shape) 정보 파싱
        # Model_Spec의 input_shapes는 dict 형태 (예: {'input_1': (1, 3, 224, 224)})
        # 통상 비전 모델은 입력을 1개만 받는다고 간주하고 첫 번째 밸류의 시각 정보(H, W)를 추출
        input_shape_tuple = next(iter(model_spec.input_shapes.values()))
        
        # (N, C, H, W) 또는 (1, 224, 224, 3) 등 다양한 포맷 중
        # 마지막 24나 224 등 공간 차원을 PIL 모델 리사이즈 기준으로 삼기 위한 휴리스틱:
        # 편의를 위해 일단 Vision 모델의 경우 224x224 등의 정사각형 또는 가장 큰 두 숫자를 추출하거나
        # 아니면 NCHW 기준 끝의 두 개(H, W)라고 가정합니다.
        if len(input_shape_tuple) >= 2:
            self.target_hw = (input_shape_tuple[-2], input_shape_tuple[-1])
        else:
            self.target_hw = (224, 224)  # 폴백

        # 3. 정규화 특화 상수(mean, std) 파싱 설계
        # 사용자가 kwargs로 지정했다면 최우선
        if "mean" in kwargs:
            self.mean = np.array(kwargs["mean"], dtype=np.float32)
        else:
            # 모델 이름을 보고 유추 (단순화: 일단 대부분 ImageNet 쓴다고 가정)
            # 추후 resnet, mobilenet 등 파싱 로직 확장 가능
            self.mean = np.array(IMAGENET_MEAN, dtype=np.float32)
            
        if "std" in kwargs:
            self.std = np.array(kwargs["std"], dtype=np.float32)
        else:
            self.std = np.array(IMAGENET_STD, dtype=np.float32)

    def load_single(self) -> Dict[str, Any]:
        if self.current_idx >= self.total_samples:
            raise StopIteration("모든 샘플이 소진되었습니다.")
            
        img_filename = self.image_files[self.current_idx]
        self.current_idx += 1
        
        img_path = os.path.join(self.image_dir, img_filename)
        label = self.labels_map.get(img_filename, -1)
        
        tensor = self.preprocess(img_path)
        return {
            "input": tensor,
            "label": label,
            "img_path": img_path
        }

    def load_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(self.load_single())
            except StopIteration:
                break
        return batch

    def get_labels(self) -> Any:
        return self.labels_map

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "dataset_path": self.base_path,
            "target_hw": self.target_hw,
            "mean": self.mean.tolist(),
            "std": self.std.tolist()
        }

    def preprocess(self, raw_input: Any) -> np.ndarray:
        if isinstance(raw_input, str):
            img = Image.open(raw_input)
        else:
            img = raw_input
            
        img = img.convert("RGB")
        img = img.resize((self.target_hw[1], self.target_hw[0]), Image.Resampling.BILINEAR) # PIL은 (W, H)
        
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = (img_array - self.mean) / self.std
        
        # (H, W, C) -> (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))
        return img_array
