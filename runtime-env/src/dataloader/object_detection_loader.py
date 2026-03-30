"""
Object Detection DataLoader

- 로컬 데이터 폴더에서 이미지를 읽고 전처리하는 순수 numpy 기반의 데이터 로더.
- Model_Spec 객체를 활용하여 모델 입력 형태를 추출.
- YOLO 형식(.txt)의 라벨 파일 형식을 기본으로 파싱.
"""

import os
import numpy as np
from PIL import Image
from typing import Dict, List, Any, Tuple
from .base import DataLoader
from ..core.model_spec import Model_Spec

class ObjectDetectionLoader(DataLoader):
    def __init__(self, model_spec: Model_Spec, **kwargs):
        """초기화 메서드."""
        self.model_spec = model_spec
        self.base_path = kwargs.get("dataset_path", "./data/coco128")
        
        # 1. 원칙 준수: 절대 경로 필수 주입 (스니핑 불가)
        self.image_dir = kwargs.get("image_dir")
        self.label_dir = kwargs.get("label_path")
        
        if not self.image_dir or not self.label_dir:
            raise ValueError("[ObjectDetectionLoader] image_dir 또는 label_path가 명시적으로 제공되지 않았습니다.")
        
        # 2. 이미지 파일 목록 로드
        self.image_files: List[str] = []
        if os.path.exists(self.image_dir):
            self.image_files = sorted([
                f for f in os.listdir(self.image_dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            
        self.total_samples = len(self.image_files)
        self.current_idx = 0
        
        # 3. 모델 정보 기반 속성 파싱
        self.target_hw = self._parse_target_shape(kwargs)
        self.mean, self.std = self._setup_normalization(kwargs)
        self.layout = kwargs.get("layout", "NCHW").upper()


    def _parse_target_shape(self, kwargs: Dict[str, Any]) -> Tuple[int, int]:
        """Model_Spec에서 입력 차원(H, W)을 안전하게 추출하는 헬퍼 메서드"""
        # 1. 사용자가 명시적으로 해상도를 덮어씌웠다면 취우선 적용 (예: target_hw=(1280, 1280))
        if "target_hw" in kwargs:
            return tuple(kwargs["target_hw"])
            
        # 2. Model_Spec의 텐서 형태에서 추론 (NCHW, NHWC 모두 대응)
        input_shape_tuple = next(iter(self.model_spec.input_shapes.values()))
        
        # 배치(보통 1)나 채널(보통 3)이 아닌, 4를 초과하는 큰 숫자들만 골라냅니다.
        # 예: (1, 640, 640, 3) 이면 spatial_dims는 [640, 640]이 됨
        spatial_dims = [dim for dim in input_shape_tuple if dim is not None and dim > 4]
        
        if len(spatial_dims) >= 2:
            return (spatial_dims[0], spatial_dims[1])
            
        # 3. 정보를 찾을 수 없으면 YOLO 기본 해상도 폴백
        return (640, 640)  # YOLO 기본 해상도 폴백

    def _setup_normalization(self, kwargs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """정규화 상수를 파싱하여 리턴하는 헬퍼 메서드"""
        mean = np.array(kwargs.get("mean", [0.0, 0.0, 0.0]), dtype=np.float32)
        std = np.array(kwargs.get("std", [1.0, 1.0, 1.0]), dtype=np.float32)
        return mean, std

    def _get_label_path(self, img_filename: str) -> str:
        """이미지 파일명에 1:1로 대응되는 YOLO 라벨 파일 경로 생성"""
        base_name = os.path.splitext(img_filename)[0]
        return os.path.join(self.label_dir, f"{base_name}.txt")

    def _parse_yolo_label(self, label_path: str) -> np.ndarray:
        """단일 텍스트 파일을 읽고 [class_id, cx, cy, w, h] Numpy 배열로 반환하는 파서"""
        labels_list = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            # '0.0' 처럼 소수점 형태의 클래스 아이디 방어 로직 추가
                            class_id = int(float(parts[0]))
                            cx, cy, w, h = map(float, parts[1:5])
                            labels_list.append([class_id, cx, cy, w, h])
                        except ValueError:
                            # 텍스트 헤더나 잘못된 라벨 포맷이 섞여있어도 크래시 내지 않고 무시합니다.
                            continue
        
        if len(labels_list) > 0:
            return np.array(labels_list, dtype=np.float32)
        return np.empty((0, 5), dtype=np.float32)

    def load_single(self) -> Dict[str, Any]:
        """단일 이미지와 매칭되는 라벨을 로드하고 최종 딕셔너리로 패키징합니다."""
        if self.current_idx >= self.total_samples:
            raise StopIteration("모든 샘플이 소진되었습니다.")
            
        img_filename = self.image_files[self.current_idx]
        self.current_idx += 1
        
        # 각 작업 위임
        img_path = os.path.join(self.image_dir, img_filename)
        label_path = self._get_label_path(img_filename)
        label_array = self._parse_yolo_label(label_path)
        tensor = self.preprocess(img_path)
        
        return {
            "input": tensor,
            "label": label_array,
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

    def load_by_index(self, index: int) -> Dict[str, Any]:
        """
        인덱스 기반 직접 접근 — LoadGen QSL 콜백 및 랜덤 접근 지원용.
        current_idx 상태를 변경하지 않습니다.
        """
        if index < 0 or index >= self.total_samples:
            raise IndexError(
                f"index {index} is out of range [0, {self.total_samples})"
            )

        img_info         = self.images_info[index]
        img_id           = img_info["id"]
        img_filename     = img_info["file_name"]
        original_width   = img_info.get("width",  None)
        original_height  = img_info.get("height", None)

        img_path = os.path.join(self.image_dir, img_filename)
        if not os.path.exists(img_path):
            return {"input": None, "targets": {"boxes": [], "labels": []},
                    "img_path": img_path, "error": "Not Found"}

        img = Image.open(img_path).convert("RGB")
        if original_width is None or original_height is None:
            original_width, original_height = img.size

        tensor  = self.preprocess(img)
        scale_x = self.target_hw[1] / original_width
        scale_y = self.target_hw[0] / original_height

        boxes, labels = [], []
        for ann in self.annotations_map.get(img_id, []):
            x_min, y_min, w, h = ann["bbox"]
            boxes.append([x_min * scale_x, y_min * scale_y, w * scale_x, h * scale_y])
            labels.append(ann["category_id"])

        return {
            "input": tensor,
            "targets": {
                "boxes":  np.array(boxes,  dtype=np.float32) if boxes  else np.empty((0, 4), dtype=np.float32),
                "labels": np.array(labels, dtype=np.int64)   if labels else np.empty((0,),   dtype=np.int64),
            },
            "img_path":      img_path,
            "original_size": (original_height, original_width),
        }

    def get_labels(self) -> Any:
        return None

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "dataset_path": self.base_path,
            "target_hw": self.target_hw,
            "mean": self.mean.tolist(),
            "std": self.std.tolist()
        }

    def preprocess(self, raw_input: Any) -> np.ndarray:
        """순수 Numpy와 PIL 위주로 이미지 리사이즈 체인 수행"""
        if isinstance(raw_input, str):
            img = Image.open(raw_input)
        else:
            img = raw_input
            
        img = img.convert("RGB")
        img = img.resize((self.target_hw[1], self.target_hw[0]), Image.Resampling.BILINEAR)
        
        # 스케일링 (0~255) -> (0.0~1.0)
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # 정규화
        img_array = (img_array - self.mean) / self.std
        
        # 3. 모델의 입맛에 맞게(NCHW vs NHWC) 메모리 레이아웃 변경 대응
        if self.layout == "NHWC":
            # 이미지가 (H, W, C) 형태이므로 그대로 통과
            pass
        else:
            # 보편적인 PyTorch/ONNX 스타일인 NCHW (C, H, W) 포맷 전환
            img_array = np.transpose(img_array, (2, 0, 1))
            
        return img_array
