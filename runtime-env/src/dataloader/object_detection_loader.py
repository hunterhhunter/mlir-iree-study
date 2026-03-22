"""
Object Detection DataLoader

- 로컬 데이터 폴더에서 이미지를 읽고 COCO 포맷의 JSON 어노테이션을 파싱합니다.
- 이미지를 모델의 입력 형태에 맞게 리사이징(Padding or Scale)하며, 
  동시에 Bounding Box의 좌표도 리사이징된 비율에 맞게 보정하여 반환합니다.
"""

import os
import json
from typing import Dict, List, Any, Tuple
import numpy as np
from PIL import Image
from .base import DataLoader
from ..core.model_spec import Model_Spec

# 범용 ImageNet 정규화 상수 (기본 폴백 용도)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class ObjectDetectionLoader(DataLoader):
    def __init__(self, model_spec: Model_Spec, **kwargs):
        """
        초기화 메서드. COCO 포맷 형태의 어노테이션을 파싱하고 매핑합니다.
        """
        self.model_spec = model_spec
        
        # 1. 경로 설정
        self.base_path = kwargs.get("dataset_path", "./data/dummy_dataset")
        self.image_dir = kwargs.get("image_dir", os.path.join(self.base_path, "images"))
        self.label_file = kwargs.get("label_file", os.path.join(self.base_path, "annotations.json"))
        
        # 2. COCO 딕셔너리 파싱 및 매핑
        self.images_info: List[Dict[str, Any]] = []
        self.annotations_map: Dict[int, List[Dict[str, Any]]] = {}
        
        if os.path.exists(self.label_file):
            print(f"[DataLoader] Parsing COCO annotations from {self.label_file}...")
            with open(self.label_file, "r") as f:
                coco_data = json.load(f)
            
            # 이미지 목록 추출
            if "images" in coco_data:
                self.images_info = coco_data["images"]
                for img in self.images_info:
                    self.annotations_map[img["id"]] = []
            
            # 어노테이션(Bbox) 목록 매핑
            if "annotations" in coco_data:
                for ann in coco_data["annotations"]:
                    img_id = ann["image_id"]
                    if img_id in self.annotations_map:
                        self.annotations_map[img_id].append(ann)
        else:
            print(f"[DataLoader] Warning: Annotation file not found at {self.label_file}")
            
        self.total_samples = len(self.images_info)
        self.current_idx = 0
        
        # 3. 형상(Shape) 정보 파싱
        input_shape_tuple = next(iter(model_spec.input_shapes.values()))
        if len(input_shape_tuple) >= 2:
            self.target_hw = (input_shape_tuple[-2], input_shape_tuple[-1])
        else:
            self.target_hw = (640, 640)  # 일반적인 객체 탐지 폴백 (e.g. YOLO)
            
        # 4. 정규화 특화 상수 파싱 설계
        config_mean, config_std = None, None
        if "onnx" in self.model_spec.model_paths:
            onnx_path = self.model_spec.model_paths["onnx"]
            config_path = os.path.join(os.path.dirname(onnx_path), "preprocessor_config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        config_data = json.load(f)
                    config_mean = config_data.get("image_mean")
                    config_std = config_data.get("image_std")
                    print(f"[DataLoader] Loaded preprocessing config from {config_path}")
                except Exception as e:
                    print(f"[DataLoader] Failed to parse config: {e}")

        # 사용자 입력 > 설정 파일 > 기본값 순의 대체(Fallback) 처리
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

    def load_single(self) -> Dict[str, Any]:
        """단일 이미지를 로드하고 바운딩 박스를 리사이즈된 크기에 맞춰 변환하여 반환합니다."""
        if self.current_idx >= self.total_samples:
            raise StopIteration("모든 샘플이 소진되었습니다.")
            
        img_info = self.images_info[self.current_idx]
        img_id = img_info["id"]
        img_filename = img_info["file_name"]
        
        original_width = img_info.get("width", None)
        original_height = img_info.get("height", None)
        
        self.current_idx += 1
        
        img_path = os.path.join(self.image_dir, img_filename)
        raw_annotations = self.annotations_map.get(img_id, [])
        
        # 이미지 로드 실패 시 에러 처리
        if not os.path.exists(img_path):
             return {"input": None, "targets": {"boxes": [], "labels": []}, "img_path": img_path, "error": "Not Found"}
             
        img = Image.open(img_path).convert("RGB")
        
        # JSON 명세에 크기값이 없는 경우 실측
        if original_width is None or original_height is None:
            original_width, original_height = img.size
            
        tensor = self.preprocess(img)
        
        # Bounding Box 위치(Scale) 보정
        # COCO 포맷: [x_min, y_min, width, height]
        scale_x = self.target_hw[1] / original_width
        scale_y = self.target_hw[0] / original_height
        
        boxes = []
        labels = []
        for ann in raw_annotations:
            x_min, y_min, w, h = ann["bbox"]
            scaled_box = [x_min * scale_x, y_min * scale_y, w * scale_x, h * scale_y]
            boxes.append(scaled_box)
            labels.append(ann["category_id"])
            
        return {
            "input": tensor,
            "targets": {
                "boxes": np.array(boxes, dtype=np.float32) if boxes else np.empty((0, 4), dtype=np.float32),
                "labels": np.array(labels, dtype=np.int64) if labels else np.empty((0,), dtype=np.int64)
            },
            "img_path": img_path,
            "original_size": (original_height, original_width)  # 원본 복원 및 평가 시 필요
        }

    def load_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        batch = []
        for _ in range(batch_size):
            try:
                data = self.load_single()
                if "error" not in data:
                     batch.append(data)
            except StopIteration:
                break
        return batch

    def get_labels(self) -> Any:
        return self.annotations_map

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
        img = img.resize((self.target_hw[1], self.target_hw[0]), Image.Resampling.BILINEAR)
        
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = (img_array - self.mean) / self.std
        
        # (H, W, C) -> (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))
        return img_array
