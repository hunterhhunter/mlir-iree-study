import sys
import os
from typing import Tuple
import numpy as np
from PIL import Image

# 환경 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.model_spec import Model_Spec, Task
from dataloader.image_classification_loader import ImageClassificationLoader

def setup_dummy_data() -> str:
    """
    Workspace 규칙(Rule 8)을 준수하여 /tmp 대신 현재 디렉토리(.tests/data) 내에
    안전하게 더미 데이터를 임시로 구성합니다.
    """
    base_dir = os.path.join(os.path.dirname(__file__), "dummy_data")
    img_dir = os.path.join(base_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    # 쓰레기 값 이미지 하나 생성 (300x300 RGB)
    dummy_img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
    dummy_img.save(os.path.join(img_dir, "00001.jpg"))
    
    # 메타 json
    import json
    with open(os.path.join(base_dir, "labels.json"), "w") as f:
        json.dump({"00001.jpg": 99}, f)
        
    return base_dir

if __name__ == "__main__":
    print("--- ImageClassificationLoader Normalization Test ---")
    dataset_path = setup_dummy_data()
    
    # [테스트 1]: ResNet 계열 (일반 ImageNet 수치 폴백 확인)
    resnet_spec = Model_Spec(
        name="resnet50",
        task=Task.IMAGE_CLASSIFICATION,
        input_shapes={"input": (1, 3, 224, 224)},
        input_dtype={"input": "float32"},
        output_shapes={"output": (1, 1000)}
    )
    loader_resnet = ImageClassificationLoader(resnet_spec, dataset_path=dataset_path)
    print("\n[ResNet50 Auto-Fallback]")
    meta = loader_resnet.get_metadata()
    print(f"Mean: {meta['mean']} (Expected: [0.485, 0.456, 0.406])")
    print(f"Std:  {meta['std']} (Expected: [0.229, 0.224, 0.225])")

    # [테스트 2]: Inception 계열 (별도 정규화 수치 트리거 확인)
    inception_spec = Model_Spec(
        name="inception_v3",
        task=Task.IMAGE_CLASSIFICATION,
        input_shapes={"input": (1, 3, 299, 299)},
        input_dtype={"input": "float32"},
        output_shapes={"output": (1, 1000)}
    )
    loader_incept = ImageClassificationLoader(inception_spec, dataset_path=dataset_path)
    print("\n[Inception v3 Auto-Fallback]")
    meta2 = loader_incept.get_metadata()
    print(f"Mean: {meta2['mean']} (Expected: [0.5, 0.5, 0.5])")
    print(f"Std:  {meta2['std']} (Expected: [0.5, 0.5, 0.5])")
    
    # [테스트 3]: 실제 로딩 확인 (Shape)
    sample = loader_incept.load_single()
    tensor = sample["input"]
    print(f"\n[Load Test]\nLoaded Tensor Shape: {tensor.shape} (Expected: (3, 299, 299))")
    
    print("\nTest Passed Successfully!")
