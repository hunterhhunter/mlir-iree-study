import sys
import os
import pytest

# 프로젝트 최상단 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.model_spec import Model_Spec, Task
# 외부(Evaluator) 입장에선 패키지(__init__.py)로부터만 깔끔하게 임포트합니다.
from dataloader import create_dataloader, ImageClassificationLoader

def test_create_image_loader():
    # 더미 데이터 경로 세팅 (실제 다운로드 된 데이터나 fallback 깡통 폴더 사용)
    dummy_dir = os.path.join(os.path.dirname(__file__), "dummy_data")

    spec = Model_Spec(
        name="resnet50",
        task=Task.IMAGE_CLASSIFICATION,
        input_shapes={"input": (1, 3, 224, 224)},
        input_dtype={"input": "float32"},
        output_shapes={"output": (1, 1000)}
    )

    loader = create_dataloader(spec, dataset_path=dummy_dir)
    assert isinstance(loader, ImageClassificationLoader)

def test_unsupported_task():
    spec = Model_Spec(
        name="bert-base",
        task=Task.NLP_CLASSIFICATION,
        input_shapes={"input_ids": (1, 128)},
        input_dtype={"input_ids": "int64"},
        output_shapes={"probs": (1, 2)}
    )

    with pytest.raises(ValueError):
        create_dataloader(spec)

if __name__ == "__main__":
    print("=== DataLoader Factory API External Integration Test ===\n")
    test_create_image_loader()
    test_unsupported_task()
    print("\n[+] Factory interface test finished!")
