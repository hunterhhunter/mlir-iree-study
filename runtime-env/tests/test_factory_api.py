import sys
import os
import tempfile
import pytest

# 프로젝트 최상단 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.model_spec import Model_Spec, Task
# 외부(Evaluator) 입장에선 패키지(__init__.py)로부터만 깔끔하게 임포트합니다.
from dataloader import create_dataloader, ImageClassificationLoader

def test_create_image_loader():
    # ImageClassificationLoader는 image_dir와 label_path를 필수 인자로 요구합니다.
    # 더미 이미지 파일과 레이블 파일로 팩토리 라우팅 검증
    with tempfile.TemporaryDirectory() as tmpdir:
        image_dir = os.path.join(tmpdir, "images")
        os.makedirs(image_dir)
        # 빈 더미 이미지 파일 (FileNotFoundError 방지)
        open(os.path.join(image_dir, "dummy.jpg"), "w").close()
        label_path = os.path.join(tmpdir, "labels.txt")
        open(label_path, "w").close()

        spec = Model_Spec(
            name="resnet50",
            task=Task.IMAGE_CLASSIFICATION,
            input_shapes={"input": (1, 3, 224, 224)},
            input_dtype={"input": "float32"},
            output_shapes={"output": (1, 1000)}
        )

        loader = create_dataloader(spec, dataset_path=tmpdir, image_dir=image_dir, label_path=label_path)
        assert isinstance(loader, ImageClassificationLoader)

def test_unsupported_task():
    # SEMANTIC_SEGMENTATION은 현재 구현되지 않은 Task → ValueError 발생해야 함
    spec = Model_Spec(
        name="seg-model",
        task=Task.SEMANTIC_SEGMENTATION,
        input_shapes={"input": (1, 3, 512, 512)},
        input_dtype={"input": "float32"},
        output_shapes={"output": (1, 21, 512, 512)}
    )

    with pytest.raises(ValueError):
        create_dataloader(spec)

if __name__ == "__main__":
    print("=== DataLoader Factory API External Integration Test ===\n")
    test_create_image_loader()
    test_unsupported_task()
    print("\n[+] Factory interface test finished!")
