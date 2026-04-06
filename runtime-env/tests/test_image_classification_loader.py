import sys
import os

# 환경 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.model_spec import Model_Spec, Task
from dataloader.image_classification_loader import ImageClassificationLoader

def test_imagenet():
    print("\n=== [1] Real ImageNet Validation Data Integration Test ===")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets/imagenet_1k"))
    val_images_dir = os.path.join(base_dir, "val")
    val_labels_file = os.path.join(base_dir, "val_labels.txt")
    
    if not os.path.exists(val_images_dir):
        import pytest
        pytest.skip(f"{val_images_dir} 이 존재하지 않습니다.")

    spec = Model_Spec(
        name="resnet50",
        task=Task.IMAGE_CLASSIFICATION,
        input_shapes={"input": (1, 3, 224, 224)},
        input_dtype={"input": "float32"},
        output_shapes={"output": (1, 1000)}
    )
    
    loader = ImageClassificationLoader(
        spec, 
        dataset_path=base_dir,
        image_dir=val_images_dir,
        label_path=val_labels_file
    )
    
    print(f"[*] Metadata: {loader.get_metadata()}")
    batch_samples = loader.load_batch(2)
    
    for i, sample in enumerate(batch_samples):
        img_path = sample.get('img_path', 'unknown')
        tensor = sample['input']
        print(f"  [{i+1}] {os.path.basename(img_path)} -> Shape: {tensor.shape}, Label: {sample['label']}")

def test_cifar10():
    print("\n=== [2] Real CIFAR-10 Test Data Integration Test ===")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets/cifar10"))
    test_images_dir = os.path.join(base_dir, "test")
    test_labels_file = os.path.join(base_dir, "test_labels.txt")
    
    if not os.path.exists(test_images_dir):
        import pytest
        pytest.skip(f"{test_images_dir} 이 존재하지 않습니다.")

    # CIFAR-10 은 보통 32x32 해상도
    spec = Model_Spec(
        name="mobilenet_v2",  # 임의 지정
        task=Task.IMAGE_CLASSIFICATION,
        input_shapes={"input": (1, 3, 32, 32)},
        input_dtype={"input": "float32"},
        output_shapes={"output": (1, 10)}
    )
    
    loader = ImageClassificationLoader(
        spec, 
        dataset_path=base_dir,
        image_dir=test_images_dir,
        label_path=test_labels_file
    )
    
    print(f"[*] Metadata: {loader.get_metadata()}")
    batch_samples = loader.load_batch(2)
    
    for i, sample in enumerate(batch_samples):
        img_path = sample.get('img_path', 'unknown')
        tensor = sample['input']
        print(f"  [{i+1}] {os.path.basename(img_path)} -> Shape: {tensor.shape}, Label: {sample['label']}")

if __name__ == "__main__":
    test_imagenet()
    test_cifar10()
    print("\n[+] All integration tests finished!")
