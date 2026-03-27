import sys
import os
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.model_spec import Model_Spec, Task
from src.dataloader.object_detection_loader import ObjectDetectionLoader

def setup_dummy_coco_format(base_dir):
    """테스트를 위한 임시 이미지 및 라벨 생성"""
    img_dir = os.path.join(base_dir, "images", "val2017")
    label_dir = os.path.join(base_dir, "labels", "val2017")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # 1. 정상 데이터
    img_path1 = os.path.join(img_dir, "000000000139.jpg")
    img1 = Image.new('RGB', (640, 640), color='black')
    img1.save(img_path1)

    label_path1 = os.path.join(label_dir, "000000000139.txt")
    with open(label_path1, "w") as f:
        f.write("0 0.5 0.5 0.2 0.2\n")
        f.write("2 0.2 0.2 0.05 0.05\n")

    # 2. 예외 처리 확인용 에지 케이스 데이터
    img_path2 = os.path.join(img_dir, "000000000140.jpg")
    img2 = Image.new('RGB', (640, 640), color='white')
    img2.save(img_path2)

    label_path2 = os.path.join(label_dir, "000000000140.txt")
    with open(label_path2, "w") as f:
        f.write("class_id cx cy w h\n")
        f.write("0.0 0.1 0.1 0.1 0.1\n")
        f.write("wrong data format without enough numbers\n")
        f.write("3.0 0.8 0.8 0.1 0.1\n")

    return img_dir, label_dir

def test_object_detection_loader():
    print("\n=== [1] NCHW Format Test ===")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", "dummy_coco"))
    img_dir, label_dir = setup_dummy_coco_format(base_dir)

    spec = Model_Spec(
        name="yolov5m",
        task=Task.OBJECT_DETECTION,
        input_shapes={"images": (1, 3, 640, 640)},
        input_dtype={"images": "float32"},
        output_shapes={"output0": (1, 25200, 85)}
    )

    loader = ObjectDetectionLoader(
        spec, dataset_path=base_dir, image_dir=img_dir, label_dir=label_dir,
        mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]
    )

    batch_samples = loader.load_batch(2)

    for i, sample in enumerate(batch_samples):
        img_path = sample.get("img_path", "unknown")
        tensor = sample["input"]
        label = sample["label"]
        print(f"  [{i+1}] {os.path.basename(img_path)}")
        print(f"      -> Input Shape: {tensor.shape}")
        print(f"      -> Parsed Labels:\n{label}")

        assert isinstance(tensor, np.ndarray) and isinstance(label, np.ndarray), "Output must be numpy.ndarray"
        assert tensor.shape == (3, 640, 640), "NCHW shape mismatch"

        # 1. 정상 케이스 확인
        if "139" in img_path:
            assert label.shape == (2, 5), "Label shape mismatch"
            assert label[0][0] == 0.0, "Class ID parsing error"

        # 2. 에지 케이스 검증 (이상 데이터 필터링 여부)
        elif "140" in img_path:
            assert label.shape == (2, 5), "Edge case parsing failed: invalid text headers were not ignored."
            assert label[0][0] == 0.0 and label[1][0] == 3.0, "Float class ID parsing error"
            print("      [*] Edge case parsing successful.")

    print("\n    [SUCCESS] NCHW format test passed. ✅")

def test_nhwc_layout_handling():
    print("\n=== [2] NHWC Layout Test ===")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", "dummy_coco"))

    spec = Model_Spec(
        name="yolov5_nhwc_custom",
        task=Task.OBJECT_DETECTION,
        input_shapes={"images": (1, 640, 640, 3)},
        input_dtype={"images": "float32"},
        output_shapes={"output0": (1, 25200, 85)}
    )

    loader = ObjectDetectionLoader(
        spec, dataset_path=base_dir,
        image_dir=os.path.join(base_dir, "images", "val2017"),
        label_dir=os.path.join(base_dir, "labels", "val2017"),
        layout="NHWC"
    )

    sample = loader.load_single()
    tensor = sample["input"]
    print(f"[*] NHWC Output Shape: {tensor.shape}")

    assert tensor.shape == (640, 640, 3), "NHWC logic failed."
    print("    [SUCCESS] NHWC format test passed! ✅")

if __name__ == "__main__":
    try:
        test_object_detection_loader()
        test_nhwc_layout_handling()
        print("\n[+] test_object_detection_loader.py Execution Finished!\n")
    except Exception as e:
        print(f"\n[-] test failed: {e}")
        sys.exit(1)
