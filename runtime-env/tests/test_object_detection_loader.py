import sys
import os
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.model_spec import Model_Spec, Task
from src.dataloader.object_detection_loader import ObjectDetectionLoader

import sys
import os
import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.model_spec import Model_Spec, Task
from src.dataloader.object_detection_loader import ObjectDetectionLoader

@pytest.fixture
def dummy_coco_dir(tmp_path):
    """Pytest fixture to create robust temporary Coco images and labels for testing."""
    img_dir = tmp_path / "images" / "val2017"
    label_dir = tmp_path / "labels" / "val2017"
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    # 1. Normal data
    img_path1 = img_dir / "000000000139.jpg"
    Image.new('RGB', (640, 640), color='black').save(img_path1)

    label_path1 = label_dir / "000000000139.txt"
    label_path1.write_text("0 0.5 0.5 0.2 0.2\n2 0.2 0.2 0.05 0.05\n")

    # 2. Edge case data (Bad formatting)
    img_path2 = img_dir / "000000000140.jpg"
    Image.new('RGB', (640, 640), color='white').save(img_path2)

    label_path2 = label_dir / "000000000140.txt"
    label_path2.write_text("class_id cx cy w h\n0.0 0.1 0.1 0.1 0.1\nwrong data format\n3.0 0.8 0.8 0.1 0.1\n")

    return tmp_path

def test_object_detection_loader_nchw(dummy_coco_dir):
    """Test object detection loader defaults to standard NCHW padding and parsing."""
    spec = Model_Spec(
        name="yolov5m",
        task=Task.OBJECT_DETECTION,
        input_shapes={"images": (1, 3, 640, 640)},
        input_dtype={"images": "float32"},
        output_shapes={"output0": (1, 25200, 85)}
    )

    img_dir = str(dummy_coco_dir / "images" / "val2017")
    label_dir = str(dummy_coco_dir / "labels" / "val2017")

    loader = ObjectDetectionLoader(
        spec, dataset_path=str(dummy_coco_dir), image_dir=img_dir, label_path=label_dir,
        mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]
    )

    batch_samples = loader.load_batch(2)
    assert len(batch_samples) == 2, "Batch loader did not return exactly 2 samples."

    for sample in batch_samples:
        img_path = sample.get("img_path", "unknown")
        tensor = sample["input"]
        label = sample["label"]

        assert isinstance(tensor, np.ndarray) and isinstance(label, np.ndarray), "Output must be strictly numpy arrays."
        assert tensor.shape == (3, 640, 640), "NCHW shape mismatch in output tensor."

        if "139" in img_path:
            assert label.shape == (2, 5), "Normal label shape mismatch."
            assert label[0][0] == 0.0, "Class ID parsing error."
        elif "140" in img_path:
            assert label.shape == (2, 5), "Edge case parsing failed to ignore invalid string lines."
            assert label[0][0] == 0.0 and label[1][0] == 3.0, "Float class ID parsing error on edge case."

def test_object_detection_loader_nhwc(dummy_coco_dir):
    """Test object detection loader supports custom NHWC layout dynamically."""
    spec = Model_Spec(
        name="yolov5_nhwc_custom",
        task=Task.OBJECT_DETECTION,
        input_shapes={"images": (1, 640, 640, 3)},
        input_dtype={"images": "float32"},
        output_shapes={"output0": (1, 25200, 85)}
    )

    img_dir = str(dummy_coco_dir / "images" / "val2017")
    label_dir = str(dummy_coco_dir / "labels" / "val2017")

    loader = ObjectDetectionLoader(
        spec, dataset_path=str(dummy_coco_dir), image_dir=img_dir, label_path=label_dir,
        layout="NHWC"
    )

    sample = loader.load_single()
    tensor = sample["input"]

    assert tensor.shape == (640, 640, 3), "NHWC logic failed to transpose the tensor correctly."
