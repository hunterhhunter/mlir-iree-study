import unittest
import os
import json
import shutil
import torch
import numpy as np
from PIL import Image
import sys
import os
# 프로젝트 루트 디렉토리를 검색 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.dataloader import get_dataloader, DetectionDataset, UniversalDataLoader

class MockProcessor:
    """테스트용 가상 이미지 프로세서 (Detection용)"""
    def __call__(self, images, return_tensors="pt"):
        # dummy pixel values (1, 3, 224, 224)
        return {"pixel_values": torch.randn(1, 3, 224, 224)}

class TestDetectionLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """실제 다운로드된 detection-datasets/coco 데이터셋 환경 설정"""
        cls.dataset_root = "dataset"
        cls.dataset_name = "detection-datasets/coco"
        cls.safe_name = cls.dataset_name.replace("/", "_")
        cls.split = "validation" # load_dataset.py 저장 규칙에 따라 validation으로 고정
        
        cls.data_path = os.path.join(cls.dataset_root, cls.safe_name)
        cls.img_dir = os.path.join(cls.data_path, "validation") # 이미지 저장 경로
        cls.label_file = os.path.join(cls.data_path, "validation_labels.json")
        
        if not os.path.exists(cls.img_dir) or not os.path.exists(cls.label_file):
            raise unittest.SkipTest(f"Actual COCO dataset not found at {cls.data_path}. Please run load_dataset.py first.")

    def test_dataset_loading(self):
        """1. DetectionDataset이 실제 COCO 어노테이션을 잘 읽는지 확인"""
        dataset = DetectionDataset(
            data_dir=self.img_dir,
            label_file=self.label_file,
            processor=MockProcessor(),
            class_map_file=os.path.join(self.data_path, "classes.json")
        )
        self.assertGreater(len(dataset), 0)
        # 클래스 맵이 존재하는 경우 클래스 명칭 확인 (COCO의 경우 0번은 person)
        if dataset.class_map:
            name = dataset.get_class_name(0)
            self.assertIsNotNone(name)

    def test_dataset_item_format(self):
        """2. Dataset이 (Tensor, Dict, str) 형식을 반환하는지 확인"""
        dataset = DetectionDataset(
            data_dir=self.img_dir,
            label_file=self.label_file,
            processor=MockProcessor()
        )
        img_tensor, target, fname = dataset[0]
        self.assertIsInstance(img_tensor, torch.Tensor)
        self.assertIsInstance(target, dict)
        self.assertIn("bbox", target)
        self.assertIn("category", target)
        self.assertTrue(fname.endswith(".jpg"))

    def test_loader_batch_with_dict(self):
        """3. UniversalDataLoader가 실제 COCO 데이터를 배칭할 때 에러 없는지 확인"""
        dataset = DetectionDataset(
            data_dir=self.img_dir,
            label_file=self.label_file,
            processor=MockProcessor()
        )
        loader = UniversalDataLoader(dataset, batch_size=1)
        
        images, targets, fnames = next(iter(loader))
        self.assertIsInstance(images, np.ndarray)
        # 배칭된 타겟은 리스트 내 딕셔너리 형태 (batch_size=1인 경우)
        self.assertIsInstance(targets, (dict, list)) 
        self.assertEqual(len(fnames), 1)

    def test_factory_get_dataloader_detection(self):
        """4. get_dataloader(task='detection') 실제 데이터셋 연동 확인"""
        loader = get_dataloader(
            dataset_name=self.dataset_name,
            split="validation",
            model_id="google/mobilenet_v2_1.0_224",
            batch_size=1,
            task="detection",
            root_dir=self.dataset_root
        )
        self.assertIsInstance(loader.dataset, DetectionDataset)
        self.assertGreater(loader.get_total_samples(), 0)

if __name__ == "__main__":
    unittest.main()

if __name__ == "__main__":
    unittest.main()
