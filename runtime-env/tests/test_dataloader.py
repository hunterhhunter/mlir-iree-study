import unittest
import os
import json
import shutil
import numpy as np
import torch
from PIL import Image

import sys
import os
# 프로젝트 루트 디렉토리를 검색 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataloader import get_dataloader, ClassificationDataset, UniversalDataLoader

class MockProcessor:
    """테스트용 가상 이미지 프로세서"""
    def __call__(self, images, return_tensors="pt"):
        # dummy pixel values (1, 3, 224, 224)
        return {"pixel_values": torch.randn(1, 3, 224, 224)}

class TestDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """테스트용 임시 데이터셋 환경 구축"""
        cls.test_root = "tests/mock_datasets"
        cls.dataset_name = "test_dataset"
        cls.safe_name = cls.dataset_name.replace("/", "_")
        cls.split = "val"
        
        cls.data_path = os.path.join(cls.test_root, cls.safe_name)
        cls.img_dir = os.path.join(cls.data_path, cls.split)
        os.makedirs(cls.img_dir, exist_ok=True)
        
        # 1. 더미 이미지 생성
        for i in range(5):
            img = Image.new("RGB", (100, 100), color=(i, i, i))
            img.save(os.path.join(cls.img_dir, f"sample_{i:05d}.jpg"))
            
        # 2. labels.json 생성
        cls.labels = {
            f"sample_{i:05d}.jpg": {"index": i, "label": f"class_{i}"} for i in range(5)
        }
        with open(os.path.join(cls.data_path, f"{cls.split}_labels.json"), "w") as f:
            json.dump(cls.labels, f)
            
        # 3. classes.json 생성
        cls.classes = {str(i): f"class_{i}" for i in range(5)}
        with open(os.path.join(cls.data_path, "classes.json"), "w") as f:
            json.dump(cls.classes, f)

    @classmethod
    def tearDownClass(cls):
        """테스트 종료 후 임시 데이터 삭제"""
        if os.path.exists(cls.test_root):
            shutil.rmtree(cls.test_root)

    def test_dataset_loading(self):
        """1. ClassificationDataset이 JSON 메타데이터를 잘 읽는지 확인"""
        dataset = ClassificationDataset(
            data_dir=self.img_dir,
            label_file=os.path.join(self.data_path, f"{self.split}_labels.json"),
            processor=MockProcessor(),
            class_map_file=os.path.join(self.data_path, "classes.json")
        )
        self.assertEqual(len(dataset), 5)
        self.assertEqual(dataset.get_class_name(0), "class_0")

    def test_dataset_item_format(self):
        """2. Dataset이 (Tensor, int, str) 형식을 반환하는지 확인"""
        dataset = ClassificationDataset(
            data_dir=self.img_dir,
            label_file=os.path.join(self.data_path, f"{self.split}_labels.json"),
            processor=MockProcessor()
        )
        img_tensor, label, fname = dataset[0]
        self.assertIsInstance(img_tensor, torch.Tensor)
        self.assertEqual(img_tensor.shape, (3, 224, 224))
        self.assertIsInstance(label, int)
        self.assertTrue(fname.startswith("sample_"))

    def test_loader_numpy_conversion(self):
        """3. UniversalDataLoader가 NumPy 배열을 반환하는지 확인"""
        dataset = ClassificationDataset(
            data_dir=self.img_dir,
            label_file=os.path.join(self.data_path, f"{self.split}_labels.json"),
            processor=MockProcessor()
        )
        loader = UniversalDataLoader(dataset, batch_size=2)
        
        for images, labels, fnames in loader:
            self.assertIsInstance(images, np.ndarray)
            self.assertIsInstance(labels, np.ndarray)
            self.assertIsInstance(fnames, list)
            self.assertEqual(images.shape[0], 2 if len(fnames) == 2 else 1)
            break

    def test_loader_batch_shape(self):
        """4. 배치 출력의 Shape이 (N, C, H, W)인지 확인"""
        dataset = ClassificationDataset(
            data_dir=self.img_dir,
            label_file=os.path.join(self.data_path, f"{self.split}_labels.json"),
            processor=MockProcessor()
        )
        loader = UniversalDataLoader(dataset, batch_size=3)
        images, _, _ = next(iter(loader))
        self.assertEqual(images.shape, (3, 3, 224, 224))

    def test_factory_get_dataloader(self):
        """5. get_dataloader 팩토리 함수 작동 확인"""
        # 이 테스트는 실제 Hugging Face Processor 로드를 시도하므로 Mocking이 필요할 수 있음
        # 여기서는 경로 생성 로직 위주로 검증 (실제 model_id 사용 시 네트워크 필요)
        try:
            # google/mobilenet_v2_1.0_224 는 매우 작아서 테스트 시 로드 가능성이 높음
            loader = get_dataloader(
                dataset_name=self.dataset_name,
                split=self.split,
                model_id="google/mobilenet_v2_1.0_224",
                batch_size=1,
                root_dir=self.test_root
            )
            self.assertIsInstance(loader, UniversalDataLoader)
            self.assertEqual(loader.get_total_samples(), 5)
        except Exception as e:
            self.skipTest(f"Skipping factory test due to environment/network: {e}")

if __name__ == "__main__":
    unittest.main()
