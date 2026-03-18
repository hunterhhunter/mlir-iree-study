import unittest
import os
import shutil
import json
import sys
# 프로젝트 루트 디렉토리를 검색 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.load_dataset import HuggingFaceSource, UniversalDatasetManager

class TestDatasetLoader(unittest.TestCase):
    def setUp(self):
        """테스트용 임시 디렉토리 설정"""
        self.test_output = "test_datasets_tmp"
        self.dataset_name = "mnist"
        self.samples_count = 2
        
        # 이전 테스트 잔해 제거
        if os.path.exists(self.test_output):
            shutil.rmtree(self.test_output)

    def tearDown(self):
        """테스트 완료 후 임시 디렉토리 제거"""
        if os.path.exists(self.test_output):
            shutil.rmtree(self.test_output)

    def test_huggingface_source(self):
        """HuggingFaceSource의 스트리밍 및 클래스 맵 추출 기능을 테스트합니다."""
        source = HuggingFaceSource(self.dataset_name)
        
        # 1. 스트리밍 확인
        samples = list(source.stream_samples(self.samples_count))
        self.assertEqual(len(samples), self.samples_count)
        
        # 2. 데이터 형식 확인 (PIL Image, int label)
        img, lbl = samples[0]
        self.assertTrue(hasattr(img, 'size'))
        self.assertIsInstance(lbl, int)
        
        # 3. 클래스 맵 확인
        class_map = source.get_class_map()
        self.assertIsNotNone(class_map)
        self.assertEqual(len(class_map), 10) # MNIST has 10 classes

    def test_manager_persistence(self):
        """UniversalDatasetManager가 파일을 규격에 맞게 저장하는지 테스트합니다."""
        source = HuggingFaceSource(self.dataset_name)
        manager = UniversalDatasetManager(source, output_root=self.test_output, dataset_tag=self.dataset_name)
        
        # 런타임 실행
        manager.run(self.samples_count)
        
        # 1. 디렉토리 구조 확인
        expected_img_dir = os.path.join(self.test_output, self.dataset_name, "validation")
        self.assertTrue(os.path.exists(expected_img_dir))
        
        # 2. 이미지 파일 존재 확인
        saved_images = os.listdir(expected_img_dir)
        self.assertEqual(len(saved_images), self.samples_count)
        self.assertTrue(all(f.endswith(".jpg") for f in saved_images))
        
        # 3. 레이블 JSON 확인
        label_file = os.path.join(self.test_output, self.dataset_name, "validation_labels.json")
        self.assertTrue(os.path.exists(label_file))
        with open(label_file, "r") as f:
            labels = json.load(f)
            self.assertEqual(len(labels), self.samples_count)
            self.assertIn("sample_00000.jpg", labels)
            
        # 4. 클래스 맵 JSON 확인
        class_file = os.path.join(self.test_output, self.dataset_name, "classes.json")
        self.assertTrue(os.path.exists(class_file))

if __name__ == "__main__":
    unittest.main()
