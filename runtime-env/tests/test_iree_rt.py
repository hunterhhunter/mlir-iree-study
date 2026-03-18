import os
import unittest
import numpy as np
import sys
import torch

# 프로젝트 루트 디렉토리를 검색 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.runtimes.iree_rt import IREERuntime
from src.dataloader import get_dataloader
from src.evaluator import get_evaluator

class TestIREERuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """테스트 환경 초기화"""
        # 모델 경로 수정: 서브 디렉토리 구조 반영
        cls.model_path = "models/mobilenetv2-10/mobilenetv2-10.onnx"
        if not os.path.exists(cls.model_path):
            raise unittest.SkipTest(f"Model file {cls.model_path} not found.")
        cls.rt = IREERuntime(cls.model_path, device="cpu")
        
        # 테스트에 사용할 공통 데이터셋 설정
        cls.dataset_args = {
            "dataset_name": "ILSVRC/imagenet-1k",
            "split": "validation",
            "model_id": "google/mobilenet_v2_1.0_224",
            "root_dir": "dataset"
        }

    def test_01_compile(self):
        """Step 1: 컴파일 파이프라인 검증"""
        print("\n[TEST] Testing IREE Compilation...")
        vmfb_path = self.rt.compile()
        
        # 아티팩트가 모델 파일과 같은 디렉토리에 생성되었는지 확인
        model_dir = os.path.dirname(self.model_path)
        self.assertTrue(os.path.exists(os.path.join(model_dir, "mobilenetv2-10.mlir")))
        self.assertTrue(os.path.exists(vmfb_path))
        self.assertTrue(vmfb_path.endswith("_cpu.vmfb"))

    def test_02_execute(self):
        """Step 2: 추론 실행 검증"""
        print("\n[TEST] Testing IREE Execution...")
        # 리팩토링된 get_dataloader 사용
        loader = get_dataloader(batch_size=1, data_size=1, **self.dataset_args)
        pixel_values, labels, _ = next(iter(loader))
        
        # 런타임 초기화 및 추론
        self.rt.initialize_runtime()
        result = self.rt.execute(pixel_values)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (1, 1000))

    def test_03_profile(self):
        """Step 3: 프로파일링 도구 연동 검증"""
        print("\n[TEST] Testing IREE Profiling...")
        try:
            # 기본 프로파일링 시나리오 실행
            self.rt.profile()
        except Exception as e:
            self.fail(f"Profiling failed with error: {e}")

    def test_04_evaluator_integration(self):
        """Step 4: 리팩토링된 에밸루에이터 및 데이터로더 연동 검증"""
        print("\n[TEST] Testing Evaluator Integration...")
        # 2개의 샘플로 통합 테스트
        loader = get_dataloader(batch_size=2, data_size=2, **self.dataset_args)
        evaluator = get_evaluator(task="classification", top_k=(1, 5))
        
        pixel_values, labels, _ = next(iter(loader))
        logits = self.rt.execute(pixel_values)
        
        # 에밸루에이터 업데이트 및 지표 산출 확인
        evaluator.update(logits, labels)
        results = evaluator.compute()
        
        self.assertIn("Top-1 Accuracy", results)
        self.assertIn("Avg Log Loss", results)
        self.assertEqual(results["Total Samples"], 2.0)
        
        # 리포트 출력 시 크래시 여부 확인
        try:
            evaluator.report()
        except Exception as e:
            self.fail(f"Evaluator report failed: {e}")

if __name__ == "__main__":
    unittest.main()
