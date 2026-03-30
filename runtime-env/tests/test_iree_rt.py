import os
import unittest
import numpy as np
import sys

# src 경로 추가 (런타임 임포트용)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from src.runtimes.iree_rt import IREERuntime
from src.dataloader.base import DataLoader

class TestIREERuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """테스트 환경 초기화"""
        cls.model_path = "models/mobilenetv2-17.onnx"
        if not os.path.exists(cls.model_path):
            raise unittest.SkipTest(f"Model file {cls.model_path} not found.")
        cls.rt = IREERuntime(cls.model_path, device="cpu")

    def test_01_compile(self):
        """Step 1: 컴파일 파이프라인 검증"""
        print("\n[TEST] Testing IREE Compilation...")
        vmfb_path = self.rt.compile()
        
        self.assertTrue(os.path.exists(self.rt.onnx_v17_path))
        self.assertTrue(os.path.exists(self.rt.mlir_path))
        self.assertTrue(os.path.exists(vmfb_path))
        self.assertTrue(vmfb_path.endswith("_cpu.vmfb"))

    def test_02_execute(self):
        """Step 2: 추론 실행 검증"""
        print("\n[TEST] Testing IREE Execution...")
        loader = DataLoader()
        input_data = loader.load_as_numpy(shape=(1, 3, 224, 224))
        
        if not os.path.exists(self.rt.vmfb_path):
            self.rt.compile()
            
        result = self.rt.execute(input_data)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (1, 1000))

    def test_03_profile(self):
        """Step 3: 프로파일링 도구 연동 검증"""
        print("\n[TEST] Testing IREE Profiling...")
        if not os.path.exists(self.rt.vmfb_path):
            self.rt.compile()
            
        try:
            self.rt.profile(input_str="1x3x224x224xf32=0.5")
        except Exception as e:
            self.fail(f"Profiling failed with error: {e}")

if __name__ == "__main__":
    unittest.main()
