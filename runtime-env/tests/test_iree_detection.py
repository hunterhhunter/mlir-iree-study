import os
import unittest
import numpy as np
import sys
import torch
import time
from tqdm import tqdm

# 프로젝트 루트 디렉토리를 검색 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.runtimes import get_runtime
from src.dataloader import get_dataloader
from src.evaluator import get_evaluator

class TestIREEDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """테스트 환경 초기화 (실제 YOLOv10s 모델 및 COCO 데이터 사용)"""
        cls.model_path = "models/jameslahm_yolov10s/model.onnx"
        if not os.path.exists(cls.model_path):
            raise unittest.SkipTest(f"[!] YOLOv10 model not found at {cls.model_path}. Run convert_to_onnx.py first.")
        
        # IREE 런타임 초기화 (LLVM-CPU 사용)
        cls.rt = get_runtime(
            task="detection", 
            engine="iree", 
            model_path=cls.model_path, 
            device="cuda" 
        )
        
        # COCO 데이터셋 설정 (1,000개 샘플 목표)
        cls.num_samples = 1000
        cls.dataset_args = {
            "dataset_name": "detection-datasets/coco",
            "split": "validation",
            "model_id": "jameslahm/yolov10s",
            "root_dir": "dataset",
            "task": "detection"
        }
        
        data_path = os.path.join("dataset", "detection-datasets_coco", "validation")
        if not os.path.exists(data_path):
            raise unittest.SkipTest("[!] COCO validation dataset not found. Run load_dataset.py first.")

    def test_01_pipeline_full_integration(self):
        """
        [Full Integration Test] 
        1,000개 샘플에 대한 컴파일 -> 루프 추론 -> 평가 통합 검증
        """
        print("\n[TEST] Step 1: Compiling YOLOv10 to IREE VMFB...")
        vmfb_path = self.rt.compile()
        self.assertTrue(os.path.exists(vmfb_path))

        print(f"\n[TEST] Step 2: Running Inference Loop ({self.num_samples} samples)...")
        loader = get_dataloader(batch_size=1, data_size=self.num_samples, **self.dataset_args)
        evaluator = get_evaluator(task="detection")
        self.rt.initialize_runtime()
        
        start_time = time.time()
        
        # tqdm을 사용한 진행률 표시
        processed_count = 0
        for pixel_values, targets, filenames in tqdm(loader, desc="Inference", total=self.num_samples):
            # 디버그: 입력 데이터가 실제로 다른지 확인 (첫 3장만)
            if processed_count < 3:
                print(f"\n[*] Debug - Input Mean: {np.mean(pixel_values):.4f}, Name: {filenames[0]}")

            # 1. 추론 수행 (데이터로더에서 이미 640x640, [0, 1]로 전처리되어 공급됨)
            predictions = self.rt.execute(pixel_values, conf_threshold=0.1) # 임계값 낮춰서 확인
            
            # 2. 에밸루에이터 업데이트
            evaluator.update(predictions, targets)
            processed_count += 1

        end_time = time.time()
        total_time = end_time - start_time
        avg_latency = (total_time / processed_count) * 1000

        print(f"\n[+] Inference Complete!")
        print(f"[*] Total Time    : {total_time:.2f}s")
        print(f"[*] Avg Latency   : {avg_latency:.2f}ms / image")
        print(f"[*] Approx. FPS   : {1.0 / (avg_latency / 1000):.2f}")

        print("\n[TEST] Step 3: Computing Final mAP Metrics...")
        results = evaluator.compute()
        
        # 최종 검증 (전체 샘플 수 확인)
        self.assertEqual(int(results["Total Samples"]), processed_count)
        
        # 최종 리포트 출력
        print("\n" + "="*50)
        print(f"   YOLOv10s IREE EVALUATION (Samples: {processed_count})")
        print("="*50)
        evaluator.report()

if __name__ == "__main__":
    unittest.main()
