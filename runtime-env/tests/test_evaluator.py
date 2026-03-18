import unittest
import torch
import numpy as np
import sys
import os
# 프로젝트 루트 디렉토리를 검색 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.evaluator import get_evaluator, ClassificationEvaluator, BaseEvaluator


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        """매 테스트마다 새로운 분류 에밸루에이터 생성"""
        self.evaluator = get_evaluator("classification", top_k=(1, 5))

    # 1-3. Factory 패턴 및 초기화 테스트
    def test_factory_creation(self):
        """1. get_evaluator가 정상적으로 인스턴스를 생성하는지 확인"""
        self.assertIsInstance(self.evaluator, ClassificationEvaluator)
        self.assertIsInstance(self.evaluator, BaseEvaluator)

    def test_factory_case_insensitivity(self):
        """2. 태스크 명칭의 대소문자 구분 없이 동작하는지 확인"""
        eval_caps = get_evaluator("CLASSIFICATION")
        self.assertIsInstance(eval_caps, ClassificationEvaluator)

    def test_factory_invalid_task(self):
        """3. 지원하지 않는 태스크 입력 시 ValueError 발생 확인"""
        with self.assertRaises(ValueError):
            get_evaluator("invalid_task_name")

    def test_init_default_top_k(self):
        """4. 기본 top_k 설정이 (1, 5)인지 확인"""
        eval_def = ClassificationEvaluator()
        self.assertEqual(eval_def.top_k, (1, 5))

    def test_init_custom_top_k(self):
        """5. 사용자 정의 top_k 설정이 반영되는지 확인"""
        eval_custom = ClassificationEvaluator(top_k=(1, 3, 10))
        self.assertEqual(eval_custom.top_k, (1, 3, 10))

    # 6. Reset 기능 테스트
    def test_reset_functionality(self):
        """6. 데이터 업데이트 후 reset 시 상태가 초기화되는지 확인"""
        logits = torch.randn(2, 10)
        targets = torch.tensor([0, 1])
        self.evaluator.update(logits, targets)
        self.evaluator.reset()
        self.assertEqual(self.evaluator.total_samples, 0)
        self.assertEqual(self.evaluator.total_loss, 0.0)
        self.assertEqual(len(self.evaluator.all_preds), 0)

    # 7-8. 데이터 타입 호환성 테스트
    def test_numpy_input_support(self):
        """7. NumPy 배열 입력 지원 여부 확인"""
        logits = np.random.randn(2, 5).astype(np.float32)
        targets = np.array([0, 1])
        try:
            self.evaluator.update(logits, targets)
        except Exception as e:
            self.fail(f"NumPy input failed: {e}")

    def test_torch_input_support(self):
        """8. PyTorch Tensor 입력 지원 여부 확인"""
        logits = torch.randn(2, 5)
        targets = torch.tensor([0, 1])
        try:
            self.evaluator.update(logits, targets)
        except Exception as e:
            self.fail(f"Torch input failed: {e}")

    # 9-11. 정확도(Accuracy) 산출 로직 테스트
    def test_accuracy_perfect(self):
        """9. 모든 예측이 정답인 경우 (100% Accuracy)"""
        logits = torch.zeros(2, 5)
        logits[0, 0] = 10.0 # class 0
        logits[1, 1] = 10.0 # class 1
        targets = torch.tensor([0, 1])
        self.evaluator.update(logits, targets)
        res = self.evaluator.compute()
        self.assertEqual(res["Top-1 Accuracy"], 100.0)
        self.assertEqual(res["Top-1 Error"], 0.0)

    def test_accuracy_zero(self):
        """10. 모든 예측이 오답인 경우 (0% Accuracy)"""
        logits = torch.zeros(2, 5)
        logits[0, 1] = 10.0 # Wrong: predicts 1 instead of 0
        logits[1, 0] = 10.0 # Wrong: predicts 0 instead of 1
        targets = torch.tensor([0, 1])
        self.evaluator.update(logits, targets)
        res = self.evaluator.compute()
        self.assertEqual(res["Top-1 Accuracy"], 0.0)
        self.assertEqual(res["Top-1 Error"], 100.0)

    def test_accuracy_top5_boundary(self):
        """11. 정답이 3순위인 경우 (Top-1 오답, Top-5 정답)"""
        logits = torch.tensor([[5.0, 4.0, 10.0, 1.0, 1.0]]) # Top-1 is class 2
        targets = torch.tensor([0]) # Target is class 0, which is 2nd best here
        self.evaluator.update(logits, targets)
        res = self.evaluator.compute()
        self.assertEqual(res["Top-1 Accuracy"], 0.0)
        self.assertEqual(res["Top-5 Accuracy"], 100.0)

    # 12-16. 신규 고도화 지표 테스트
    def test_log_loss_calculation(self):
        """12. Avg Log Loss가 유효한 범위 내에서 계산되는지 확인"""
        self.evaluator.update(torch.randn(5, 10), torch.randint(0, 10, (5,)))
        res = self.evaluator.compute()
        self.assertGreater(res["Avg Log Loss"], 0.0)

    def test_weighted_metrics(self):
        """13. Weighted Precision/Recall/F1 산출 여부 및 범위 확인"""
        self.evaluator.update(torch.randn(10, 10), torch.randint(0, 10, (10,)))
        res = self.evaluator.compute()
        self.assertIn("Precision (Weighted)", res)
        self.assertTrue(0 <= res["Precision (Weighted)"] <= 100)
        self.assertTrue(0 <= res["F1-Score (Weighted)"] <= 100)

    def test_top1_error_rate_logic(self):
        """14. Accuracy + Error = 100 산술 논리 검증"""
        self.evaluator.update(torch.randn(10, 100), torch.randint(0, 100, (10,)))
        res = self.evaluator.compute()
        self.assertAlmostEqual(res["Top-1 Accuracy"] + res["Top-1 Error"], 100.0, places=5)

    def test_error_analysis_extraction(self):
        """15. 가장 많이 틀린 클래스(Hardest Classes) 식별 로직 검증"""
        # 클래스 개수를 10개로 늘려 Top-5 연산 범위 확보
        num_classes = 10
        logits = torch.zeros(4, num_classes)
        
        # Class 0: 0% Acc (2 samples), Class 1: 100% Acc (2 samples)
        logits[0:2, 1] = 10.0 # Target 0인데 Class 1로 예측 (오답)
        logits[2:4, 1] = 10.0 # Target 1인데 Class 1로 예측 (정답)
        targets = torch.tensor([0, 0, 1, 1])
        
        self.evaluator.update(logits, targets)
        worst = self.evaluator._get_error_analysis(top_n=1)
        
        self.assertEqual(len(worst), 1)
        self.assertEqual(worst[0]["class_id"], 0) # 가장 많이 틀린 클래스는 0
        self.assertEqual(worst[0]["accuracy"], 0.0)

    def test_report_visual_no_crash(self):
        """16. 확장된 report() 출력 시 예외 미발생 확인"""
        self.evaluator.update(torch.randn(5, 1000), torch.randint(0, 1000, (5,)))
        try:
            self.evaluator.report()
        except Exception as e:
            self.fail(f"Enhanced report() crashed: {e}")

    # 17-23. 안정성 및 키 일치 테스트
    def test_empty_compute(self):
        """17. 데이터 없이 compute 호출 시 빈 딕셔너리 반환 확인"""
        res = self.evaluator.compute()
        self.assertEqual(res, {})

    def test_zero_division_stability(self):
        """18. 특정 클래스 샘플이 없는 상황에서의 매크로 지표 안정성"""
        self.evaluator.update(torch.randn(5, 10), torch.zeros(5).long())
        try:
            res = self.evaluator.compute()
            self.assertIsInstance(res["F1-Score (Macro)"], float)
        except Exception as e:
            self.fail(f"Zero division stability check failed: {e}")

    def test_result_keys_comprehensive(self):
        """19. 모든 신규 지표 키가 포함되어 있는지 전수 조사"""
        self.evaluator.update(torch.randn(2, 10), torch.tensor([0, 1]))
        res = self.evaluator.compute()
        expected_keys = [
            "Top-1 Accuracy", "Top-5 Accuracy", "Top-1 Error", "Avg Log Loss",
            "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)",
            "Precision (Weighted)", "Recall (Weighted)", "F1-Score (Weighted)",
            "Total Samples"
        ]
        for key in expected_keys:
            self.assertIn(key, res, f"Missing expected metric key: {key}")

if __name__ == "__main__":
    unittest.main()

if __name__ == "__main__":
    unittest.main()
