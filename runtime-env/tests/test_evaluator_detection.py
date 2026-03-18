import unittest
import numpy as np
import sys
import os
# 프로젝트 루트 디렉토리를 검색 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.evaluator import get_evaluator, DetectionEvaluator

class TestDetectionEvaluator(unittest.TestCase):
    def setUp(self):
        """테스트마다 에밸루에이터 초기화"""
        self.evaluator = get_evaluator("detection", iou_thresholds=[0.5])

    def test_iou_calculation(self):
        """1. IoU 계산 로직의 수치 정확성 확인"""
        box1 = np.array([0, 0, 10, 10]) # [x, y, w, h]
        box2 = np.array([5, 0, 10, 10]) # 50% overlap
        iou = self.evaluator._calculate_iou(box1, box2)
        # Intersection: [5, 0, 5, 10] = 50
        # Union: 100 + 100 - 50 = 150
        # IoU = 50 / 150 = 0.3333
        self.assertAlmostEqual(iou, 0.3333, places=4)

    def test_perfect_mAP(self):
        """2. 완벽한 예측 시 mAP@0.5가 100%인지 확인"""
        targets = [{"bbox": [[10, 10, 20, 20]], "category": [0]}]
        preds = [[{"bbox": [10, 10, 20, 20], "score": 0.95, "category": 0}]]
        
        self.evaluator.update(preds, targets)
        res = self.evaluator.compute()
        self.assertEqual(res["mAP@0.5"], 100.0)

    def test_zero_mAP(self):
        """3. 모든 예측이 오답(IoU < 0.5)일 때 mAP가 0%인지 확인"""
        targets = [{"bbox": [[10, 10, 20, 20]], "category": [0]}]
        # IoU = 0 (박스가 전혀 겹치지 않음)
        preds = [[{"bbox": [100, 100, 20, 20], "score": 0.99, "category": 0}]]
        
        self.evaluator.update(preds, targets)
        res = self.evaluator.compute()
        self.assertEqual(res["mAP@0.5"], 0.0)

    def test_iou_threshold_boundary(self):
        """4. IoU 임계값 경계(0.49 vs 0.5)에서 TP/FP 판별 확인"""
        targets = [{"bbox": [[0, 0, 10, 10]], "category": [0]}]
        
        # Case 1: IoU 0.49 (FP)
        preds_fp = [[{"bbox": [5.1, 0, 10, 10], "score": 0.99, "category": 0}]]
        self.evaluator.update(preds_fp, targets)
        self.assertEqual(self.evaluator.compute()["mAP@0.5"], 0.0)
        self.evaluator.reset()
        
        # Case 2: IoU > 0.5 (TP)
        preds_tp = [[{"bbox": [2.0, 0, 10, 10], "score": 0.99, "category": 0}]]
        self.evaluator.update(preds_tp, targets)
        self.assertEqual(self.evaluator.compute()["mAP@0.5"], 100.0)

    def test_multiple_objects_matching(self):
        """5. 다중 객체 매칭 시 중복 검출 방지 로직 확인"""
        # 정답은 하나, 예측은 둘
        targets = [{"bbox": [[0, 0, 10, 10]], "category": [0]}]
        
        # FP가 TP보다 점수가 높아야 Precision이 초반에 떨어져 mAP < 100% 가 됨
        preds = [[
            {"bbox": [100, 100, 10, 10], "score": 0.95, "category": 0}, # FP (Wrong box, high score)
            {"bbox": [0, 0, 10, 10], "score": 0.90, "category": 0}      # TP (Correct box, lower score)
        ]]
        
        self.evaluator.update(preds, targets)
        res = self.evaluator.compute()
        # 1st pred (FP): Precision 0/1 = 0.0, Recall 0/1 = 0.0
        # 2nd pred (TP): Precision 1/2 = 0.5, Recall 1/1 = 1.0
        # AP = (1.0 - 0.0) * max(precisions for r >= 1.0) = 1.0 * 0.5 = 0.5
        self.assertTrue(res["mAP@0.5"] < 100.0)

    def test_ap_interpolation(self):
        """6. Precision-Recall 곡선 면적(AP) 산출 로직 확인"""
        # (Recall 0.5, Precision 1.0), (Recall 1.0, Precision 0.5)
        # m_rec = [0.0, 0.5, 1.0, 1.0]
        # m_pre = [1.0, 1.0, 0.5, 0.0] (after interp)
        # ap = (0.5-0.0)*1.0 + (1.0-0.5)*0.5 = 0.5 + 0.25 = 0.75
        recalls = np.array([0.5, 1.0])
        precisions = np.array([1.0, 0.5])
        ap = self.evaluator._calculate_ap(recalls, precisions)
        self.assertEqual(ap, 0.75)

    def test_reset(self):
        """7. reset() 호출 후 데이터 초기화 확인"""
        self.evaluator.update([[{"bbox":[0,0,10,10], "score":0.9, "category":0}]], [{"bbox":[[0,0,10,10]], "category":[0]}])
        self.evaluator.reset()
        self.assertEqual(len(self.evaluator.all_predictions), 0)
        self.assertEqual(len(self.evaluator.all_targets), 0)

    def test_report_visual(self):
        """8. report() 실행 시 예외 미발생 확인"""
        self.evaluator.update([[{"bbox":[0,0,10,10], "score":0.9, "category":0}]], [{"bbox":[[0,0,10,10]], "category":[0]}])
        try:
            self.evaluator.report()
        except Exception as e:
            self.fail(f"report() crashed: {e}")

if __name__ == "__main__":
    unittest.main()
