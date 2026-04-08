import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

class MobileNetEvaluator:
    """
    MobileNetV2 성능 평가 모듈 (Accuracy, Precision, Recall, F1-Score).
    PyTorch 텐서 연산과 Scikit-learn 지표를 통합하여 상세 분석을 제공합니다.
    """
    def __init__(self, top_k=(1, 5)):
        self.top_k = top_k
        self.reset()

    def reset(self):
        """평가 데이터 초기화"""
        self.total_samples = 0
        self.correct_counts = {k: 0 for k in self.top_k}
        
        # 상세 지표 계산을 위한 예측값/정답 저장소
        self.all_preds = []
        self.all_targets = []

    def update(self, logits, targets):
        """
        배치 단위로 추론 결과 업데이트.
        
        Args:
            logits (np.ndarray or torch.Tensor): (Batch, 1000)
            targets (np.ndarray or torch.Tensor): (Batch,)
        """
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets)

        batch_size = targets.size(0)
        max_k = max(self.top_k)
        
        # 1. Top-K Accuracy 계산용 연산
        _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
        pred_t = pred.t()
        correct = pred_t.eq(targets.view(1, -1).expand_as(pred_t))

        for k in self.top_k:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            self.correct_counts[k] += correct_k.item()
            
        self.total_samples += batch_size

        # 2. 상세 지표(P/R/F1)를 위해 Top-1 예측값 및 정답 저장
        self.all_preds.extend(pred[:, 0].tolist())
        self.all_targets.extend(targets.tolist())

    def get_results(self):
        """정확도 및 상세 지표(Macro-average) 산출"""
        if self.total_samples == 0:
            return {}

        # Top-K Accuracy
        results = {
            f"Top-{k} Accuracy": (self.correct_counts[k] / self.total_samples) * 100 
            for k in self.top_k
        }

        # Precision, Recall, F1-Score (Macro-average)
        # 1,000개 클래스에 대해 각 클래스별 성능을 평균냄
        p, r, f1, _ = precision_recall_fscore_support(
            self.all_targets, 
            self.all_preds, 
            average='macro', 
            zero_division=0
        )

        results.update({
            "Precision (Macro)": p * 100,
            "Recall (Macro)": r * 100,
            "F1-Score (Macro)": f1 * 100,
            "Total Samples": self.total_samples
        })
        
        return results

    def report(self):
        """상세 평가 결과 출력"""
        res = self.get_results()
        if not res:
            print("[!] No evaluation data available.")
            return

        print("\n" + "═"*50)
        print("  MobileNetV2 Comprehensive Evaluation Report")
        print("═"*50)
        print(f"  Total Samples     : {int(res['Total Samples']):,}")
        print("-" * 50)
        # Accuracy 섹션
        print(f"  Total Accuracy    : {res['Top-1 Accuracy']:>7.2f}% (Top-1)")
        for k in self.top_k:
            if k == 1: continue # Total Accuracy와 중복 방지
            print(f"  Top-{k} Accuracy    : {res[f'Top-{k} Accuracy']:>7.2f}%")
        
        print("-" * 50)
        # 상세 분석 섹션 (Macro Average)
        print(f"  Precision (Macro) : {res['Precision (Macro)']:>7.2f}%")
        print(f"  Recall (Macro)    : {res['Recall (Macro)']:>7.2f}%")
        print(f"  F1-Score (Macro)  : {res['F1-Score (Macro)']:>7.2f}%")
        print("═"*50)
