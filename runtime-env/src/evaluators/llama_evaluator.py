import re
import string
from collections import Counter
from typing import Dict, Any, List

import numpy as np
from transformers import AutoTokenizer

from .base import Evaluator
from ..core.model_spec import Model_Spec, Task
from ..core.inference_result import InferenceResult


class LlamaEvaluator(Evaluator):
    """
    LLaMA 3.1 8B 모델의 SQuAD 2.0 QA 태스크 추론 결과를 평가하는 모듈.

    평가 지표:
        - Exact Match (EM): 예측 텍스트가 정답과 완전히 일치하는 비율
        - F1 Score: 토큰 수준 overlap 기반 점수
        - Average Latency (ms): 평균 추론 지연 시간
        - P99 Latency (ms): 99번째 백분위 추론 지연 시간
    """

    def __init__(self, **eval_options):
        """
        Args:
            tokenizer_path (str): AutoTokenizer 로드 경로 (HuggingFace 모델 ID 또는 로컬 경로)
        """
        tokenizer_path = eval_options.get("tokenizer_path", "meta-llama/Llama-3.1-8B-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def evaluate(self, result: InferenceResult) -> Dict[str, Any]:
        """
        InferenceResult를 받아 EM, F1, Latency 지표를 계산하여 반환함.

        Args:
            result: InferenceResult
                - outputs: {"logits": np.ndarray (batch, seq_len, vocab_size)}
                - labels: List[Dict]  각 원소 = {"answers": [...], "is_impossible": bool, ...}
                - timing_records: List[float] (ms)

        Returns:
            Dict[str, Any]: 지표 딕셔너리
        """
        logits_key = list(result.outputs.keys())[0]
        logits = result.outputs[logits_key]  # (batch, seq_len, vocab_size)

        labels = result.labels  # List[Dict] or single Dict
        if isinstance(labels, dict):
            labels = [labels]

        num_samples = logits.shape[0]
        if num_samples == 0:
            return {"Exact Match": 0.0, "F1 Score": 0.0, "num_samples": 0}

        if len(labels) != num_samples:
            raise ValueError(
                f"logits 배치 크기({num_samples})와 labels 길이({len(labels)})가 일치하지 않습니다."
            )

        em_scores, f1_scores = [], []

        for i in range(num_samples):
            pred_text = self._decode_logits(logits[i])  # (seq_len, vocab_size)
            label = labels[i]

            gold_answers = self._extract_gold_answers(label)
            em_scores.append(self._compute_exact_match(pred_text, gold_answers))
            f1_scores.append(self._compute_f1(pred_text, gold_answers))

        metrics = {
            "Exact Match": float(np.mean(em_scores)) * 100,
            "F1 Score": float(np.mean(f1_scores)) * 100,
            "num_samples": num_samples,
        }
        metrics.update(self._compute_latency_metrics(result.timing_records))
        return metrics

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _decode_logits(self, logits_2d: np.ndarray) -> str:
        """
        (seq_len, vocab_size) logits → greedy decoding → 정규화 전 텍스트 반환.
        특수 토큰은 제거하고 반환함.
        """
        token_ids = np.argmax(logits_2d, axis=-1).tolist()
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return text

    def _normalize_answer(self, s: str) -> str:
        """SQuAD 공식 normalizer: 소문자 변환, 관사·구두점·여분 공백 제거."""
        s = s.lower()
        # 구두점 제거
        s = s.translate(str.maketrans("", "", string.punctuation))
        # 관사 제거 (a, an, the)
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        # 여분 공백 정리
        s = " ".join(s.split())
        return s

    def _extract_gold_answers(self, label: Dict) -> List[str]:
        """
        LlamaLoader 라벨에서 정답 텍스트 목록을 추출함.
        is_impossible=True인 경우 빈 문자열을 정답으로 취급함.
        """
        if label.get("is_impossible", False):
            return [""]
        answers = label.get("answers", [])
        texts = [a["text"] for a in answers if "text" in a]
        return texts if texts else [""]

    def _compute_exact_match(self, pred: str, gold_list: List[str]) -> float:
        """gold_list 중 하나라도 정규화 후 완전 일치하면 1.0, 아니면 0.0 반환."""
        norm_pred = self._normalize_answer(pred)
        return float(any(norm_pred == self._normalize_answer(g) for g in gold_list))

    def _compute_f1(self, pred: str, gold_list: List[str]) -> float:
        """gold_list 중 가장 높은 토큰 수준 F1 점수를 반환함."""
        return max(self._token_f1(pred, g) for g in gold_list)

    def _token_f1(self, pred: str, gold: str) -> float:
        """두 문자열 사이의 토큰 수준 F1 점수를 계산함."""
        pred_tokens = self._normalize_answer(pred).split()
        gold_tokens = self._normalize_answer(gold).split()

        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            return 0.0

        precision = num_common / len(pred_tokens)
        recall = num_common / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)

    def _compute_latency_metrics(self, timing_records: List[float]) -> Dict[str, float]:
        """평균 및 P99 latency를 계산함."""
        if not timing_records:
            return {}
        return {
            "Average Latency (ms)": float(np.mean(timing_records)),
            "P99 Latency (ms)": float(np.percentile(timing_records, 99)),
        }

    # ------------------------------------------------------------------ #
    #  Abstract method implementations                                     #
    # ------------------------------------------------------------------ #

    def is_applicable(self, device_spec: Dict[str, Any], model_spec: Model_Spec) -> bool:
        return model_spec.task == Task.NLP_GENERATION

    def get_metric_names(self) -> List[str]:
        return [
            "Exact Match",
            "F1 Score",
            "Average Latency (ms)",
            "P99 Latency (ms)",
            "num_samples",
        ]
