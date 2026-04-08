import re
import string
from collections import Counter
from typing import Dict, Any, List

import numpy as np
from transformers import AutoTokenizer

from .base import Evaluator
from core.model_spec import Model_Spec, Task
from core.inference_result import InferenceResult


class LlamaEvaluator(Evaluator):
    """
    LLaMA 3.1 8B 모델의 SQuAD 2.0 QA 태스크 추론 결과를 평가하는 모듈.

    스트리밍 평가를 지원합니다.
    add_batch()에서 (seq_len, vocab_size) logits를 greedy decoding하여 텍스트로 변환한 뒤
    EM·F1 점수만 누산하고, 원본 logits 텐서는 즉시 폐기합니다.
    배치당 저장량: float 2개(em, f1) × 배치크기 — logits 대비 수천 배 절약.

    평가 지표:
        - Exact Match (EM): 예측 텍스트가 정답과 완전히 일치하는 비율
        - F1 Score: 토큰 수준 overlap 기반 점수
        - Average Latency (ms): 평균 추론 지연 시간
        - P99 Latency (ms): 99번째 백분위 추론 지연 시간
    """

    def __init__(self, **eval_options):
        tokenizer_path = eval_options.get("tokenizer_path", "meta-llama/Llama-3.1-8B-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # debug=True 이면 샘플마다 예측/정답/점수를 출력합니다.
        self.debug: bool = eval_options.get("debug", False)
        self._reset()

    # ------------------------------------------------------------------
    # 내부 상태 초기화
    # ------------------------------------------------------------------

    def _reset(self):
        """누산 상태를 초기화합니다."""
        self._em_scores: List[float] = []
        self._f1_scores: List[float] = []
        self._timing_records: List[float] = []  # total_ms
        self._ttft_records: List[float] = []
        self._tpot_records: List[float] = []
        self._total_tokens: int = 0

    # ------------------------------------------------------------------
    # 스트리밍 인터페이스
    # ------------------------------------------------------------------

    def add_batch(self, outputs: Dict[str, np.ndarray], labels: Any, timing_ms: float) -> None:
        """
        배치의 출력을 EM·F1 점수만 누산하고 원본 텐서는 즉시 폐기합니다.

        두 가지 경로를 지원합니다:
        - vLLM 경로: outputs에 "generated_ids" 키가 있으면 이미 생성된 토큰 ID를 직접 디코딩
        - ONNX 경로: logits (B, seq_len, vocab_size)에서 greedy decoding
        """
        flat_labels = self._flatten_labels(labels)

        if "generated_ids" in outputs:
            # vLLM 경로: 배치 크기 1 가정 (VllmRuntime.generate() 는 단일 샘플 반환)
            generated_ids = outputs["generated_ids"]
            self._total_tokens += len(generated_ids)
            label = flat_labels[0]
            raw_text = self.tokenizer.decode(
                generated_ids.tolist(), skip_special_tokens=True
            ).strip()
            pred_text = self._postprocess_response(raw_text)
            gold_answers = self._extract_gold_answers(label)
            em = self._compute_exact_match(pred_text, gold_answers)
            f1 = self._compute_f1(pred_text, gold_answers)
            self._em_scores.append(em)
            self._f1_scores.append(f1)
            self._log_sample(len(self._em_scores), label, pred_text, gold_answers, em, f1)
        else:
            # ONNX logits 경로
            logits_key = list(outputs.keys())[0]
            logits = outputs[logits_key]  # (B, seq_len, vocab_size)

            for i in range(logits.shape[0]):
                label = flat_labels[i]
                prompt_length = label.get("prompt_length", None)
                raw_text = self._decode_logits(logits[i], prompt_length)
                pred_text = self._postprocess_response(raw_text)
                gold_answers = self._extract_gold_answers(label)
                em = self._compute_exact_match(pred_text, gold_answers)
                f1 = self._compute_f1(pred_text, gold_answers)
                self._em_scores.append(em)
                self._f1_scores.append(f1)
                self._log_sample(len(self._em_scores), label, pred_text, gold_answers, em, f1)

        # timing_ms: float(non-LLM) 또는 dict {"total_ms", "ttft_ms", "tpot_ms"}(LLM)
        if isinstance(timing_ms, dict):
            self._timing_records.append(timing_ms.get("total_ms", 0.0))
            self._ttft_records.append(timing_ms.get("ttft_ms", 0.0))
            self._tpot_records.append(timing_ms.get("tpot_ms", 0.0))
        else:
            self._timing_records.append(float(timing_ms))
        # outputs 변수가 스코프를 벗어나면 GC 대상이 됩니다.

    def compute(self) -> Dict[str, Any]:
        """누산된 EM·F1 점수로 최종 메트릭을 계산합니다."""
        num_samples = len(self._em_scores)
        if num_samples == 0:
            return {"Exact Match": 0.0, "F1 Score": 0.0, "num_samples": 0}

        metrics = {
            "Exact Match": float(np.mean(self._em_scores)) * 100,
            "F1 Score": float(np.mean(self._f1_scores)) * 100,
            "num_samples": num_samples,
        }
        metrics.update(self._compute_latency_metrics(self._timing_records))
        if self._ttft_records:
            metrics["Avg TTFT (ms)"] = float(np.mean(self._ttft_records))
            metrics["P99 TTFT (ms)"] = float(np.percentile(self._ttft_records, 99))
        if self._tpot_records:
            # KV 캐시 없는 ONNX 경로에서는 시퀀스가 길어질수록 decode step이 느려짐.
            # 실제 서빙 환경(KV 캐시)의 TPOT와 다르므로 이름으로 구분합니다.
            metrics["Avg Decode Step (no KV cache) (ms)"] = float(np.mean(self._tpot_records))
            metrics["P99 Decode Step (no KV cache) (ms)"] = float(np.percentile(self._tpot_records, 99))
        if self._total_tokens > 0 and self._timing_records:
            total_time_s = sum(self._timing_records) / 1000.0
            metrics["Throughput (tokens/s)"] = self._total_tokens / total_time_s
            metrics["Total Tokens Generated"] = self._total_tokens
        return metrics

    # ------------------------------------------------------------------
    # 배치 호환 인터페이스 (단위 테스트 및 레거시 지원)
    # ------------------------------------------------------------------

    def evaluate(self, result: InferenceResult) -> Dict[str, Any]:
        """InferenceResult 전체를 받아 스트리밍 내부 로직으로 채점합니다."""
        self._reset()

        logits_key = list(result.outputs.keys())[0]
        logits = result.outputs[logits_key]  # (N, seq_len, vocab_size)

        flat_labels = self._flatten_labels(result.labels)

        num_samples = logits.shape[0]
        if num_samples == 0:
            return {"Exact Match": 0.0, "F1 Score": 0.0, "num_samples": 0}

        if len(flat_labels) != num_samples:
            raise ValueError(
                f"logits 배치 크기({num_samples})와 labels 길이({len(flat_labels)})가 일치하지 않습니다."
            )

        for i in range(num_samples):
            label = flat_labels[i]
            prompt_length = label.get("prompt_length", None)
            pred_text = self._decode_logits(logits[i], prompt_length)
            gold_answers = self._extract_gold_answers(label)
            self._em_scores.append(self._compute_exact_match(pred_text, gold_answers))
            self._f1_scores.append(self._compute_f1(pred_text, gold_answers))

        self._timing_records = list(result.timing_records)
        return self.compute()

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _flatten_labels(self, labels: Any) -> List[Dict]:
        """레이블을 1D 딕셔너리 리스트로 평탄화합니다. 중첩 배치 형식을 지원합니다."""
        if isinstance(labels, dict):
            return [labels]
        if isinstance(labels, list):
            # 중첩 배치: [[label_dict, ...], [...]] → [label_dict, ...]
            if labels and isinstance(labels[0], list):
                return [lbl for batch in labels for lbl in batch]
            return labels
        return [labels]

    def _decode_logits(self, logits_2d: np.ndarray, prompt_length: int = None) -> str:
        """
        (seq_len, vocab_size) logits → greedy decoding → 정규화 전 텍스트 반환.

        causal LM에서 logits[t]는 위치 t+1의 토큰을 예측합니다.
        따라서 모델이 생성한 답변은 logits[prompt_length-1:] 에 위치합니다.
        prompt_length가 없으면 전체 시퀀스를 디코딩합니다(레거시/테스트 호환).
        """
        if prompt_length is not None and prompt_length > 0:
            # 답변 생성 구간: logits[prompt_length-1] 이 첫 번째 답변 토큰을 예측
            answer_logits = logits_2d[prompt_length - 1:]
        else:
            answer_logits = logits_2d

        token_ids = np.argmax(answer_logits, axis=-1).tolist()

        # EOS 토큰에서 조기 종료
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None and eos_id in token_ids:
            token_ids = token_ids[:token_ids.index(eos_id)]

        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return text.strip()

    def _normalize_answer(self, s: str) -> str:
        """SQuAD 공식 normalizer: 소문자 변환, 관사·구두점·여분 공백 제거."""
        s = s.lower()
        s = s.translate(str.maketrans("", "", string.punctuation))
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        s = " ".join(s.split())
        return s

    def _extract_gold_answers(self, label: Dict) -> List[str]:
        """라벨에서 정답 텍스트 목록을 추출합니다. is_impossible=True이면 빈 문자열을 반환합니다."""
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
        """gold_list 중 가장 높은 토큰 수준 F1 점수를 반환합니다."""
        return max(self._token_f1(pred, g) for g in gold_list)

    def _token_f1(self, pred: str, gold: str) -> float:
        """두 문자열 사이의 토큰 수준 F1 점수를 계산합니다."""
        pred_tokens = self._normalize_answer(pred).split()
        gold_tokens = self._normalize_answer(gold).split()

        # 둘 다 빈 문자열이면 완벽한 일치 (unanswerable 질문을 올바르게 예측한 경우)
        if not pred_tokens and not gold_tokens:
            return 1.0

        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            return 0.0

        precision = num_common / len(pred_tokens)
        recall = num_common / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)

    # unanswerable로 간주할 문자열 집합
    _NO_ANS_MARKERS = frozenset({
        "unanswerable", "no answer", "cannot answer", "not answerable",
        "null", "none", "n/a", "unknown", "",
    })

    def _postprocess_response(self, text: str) -> str:
        """
        모델 생성 텍스트 → SQuAD2 평가용 예측 텍스트 변환.

        Stop token으로 1차 차단 후에도 남은 패턴을 정리합니다.
        1. "Passage:" / "Question:" 등 컨텍스트 반복 패턴 제거
        2. 첫 줄만 추출
        3. unanswerable 마커 감지 → 빈 문자열 반환
        4. 비정상 길이(너무 짧거나 너무 김) → 빈 문자열
        """
        # 1. 이스케이프된 개행 정규화
        text = text.replace("\\n", "\n").strip()

        # 2. "Passage:", "Question:", "Context:" 이후 내용 제거
        text = re.sub(r"(?i)(passage|question|context)[:\s].*$", "", text, flags=re.DOTALL).strip()

        # 3. 첫 줄만 추출
        first_line = text.split("\n")[0].strip()

        # 4. unanswerable 마커
        if first_line.lower() in self._NO_ANS_MARKERS:
            return ""

        # 5. 비정상 길이 필터 (1자 미만 또는 200자 초과는 hallucination으로 간주)
        # SQuAD 2.0 정답 최대 길이 기준으로 200자 설정 (기존 100자는 일부 긴 정답을 오탈락시킴)
        if len(first_line) < 1 or len(first_line) > 200:
            return ""

        return first_line

    def _log_sample(self, idx: int, label: Dict, pred: str, golds: List[str],
                    em: float, f1: float) -> None:
        """debug=True 일 때 샘플별 예측/정답/점수를 출력합니다."""
        if not self.debug:
            return
        qa_id = label.get("id", "?")
        gold_str = " | ".join(golds) if golds else "(none)"
        # 긴 텍스트는 잘라서 출력
        pred_display = (pred[:120] + "…") if len(pred) > 120 else pred
        print(
            f"[LlamaEval #{idx}] id={qa_id}\n"
            f"  PRED : {pred_display!r}\n"
            f"  GOLD : {gold_str!r}\n"
            f"  EM={em:.0f}  F1={f1:.3f}"
        )

    def _compute_latency_metrics(self, timing_records: List[float]) -> Dict[str, float]:
        """평균 및 P99 latency를 계산합니다."""
        if not timing_records:
            return {}
        return {
            "Average Latency (ms)": float(np.mean(timing_records)),
            "P99 Latency (ms)": float(np.percentile(timing_records, 99)),
        }

    def is_applicable(self, device_spec: Dict[str, Any], model_spec: Model_Spec) -> bool:
        return model_spec.task == Task.NLP_GENERATION

    def get_metric_names(self) -> List[str]:
        return [
            "Exact Match",
            "F1 Score",
            "Average Latency (ms)",
            "P99 Latency (ms)",
            "Avg TTFT (ms)",
            "P99 TTFT (ms)",
            "Avg Decode Step (no KV cache) (ms)",
            "P99 Decode Step (no KV cache) (ms)",
            "Throughput (tokens/s)",
            "Total Tokens Generated",
            "num_samples",
        ]
