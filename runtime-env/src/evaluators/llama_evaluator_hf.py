from typing import Dict, Any, List

import evaluate
import numpy as np
from transformers import AutoTokenizer

from .base import Evaluator
from core.model_spec import Model_Spec, Task
from core.inference_result import InferenceResult


class LlamaEvaluatorHF(Evaluator):
    """
    LlamaEvaluator의 HuggingFace evaluate 버전.

    EM·F1 계산을 직접 구현 대신 `evaluate.load("squad")`에 위임하여
    기존 구현과 수치 차이가 있는지 비교 실험용으로 작성.

    토크나이저, 후처리, 레이턴시 측정 로직은 LlamaEvaluator와 동일.
    차이점: add_batch()에서 점수를 즉시 계산하지 않고 예측/정답 쌍을 누산한 뒤
    compute()에서 HF squad metric으로 일괄 계산.
    """

    def __init__(self, **eval_options):
        tokenizer_path = eval_options.get("tokenizer_path", "meta-llama/Llama-3.1-8B-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self._squad_metric = evaluate.load("squad")
        self._reset()

    # ------------------------------------------------------------------
    # 내부 상태 초기화
    # ------------------------------------------------------------------

    def _reset(self):
        self._predictions: List[Dict] = []   # [{"id": str, "prediction_text": str}]
        self._references: List[Dict] = []    # [{"id": str, "answers": {"text": [...], "answer_start": [...]}}]
        self._timing_records: List[float] = []
        self._ttft_records: List[float] = []
        self._tpot_records: List[float] = []
        self._tokens_generated: List[int] = []
        self._sample_id = 0

    _EOT_ID: int = 128009

    @property
    def eos_token_id(self) -> List[int]:
        base = self.tokenizer.eos_token_id
        ids = base if isinstance(base, list) else [base]
        if self._EOT_ID not in ids:
            ids = ids + [self._EOT_ID]
        return ids

    # ------------------------------------------------------------------
    # 후처리 (LlamaEvaluator와 동일)
    # ------------------------------------------------------------------

    def _postprocess(self, decoded: str) -> str:
        answer = decoded.strip().split("\n")[0].strip()
        if answer.lower().startswith("answer:"):
            answer = answer[7:].strip()
        unanswerable_patterns = [
            "unanswerable", "cannot be answered", "not provided",
            "does not contain", "no information", "not mentioned",
            "cannot find", "not enough information"
        ]
        if any(p in answer.lower() for p in unanswerable_patterns):
            return ""
        return answer

    # ------------------------------------------------------------------
    # 스트리밍 인터페이스
    # ------------------------------------------------------------------

    def add_batch(self, outputs: Dict[str, np.ndarray], labels: Any, timing_ms: float, gen_timing=None) -> None:
        flat_labels = self._flatten_labels(labels)

        if "generated_ids" in outputs:
            token_ids = outputs["generated_ids"]
            if isinstance(token_ids, np.ndarray):
                token_ids = token_ids.tolist()
            stop_ids = set(self.eos_token_id)
            for i, tid in enumerate(token_ids):
                if tid in stop_ids:
                    token_ids = token_ids[:i]
                    break
            raw_text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            pred_text = self._postprocess(raw_text)
            label = flat_labels[0]
            gold_answers = self._extract_gold_answers(label)
            self._append_pair(pred_text, gold_answers)
        else:
            logits_key = list(outputs.keys())[0]
            logits = outputs[logits_key]
            for i in range(logits.shape[0]):
                label = flat_labels[i]
                prompt_length = label.get("prompt_length", None)
                raw_text = self._decode_logits(logits[i], prompt_length)
                pred_text = self._postprocess(raw_text)
                gold_answers = self._extract_gold_answers(label)
                self._append_pair(pred_text, gold_answers)

        self._timing_records.append(timing_ms)

        if gen_timing is not None:
            self._ttft_records.append(gen_timing.ttft_ms)
            if gen_timing.tpot_ms > 0.0:
                self._tpot_records.append(gen_timing.tpot_ms)
            self._tokens_generated.append(gen_timing.num_tokens)

    def _append_pair(self, pred_text: str, gold_answers: List[str]) -> None:
        """예측/정답 쌍을 HF squad 포맷으로 누산."""
        sample_id = str(self._sample_id)
        self._predictions.append({"id": sample_id, "prediction_text": pred_text})
        self._references.append({
            "id": sample_id,
            "answers": {
                "text": gold_answers,
                "answer_start": [0] * len(gold_answers),
            },
        })
        self._sample_id += 1

    def compute(self) -> Dict[str, Any]:
        num_samples = len(self._predictions)
        if num_samples == 0:
            return {"Exact Match": 0.0, "F1 Score": 0.0, "num_samples": 0}

        # HF squad metric은 EM/F1을 0~100 백분율로 반환
        hf_result = self._squad_metric.compute(
            predictions=self._predictions,
            references=self._references,
        )

        metrics = {
            "Exact Match": hf_result["exact_match"],
            "F1 Score": hf_result["f1"],
            "num_samples": num_samples,
        }
        metrics.update(self._compute_latency_metrics(self._timing_records))

        if self._ttft_records:
            metrics["TTFT Mean (ms)"] = float(np.mean(self._ttft_records))
            metrics["TTFT P99 (ms)"] = float(np.percentile(self._ttft_records, 99))
        if self._tpot_records:
            metrics["TPOT Mean (ms)"] = float(np.mean(self._tpot_records))
        if self._tokens_generated:
            total_tokens = sum(self._tokens_generated)
            total_time_s = sum(self._timing_records) / 1000.0
            metrics["Throughput (tokens/s)"] = (
                total_tokens / total_time_s if total_time_s > 0 else 0.0
            )
            metrics["Total Tokens Generated"] = total_tokens

        return metrics

    # ------------------------------------------------------------------
    # 배치 호환 인터페이스
    # ------------------------------------------------------------------

    def evaluate(self, result: InferenceResult) -> Dict[str, Any]:
        self._reset()

        logits_key = list(result.outputs.keys())[0]
        logits = result.outputs[logits_key]
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
            raw_text = self._decode_logits(logits[i], prompt_length)
            pred_text = self._postprocess(raw_text)
            gold_answers = self._extract_gold_answers(label)
            self._append_pair(pred_text, gold_answers)

        self._timing_records = list(result.timing_records)
        return self.compute()

    # ------------------------------------------------------------------
    # 내부 헬퍼 (LlamaEvaluator와 동일)
    # ------------------------------------------------------------------

    def _flatten_labels(self, labels: Any) -> List[Dict]:
        if isinstance(labels, dict):
            return [labels]
        if isinstance(labels, list):
            if labels and isinstance(labels[0], list):
                return [lbl for batch in labels for lbl in batch]
            return labels
        return [labels]

    def _decode_logits(self, logits_2d: np.ndarray, prompt_length: int = None) -> str:
        if prompt_length is not None and prompt_length > 0:
            answer_logits = logits_2d[prompt_length - 1:]
        else:
            answer_logits = logits_2d

        token_ids = np.argmax(answer_logits, axis=-1).tolist()

        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None and eos_id in token_ids:
            token_ids = token_ids[:token_ids.index(eos_id)]

        return self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    def _extract_gold_answers(self, label: Dict) -> List[str]:
        if label.get("is_impossible", False):
            return [""]
        answers = label.get("answers", [])
        texts = [a["text"] for a in answers if "text" in a]
        return texts if texts else [""]

    def _compute_latency_metrics(self, timing_records: List[float]) -> Dict[str, float]:
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
            "TTFT Mean (ms)",
            "TTFT P99 (ms)",
            "TPOT Mean (ms)",
            "Throughput (tokens/s)",
            "Total Tokens Generated",
            "num_samples",
        ]
