"""
LlamaEvaluator 통합 테스트

LlamaLoader로 샘플을 로드하고, 그 레이블을 그대로 LlamaEvaluator에 입력하여
EM · F1 · Latency 지표가 올바르게 계산되는지 검증합니다.

실제 LLaMA 모델 / 토크나이저 없이 실행되도록:
  - LlamaLoader: MockPreprocessStrategy 주입 (tokenizer 불필요)
  - LlamaEvaluator: AutoTokenizer.from_pretrained를 Mock으로 교체
"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.inference_result import InferenceResult
from core.model_spec import Model_Spec, Task
from dataloader.llama_loader import LlamaLoader
from evaluators import LlamaEvaluator, create_evaluator

# ─────────────────────────────────────────────────────────────────────────────
# 공통 픽스처
# ─────────────────────────────────────────────────────────────────────────────

MAX_LEN = 16
VOCAB_SIZE = 100


class MockPreprocessStrategy:
    """실제 토크나이저 없이 고정 더미 텐서를 반환하는 전처리 전략."""

    max_length = MAX_LEN

    class _FakeTokenizer:
        name_or_path = "mock-tokenizer"

    tokenizer = _FakeTokenizer()

    def tokenize(self, question: str, context: str, **kwargs):
        return {
            "input_ids":      np.zeros((1, MAX_LEN), dtype=np.int64),
            "attention_mask": np.ones((1, MAX_LEN),  dtype=np.int64),
        }


def _make_squad_json(samples: list) -> str:
    """samples 리스트로 최소 SQuAD 2.0 JSON을 임시 파일에 저장 후 경로를 반환."""
    squad = {
        "data": [
            {
                "title": "Test",
                "paragraphs": [
                    {
                        "context": s["context"],
                        "qas": [
                            {
                                "id":               s["id"],
                                "question":         s["question"],
                                "answers":          s.get("answers", []),
                                "is_impossible":    s.get("is_impossible", False),
                                "plausible_answers": s.get("plausible_answers", []),
                            }
                        ],
                    }
                ],
            }
            for s in samples
        ]
    }
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(squad, tmp)
    tmp.close()
    return tmp.name


def _make_loader(samples: list) -> LlamaLoader:
    """MockPreprocessStrategy가 주입된 LlamaLoader를 반환."""
    spec = Model_Spec(
        name="llama-3.1-8b",
        task=Task.NLP_GENERATION,
        input_shapes={"input_ids": (1, MAX_LEN), "attention_mask": (1, MAX_LEN)},
        input_dtype={"input_ids": "int64", "attention_mask": "int64"},
        output_shapes={"logits": (1, MAX_LEN, VOCAB_SIZE)},
    )
    squad_path = _make_squad_json(samples)
    return LlamaLoader(
        spec,
        squad_json=squad_path,
        preprocess_strategy=MockPreprocessStrategy(),
    )


def _make_evaluator(decode_fn) -> LlamaEvaluator:
    """
    AutoTokenizer를 Mock으로 교체한 LlamaEvaluator를 반환.
    decode_fn: (token_ids, skip_special_tokens) → str
    """
    mock_tok = MagicMock()
    mock_tok.decode.side_effect = lambda ids, **kw: decode_fn(ids)
    with patch("src.evaluators.llama_evaluator.AutoTokenizer") as mock_cls:
        mock_cls.from_pretrained.return_value = mock_tok
        evaluator = LlamaEvaluator(tokenizer_path="mock")
    return evaluator


def _make_logits(batch: int = 1) -> np.ndarray:
    """(batch, seq_len, vocab_size) 형태의 더미 logits."""
    rng = np.random.default_rng(42)
    return rng.random((batch, MAX_LEN, VOCAB_SIZE)).astype(np.float32)


def _build_result(labels, decode_fn, timing=None) -> tuple:
    """(InferenceResult, LlamaEvaluator) 쌍을 반환하는 헬퍼."""
    evaluator = _make_evaluator(decode_fn)
    result = InferenceResult(
        outputs={"logits": _make_logits(len(labels))},
        labels=labels,
        timing_records=timing if timing is not None else [10.0, 20.0, 30.0],
    )
    return result, evaluator


# ─────────────────────────────────────────────────────────────────────────────
# 테스트: EM (Exact Match)
# ─────────────────────────────────────────────────────────────────────────────

def test_exact_match_perfect():
    """모델이 정답을 정확히 예측하면 EM=100."""
    samples = [{"id": "q1", "question": "What city?", "context": "Paris is the capital.",
                "answers": [{"text": "Paris", "answer_start": 0}], "is_impossible": False}]
    loader = _make_loader(samples)
    item = loader.load_single()
    label = item["label"]

    result, evaluator = _build_result([label], decode_fn=lambda _: "Paris")
    metrics = evaluator.evaluate(result)

    assert metrics["Exact Match"] == 100.0, f"Expected EM=100, got {metrics['Exact Match']}"
    print(f"[PASS] test_exact_match_perfect: EM={metrics['Exact Match']}")


def test_exact_match_zero():
    """모델이 완전히 틀린 답변을 내면 EM=0."""
    samples = [{"id": "q2", "question": "What city?", "context": "Paris is the capital.",
                "answers": [{"text": "Paris", "answer_start": 0}], "is_impossible": False}]
    loader = _make_loader(samples)
    item = loader.load_single()

    result, evaluator = _build_result([item["label"]], decode_fn=lambda _: "London")
    metrics = evaluator.evaluate(result)

    assert metrics["Exact Match"] == 0.0, f"Expected EM=0, got {metrics['Exact Match']}"
    print(f"[PASS] test_exact_match_zero: EM={metrics['Exact Match']}")


def test_exact_match_normalization():
    """대소문자·구두점이 달라도 정규화 후 일치하면 EM=100."""
    samples = [{"id": "q3", "question": "What is it?", "context": "The answer is A.",
                "answers": [{"text": "the answer", "answer_start": 0}], "is_impossible": False}]
    loader = _make_loader(samples)
    item = loader.load_single()

    # 대문자, 마침표 포함 — 정규화 후 "answer"로 같아져야 함
    result, evaluator = _build_result([item["label"]], decode_fn=lambda _: "The Answer.")
    metrics = evaluator.evaluate(result)

    assert metrics["Exact Match"] == 100.0, f"Expected EM=100 after normalization, got {metrics['Exact Match']}"
    print(f"[PASS] test_exact_match_normalization: EM={metrics['Exact Match']}")


# ─────────────────────────────────────────────────────────────────────────────
# 테스트: is_impossible (SQuAD 2.0 답변 불가 질문)
# ─────────────────────────────────────────────────────────────────────────────

def test_is_impossible_correct_abstention():
    """is_impossible=True인 질문에서 빈 문자열을 예측하면 EM=100."""
    samples = [{"id": "q4", "question": "Unknown?", "context": "No answer here.",
                "answers": [], "is_impossible": True}]
    loader = _make_loader(samples)
    item = loader.load_single()

    result, evaluator = _build_result([item["label"]], decode_fn=lambda _: "")
    metrics = evaluator.evaluate(result)

    assert metrics["Exact Match"] == 100.0, \
        f"Expected EM=100 for correct abstention, got {metrics['Exact Match']}"
    print(f"[PASS] test_is_impossible_correct_abstention: EM={metrics['Exact Match']}")


def test_is_impossible_wrong_answer():
    """is_impossible=True인 질문에서 답변을 내면 EM=0."""
    samples = [{"id": "q5", "question": "Unknown?", "context": "No answer here.",
                "answers": [], "is_impossible": True}]
    loader = _make_loader(samples)
    item = loader.load_single()

    result, evaluator = _build_result([item["label"]], decode_fn=lambda _: "some wrong answer")
    metrics = evaluator.evaluate(result)

    assert metrics["Exact Match"] == 0.0, \
        f"Expected EM=0 for wrong answer on unanswerable question, got {metrics['Exact Match']}"
    print(f"[PASS] test_is_impossible_wrong_answer: EM={metrics['Exact Match']}")


# ─────────────────────────────────────────────────────────────────────────────
# 테스트: F1 Score
# ─────────────────────────────────────────────────────────────────────────────

def test_f1_perfect():
    """정답을 정확히 예측하면 F1=100."""
    samples = [{"id": "q6", "question": "Q?", "context": "ctx",
                "answers": [{"text": "the quick brown fox", "answer_start": 0}],
                "is_impossible": False}]
    loader = _make_loader(samples)
    item = loader.load_single()

    result, evaluator = _build_result([item["label"]], decode_fn=lambda _: "the quick brown fox")
    metrics = evaluator.evaluate(result)

    assert metrics["F1 Score"] == pytest.approx(100.0), \
        f"Expected F1=100, got {metrics['F1 Score']}"
    print(f"[PASS] test_f1_perfect: F1={metrics['F1 Score']:.2f}")


def test_f1_partial():
    """부분 일치 시 0 < F1 < 100."""
    samples = [{"id": "q7", "question": "Q?", "context": "ctx",
                "answers": [{"text": "quick brown fox", "answer_start": 0}],
                "is_impossible": False}]
    loader = _make_loader(samples)
    item = loader.load_single()

    # "quick" 1개만 일치 → precision=1/1, recall=1/3 → F1 = 2*(1)*(1/3)/(1+1/3) = 0.5
    result, evaluator = _build_result([item["label"]], decode_fn=lambda _: "quick")
    metrics = evaluator.evaluate(result)

    assert 0.0 < metrics["F1 Score"] < 100.0, \
        f"Expected 0 < F1 < 100, got {metrics['F1 Score']}"
    print(f"[PASS] test_f1_partial: F1={metrics['F1 Score']:.2f}")


def test_f1_zero():
    """겹치는 토큰이 없으면 F1=0."""
    samples = [{"id": "q8", "question": "Q?", "context": "ctx",
                "answers": [{"text": "alpha beta", "answer_start": 0}],
                "is_impossible": False}]
    loader = _make_loader(samples)
    item = loader.load_single()

    result, evaluator = _build_result([item["label"]], decode_fn=lambda _: "gamma delta")
    metrics = evaluator.evaluate(result)

    assert metrics["F1 Score"] == pytest.approx(0.0), \
        f"Expected F1=0, got {metrics['F1 Score']}"
    print(f"[PASS] test_f1_zero: F1={metrics['F1 Score']:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 테스트: 다중 정답 (gold_list)
# ─────────────────────────────────────────────────────────────────────────────

def test_multiple_gold_answers_best_match():
    """여러 정답 중 하나와 일치하면 EM=100."""
    samples = [{"id": "q9", "question": "Q?", "context": "ctx",
                "answers": [
                    {"text": "answer one", "answer_start": 0},
                    {"text": "answer two", "answer_start": 0},
                ],
                "is_impossible": False}]
    loader = _make_loader(samples)
    item = loader.load_single()

    result, evaluator = _build_result([item["label"]], decode_fn=lambda _: "answer two")
    metrics = evaluator.evaluate(result)

    assert metrics["Exact Match"] == 100.0, \
        f"Expected EM=100 with second gold match, got {metrics['Exact Match']}"
    print(f"[PASS] test_multiple_gold_answers_best_match: EM={metrics['Exact Match']}")


# ─────────────────────────────────────────────────────────────────────────────
# 테스트: Latency
# ─────────────────────────────────────────────────────────────────────────────

def test_latency_average():
    """timing_records의 평균이 올바르게 계산되는지 검증."""
    samples = [{"id": "q10", "question": "Q?", "context": "ctx",
                "answers": [{"text": "x", "answer_start": 0}], "is_impossible": False}]
    loader = _make_loader(samples)
    item = loader.load_single()

    timing = [10.0, 20.0, 30.0, 40.0]
    result, evaluator = _build_result([item["label"]], decode_fn=lambda _: "x", timing=timing)
    metrics = evaluator.evaluate(result)

    assert metrics["Average Latency (ms)"] == pytest.approx(25.0), \
        f"Expected avg=25.0, got {metrics['Average Latency (ms)']}"
    print(f"[PASS] test_latency_average: avg={metrics['Average Latency (ms)']:.1f} ms")


def test_latency_p99():
    """P99 latency가 올바르게 계산되는지 검증."""
    samples = [{"id": "q11", "question": "Q?", "context": "ctx",
                "answers": [{"text": "x", "answer_start": 0}], "is_impossible": False}]
    loader = _make_loader(samples)
    item = loader.load_single()

    timing = list(range(1, 101))  # 1~100 ms
    result, evaluator = _build_result([item["label"]], decode_fn=lambda _: "x", timing=timing)
    metrics = evaluator.evaluate(result)

    expected_p99 = float(np.percentile(timing, 99))
    assert metrics["P99 Latency (ms)"] == pytest.approx(expected_p99), \
        f"Expected P99={expected_p99}, got {metrics['P99 Latency (ms)']}"
    print(f"[PASS] test_latency_p99: P99={metrics['P99 Latency (ms)']:.1f} ms")


# ─────────────────────────────────────────────────────────────────────────────
# 테스트: 배치 (다중 샘플)
# ─────────────────────────────────────────────────────────────────────────────

def test_batch_evaluation():
    """배치 내 일부만 정답인 경우 EM·F1이 평균으로 반환되는지 검증."""
    samples = [
        {"id": "b1", "question": "Q1?", "context": "ctx1",
         "answers": [{"text": "alpha", "answer_start": 0}], "is_impossible": False},
        {"id": "b2", "question": "Q2?", "context": "ctx2",
         "answers": [{"text": "beta", "answer_start": 0}], "is_impossible": False},
    ]
    loader = _make_loader(samples)
    batch = loader.load_batch(2)
    labels = [item["label"] for item in batch]

    # 첫 번째 정답(alpha), 두 번째 오답(wrong)
    call_count = [0]
    def decode_fn(_):
        idx = call_count[0]
        call_count[0] += 1
        return ["alpha", "wrong"][idx % 2]

    evaluator = _make_evaluator(decode_fn)
    result = InferenceResult(
        outputs={"logits": _make_logits(2)},
        labels=labels,
        timing_records=[15.0, 25.0],
    )
    metrics = evaluator.evaluate(result)

    # EM: 1개 정답 / 2개 샘플 = 50.0
    assert metrics["Exact Match"] == pytest.approx(50.0), \
        f"Expected EM=50.0 for batch, got {metrics['Exact Match']}"
    assert metrics["num_samples"] == 2
    print(f"[PASS] test_batch_evaluation: EM={metrics['Exact Match']}, samples={metrics['num_samples']}")


# ─────────────────────────────────────────────────────────────────────────────
# 테스트: 팩토리 라우팅
# ─────────────────────────────────────────────────────────────────────────────

def test_factory_routes_nlp_generation():
    """create_evaluator가 Task.NLP_GENERATION에서 LlamaEvaluator를 반환하는지 검증."""
    spec = Model_Spec(
        name="llama-3.1-8b",
        task=Task.NLP_GENERATION,
        input_shapes={"input_ids": (1, MAX_LEN)},
        input_dtype={"input_ids": "int64"},
        output_shapes={"logits": (1, MAX_LEN, VOCAB_SIZE)},
    )
    with patch("src.evaluators.llama_evaluator.AutoTokenizer") as mock_cls:
        mock_cls.from_pretrained.return_value = MagicMock()
        evaluator = create_evaluator(spec, tokenizer_path="mock")

    assert isinstance(evaluator, LlamaEvaluator), \
        f"Expected LlamaEvaluator, got {type(evaluator)}"
    print(f"[PASS] test_factory_routes_nlp_generation: {type(evaluator).__name__}")


def test_is_applicable():
    """is_applicable이 NLP_GENERATION에서 True를 반환하는지 검증."""
    spec = Model_Spec(
        name="llama-3.1-8b",
        task=Task.NLP_GENERATION,
        input_shapes={"input_ids": (1, MAX_LEN)},
        input_dtype={"input_ids": "int64"},
        output_shapes={"logits": (1, MAX_LEN, VOCAB_SIZE)},
    )
    with patch("src.evaluators.llama_evaluator.AutoTokenizer") as mock_cls:
        mock_cls.from_pretrained.return_value = MagicMock()
        evaluator = LlamaEvaluator(tokenizer_path="mock")

    assert evaluator.is_applicable({}, spec) is True
    print("[PASS] test_is_applicable: NLP_GENERATION → True")


def test_get_metric_names():
    """get_metric_names가 기대하는 지표 이름을 반환하는지 검증."""
    with patch("src.evaluators.llama_evaluator.AutoTokenizer") as mock_cls:
        mock_cls.from_pretrained.return_value = MagicMock()
        evaluator = LlamaEvaluator(tokenizer_path="mock")

    names = evaluator.get_metric_names()
    for expected in ["Exact Match", "F1 Score", "Average Latency (ms)", "P99 Latency (ms)"]:
        assert expected in names, f"'{expected}' not in get_metric_names()"
    print(f"[PASS] test_get_metric_names: {names}")


# ─────────────────────────────────────────────────────────────────────────────
# 직접 실행
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("LlamaEvaluator 통합 테스트")
    print("=" * 60)

    test_exact_match_perfect()
    test_exact_match_zero()
    test_exact_match_normalization()
    test_is_impossible_correct_abstention()
    test_is_impossible_wrong_answer()
    test_f1_perfect()
    test_f1_partial()
    test_f1_zero()
    test_multiple_gold_answers_best_match()
    test_latency_average()
    test_latency_p99()
    test_batch_evaluation()
    test_factory_routes_nlp_generation()
    test_is_applicable()
    test_get_metric_names()

    print("\n[+] 모든 테스트 통과!")
