"""
LlamaLoader + SQuADPreprocessStrategy 단위 테스트

실제 토크나이저 없이도 동작하는 Mock 기반 테스트와
실제 SQuAD JSON 파싱 테스트를 분리합니다.

실행:
    python -m pytest tests/test_llama_loader.py -v
"""

import os
import sys
import json
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.core.model_spec import Model_Spec, Task
from src.dataloader import LlamaLoader, SQuADPreprocessStrategy, create_dataloader


# ------------------------------------------------------------------
# 공용 픽스처
# ------------------------------------------------------------------

LLAMA_INPUT_SHAPES = {
    "input_ids":      (1, 128),   # 테스트용 짧은 길이
    "attention_mask": (1, 128),
}

LLAMA_SPEC = Model_Spec(
    name="llama-3.1-8b",
    task=Task.NLP_GENERATION,
    input_shapes=LLAMA_INPUT_SHAPES,
    input_dtype={"input_ids": "int64", "attention_mask": "int64"},
    output_shapes={"logits": (1, 128, 32000)},
    model_paths={"onnx": "models/llama.onnx"},
)


def _make_squad_json(tmp_dir: str, num_samples: int = 5) -> str:
    """테스트용 최소 SQuAD 2.0 형식 JSON 파일을 생성합니다."""
    qas = []
    for i in range(num_samples):
        is_imp = i % 2 == 0
        qas.append({
            "id": f"qa_{i:04d}",
            "question": f"What is the capital of country {i}?",
            "answers": [] if is_imp else [{"text": f"City{i}", "answer_start": 10}],
            "is_impossible": is_imp,
            "plausible_answers": [{"text": f"Maybe{i}", "answer_start": 5}] if is_imp else [],
        })

    squad = {
        "data": [
            {
                "title": "TestArticle",
                "paragraphs": [
                    {
                        "context": "The capital of the test country is TestCity located at the center.",
                        "qas": qas,
                    }
                ],
            }
        ]
    }
    path = os.path.join(tmp_dir, "val.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(squad, f)
    return path


def _make_mock_strategy(max_length: int = 128) -> MagicMock:
    """실제 토크나이저 없이 동작하는 Mock SQuADPreprocessStrategy를 반환합니다."""
    strategy = MagicMock(spec=SQuADPreprocessStrategy)
    strategy.max_length = max_length
    strategy.tokenizer = MagicMock()
    strategy.tokenizer.name_or_path = "mock-tokenizer"
    strategy.tokenize.return_value = {
        "input_ids":      np.zeros((1, max_length), dtype=np.int64),
        "attention_mask": np.ones((1, max_length),  dtype=np.int64),
    }
    return strategy


# ------------------------------------------------------------------
# SQuADPreprocessStrategy 테스트
# ------------------------------------------------------------------

class TestSQuADPreprocessStrategy:

    def test_build_prompt_contains_question_and_context(self):
        """_build_prompt가 question과 context를 포함하는지 확인합니다."""
        with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok:
            mock_tok.return_value = MagicMock()
            mock_tok.return_value.pad_token = "[PAD]"
            strategy = SQuADPreprocessStrategy.__new__(SQuADPreprocessStrategy)
            strategy.tokenizer = mock_tok.return_value
            strategy.max_length = 128
            strategy.SYSTEM_PROMPT = SQuADPreprocessStrategy.SYSTEM_PROMPT

            prompt = strategy._build_prompt("Who is Alice?", "Alice is a person.")
            assert "Who is Alice?" in prompt
            assert "Alice is a person." in prompt
            assert "<|begin_of_text|>" in prompt
            assert "<|eot_id|>" in prompt

    def test_call_raises_type_error(self):
        """이미지 파이프라인용 __call__ 호출 시 TypeError를 발생시켜야 합니다."""
        with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok:
            mock_tok.return_value = MagicMock()
            mock_tok.return_value.pad_token = None
            mock_tok.return_value.eos_token = "<eos>"

            strategy = SQuADPreprocessStrategy("mock-path", max_length=128)
            with pytest.raises(TypeError, match="tokenize"):
                strategy(None, (224, 224), np.zeros(3), np.ones(3))

    def test_tokenize_output_shapes(self):
        """tokenize()가 올바른 shape의 numpy 배열을 반환하는지 확인합니다."""
        max_length = 64
        with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok:
            tokenizer = MagicMock()
            tokenizer.pad_token = "[PAD]"
            # tokenizer(prompt, ...) 호출 시 반환값을 dict로 직접 지정
            tokenizer.return_value = {
                "input_ids":      np.zeros((1, max_length), dtype=np.int64),
                "attention_mask": np.ones((1, max_length),  dtype=np.int64),
            }
            mock_tok.return_value = tokenizer

            strategy = SQuADPreprocessStrategy.__new__(SQuADPreprocessStrategy)
            strategy.tokenizer = tokenizer
            strategy.max_length = max_length
            strategy.SYSTEM_PROMPT = SQuADPreprocessStrategy.SYSTEM_PROMPT

            result = strategy.tokenize("What?", "Some context.")
            assert result["input_ids"].shape      == (1, max_length)
            assert result["attention_mask"].shape == (1, max_length)
            assert result["input_ids"].dtype      == np.int64
            assert result["attention_mask"].dtype == np.int64


# ------------------------------------------------------------------
# LlamaLoader 테스트
# ------------------------------------------------------------------

class TestLlamaLoader:

    def test_parse_squad_json(self, tmp_path):
        """SQuAD JSON이 올바르게 파싱되어 samples 리스트가 구성되는지 확인합니다."""
        squad_json = _make_squad_json(str(tmp_path), num_samples=5)
        strategy = _make_mock_strategy()

        loader = LlamaLoader(
            LLAMA_SPEC,
            squad_json=squad_json,
            preprocess_strategy=strategy,
        )
        assert loader.total_samples == 5
        assert all("qa_id" in s for s in loader.samples)
        assert all("question" in s for s in loader.samples)
        assert all("context" in s for s in loader.samples)

    def test_load_single_keys_and_shapes(self, tmp_path):
        """load_single()이 올바른 키와 텐서 shape을 반환하는지 확인합니다."""
        squad_json = _make_squad_json(str(tmp_path), num_samples=3)
        strategy = _make_mock_strategy(max_length=128)

        loader = LlamaLoader(
            LLAMA_SPEC,
            squad_json=squad_json,
            preprocess_strategy=strategy,
        )
        sample = loader.load_single()

        assert "input"          in sample
        assert "input_ids"      in sample["input"]
        assert "attention_mask" in sample["input"]
        assert "label"          in sample
        assert "qa_id"          in sample

        assert sample["input"]["input_ids"].shape      == (1, 128)
        assert sample["input"]["attention_mask"].shape == (1, 128)
        assert sample["input"]["input_ids"].dtype      == np.int64
        assert sample["input"]["attention_mask"].dtype == np.int64

    def test_load_single_increments_index(self, tmp_path):
        """load_single() 호출 시 current_idx가 증가하는지 확인합니다."""
        squad_json = _make_squad_json(str(tmp_path), num_samples=3)
        loader = LlamaLoader(LLAMA_SPEC, squad_json=squad_json,
                             preprocess_strategy=_make_mock_strategy())
        assert loader.current_idx == 0
        loader.load_single()
        assert loader.current_idx == 1
        loader.load_single()
        assert loader.current_idx == 2

    def test_load_single_raises_stop_iteration(self, tmp_path):
        """모든 샘플 소진 후 StopIteration이 발생하는지 확인합니다."""
        squad_json = _make_squad_json(str(tmp_path), num_samples=2)
        loader = LlamaLoader(LLAMA_SPEC, squad_json=squad_json,
                             preprocess_strategy=_make_mock_strategy())
        loader.load_single()
        loader.load_single()
        with pytest.raises(StopIteration):
            loader.load_single()

    def test_load_batch_size(self, tmp_path):
        """load_batch()가 요청한 크기만큼 샘플을 반환하는지 확인합니다."""
        squad_json = _make_squad_json(str(tmp_path), num_samples=5)
        loader = LlamaLoader(LLAMA_SPEC, squad_json=squad_json,
                             preprocess_strategy=_make_mock_strategy())
        batch = loader.load_batch(3)
        assert len(batch) == 3

    def test_load_batch_partial_at_end(self, tmp_path):
        """데이터셋 끝에서 load_batch()가 남은 샘플만 반환하는지 확인합니다."""
        squad_json = _make_squad_json(str(tmp_path), num_samples=3)
        loader = LlamaLoader(LLAMA_SPEC, squad_json=squad_json,
                             preprocess_strategy=_make_mock_strategy())
        batch = loader.load_batch(10)  # 3개만 남아 있음
        assert len(batch) == 3

    def test_load_by_index_does_not_change_current_idx(self, tmp_path):
        """load_by_index()가 current_idx를 변경하지 않는지 확인합니다."""
        squad_json = _make_squad_json(str(tmp_path), num_samples=5)
        loader = LlamaLoader(LLAMA_SPEC, squad_json=squad_json,
                             preprocess_strategy=_make_mock_strategy())
        loader.load_single()  # current_idx = 1
        loader.load_by_index(4)
        assert loader.current_idx == 1

    def test_load_by_index_out_of_range(self, tmp_path):
        """범위 밖 인덱스 접근 시 IndexError가 발생하는지 확인합니다."""
        squad_json = _make_squad_json(str(tmp_path), num_samples=3)
        loader = LlamaLoader(LLAMA_SPEC, squad_json=squad_json,
                             preprocess_strategy=_make_mock_strategy())
        with pytest.raises(IndexError):
            loader.load_by_index(10)

    def test_get_labels_structure(self, tmp_path):
        """get_labels()가 qa_id 키를 가진 딕셔너리를 반환하는지 확인합니다."""
        squad_json = _make_squad_json(str(tmp_path), num_samples=5)
        loader = LlamaLoader(LLAMA_SPEC, squad_json=squad_json,
                             preprocess_strategy=_make_mock_strategy())
        labels = loader.get_labels()
        assert isinstance(labels, dict)
        assert len(labels) == 5
        for qa_id, label in labels.items():
            assert "answers"       in label
            assert "is_impossible" in label

    def test_get_metadata_keys(self, tmp_path):
        """get_metadata()가 필수 키를 모두 포함하는지 확인합니다."""
        squad_json = _make_squad_json(str(tmp_path), num_samples=5)
        loader = LlamaLoader(LLAMA_SPEC, squad_json=squad_json,
                             preprocess_strategy=_make_mock_strategy())
        meta = loader.get_metadata()
        for key in ("total_samples", "answerable_samples", "impossible_samples",
                    "max_length", "tokenizer_path", "cache_dir", "preprocess_strategy"):
            assert key in meta, f"메타데이터에 '{key}' 키가 없습니다."

    def test_get_metadata_sample_counts(self, tmp_path):
        """get_metadata()의 answerable/impossible 카운트가 정확한지 확인합니다."""
        squad_json = _make_squad_json(str(tmp_path), num_samples=5)
        # 인덱스 0, 2, 4 → is_impossible=True (3개), 1, 3 → False (2개)
        loader = LlamaLoader(LLAMA_SPEC, squad_json=squad_json,
                             preprocess_strategy=_make_mock_strategy())
        meta = loader.get_metadata()
        assert meta["total_samples"]      == 5
        assert meta["impossible_samples"] == 3
        assert meta["answerable_samples"] == 2

    def test_cache_roundtrip(self, tmp_path):
        """.npz 캐시 저장 후 재로드 시 배열이 동일한지 확인합니다."""
        squad_json = _make_squad_json(str(tmp_path), num_samples=2)
        cache_dir  = str(tmp_path / "cache")
        strategy   = _make_mock_strategy(max_length=128)

        loader = LlamaLoader(
            LLAMA_SPEC,
            squad_json=squad_json,
            preprocess_strategy=strategy,
            cache_dir=cache_dir,
        )
        # 첫 번째 호출 → 토큰화 후 .npz 저장
        sample1 = loader.load_single()

        # tokenize 호출 횟수 초기화 후 다시 load_by_index → 캐시에서 로드
        strategy.tokenize.reset_mock()
        sample2 = loader.load_by_index(0)

        np.testing.assert_array_equal(sample1["input"]["input_ids"], sample2["input"]["input_ids"])
        np.testing.assert_array_equal(sample1["input"]["attention_mask"], sample2["input"]["attention_mask"])
        # 캐시 히트이므로 tokenize가 호출되지 않아야 함
        strategy.tokenize.assert_not_called()

    def test_preprocess_with_dict(self, tmp_path):
        """preprocess()에 dict 입력이 정상 동작하는지 확인합니다."""
        squad_json = _make_squad_json(str(tmp_path), num_samples=1)
        strategy = _make_mock_strategy()
        loader = LlamaLoader(LLAMA_SPEC, squad_json=squad_json,
                             preprocess_strategy=strategy)

        result = loader.preprocess({"question": "Who?", "context": "Someone."})
        assert "input_ids" in result
        assert "attention_mask" in result

    def test_preprocess_with_tuple(self, tmp_path):
        """preprocess()에 tuple 입력이 정상 동작하는지 확인합니다."""
        squad_json = _make_squad_json(str(tmp_path), num_samples=1)
        strategy = _make_mock_strategy()
        loader = LlamaLoader(LLAMA_SPEC, squad_json=squad_json,
                             preprocess_strategy=strategy)

        result = loader.preprocess(("Who?", "Someone."))
        assert "input_ids" in result

    def test_missing_squad_file_does_not_raise(self, tmp_path):
        """SQuAD 파일이 없을 때 경고만 출력하고 예외를 던지지 않는지 확인합니다."""
        loader = LlamaLoader(
            LLAMA_SPEC,
            squad_json=str(tmp_path / "nonexistent.json"),
            preprocess_strategy=_make_mock_strategy(),
        )
        assert loader.total_samples == 0

    def test_include_impossible_false(self, tmp_path):
        """include_impossible=False 시 unanswerable 샘플이 제외되는지 확인합니다."""
        squad_json = _make_squad_json(str(tmp_path), num_samples=5)
        loader = LlamaLoader(
            LLAMA_SPEC,
            squad_json=squad_json,
            preprocess_strategy=_make_mock_strategy(),
            include_impossible=False,
        )
        # 인덱스 1, 3만 answerable (is_impossible=False)
        assert loader.total_samples == 2
        assert all(not s["is_impossible"] for s in loader.samples)


# ------------------------------------------------------------------
# 팩토리 테스트
# ------------------------------------------------------------------

class TestFactory:

    def test_create_dataloader_nlp_generation(self, tmp_path):
        """create_dataloader()가 NLP_GENERATION Task에 LlamaLoader를 반환하는지 확인합니다."""
        squad_json = _make_squad_json(str(tmp_path), num_samples=2)
        loader = create_dataloader(
            LLAMA_SPEC,
            squad_json=squad_json,
            preprocess_strategy=_make_mock_strategy(),
        )
        assert isinstance(loader, LlamaLoader)

    def test_create_dataloader_unsupported_task_raises(self):
        """지원하지 않는 Task에 ValueError가 발생하는지 확인합니다."""
        spec = Model_Spec(
            name="unknown",
            task=Task.SPEECH_RECOGNITION,
            input_shapes={"input": (1, 80, 3000)},
            input_dtype={"input": "float32"},
            output_shapes={"output": (1, 100)},
            model_paths={"onnx": "models/dummy.onnx"},
        )
        with pytest.raises(ValueError):
            create_dataloader(spec)
