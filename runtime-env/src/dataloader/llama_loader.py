"""
LLaMA DataLoader — SQuAD 2.0 / LLaMA 3.1 8B 전용

SQuAD 2.0 val.json을 파싱하여 LLaMA 3.1 8B 추론에 필요한
input_ids / attention_mask numpy 배열을 공급합니다.

출력 텐서:
    input_ids      : shape (1, max_length), dtype int64
    attention_mask : shape (1, max_length), dtype int64

바이너리 캐싱:
    cache_dir 지정 시 qa_id 기반으로 .npz 파일로 저장하여
    반복 실행 시 토큰화 비용을 제거합니다.
"""

import os
import json
from typing import Any, Dict, List, Optional

import numpy as np

from .base import DataLoader
from .preprocess_strategies import SQuADPreprocessStrategy
from ..core.model_spec import Model_Spec


class LlamaLoader(DataLoader):
    """
    SQuAD 2.0 val 데이터를 LLaMA 3.1 8B 입력 텐서로 변환하는 데이터로더.

    ImageClassificationLoader의 캐싱·순차접근 패턴을 NLP 도메인으로 확장합니다.
    """

    def __init__(self, model_spec: Model_Spec, **kwargs):
        """
        Args:
            model_spec (Model_Spec): NLP_GENERATION Task 스펙.
                input_shapes 예시: {'input_ids': (1, 4096), 'attention_mask': (1, 4096)}

            **kwargs:
                dataset_path (str)       : SQuAD JSON이 있는 디렉토리
                                           (squad_json 미지정 시 {dataset_path}/val.json 탐색)
                squad_json   (str)       : SQuAD val.json 직접 경로 (dataset_path보다 우선)
                tokenizer_path (str)     : 필수. HuggingFace 토크나이저 디렉토리 경로
                max_length   (int)       : 최대 시퀀스 길이 (기본 4096)
                cache_dir    (str|None)  : .npz 캐시 디렉토리 (None이면 비활성)
                include_impossible (bool): unanswerable 샘플 포함 여부 (기본 True)
                preprocess_strategy      : 외부 주입 전략 (기본 SQuADPreprocessStrategy)
        """
        self.model_spec = model_spec

        # 1. 경로 설정
        self.base_path = kwargs.get("dataset_path", "./datasets/SQuAD_2")
        squad_json = kwargs.get(
            "squad_json", os.path.join(self.base_path, "val.json")
        )

        # 2. 전처리 전략 초기화
        if "preprocess_strategy" in kwargs:
            self.preprocess_strategy: SQuADPreprocessStrategy = kwargs["preprocess_strategy"]
        else:
            tokenizer_path = kwargs.get("tokenizer_path")
            if tokenizer_path is None:
                raise ValueError(
                    "tokenizer_path가 필요합니다. "
                    "예: LlamaLoader(spec, tokenizer_path='models/meta-llama/...')"
                )
            max_length = kwargs.get("max_length", 4096)
            self.preprocess_strategy = SQuADPreprocessStrategy(
                tokenizer_path=tokenizer_path,
                max_length=max_length,
            )

        # 3. 캐시 설정
        self.cache_dir: Optional[str] = kwargs.get("cache_dir", None)
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"[LlamaLoader] 토큰화 캐시 활성화: {self.cache_dir}")

        # 4. SQuAD JSON 파싱
        include_impossible = kwargs.get("include_impossible", True)
        self.samples: List[Dict] = []
        self._qa_labels: Dict[str, Dict] = {}
        if os.path.exists(squad_json):
            self._parse_squad_json(squad_json, include_impossible)
        else:
            print(f"[LlamaLoader] 경고: SQuAD 파일을 찾을 수 없습니다: {squad_json}")

        self.total_samples = len(self.samples)
        self.current_idx = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_squad_json(self, json_path: str, include_impossible: bool) -> None:
        """
        SQuAD 2.0 JSON을 파싱하여 플랫한 샘플 리스트를 구성합니다.

        SQuAD 구조:
            data[].paragraphs[].qas[].{id, question, answers, is_impossible}
            data[].paragraphs[].context

        각 QA 쌍을 독립 샘플로 펼쳐 context를 인라인으로 저장합니다.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            squad = json.load(f)

        for article in squad["data"]:
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    if not include_impossible and qa["is_impossible"]:
                        continue
                    sample = {
                        "qa_id":             qa["id"],
                        "question":          qa["question"],
                        "context":           context,
                        "answers":           qa.get("answers", []),
                        "is_impossible":     qa["is_impossible"],
                        "plausible_answers": qa.get("plausible_answers", []),
                    }
                    self.samples.append(sample)
                    self._qa_labels[qa["id"]] = {
                        "answers":           qa.get("answers", []),
                        "is_impossible":     qa["is_impossible"],
                        "plausible_answers": qa.get("plausible_answers", []),
                    }

    def _get_cache_path(self, qa_id: str) -> Optional[str]:
        """qa_id 기반 .npz 캐시 파일 경로. 캐시 비활성 시 None."""
        if not self.cache_dir:
            return None
        # qa_id는 영숫자와 하이픈으로 구성되어 파일명으로 안전하게 사용 가능
        return os.path.join(self.cache_dir, f"{qa_id}.npz")

    def _load_or_tokenize(self, sample: Dict) -> Dict[str, np.ndarray]:
        """
        .npz 캐시가 있으면 로드, 없으면 토큰화 후 캐시에 저장합니다.
        (ImageClassificationLoader._load_or_preprocess 패턴과 동일)
        """
        cache_path = self._get_cache_path(sample["qa_id"])

        # 캐시 히트
        if cache_path and os.path.exists(cache_path):
            loaded = np.load(cache_path)
            return {
                "input_ids":      loaded["input_ids"],
                "attention_mask": loaded["attention_mask"],
            }

        # 토큰화 실행
        tensors = self.preprocess_strategy.tokenize(
            question=sample["question"],
            context=sample["context"],
        )

        # 캐시 저장
        if cache_path:
            np.savez(cache_path, **tensors)

        return tensors

    # ------------------------------------------------------------------
    # DataLoader ABC 구현
    # ------------------------------------------------------------------

    def load_single(self) -> Dict[str, Any]:
        if self.current_idx >= self.total_samples:
            raise StopIteration("모든 샘플이 소진되었습니다.")

        sample = self.samples[self.current_idx]
        self.current_idx += 1

        tensors = self._load_or_tokenize(sample)
        attn = tensors["attention_mask"]
        pos = np.maximum(np.cumsum(attn, axis=-1) - 1, 0).astype(np.int64)
        return {
            "input": {
                "input_ids":      tensors["input_ids"].reshape(-1),
                "attention_mask": attn.reshape(-1),
                "position_ids":   pos.reshape(-1),
            },
            "label": {
                "id":                sample["qa_id"],
                "answers":           sample["answers"],
                "is_impossible":     sample["is_impossible"],
                "plausible_answers": sample["plausible_answers"],
                # 프롬프트 실제 토큰 수: Evaluator가 답변 생성 위치를 찾기 위해 필요.
                # logits[prompt_length-1] 이 모델이 첫 답변 토큰을 예측하는 위치.
                "prompt_length":     int(attn.sum()),
            },
            "qa_id": sample["qa_id"],
        }

    def load_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(self.load_single())
            except StopIteration:
                break
        return batch

    def load_by_index(self, index: int) -> Dict[str, Any]:
        """
        인덱스 기반 직접 접근 — LoadGen QSL 콜백 및 랜덤 접근 지원용.
        current_idx 상태를 변경하지 않습니다.
        """
        if index < 0 or index >= self.total_samples:
            raise IndexError(
                f"index {index} is out of range [0, {self.total_samples})"
            )
        sample = self.samples[index]
        tensors = self._load_or_tokenize(sample)
        attn = tensors["attention_mask"]
        pos = np.maximum(np.cumsum(attn, axis=-1) - 1, 0).astype(np.int64)
        return {
            "input": {
                "input_ids":      tensors["input_ids"].reshape(-1),
                "attention_mask": attn.reshape(-1),
                "position_ids":   pos.reshape(-1),
            },
            "label": {
                "id":                sample["qa_id"],
                "answers":           sample["answers"],
                "is_impossible":     sample["is_impossible"],
                "plausible_answers": sample["plausible_answers"],
                "prompt_length":     int(attn.sum()),
            },
            "qa_id": sample["qa_id"],
        }

    def get_labels(self) -> Dict[str, Dict]:
        """
        qa_id → 정답 정보 딕셔너리를 반환합니다.
        표준 SQuAD 평가 스크립트 형식과 호환됩니다.
        """
        return self._qa_labels

    def _build_stop_token_ids(self) -> list:
        """EOS + 줄바꿈 계열 토큰 ID 목록을 반환합니다."""
        tok = self.preprocess_strategy.tokenizer
        ids = set()
        if tok.eos_token_id is not None:
            ids.add(tok.eos_token_id)
        for text in ["\n", "\n\n"]:
            encoded = tok.encode(text, add_special_tokens=False)
            if encoded:
                ids.add(encoded[0])
        return list(ids)

    def get_metadata(self) -> Dict[str, Any]:
        impossible_count = sum(1 for s in self.samples if s["is_impossible"])
        return {
            "total_samples":       self.total_samples,
            "answerable_samples":  self.total_samples - impossible_count,
            "impossible_samples":  impossible_count,
            "dataset_path":        self.base_path,
            "max_length":          self.preprocess_strategy.max_length,
            "tokenizer_path":      self.preprocess_strategy.tokenizer.name_or_path,
            "eos_token_id":        self.preprocess_strategy.tokenizer.eos_token_id,
            # 줄바꿈/이중줄바꿈 토큰: "Answer:" 이후 한 줄 완성을 유도하는 프롬프트와 쌍을 이룸
            "stop_token_ids":      self._build_stop_token_ids(),
            "cache_dir":           self.cache_dir,
            "preprocess_strategy": type(self.preprocess_strategy).__name__,
        }

    def preprocess(self, raw_input: Any) -> Dict[str, np.ndarray]:
        """
        단일 raw 입력을 토큰화합니다.

        Args:
            raw_input: dict {'question': str, 'context': str}
                       또는 tuple (question: str, context: str)

        Returns:
            dict: {'input_ids': ndarray, 'attention_mask': ndarray}
        """
        if isinstance(raw_input, dict):
            question = raw_input["question"]
            context  = raw_input["context"]
        else:
            question, context = raw_input
        return self.preprocess_strategy.tokenize(question=question, context=context)