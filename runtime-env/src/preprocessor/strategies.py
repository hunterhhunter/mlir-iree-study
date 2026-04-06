"""
Preprocess Strategy Definitions

MLPerf 참조 알고리즘을 포함한 전처리 전략(Strategy) 패턴 구현체 모음.
mlperf_loadgen이나 mlperf 라이브러리를 직접 import하지 않고,
공개된 참조 구현의 알고리즘만 자체 코드로 재구현합니다.

Ref: https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection
"""

from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
from typing import Dict, Tuple


class PreprocessStrategy(ABC):
    """
    전처리 전략 인터페이스.
    PIL Image → 모델 입력 Numpy 텐서로 변환하는 단일 책임을 가집니다.
    """

    @abstractmethod
    def __call__(
        self,
        img: Image.Image,
        target_hw: tuple,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> np.ndarray:
        """
        Args:
            img       : 이미 open된 PIL.Image 객체 (RGB 변환 전이어도 무방)
            target_hw : 모델이 요구하는 (H, W) 형태의 튜플 (예: (224, 224))
            mean      : 채널별 정규화 평균값 np.ndarray (shape: [3])
            std       : 채널별 정규화 표준편차 np.ndarray (shape: [3])

        Returns:
            np.ndarray: shape (C, H, W), dtype float32
        """
        pass


class DirectResizePreprocess(PreprocessStrategy):
    """
    기존(레거시) 파이프라인: 비율 왜곡을 감수하고 target_hw로 직접 Resize.

    빠른 디버깅·호환성이 필요한 경우 사용합니다.
    mAP/Acc 정합성을 공식 벤치마크 수준으로 맞출 필요가 없을 때 적합합니다.
    """

    def __call__(
        self,
        img: Image.Image,
        target_hw: tuple,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> np.ndarray:
        img = img.convert("RGB")
        img = img.resize((target_hw[1], target_hw[0]), Image.Resampling.BILINEAR)

        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - mean) / std

        # HWC → CHW
        return np.transpose(arr, (2, 0, 1))


class MLPerfResNet50Preprocess(PreprocessStrategy):
    """
    MLPerf Inference 참조 구현 기반 ResNet-50 전처리.

    알고리즘 (OpenImages Dataset용):
      1. Shortest side를 256으로 비율 유지 Resize
      2. 중앙 224×224 Center Crop
      3. [0, 255] → [0.0, 1.0] 정규화 후 ImageNet mean/std 적용
      4. HWC → CHW 전치

    Ref:
      mlperf/inference/vision/classification_and_detection/python/dataset.py
      (pre_process_vgg 함수의 NHWC→NCHW 변형)
    """

    def __call__(
        self,
        img: Image.Image,
        target_hw: tuple,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> np.ndarray:
        img = img.convert("RGB")

        # Step 1: Shortest side → 256으로 비율 유지 Resize
        short_side = 256
        w, h = img.size
        if h < w:
            new_h = short_side
            new_w = int(round(w * short_side / h))
        else:
            new_w = short_side
            new_h = int(round(h * short_side / w))
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Step 2: Center Crop → target_hw (보통 224×224)
        crop_h, crop_w = target_hw
        if new_w < crop_w or new_h < crop_h:
            raise ValueError(
                f"Image size ({new_w}x{new_h}) after resize is smaller than crop size "
                f"({crop_w}x{crop_h}). Increase short_side or decrease target_hw."
            )
        left = (new_w - crop_w) // 2
        top  = (new_h - crop_h) // 2
        img  = img.crop((left, top, left + crop_w, top + crop_h))

        # Step 3: 정규화
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - mean) / std

        # Step 4: HWC → CHW
        return np.transpose(arr, (2, 0, 1))


class SQuADPreprocessStrategy(PreprocessStrategy):
    """
    SQuAD 2.0 QA 전처리 전략 — LLaMA 3.1 8B 전용.

    PreprocessStrategy를 상속하되, NLP 전용 tokenize() 메서드를 통해 동작합니다.
    __call__()은 이미지 파이프라인과의 인터페이스 호환성을 위해 구현을 보존하나,
    실제 호출 시 명시적 TypeError를 발생시켜 오용을 방지합니다.

    LLaMA 3.1 Chat Template 형식으로 프롬프트를 조립한 뒤 HuggingFace tokenizer로
    토큰화하여 input_ids / attention_mask numpy 배열을 반환합니다.
    """

    def __init__(self, tokenizer_path: str, max_length: int = 4096):
        """
        Args:
            tokenizer_path (str): HuggingFace 로컬 토크나이저 디렉토리 경로
                                  예: 'models/meta-llama/Meta-Llama-3.1-8B-Instruct'
            max_length (int): 최대 시퀀스 길이 (기본값 4096)
        """
        from transformers import AutoTokenizer
        self.tokenizer_path = tokenizer_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # LLaMA에는 pad_token이 없으므로 eos_token으로 대체 (표준 관행)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    def __call__(
        self,
        img: Image.Image,
        target_hw: tuple,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> np.ndarray:
        raise TypeError(
            "SQuADPreprocessStrategy는 이미지 파이프라인의 __call__()을 지원하지 않습니다. "
            "NLP 전처리는 tokenize(question, context) 메서드를 사용하세요."
        )

    def _build_prompt(self, question: str, context: str) -> str:
        """
        SQuAD 2.0 QA 전용 plain-text few-shot 프롬프트.

        Chat-template 토큰(<|begin_of_text|> 등)을 사용하지 않습니다.
        Base 모델과 Instruct 모델 모두에서 동작하며, 모델이 "Answer:" 이후를
        짧은 추출 답변으로 완성(completion)하도록 유도합니다.
        """
        return (
            "Extract the shortest possible answer from the passage.\n"
            "If the passage does not contain enough information, respond with \"unanswerable\".\n\n"
            "Passage: The capital of France is Paris.\n"
            "Question: What is the capital of France?\n"
            "Answer: Paris\n\n"
            f"Passage: {context}\n"
            f"Question: {question}\n"
            "Answer:"
        )

    def tokenize(
        self,
        question: str,
        context: str,
        padding: str = "max_length",
        truncation: bool = True,
    ) -> dict:
        """
        SQuAD QA 샘플을 LLaMA 3.1 8B 입력 텐서로 변환합니다.

        Args:
            question  (str): SQuAD QA 쌍의 질문 텍스트
            context   (str): SQuAD 단락의 컨텍스트 텍스트
            padding   (str): 'max_length' | 'longest'
            truncation (bool): max_length 초과 시 잘라내기 여부

        Returns:
            dict:
                'input_ids'      : np.ndarray, shape (1, max_length), dtype int64
                'attention_mask' : np.ndarray, shape (1, max_length), dtype int64
        """
        prompt = self._build_prompt(question, context)
        encoded = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="np",
        )
        return {
            "input_ids":      encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }


class TimeSeriesPreprocessStrategy:
    """
    (T, C) numpy 배열 → RevIN 정규화 후 past_values / past_observed_mask 반환.

    이미지 파이프라인과 인터페이스가 달라 PreprocessStrategy를 상속하지 않습니다.
    ETTmLoader가 직접 호출합니다.
    """

    def __call__(
        self,
        window: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Args:
            window: (context_length, C) float32 — 정규화 전 원본 윈도우

        Returns:
            dict:
                'past_values'        : (T, C) float32, RevIN 정규화 후
                'past_observed_mask' : (T, C) bool, 전부 True (ETTm 결측 없음)
                'norm_stats'         : {'mean': (C,), 'std': (C,)}
        """
        mean = window.mean(axis=0)              # (C,)
        std  = window.std(axis=0) + 1e-8        # (C,) 분모 0 방지

        past_norm = ((window - mean) / std).astype(np.float32)  # (T, C)

        return {
            "past_values":        past_norm,                          # (T, C)
            "past_observed_mask": np.ones_like(past_norm, dtype=bool),# (T, C)
            "norm_stats": {
                "mean": mean.astype(np.float32),
                "std":  std.astype(np.float32),
            },
        }
