"""
Preprocessor Package — 모델별 전처리기 모음

각 모델 타입에 특화된 전처리기 클래스를 제공합니다.
모든 전처리기는 BasePreprocessor를 상속합니다.

핵심 원칙:
  - numpy 캐시가 존재하면 전처리 없이 로드합니다.
  - 캐시가 없으면 전처리를 실행하고 numpy 파일로 저장합니다.

전처리기 유형:
  샘플 단위 (.npy / .npz 캐싱):
    - ImagePreprocessor             : 이미지 분류
    - ObjectDetectionPreprocessor   : 객체 탐지 (YOLO)
    - LlamaPreprocessor             : LLaMA SQuAD QA 토큰화
    - ETTmPreprocessor              : 시계열 RevIN 정규화

  데이터셋 단위 (전체 데이터셋 .npy 저장):
    - BertClassificationPreprocessor : BERT SST-2 등 텍스트 분류
    - BertQAPreprocessor             : BERT SQuAD QA
"""

from .base import BasePreprocessor
from .strategies import (
    PreprocessStrategy,
    DirectResizePreprocess,
    MLPerfResNet50Preprocess,
    SQuADPreprocessStrategy,
    TimeSeriesPreprocessStrategy,
)
from .image_preprocessor import ImagePreprocessor
from .object_detection_preprocessor import ObjectDetectionPreprocessor
from .llama_preprocessor import LlamaPreprocessor
from .ettm_preprocessor import ETTmPreprocessor
from .bert_classification_preprocessor import BertClassificationPreprocessor
from .bert_qa_preprocessor import BertQAPreprocessor

__all__ = [
    "BasePreprocessor",
    "PreprocessStrategy",
    "DirectResizePreprocess",
    "MLPerfResNet50Preprocess",
    "SQuADPreprocessStrategy",
    "TimeSeriesPreprocessStrategy",
    "ImagePreprocessor",
    "ObjectDetectionPreprocessor",
    "LlamaPreprocessor",
    "ETTmPreprocessor",
    "BertClassificationPreprocessor",
    "BertQAPreprocessor",
]
