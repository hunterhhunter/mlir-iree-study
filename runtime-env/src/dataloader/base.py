"""
Abstract Base Class for DataLoader in the Benchmark Framework.

Task 별로 달라지는 원시(raw) 데이터를 통일된 인터페이스로 벤치마킹 시스템에
일관되게 공급하기 위한 최상위 추상화 계층입니다.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np

class DataLoader(ABC):
    @abstractmethod
    def __init__(self, model_spec: dict, **kwargs):
        """
        초기화 메서드.
        model_spec에 정의된 데이터 설정 사항, 전처리 파라미터 등을 기반으로 로더 환경을 세팅합니다.
        
        Args:
            model_spec (dict): Task 종류, 데이터 경로 등의 명세서
            **kwargs: 추가 커스텀 인자
        """
        pass

    @abstractmethod
    def load_single(self) -> Dict[str, Any]:
        """
        단일 데이터 샘플 하나를 메모리에 올리고 메타 데이터와 함께 반환합니다.
        
        Returns:
            dict: 'input'(tensor), 'label' 등을 포함한 데이터 딕셔너리
        """
        pass

    @abstractmethod
    def load_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        지정된 크기만큼의 배치 데이터를 리스트 형태로 반환합니다.
        대용량 데이터로 인한 메모리 오버플로우를 막기 위해 Lazy Loading 기법 적용을 권장합니다.
        
        Args:
            batch_size (int): 요청할 데이터 아이템 개수
            
        Returns:
            List[dict]: batch_size 만큼의 데이터 딕셔너리 리스트
        """
        pass

    @abstractmethod
    def get_labels(self) -> Any:
        """
        학습 혹은 평가 목적의 전체/부분 정답지(Label) 데이터를 반환합니다.
        평가기(Evaluator)와 결합할 때 주로 사용됩니다.
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        총 샘플 개수, 클래스 수, 요구 전처리 정규화 방식(mean, std) 등
        해당 데이터셋의 포괄적인 메타 정보를 반환합니다.
        """
        pass

    @abstractmethod
    def preprocess(self, raw_input: Any) -> np.ndarray:
        """
        원시(raw) 데이터를 모델/컴파일러 로직이 요구하는 Tensor 구조(shape, dtype)로 가공합니다.
        (예: Resize, Normalization, 차원 변경 HWC -> NCHW 등)
        
        Args:
            raw_input (Any): 디스크나 메모리에서 스트리밍 된 로우 파일 데이터
            
        Returns:
            np.ndarray: 모델 입력에 적합한 다차원 배열 데이터
        """
        pass

    @abstractmethod
    def load_by_index(self, index: int) -> Dict[str, Any]:
        """
        순서와 무관하게 특정 인덱스의 샘플 하나를 직접 반환합니다.
        LoadGen QSL의 issue_queries 콜백처럼 랜덤 접근이 필요할 때 사용합니다.

        Args:
            index (int): 데이터셋 내 샘플의 절대 인덱스 (0-based)

        Returns:
            dict: 'input'(tensor), 'label' 등을 포함한 데이터 딕셔너리
        """
        pass
