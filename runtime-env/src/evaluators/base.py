import abc
import numpy as np
from typing import Dict, Any, List
from ..core.model_spec import Model_Spec
from ..core.inference_result import InferenceResult

class Evaluator(abc.ABC):
    """
    모든 평가(Evaluation) 클래스들이 상속받아야 하는 추상 기본 클래스.
    순수 Numpy 배열과 실행 시간을 받아 최종 벤치마크 지표를 산출하는 기준점 역할을 함.
    """

    @abc.abstractmethod
    def __init__(self, **eval_options):
        """특화된 평가 파라미터(예: top_k)를 동적인 딕셔너리로 받아 초기화함."""
        pass

    @abc.abstractmethod
    def evaluate(self, result: InferenceResult) -> Dict[str, Any]:
        """
        DTO를 통해 예측값(outputs), 시간(timing_records), 정답지(labels)를 꺼낸 뒤
        최종 점수 딕셔너리를 산출함.
        """
        pass

    @abc.abstractmethod
    def is_applicable(self, device_spec: Dict[str, Any], model_spec: Model_Spec) -> bool:
        pass

    @abc.abstractmethod
    def get_metric_names(self) -> List[str]:
        """이 평가기가 뱉어낼 지표 목록을 미리 반환."""
        pass
