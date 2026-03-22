"""
Evaluator Package Initialization & Factory

이 모듈은 벤치마크 프레임워크의 다양한 Task(이미지 분류, 객체 탐지 등)에 맞는
평가기(Evaluator) 인스턴스를 동적으로 생성하고 접근할 수 있는 단일 진입점 API를 제공합니다.
"""

from ..core.model_spec import Model_Spec, Task
from .base import Evaluator
from .image_classification_evaluator import ImageClassificationEvaluator

def create_evaluator(model_spec: Model_Spec, **kwargs) -> Evaluator:
    """
    Factory Method for Evaluator
    
    Model_Spec의 Task 종류를 분석하여 해당 Task를 평가할 수 있는
    적절한 구체 평가기(Concrete Evaluator) 객체를 생성합니다.
    
    Args:
        model_spec (Model_Spec): 평가할 모델의 코어 스펙 규격서
        **kwargs: top_k 등 평가기에 전달될 추가 옵션 인자
        
    Returns:
        Evaluator: 추상 베이스 클래스를 상속받은 구체 평가기(Metric Calculator)
    """
    task = model_spec.task
    
    if task == Task.IMAGE_CLASSIFICATION:
        # 단일 책임 원칙: 이미지 분류 테스크는 ImageClassificationEvaluator가 전담
        # 추후 MobileNet 특화 로직이 별도로 필요하면 model_spec.name 등을 통해 분기 가능
        return ImageClassificationEvaluator(**kwargs)
    
    # 추후 NLP, Object Detection 등의 Task 평가기가 확장되면 여기에 추가
    elif task == Task.OBJECT_DETECTION:
        return ObjectDetectionEvaluator(**kwargs)
    else:
        raise ValueError(f"현재 '{task.name}' Task를 지원하는 Evaluator가 구현되어 있지 않습니다.")

__all__ = [
    "Evaluator",
    "ImageClassificationEvaluator",
    "create_evaluator",
    "ObjectDetectionEvaluator"
]
