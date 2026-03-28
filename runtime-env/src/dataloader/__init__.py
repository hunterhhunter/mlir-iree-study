"""
DataLoader Package Initialization & Factory

이 모듈은 벤치마크 프레임워크의 다른 컴포넌트(Runtime, Evaluators 등)에게
DataLoader 클래스들에 대한 손쉬운 접근(단일 진입점 API)을 제공합니다.
"""

from ..core.model_spec import Model_Spec, Task
from .base import DataLoader
from .image_classification_loader import ImageClassificationLoader
from .object_detection_loader import ObjectDetectionLoader
from .llama_loader import LlamaLoader
from .preprocess_strategies import (
    PreprocessStrategy,
    MLPerfResNet50Preprocess,
    DirectResizePreprocess,
    SQuADPreprocessStrategy,
)

def create_dataloader(model_spec: Model_Spec, **kwargs) -> DataLoader:
    """
    Factory Method for DataLoader
    
    Model_Spec의 Task 종류(IMAGE_CLASSIFICATION 등)를 분석하여
    해당 Task에 알맞은 구체 로더(Concrete Loader) 객체를 초기화하여 반환합니다.
    
    Args:
        model_spec (Model_Spec): 로드할 모델의 코어 스펙 규격서
        **kwargs: dataset_path, image_dir, mean, std 등 구체 로더에 필요한 추가 인자
        
    Returns:
        DataLoader: 추상 베이스 클래스를 상속받은 구체 로더 인스턴스
        
    Raises:
        ValueError: 모델의 Task에 알맞은 로더가 구현되어 있지 않을 경우 발생
    """
    task = model_spec.task
    
    if task == Task.IMAGE_CLASSIFICATION:
        return ImageClassificationLoader(model_spec, **kwargs)
    elif task == Task.OBJECT_DETECTION:
        return ObjectDetectionLoader(model_spec, **kwargs)
    elif task == Task.NLP_GENERATION:
        return LlamaLoader(model_spec, **kwargs)
    else:
        raise ValueError(f"현재 '{task.name}' Task를 지원하는 DataLoader가 구현되어 있지 않습니다.")

__all__ = [
    "DataLoader",
    "ImageClassificationLoader",
    "ObjectDetectionLoader",
    "LlamaLoader",
    "create_dataloader",
    "PreprocessStrategy",
    "MLPerfResNet50Preprocess",
    "DirectResizePreprocess",
    "SQuADPreprocessStrategy",
]
