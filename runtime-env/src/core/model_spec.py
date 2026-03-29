from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Tuple

class Task(Enum):
    """
    ML 벤치마킹 프레임워크에서 지원하는 작업 유형(Task Type) 정의.
    DataLoader와 Evaluator가 자신의 적용 가능 여부를 판단하는 기준으로 사용됨.
    """
    IMAGE_CLASSIFICATION = auto()
    OBJECT_DETECTION = auto()
    SEMANTIC_SEGMENTATION = auto()
    NLP_CLASSIFICATION = auto()
    NLP_GENERATION = auto()
    SPEECH_RECOGNITION = auto()
    MATMUL = auto()
    CONV2D = auto()

@dataclass(frozen=True)
class Model_Spec:
    """
    모델의 메타데이터 및 아티팩트 경로를 관리하는 핵심 데이터 클래스 (SSOT).
    
    Attributes:
        name: 모델 식별자 (예: 'resnet50')
        task: 지원하는 Task 유형 (Enum)
        input_shapes: 입력 텐서 이름과 형태 (예: {'input': (1, 3, 224, 224)})
        input_dtype: 입력 텐서의 데이터 타입 (예: {'input': 'float32'})
        output_shapes: 출력 텐서 이름과 기대 형태 (예: {'output': (1, 1000)})
        model_paths: 아티팩트 경로 딕셔너리 (예: {'onnx': 'path/to/model.onnx', 'vmfb': 'path/to/model.vmfb'})
    """
    name: str
    task: Task
    input_shapes: Dict[str, Tuple[int, ...]]
    input_dtype: Dict[str, str]
    output_shapes: Dict[str, Tuple[int, ...]]
    model_paths: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """인스턴스 생성 후 최소한의 유효성 검증 수행"""
        if not self.input_shapes:
            raise ValueError(f"Model_Spec '{self.name}' must have at least one input shape.")
        if not self.output_shapes:
            raise ValueError(f"Model_Spec '{self.name}' must have at least one output shape.")

