import abc
import numpy as np
from typing import Dict, Any, List
from ..core.model_spec import Model_Spec
from ..core.inference_result import InferenceResult

class Evaluator(abc.ABC):
    """
    모든 평가(Evaluation) 클래스들이 상속받아야 하는 추상 기본 클래스.

    스트리밍 평가(Streaming Evaluation) 인터페이스를 채택합니다.
    BenchmarkRunner는 배치마다 add_batch()를 호출하여 무거운 텐서를 즉시 처리·폐기하고,
    모든 배치 완료 후 compute()로 최종 메트릭을 산출합니다.
    이 방식으로 수백만 샘플을 처리해도 RAM이 O(N) 누적되지 않습니다.
    """

    @abc.abstractmethod
    def __init__(self, **eval_options):
        """특화된 평가 파라미터(예: top_k)를 동적인 딕셔너리로 받아 초기화함."""
        pass

    @abc.abstractmethod
    def add_batch(self, outputs: Dict[str, np.ndarray], labels: Any, timing_ms: float) -> None:
        """
        [스트리밍 인터페이스] 배치 하나의 추론 결과를 즉시 처리합니다.

        무거운 outputs 텐서(logits 등)를 받아 필요한 경량 통계(argmax, 점수 합산 등)만
        내부 상태에 축적하고, 원본 텐서는 즉시 GC에 반환합니다.

        Args:
            outputs:   런타임이 반환한 한 배치의 출력 텐서 딕셔너리
            labels:    해당 배치의 정답 레이블 (DataLoader 규약에 따라 다름)
            timing_ms: 해당 배치의 추론 소요 시간 (밀리초)
        """
        pass

    @abc.abstractmethod
    def compute(self) -> Dict[str, Any]:
        """
        [스트리밍 인터페이스] add_batch()로 축적된 경량 통계로 최종 메트릭을 산출합니다.

        Returns:
            메트릭 이름 → 값 딕셔너리
        """
        pass

    @abc.abstractmethod
    def evaluate(self, result: InferenceResult) -> Dict[str, Any]:
        """
        [배치 호환 인터페이스] InferenceResult 전체를 받아 최종 점수 딕셔너리를 산출함.
        단위 테스트 및 레거시 호출을 위해 유지됩니다.
        내부적으로 _reset() → 데이터 처리 → compute() 흐름으로 구현합니다.
        """
        pass

    @abc.abstractmethod
    def is_applicable(self, device_spec: Dict[str, Any], model_spec: Model_Spec) -> bool:
        pass

    @abc.abstractmethod
    def get_metric_names(self) -> List[str]:
        """이 평가기가 뱉어낼 지표 목록을 미리 반환."""
        pass
