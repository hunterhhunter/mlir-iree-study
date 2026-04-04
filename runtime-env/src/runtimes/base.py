import abc
import dataclasses
import numpy as np
from typing import Dict, Any, List, Optional
from core.compiled_model import CompiledModel


@dataclasses.dataclass(frozen=True)
class GenerationResult:
    """자기회귀 생성 결과 컨테이너."""
    generated_ids: np.ndarray  # shape (num_tokens,), dtype int64
    ttft_ms: float             # Time To First Token (근사값)
    tpot_ms: float             # Time Per Output Token (근사값)
    total_ms: float            # 전체 생성 시간
    num_tokens: int            # 생성된 토큰 수


class Runtime(abc.ABC):
    """
    하드웨어 종속 추론 실행을 위한 추상 클래스.
    모든 백엔드(IREE, ONNX 등)의 런타임 구현체는 이 클래스를 상속받야 함.
    """

    @abc.abstractmethod
    def __init__(self, **runtime_options):
        """런타임 전용 환경 설정 및 옵션을 받아 초기화함."""
        pass
    @abc.abstractmethod
    def load(self, compiled_model: CompiledModel) -> None:
        """
        문자열 경로나 스펙 대신 오직 'CompiledModel' 아티팩트만을 인자로 받음.
        내부에서 파일이 존재하는지와 같은 컴파일러 측 로직을 검증하지 않음.
        """
        pass

    @abc.abstractmethod
    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Numpy 배열로 입력을 받아 추론을 수행한 뒤, 
        프레임워크 특화 텐서를 모두 제거하고 오직 Numpy 딕셔너리로 결괏값을 반환.
        """
        pass

    @abc.abstractmethod
    def warmup(self, inputs: Dict[str, np.ndarray], num_runs: int = 1) -> None:
        """캐시 및 런타임 엔진 웜업을 힌트로 받아 수행함."""
        pass

    @abc.abstractmethod
    def unload(self) -> None:
        """할당된 타겟 디바이스의 데이터 메모리를 반환/해제함."""
        pass

    @abc.abstractmethod
    def get_device_spec(self) -> Dict[str, Any]:
        """현재 모델이 구동 중인 하드웨어(CPU/GPU/NPU) 메타데이터를 반환함."""
        pass

    @abc.abstractmethod
    def is_compatible(self, compiled_model: CompiledModel) -> bool:
        """현재 로드된 런타임 옵션으로 해당 아티팩트의 추론이 가능한지 동적으로 검사함."""
        pass

    def supports_generate(self) -> bool:
        """이 런타임이 generate()를 실제로 지원하는지 여부. 구현체에서 True로 오버라이드합니다."""
        return False

    def generate(self, inputs: Dict[str, np.ndarray], max_new_tokens: int = 256,
                 stop_token_ids: Optional[List[int]] = None) -> GenerationResult:
        """자기회귀 텍스트 생성. 이 메서드를 지원하는 백엔드만 오버라이드합니다."""
        raise NotImplementedError("이 런타임은 generate()를 지원하지 않습니다.")
