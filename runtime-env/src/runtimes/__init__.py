"""
Runtime Package Initialization & Factory

이 모듈은 벤치마크 프레임워크의 다른 컴포넌트에게
다양한 Runtime 엔진(ONNX, IREE 등)에 대한 손쉬운 접근(단일 진입점 API)을 제공합니다.
"""

from .base import Runtime, GenerationResult
from .onnx_rt import OnnxRuntime
from .iree_rt import IREERuntime
from .vllm_rt import VllmRuntime

def create_runtime(backend_name: str, device: str = "cpu", **kwargs) -> Runtime:
    """
    Factory Method for Runtime
    
    주어진 백엔드 이름(예: 'onnxruntime', 'iree')에 맞는 
    구체 Runtime 객체를 초기화하여 반환합니다.
    
    Args:
        backend_name (str): 실행할 백엔드 엔진 이름
        device (str): 실행 디바이스 (기본값: "cpu")
        **kwargs: 기타 런타임 초기화에 필요한 추가 인자
        
    Returns:
        Runtime: 추상 베이스 클래스를 상속받은 구체 런타임 인스턴스
    """
    backend = backend_name.lower()
    
    if backend in ["onnx", "onnxruntime"]:
        return OnnxRuntime(device=device, **kwargs)
    elif backend in ["iree", "mlir"]:
        # 참고: 현재 iree_rt.py의 IREERuntime은 구형 스크립트로 동작하므로
        # Base Runtime 인터페이스 호환을 위한 리팩토링이 선행되어야 완벽히 동작합니다.
        # return IREERuntime(device=device, **kwargs)
        raise NotImplementedError("IREE 런타임은 현재 공통 인터페이스 맞춤 리팩토링 중입니다.")
    elif backend in ["vllm"]:
        return VllmRuntime(device=device, **kwargs)
    else:
        raise ValueError(f"지원하지 않는 백엔드입니다: {backend_name}")

__all__ = [
    "Runtime",
    "GenerationResult",
    "OnnxRuntime",
    "IREERuntime",
    "VllmRuntime",
    "create_runtime"
]
