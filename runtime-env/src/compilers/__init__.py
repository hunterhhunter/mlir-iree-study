"""
Compilers Package Initialization & Factory

벤치마킹 프레임워크 타겟 모델을 다양한 디바이스(CPU/GPU 등) 네이티브 바이너리로 컴파일할 수 있도록,
다양한 백엔드 프레임워크(IREE, TVM) 컴파일러 인스턴스를 손쉽게 생성하는 단일 진입점을 제공합니다.
"""

from .base import Compiler
from .iree_compiler import IREECompiler

def get_compiler(compiler_name: str, **compile_options) -> Compiler:
    """
    Factory Method for Compiler
    
    이름(compiler_name)을 입력받아 해당 백엔드 구체 컴파일러(Concrete Compiler) 인스턴스를 반환합니다.
    
    Args:
        compiler_name (str): 사용할 AI 컴파일러 이름 (예: "iree", "tvm")
        **compile_options: target_backend (예: llvm-cpu, cuda), 최적화 레벨 등의 컴파일 인자
        
    Returns:
        Compiler: 추상 베이스 클래스를 상속받은 구체 컴파일러 인스턴스
        
    Raises:
        ValueError: 지원하지 않는 컴파일러 이름이 들어명 예외 발생
    """
    compiler_name = compiler_name.strip().lower()
    
    if compiler_name == "iree":
        return IREECompiler(**compile_options)
        
    # 추후 TVM, TensorRT 컴파일러가 추가되면 elif 로 분기 확장
    # elif compiler_name == "tvm":
    #     return TVMCompiler(**compile_options)
        
    else:
        raise ValueError(f"현재 '{compiler_name}' 컴파일러 백엔드는 지원되지 않습니다. 지원 목록: ['iree']")

__all__ = [
    "Compiler",
    "IREECompiler",
    "get_compiler"
]
