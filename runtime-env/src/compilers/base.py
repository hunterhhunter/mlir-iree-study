from abc import ABC, abstractmethod
from typing import Any, Dict
import os

# ..core.model_spec 가 존재한다고 가정
from ..core.model_spec import Model_Spec
from ..core.compiled_model import CompiledModel

class Compiler(ABC):
    """
    벤치마킹 프레임워크의 컴파일러 추상 계층 클래스.
    
    다양한 딥러닝 컴파일러(IREE, TVM, TensorRT 등)를 동일한 인터페이스로
    포장하여 사용할 수 있도록 보장하는 역할을 합니다.
    오로지 소스 모델(.onnx 등)을 컴파일러 프레임워크의 네이티브 바이너리(.vmfb 등)로
    낮추는(Lowering) 단계까지만 책임을 집니다.
    """
    
    @abstractmethod
    def __init__(self, **compile_options):
        """
        컴파일러 인스턴스 초기화.
        
        Args:
            **compile_options: target_backend(cpu/cuda), 최적화 레벨 등 
                               프레임워크별로 요구되는 고유 옵션들
        """
        self.compile_options = compile_options
        pass
        
    @abstractmethod
    def compile(self, model_spec: Model_Spec, output_dir: str) -> CompiledModel:
        """
        주어진 모델 스펙을 바탕으로 실제 컴파일을 수행합니다.
        
        Args:
            model_spec (Model_Spec): 컴파일할 대상 모델의 정보 집합체
            output_dir (str): 컴파일된 바이너리가 저장될 대상 폴더 경로
            
        Returns:
            CompiledModel: 성공적으로 컴파일된 결과 바이너리 정보가 담긴 DTO 객체
        """
        pass
        
    @abstractmethod
    def get_artifact_name(self, model_spec: Model_Spec) -> str:
        """
        해당 모델 스펙과 현재 컴파일 옵션이 주어졌을 때,
        산출될 예상 바이너리 파일명(확장자 포함)을 반환합니다.
        (예: 'resnet50_cuda_sm86.vmfb')
        
        Args:
            model_spec (Model_Spec): 대상 모델 스펙
            
        Returns:
            str: 예상되는 결과 파일명
        """
        pass
        
    def is_cached(self, model_spec: Model_Spec, output_dir: str) -> bool:
        """
        이미 동일한 조건(옵션)으로 컴파일이 수행되어 결과물이 존재하는지 검사합니다.
        재도입 시 중복 컴파일로 인한 시간 낭비를 막는 데 쓰일 수 있습니다.
        
        Args:
            model_spec (Model_Spec): 컴파일 대상 모델 스펙
            output_dir (str): 바이너리 파일이 위치한 경로
            
        Returns:
            bool: 캐시 히트(이미 존재함) 여부
        """
        expected_file = os.path.join(output_dir, self.get_artifact_name(model_spec))
        return os.path.exists(expected_file)

    def get_compile_config(self) -> Dict[str, Any]:
        """
        현재 컴파일러의 세팅(버전, 옵션, backend 타겟 등)을 반환합니다.
        벤치마킹 리포트에 남길 메타데이터로 활용됩니다.
        """
        return {
            "compile_options": self.compile_options
        }
