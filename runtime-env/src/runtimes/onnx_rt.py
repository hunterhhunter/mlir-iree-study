import ctypes
import glob
import importlib
import os
import site
import numpy as np
from typing import Dict, Any

# nvidia-*-cu12 pip 패키지로 설치된 CUDA 라이브러리를 사전 로드합니다.
# onnxruntime-gpu는 libcublasLt.so.12 등을 dlopen으로 찾는데, 이 라이브러리들이
# Python site-packages 안에 있어 ld.so 기본 탐색 경로에 없는 경우 CUDAExecutionProvider가
# 비활성화됩니다. 이 코드는 onnxruntime import 전에 실행해야 합니다.
def _preload_cuda_libs() -> None:
    _LIBS = [
        ("nvidia.cuda_runtime.lib", "libcudart.so.12"),
        ("nvidia.cublas.lib",       "libcublas.so.12"),
        ("nvidia.cublas.lib",       "libcublasLt.so.12"),
        ("nvidia.cufft.lib",        "libcufft.so.11"),
        ("nvidia.curand.lib",       "libcurand.so.10"),
        ("nvidia.cusolver.lib",     "libcusolver.so.11"),
        ("nvidia.cusparse.lib",     "libcusparse.so.12"),
        ("nvidia.nvjitlink.lib",    "libnvJitLink.so.12"),
    ]
    for mod_name, lib_name in _LIBS:
        try:
            mod = importlib.import_module(mod_name)
            lib_dir = os.path.dirname(mod.__file__) if mod.__file__ else None
            if lib_dir:
                lib_path = os.path.join(lib_dir, lib_name)
                if os.path.exists(lib_path):
                    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        except Exception:
            pass
    # cudnn은 __file__이 None인 경우가 있어 glob으로 탐색
    for sp in site.getsitepackages():
        for lib_path in glob.glob(os.path.join(sp, "nvidia", "cudnn", "lib", "libcudnn.so.9")):
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass

_preload_cuda_libs()

import onnxruntime as ort

from .base import Runtime
from ..core.compiled_model import CompiledModel

class OnnxRuntime(Runtime):
    """
    ONNX Runtime 기반의 실행 엔진 래퍼.
    """
    def __init__(self, **runtime_options):
        """
        [1. Hardware Provisioning & Context Initialization]
        """
        # 실행 디바이스 환경 변수 받기 (기본값 cpu)
        self.device = runtime_options.get("device", "cpu")
        
        # ONNX Runtime의 Execution Provider 설정
        _SUPPORTED_DEVICES = {"cpu", "cuda"}
        if self.device not in _SUPPORTED_DEVICES:
            raise ValueError(
                f"지원하지 않는 device입니다: '{self.device}'. "
                f"지원 목록: {sorted(_SUPPORTED_DEVICES)}"
            )

        if self.device == "cuda":
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    "CUDAExecutionProvider를 사용할 수 없습니다. "
                    "onnxruntime-gpu 설치 여부와 CUDA 환경을 확인하세요. "
                    f"현재 가용 Provider: {available}"
                )
            self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            self.providers = ['CPUExecutionProvider']
            
        # 런타임의 상태 변수들 초기화
        self.session = None
        self.input_names = []
        self.output_names = []
        self.compiled_model = None

    def load(self, compiled_model: CompiledModel) -> None:
        """
        [2. Artifact Deserialization & Memory Mapping]
        .onnx 파일을 computation graph로 로드.
        """
        if not self.is_compatible(compiled_model):
            raise ValueError(f"Incompatible backend: {compiled_model.backend_name}")
            
        self.compiled_model = compiled_model
        
        # ONNX Runtime은 내부적으로 mmap 최적화 및 직렬화 해제를 자체 지원.
        self.session = ort.InferenceSession(
            str(self.compiled_model.artifact_path), 
            providers=self.providers
        )
        
        # 모델이 요구하는 입출력 텐서의 이름표를 추출.
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        [4. Kernel Dispatch & Forward Pass (Inference)]
        순수 Numpy 배열을 던져주고 결과도 Numpy로 추출.
        """
        if self.session is None:
            raise RuntimeError("ONNX Runtime session is not loaded. Call load() first.")
            
        # 입력된 텐서들 중 모델이 실제로 필요로 하는 이름표만 매핑해서 넣음
        ort_inputs = {name: inputs[name] for name in self.input_names if name in inputs}
        
        # 실제 하드웨어 커널에 연산 지시
        results = self.session.run(self.output_names, ort_inputs)
        
        # 결과 리스트를 이름표와 함께 묶어 Dict 타입의 Numpy로 반환
        return {out_name: np.array(res) for out_name, res in zip(self.output_names, results)}

    def warmup(self, inputs: Dict[str, np.ndarray], num_runs: int = 1) -> None:
        """
        [3. JIT Triggering & Cache Warming]
        실제 측정 전, Cold-start 지연 시간을 제거.
        """
        print(f"[ONNX Runtime] Warming up {num_runs} times on {self.device}...")
        for _ in range(num_runs):
            self.run(inputs)

    def unload(self) -> None:
        """
        [5. Resource Deallocation & Teardown]
        메모리 누수 및 다른 모델 테스트 시 발생할 수 있는 VRAM OOM 에러를 방지.
        """
        self.session = None
        self.input_names = []
        self.output_names = []
        self.compiled_model = None

    def get_device_spec(self) -> Dict[str, Any]:
        """현재 런타임이 구동 중인 하드웨어 명세를 반환."""
        return {
            "backend": "onnxruntime", 
            "device": self.device, 
            "active_providers": self.providers
        }

    def is_compatible(self, compiled_model: CompiledModel) -> bool:
        """이 런타임이 실행할 수 있는 '.onnx' 확장자 모델이 맞는지 검사함."""
        backend_match = compiled_model.backend_name.startswith("onnx")
        extension_match = str(compiled_model.artifact_path).endswith(".onnx")
        return backend_match or extension_match
