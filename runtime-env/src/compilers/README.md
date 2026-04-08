# Benchmark Framework - Compilers

이 모듈은 벤치마킹될 머신러닝 스펙(`Model_Spec`)과 원본 소스 파일(.onnx 등)을 입력받아,
타겟 하드웨어 디바이스(CPU, CUDA 등)에 최적화된 **네이티브 실행 바이너리(예: `.vmfb`)로 최종 컴파일**해 주는 독립된 역할을 수행합니다.

## 🚀 외부에서 API 호출하는 방법

다른 컴포넌트(Evaluator 등) 개발자는 개별 컴파일러 로직을 고민할 필요 없이, 지정된 **팩토리 함수** 호출 한 번으로 복잡한 파이프라인(Opset 변환, MLIR 추출 등)이 캡슐화된 인스턴스를 얻고 구동할 수 있습니다.

```python
import sys
from src.core.model_spec import Model_Spec, Task
from src.compilers import get_compiler

# 1. 실행할 대상 모델의 핵심 스펙 정의
spec = Model_Spec(
    name="resnet50",
    model_paths={"onnx": "/path/to/resnet50.onnx"},
    task=Task.IMAGE_CLASSIFICATION,
    input_shapes={"input": (1, 3, 224, 224)},
    input_dtype={"input": "float32"},
    output_shapes={"output": (1, 1000)}
)

# 2. 팩토리 호출: 원하는 백엔드 컴파일러와 최적화 옵션을 주입
# (예: IREE 컴파일러, LLVM-CPU 타겟)
compiler = get_compiler(
    compiler_name="iree", 
    target_backend="llvm-cpu"  # 필요하면 "cuda" 등
)

# 3. 컴파일 수행 (VMFB 바이너리 생성)
# 캐시에 이미 존재하면 중복 수행 없이 파일 경로만 반환합니다.
try:
    binary_path = compiler.compile(spec, output_dir="/path/to/save/dir")
    print(f"[*] 성공적으로 조립된 바이너리 경로: {binary_path}")    
except Exception as e:
    print(f"[!] 컴파일 도중 에러가 발생했습니다: {e}")
```

---

## 🛠 새로운 타겟 컴파일러(Compiler)를 생성하는 방법

추후 IREE가 아닌 `TVM` 이나 `TensorRT`, `LLVM` 등 전혀 다른 파이프라인과 중간 언어를 거치는 AI 컴파일러를 새롭게 벤치마킹해야 한다면, 아래 순서에 따라 프레임워크를 유연하게 확장할 수 있습니다.

### Step 1. 구체 컴파일러 클래스 작성
`src/compilers/` 내부에 새로운 스크립트(예: `tvm_compiler.py`)를 파고, `base.py`에 정의된 `Compiler` 추상 클래스를 상속받습니다.

```python
# src/compilers/tvm_compiler.py
from .base import Compiler
from ..core.model_spec import Model_Spec
import os

class TVMCompiler(Compiler):
    def __init__(self, **compile_options):
        super().__init__(**compile_options)
        self.target = self.compile_options.get("target", "llvm")
        
    def get_artifact_name(self, model_spec: Model_Spec) -> str:
        # 파일명 규칙을 자유롭게 프레임워크 특징에 맞게 정합니다.
        return f"{model_spec.name}_tvm_{self.target}.so"

    def compile(self, model_spec: Model_Spec, output_dir: str) -> str:
        # 0. 중복 캐시 점검 (권장 방어 코드)
        if self.is_cached(model_spec, output_dir):
            return os.path.join(output_dir, self.get_artifact_name(model_spec))
            
        # 1. 모델 읽어오기
        onnx_path = model_spec.model_paths.get("onnx")
        
        # 2. Relay 변환 및 TVM Target Build 시퀀스 등 무거운 작업 수행
        print("Compiling via Apache TVM...")
        # ... tvm.relay.build(...) 로직 ...
        
        # 3. .so 바이너리를 output_dir에 저장하고 최종 절대 경로를 반환합니다.
        final_path = os.path.join(output_dir, self.get_artifact_name(model_spec))
        return final_path
```

### Step 2. 팩토리 (Factory API) 경로 개척하기
새로 만든 클래스를 팩토리에서 알아서 조립해 반환할 수 있도록 `src/compilers/__init__.py` 파일을 수정합니다.

```python
# src/compilers/__init__.py 수정
from .tvm_compiler import TVMCompiler

def get_compiler(compiler_name: str, **compile_options) -> Compiler:
    compiler_name = compiler_name.strip().lower()
    
    if compiler_name == "iree":
        return IREECompiler(**compile_options)
        
    # !!! 여기에 새로운 분기 추가 !!!
    elif compiler_name == "tvm":
        return TVMCompiler(**compile_options)
        
    else:
        raise ValueError("현재 지원하지 않는 컴파일러 백엔드입니다.")
```

단 두 단계의 작업만 마치면, 앞으로 프레임워크를 이용하는 사람은 `get_compiler("tvm")` 이란 단어 하나만 바꿔 끼워 모든 엔진에서 똑같은 바이너리 변환 혜택(캐싱 및 스펙 바인딩)을 받을 수 있게 됩니다!
