# Runtimes Package

`runtimes` 패키지는 ONNX Runtime, IREE, TensorRT 등 기술 스택과 가속 방식이 서로 다른 다양한 **하드웨어 추론 엔진**들을 클라이언트(BenchmarkRunner) 입장에서 투명하게 제어할 수 있도록 캡슐화한 모듈 모음입니다.

## Architecture

- **`base.py` (`Runtime`)**: 모든 외부 런타임 라이브러리 래퍼(Wrapper)가 반드시 준수해야 하는 추상 인터페이스(Abstract Base Class)입니다.
  - `load()`: 컴파일 아티팩트(ONNX, VMFB 등)를 타겟 메모리에 로드
  - `warmup()`: 추론 초기의 성능 측정 왜곡(JIT 보정 등)을 방지하기 위한 예열 기능
  - `run()`: 실제 배치 단위 데이터의 추론을 단일 인터페이스로 수행하고 결괏값을 Numpy Dictionary로 반환
  - `unload()`: 메모리 해제

- **`__init__.py` (Facade & Factory)**
  - 외부 시스템(`main.py` 등)에서 내부 패키지 복잡도를 완전히 무시할 수 있도록 지원하는 **게이트웨이**입니다.
  - 사용자는 그저 `create_runtime("onnx")` 함수명 하나만 사용하여 적절한 런타임 인터페이스 객체를 생성받을 수 있습니다.

## How to add a new Runtime
추후 TensorRT나 TFLite 백엔드를 확장하고 싶다면 아래 절차를 따릅니다.
1. `base.py`의 `Runtime` 클래스를 상속받는 구체 클래스(예: `TensorRTRuntime`)를 내부 파일로 작성합니다.
2. `__init__.py` 안의 `create_runtime()` 팩토리 분기(if-else) 조건문에 새 구체 클래스를 등록(매핑)합니다.
3. 코어 애플리케이션 코드는 일체 수정할 필요가 없습니다!
