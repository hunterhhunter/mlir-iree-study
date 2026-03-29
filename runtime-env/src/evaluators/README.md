# Evaluators Package

`evaluators` 패키지는 Runtime 엔진이 배출한 추론 결과물 로짓(Logits)을 정답 스펙(Ground Truth)과 매칭하여 **정확도(Accuracy)** 와 **속도(Latency)** 등 종합 통계(Metrics)를 도출하고 채점하는 독립적인 모듈 그룹입니다.

## Architecture

- **`base.py` (`Evaluator`)**: 모든 채점 모듈이 반드시 상속받고 구현을 보장해야 하는 추상 인터페이스.
  - `evaluate(result: InferenceResult)`: 실행 결과 DTO 객체를 해부하여 딕셔너리 형태의 종합 통계를 반환.
  - `is_applicable(device_spec, model_spec)`: 현재 평가기가 투입된 모델 테스크(Task)에 대응 가능한 로직인지 여부 판단.

- **`image_classification_evaluator.py`**:
  - ImageNet-1K 규격의 모델에서 Top-1, Top-5 Accuracy, Precision(Macro), Recall, F1-Score 연산을 담당하는 단일 책임 구체(Concrete) 클래스.
  - (구글 MobileNet 1001 차원 출력 편향 보정 등의 각종 예외 처리를 보유합니다)

- **`__init__.py` (Facade & Factory)**
  - 입력받은 `Model_Spec` 내부의 `Task` 필드(`IMAGE_CLASSIFICATION` 등)만을 분석하여 상황에 딱 알맞은 평가기(Evaluator) 인스턴스를 동적으로 찍어내고 반환하는 `create_evaluator()` 팩토리(Factory)를 제공합니다.

## 확장 방법 (NLP, Object Detection 지원 시)
추후 자연어나 욜로(YOLO) 객체 탐지 태스크를 지원해야 할 때:
1. `Evaluator`를 상속하는 새 채점기 파일(`object_detection_evaluator.py`)을 생성 및 구현.
2. `__init__.py` 안의 `create_evaluator()` 팩토리 함수에 `if task == Task.OBJECT_DETECTION:` 조건을 추가하여 바인딩.
