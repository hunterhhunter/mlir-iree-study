# Evaluator 모듈

IREE 런타임에서 추론된 결과를 바탕으로 모델의 성능 지표(Accuracy, F1-Score 등)를 통합 산출하고 분석 리포트를 생성하는 모듈입니다.

## 디렉토리 구조

- `base_evaluator.py`: 모든 평가 모듈의 규격을 정의하는 추상 기반 클래스.
- `classification.py`: 이미지 분류 태스크 전용 지표(Top-K, Precision/Recall/F1) 산출 구현체.
- `__init__.py`: 팩토리 인터페이스(`get_evaluator`)를 포함한 패키지 진입점.

## API 명세 (API Specification)

### get_evaluator

수행할 태스크에 적합한 Evaluator 인스턴스를 반환하는 팩토리 함수입니다.

**매개변수:**
- `task` (str): 평가할 태스크 유형 (예: 'classification'). 기본값은 'classification'입니다.
- `**kwargs`: 해당 Evaluator 생성자에 전달할 추가 인자 (예: `top_k=(1, 5)`).

**반환값:**
- `BaseEvaluator`: 지정된 태스크에 최적화된 평가 객체.

### BaseEvaluator (인터페이스)

모든 평가 클래스가 구현해야 할 표준 메서드입니다.

- `update(predictions, targets)`: 배치 단위 추론 결과와 정답을 입력받아 상태를 업데이트합니다.
- `compute()`: 누적된 데이터를 바탕으로 최종 지표(딕셔너리 형태)를 산출합니다.
- `report()`: 포맷팅된 텍스트 형태로 상세 평가 결과를 출력합니다.
- `reset()`: 누적된 데이터를 초기화합니다.

## 사용 예시

```python
from src.evaluator import get_evaluator

# 에밸루에이터 초기화
evaluator = get_evaluator(task="classification", top_k=(1, 5))

# 루프 내에서 지표 업데이트
for logits, labels, _ in loader:
    evaluator.update(logits, labels)

# 최종 결과 산출 및 리포트 출력
results = evaluator.compute()
evaluator.report()
```

## 주요 산출 지표

- **Top-1 / Top-5 Accuracy**: 전체 샘플 대비 상위 K개 예측값 내 정답 포함 비율.
- **Precision / Recall / F1-Score**: 클래스 간 불균형을 고려한 Macro-average 지표.
