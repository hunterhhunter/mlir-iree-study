# Benchmark Framework - DataLoader

이 모듈은 벤치마킹 파이프라인에서 Evaluator(평가기)나 런타임(Runtime) 환경에
모델별/Task별로 다양한 형식의 원시(Raw) 데이터를 통일된 인터페이스로 가공해 공급하는 중추 역할을 합니다.

## 🚀 외부에서 API 호출하는 방법

다른 컴포넌트(Evaluator 등) 개발자는 개별 로더(`ImageClassificationLoader` 등)를 직접 가져올 필요 없이,
정리된 **팩토리 함수 단 하나**만 호출하여 로더 인스턴스를 획득할 수 있습니다.

```python
import sys
from src.core.model_spec import Model_Spec, Task
from src.dataloader import create_dataloader

# 1. 실행할 대상 모델의 핵심 스펙 정의
spec = Model_Spec(
    name="resnet50",
    task=Task.IMAGE_CLASSIFICATION,
    input_shapes={"input": (1, 3, 224, 224)},
    input_dtype={"input": "float32"},
    output_shapes={"output": (1, 1000)}
)

# 2. 팩토리 호출
# -> model_spec 의 "task" 속성을 읽어 알아서 적합한 로더를 선택해 줍니다.
loader = create_dataloader(spec, dataset_path="/path/to/dataset")

# 3. 데이터 로딩 파이프라인 실행
try:
    # 배치 사이즈 32씩 가져오기 모듈레이션
    batch_samples = loader.load_batch(32)
    for sample in batch_samples:
        tensor = sample["input"]   # np.ndarray 형태
        label = sample["label"]    # 정답 라벨
        # ... Run evaluation ...
except StopIteration:
    pass
```

---

## 🛠 새로운 Task의 로더(Loader)를 생성하는 방법

추후 영상 처리가 아닌 `NLP`나 `Object Detection` 처럼 데이터셋 폴더 구조와 전처리 로직이 완전히 다른 Task를 지원해야 한다면 다음 순서에 따라 확장이 가능합니다.

### Step 1. 구체 로더 클래스 작성
`src/dataloader/` 내부 어딘가(예: `nlp_loader.py`)에 새로운 스크립트를 파고,
`base.py`에 정의된 `DataLoader` 추상 클래스를 상속받습니다.

```python
# src/dataloader/nlp_loader.py
from .base import DataLoader
from typing import Dict, Any, List
import numpy as np

class NLPLoader(DataLoader):
    def __init__(self, model_spec, **kwargs):
        super().__init__(model_spec, **kwargs)
        # 텍스트 코퍼스 파일 등 로컬 경로 세팅
        pass
        
    def load_single(self) -> Dict[str, Any]:
        # 한 줄(문장)을 읽어 반환
        pass

    def load_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        pass

    def preprocess(self, raw_input: Any) -> np.ndarray:
        # 토크나이저를 돌려서 Tensor화 시키는 로직
        pass
        
    def get_labels(self): pass
    def get_metadata(self): pass
```

### Step 2. 팩토리 (Factory API) 경로 개척하기
새로 만든 클래스를 팩토리에서 반환할 수 있도록 `src/dataloader/__init__.py` 파일을 수정합니다.

```python
# src/dataloader/__init__.py 수정
from .nlp_loader import NLPLoader

def create_dataloader(model_spec: Model_Spec, **kwargs) -> DataLoader:
    task = model_spec.task
    
    if task == Task.IMAGE_CLASSIFICATION:
        return ImageClassificationLoader(model_spec, **kwargs)
        
    # !!! 여기에 새로운 분기 추가 !!!
    elif task == Task.NLP_CLASSIFICATION:
        return NLPLoader(model_spec, **kwargs)
        
    else:
        raise ValueError("지원하지 않는 Task입니다.")
```

이렇게 팩토리에 등록해 주기만 하면, 기존 벤치마킹 시스템 전체가 변경 없이 새로운 NLP 로더 파이프라인을 그대로 사용할 수 있게 됩니다!
