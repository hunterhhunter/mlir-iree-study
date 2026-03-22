from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np

@dataclass(frozen=True)
class InferenceResult:
    outputs: Dict[str, np.ndarray]  # Runtime이 반환한 프레임워크 독립적인 순수 Numpy 예측값 딕셔너리
    timing_records: List[float]     # 모델 실행에 소요된 지연 시간(Latency) 목록 (단위: ms 등)
    labels: Any                     # DataLoader가 제공한 정답지 

    def __post_init__(self):
        # 1. 런타임이 넣은 outputs 검증: 모조리 Numpy 배열이어야 함
        for key, value in self.outputs.items():
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"[Framework Leakage] Runtime output '{key}' must be a pure numpy.ndarray, "
                    f"but got {type(value).__name__}!"
                )
                
        # 2. 데이터로더가 넣은 labels 검증: 이름에 'Tensor'나 'DeviceArray'가 포함되었는지 확인
        label_type_name = type(self.labels).__name__
        if "Tensor" in label_type_name or "DeviceArray" in label_type_name:
            raise TypeError(
                f"[Framework Leakage] DataLoader output 'labels' cannot be a framework-specific tensor "
                f"({label_type_name}). It must be converted to numpy.ndarray or native Python types!"
            )
