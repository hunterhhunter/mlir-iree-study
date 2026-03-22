from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np

@dataclass(frozen=True)
class InferenceResult:
    outputs: Dict[str, np.ndarray]  # Runtime이 반환한 프레임워크 독립적인 순수 Numpy 예측값 딕셔너리
    timing_records: List[float]     # 모델 실행에 소요된 지연 시간 목록 (단위: ms 등)
    labels: Any                     # DataLoader가 제공한 정답지 (Ground Truth)
