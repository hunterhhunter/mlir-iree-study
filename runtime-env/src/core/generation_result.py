import dataclasses
import numpy as np

@dataclasses.dataclass(frozen=True)
class GenerationResult:
    """자기회귀 생성 결과 컨테이너."""
    generated_ids: np.ndarray  # shape (num_tokens,), dtype int64
    ttft_ms: float             # Time To First Token (근사값)
    tpot_ms: float             # Time Per Output Token (근사값)
    total_ms: float            # 전체 생성 시간
    num_tokens: int            # 생성된 토큰 수
