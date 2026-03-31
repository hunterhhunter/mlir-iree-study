import os
import numpy as np
from typing import Dict, Any

from .base import DataLoader
from src.core.model_spec import Model_Spec

class BertQALoader(DataLoader):
    """
    BERT SQuAD 질의응답(Question Answering) 자연어 텍스트 데이터를 벤치마크 엔진에 공급하는 로더.
    Zero-Latency 원칙에 따라, 전처리 연산을 수행하지 않고 디스크에 오프라인으로 베이킹(Baked)된
    Numpy 배열(Memory-mapped)을 O(1) 시간복잡도로 슬라이싱하여 반환합니다.
    """
    def __init__(self, model_spec: Model_Spec, dataset_path: str, **kwargs):
        self.model_spec = model_spec
        self.dataset_path = dataset_path
        self.current_idx = 0

        # SQuAD 오프라인 베이킹(prepare_squad_numpy.py)으로 구워진 4개의 파일 타겟팅
        self._require_files = {
            "id": os.path.join(dataset_path, "input_ids.npy"),
            "mask": os.path.join(dataset_path, "attention_mask.npy"),
            "start": os.path.join(dataset_path, "start_positions.npy"),
            "end": os.path.join(dataset_path, "end_positions.npy"),
        }

        for name, path in self._require_files.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"[Error] 필수 SQuAD 배열 파일 누락: {path}")

        # O(1) 로딩 (mmap_mode='r'):
        # 디스크의 거대한 Numpy 배열을 실제 RAM에 상주시키지 않고 가상 C포인터 체계로 캐싱.
        self.input_ids = np.load(self._require_files["id"], mmap_mode='r')
        self.attention_mask = np.load(self._require_files["mask"], mmap_mode='r')
        self.start_positions = np.load(self._require_files["start"], mmap_mode='r')
        self.end_positions = np.load(self._require_files["end"], mmap_mode='r')

        self.total_samples = len(self.start_positions)

    def _build_payload(self, id_array: np.ndarray, mask_array: np.ndarray, 
                       start_array: np.ndarray, end_array: np.ndarray) -> Dict[str, Any]:
        """
        [SRP 준수] 다중 입력 및 다중 정답지(Multi-label)를 규격화된 DTO 딕셔너리 페이로드로 포장하는 헬퍼 메서드.
        """
        return {
            "input": {
                "input_ids": id_array,
                "attention_mask": mask_array
            },
            "label": {
                "start_positions": start_array,
                "end_positions": end_array
            }
        }

    def load_single(self) -> Dict[str, Any]:
        """단일 배치 처리를 위해 1개의 샘플을 반환."""
        if self.current_idx >= self.total_samples:
            raise StopIteration("모든 샘플이 소진되었습니다.")
            
        sample = self._build_payload(
            self.input_ids[self.current_idx],
            self.attention_mask[self.current_idx],
            self.start_positions[self.current_idx],
            self.end_positions[self.current_idx]
        )
        self.current_idx += 1
        return sample

    def load_batch(self, batch_size: int) -> Dict[str, Any]:
        """
        [Fast-Path] 주어진 사이즈만큼 데이터를 O(1) Slicing하여 통짜 덩어리(Batch) 반환.
        리스트를 순회하지 않음으로써 BenchmarkRunner의 collate_batch 병목을 회피합니다.
        """
        if self.current_idx >= self.total_samples:
            return {}
            
        end_idx = min(self.current_idx + batch_size, self.total_samples)
        
        batch_samples = self._build_payload(
            self.input_ids[self.current_idx:end_idx],
            self.attention_mask[self.current_idx:end_idx],
            self.start_positions[self.current_idx:end_idx],
            self.end_positions[self.current_idx:end_idx]
        )
            
        self.current_idx = end_idx
        return batch_samples

    def get_labels(self) -> Dict[str, np.ndarray]:
        """
        평가기가 채점할 수 있도록 전체 정답 데이터 반환.
        """
        return {
            "start_positions": self.start_positions,
            "end_positions": self.end_positions
        }

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "dataset_path": self.dataset_path,
            "seq_len": self.input_ids.shape[1],
            "is_static_batched": True  # BenchmarkRunner에 Collation 무시 신호 전송
        }

    def preprocess(self, raw_input: Any) -> np.ndarray:
        # AOT 로드 방식이므로 런타임 전처리는 절대 수행하지 않음 (O(1) 원칙 고수)
        return raw_input

    def load_by_index(self, index: int) -> Dict[str, Any]:
        """
        [MLPerf QSL 전용] Worker 쓰레드 충돌을 막기 위한 상태 비저장(Stateless) 임의 인덱스 접근용 메서드.
        """
        if index < 0 or index >= self.total_samples:
            raise IndexError(f"요청 인덱스 {index} 범위 초과 (0 ~ {self.total_samples - 1})")
            
        return self._build_payload(
            self.input_ids[index],
            self.attention_mask[index],
            self.start_positions[index],
            self.end_positions[index]
        )
