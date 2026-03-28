import os
import numpy as np
from typing import Dict, List, Any
from .base import DataLoader
from src.core.model_spec import Model_Spec

class BertClassificationLoader(DataLoader):
    """
    BERT (SST-2 등) 자연어 텍스트 데이터를 벤치마크 엔진에 공급하는 로더.
    Zero-Latency 원칙에 따라, 전처리 연산을 수행하지 않고 디스크에 오프라인으로 구운
    Numpy 배열(Memory-mapped)을 O(1) 슬라이싱하여 반환.
    """
    def __init__(self, model_spec: Model_Spec, dataset_path: str, **kwargs):
        self.model_spec = model_spec
        self.dataset_path = dataset_path
        self.current_idx = 0

        id_path = os.path.join(dataset_path, "input_ids.npy")
        mask_path = os.path.join(dataset_path, "attention_mask.npy")
        label_path = os.path.join(dataset_path, "labels.npy")

        for path in [id_path, mask_path, label_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"[Error] 필수 배열 파일 누락: {path}. tokenize_to_numpy.py를 먼저 실행하세요.")

        # O(1) 로딩 (mmap_mode='r'):
        # 디스크의 거대한 Numpy 배열을 실제 RAM에 올리지 않고, C언어 포인터처럼 가상 주소 체계로 연결.
        # 인덱스 슬라이싱이 발생할 때만 운영체제가 해당 페이지만 디스크에서 RAM으로 캐싱.
        self.input_ids = np.load(id_path, mmap_mode='r')
        self.attention_mask = np.load(mask_path, mmap_mode='r')
        self.labels = np.load(label_path, mmap_mode='r')

        self.total_samples = len(self.labels)

    def _build_payload(self, id_array: np.ndarray, mask_array: np.ndarray, label_scalar: Any) -> Dict[str, Any]:
        """
        중복 로직 제거용 헬퍼 메서드: 
        다중 입력 텐서(Multi-input)를 지원하기 위한 딕셔너리 포장 작업을 캡슐화.
        """
        return {
            "input": {
                "input_ids": id_array,
                "attention_mask": mask_array
            },
            "label": label_scalar
        }

    def load_single(self) -> Dict[str, Any]:
        """단일 배치 처리를 위해 1개의 샘플을 반환."""
        if self.current_idx >= self.total_samples:
            raise StopIteration("모든 샘플이 소진되었습니다.")
            
        sample = self._build_payload(
            self.input_ids[self.current_idx],
            self.attention_mask[self.current_idx],
            self.labels[self.current_idx]
        )
        self.current_idx += 1
        return sample

    def load_batch(self, batch_size: int) -> Any:
        """
        주어진 사이즈만큼 데이터를 Slicing하여 반환.
        파이썬 리스트 포장 병목을 우회하여 텐서 차원 덩어리(Batch)를 통째로 반환.
        추후 BenchmarkRunner가 메타데이터를 감지하고 _collate_batch를 건너뜀.
        """
        if self.current_idx >= self.total_samples:
            return {}
            
        end_idx = min(self.current_idx + batch_size, self.total_samples)
        
        # 슬라이싱(Slicing) 연산 (O(1))
        # mmap_mode 상태에서는 디스크에서 정확히 요청된 데이터 덩어리만 캐싱.
        batch_ids = self.input_ids[self.current_idx:end_idx]
        batch_masks = self.attention_mask[self.current_idx:end_idx]
        batch_labels = self.labels[self.current_idx:end_idx]
        
        # 낱개로 나누지 않고, 통짜 배열 덩어리를 헬퍼 메서드에 그대로 전달
        batch_samples = self._build_payload(batch_ids, batch_masks, batch_labels)
            
        self.current_idx = end_idx
        return batch_samples

    def get_labels(self) -> Any:
        return self.labels

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "dataset_path": self.dataset_path,
            "seq_len": self.input_ids.shape[1],
            "is_static_batched": True  # 벤치마크 엔진에 Fast-Path (collate 우회) 배정을 요청하는 식별표
        }

    def preprocess(self, raw_input: Any) -> np.ndarray:
        # AOT 로드 방식이므로 런타임 전처리는 절대 수행하지 않음 (O(1) 원칙 고수)
        return raw_input
