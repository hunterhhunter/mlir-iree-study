import os
import sys
# pytest 모듈 인식 오류(ModuleNotFoundError: No module named 'src') 방지용 강제 경로 주입
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from unittest.mock import MagicMock

from src.dataloader.bert_classification_loader import BertClassificationLoader
from src.core.model_spec import Model_Spec

@pytest.fixture
def dummy_bert_spec():
    """Model_Spec 인터페이스에 맞춘 Mock 스펙 반환"""
    spec = MagicMock()
    spec.input_shapes = {
        "input_ids": (1, 128),
        "attention_mask": (1, 128)
    }
    return spec

def create_mock_dataset(tmp_path, total_samples: int = 100, seq_len: int = 128) -> str:
    """테스트용 Mock 데이터셋 생성 및 임시 경로 반환"""
    input_ids = np.ones((total_samples, seq_len), dtype=np.int64)
    attention_mask = np.ones((total_samples, seq_len), dtype=np.int64)
    labels = np.zeros((total_samples,), dtype=np.int64)
    
    np.save(tmp_path / "input_ids.npy", input_ids)
    np.save(tmp_path / "attention_mask.npy", attention_mask)
    np.save(tmp_path / "labels.npy", labels)
    
    return str(tmp_path)


def test_bert_loader_metadata(dummy_bert_spec, tmp_path):
    """메타데이터 생성 로직 및 통짜 배치(Fast-Path) 프로토콜 검증"""
    dataset_dir = create_mock_dataset(tmp_path, total_samples=50)
    loader = BertClassificationLoader(model_spec=dummy_bert_spec, dataset_path=dataset_dir)
    
    meta = loader.get_metadata()
    assert meta.get("is_static_batched") is True
    assert meta["total_samples"] == 50
    assert meta["seq_len"] == 128


@pytest.mark.parametrize("total_samples, batch_size, expected_batch_counts, expected_last_batch_size", [
    (100, 32, 4, 4),    # 32, 32, 32, 4 (Partial Batch 발생)
    (128, 32, 4, 32),   # 32, 32, 32, 32 (정확한 배수)
    (100, 150, 1, 100), # 배치 크기가 전체 데이터보다 클 경우
    (5, 32, 1, 5),      # 샘플 개수가 배치 사이즈보다 적을 경우
])
def test_bert_loader_batch_slicing_boundaries(dummy_bert_spec, tmp_path, total_samples, batch_size, expected_batch_counts, expected_last_batch_size):
    """
    다양한 배치 사이즈와 데이터셋 크기 조건에서 load_batch()의 
    경계값(Boundary) 슬라이싱이 정상적으로 수행되는지 검증 (OOM 및 IndexError 방어)
    """
    dataset_dir = create_mock_dataset(tmp_path, total_samples)
    loader = BertClassificationLoader(model_spec=dummy_bert_spec, dataset_path=dataset_dir)
    
    batches = []
    while True:
        batch = loader.load_batch(batch_size)
        if not batch: # 빈 Dictionary 반환 시 종료
            break
        batches.append(batch)
        
    # 총 생성된 배치 개수 검증
    assert len(batches) == expected_batch_counts
    
    # 마지막 배치의 크기가 예상된 잔여(Partial) 샘플 개수와 일치하는지 검증
    last_batch = batches[-1]
    
    # 리스트 파이프라인인지 딕셔너리(Fast-Path) 파이프라인인지에 따라 동적 타입 체크
    if isinstance(last_batch, list):
        assert len(last_batch) == expected_last_batch_size
    else:
        assert last_batch["input"]["input_ids"].shape == (expected_last_batch_size, 128)
        assert last_batch["input"]["attention_mask"].shape == (expected_last_batch_size, 128)
        assert last_batch["label"].shape == (expected_last_batch_size,)


def test_bert_loader_load_single_raises_stop_iteration(dummy_bert_spec, tmp_path):
    """
    단일 샘플 조회 시 전체 데이터를 소진한 후 
    StopIteration 예외가 정상적으로 발생하는지 검증
    """
    total_samples = 10
    dataset_dir = create_mock_dataset(tmp_path, total_samples)
    loader = BertClassificationLoader(model_spec=dummy_bert_spec, dataset_path=dataset_dir)
    
    # 정상 순회 검증
    for _ in range(total_samples):
        sample = loader.load_single()
        assert sample["input"]["input_ids"].shape == (128,)
        
    # 소진 후 예외 발생 검증
    with pytest.raises(StopIteration, match="모든 샘플이 소진되었습니다"):
        loader.load_single()
