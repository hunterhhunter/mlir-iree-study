import os
import tempfile
import numpy as np
import pytest
from unittest.mock import Mock

from src.dataloader.bert_qa_loader import BertQALoader
from src.core.model_spec import Model_Spec, Task

@pytest.fixture
def dummy_squad_data():
    """테스트를 위한 임시 SQuAD Numpy 데이터셋 환경 구축"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # 가상의 데이터 (N=10, seq_len=384)
        N = 10
        seq_len = 384
        input_ids = np.random.randint(0, 30000, size=(N, seq_len), dtype=np.int64)
        attention_mask = np.ones((N, seq_len), dtype=np.int64)
        start_positions = np.random.randint(0, 30, size=(N,), dtype=np.int64)
        end_positions = np.random.randint(30, 60, size=(N,), dtype=np.int64)
        
        np.save(os.path.join(temp_dir, "input_ids.npy"), input_ids)
        np.save(os.path.join(temp_dir, "attention_mask.npy"), attention_mask)
        np.save(os.path.join(temp_dir, "start_positions.npy"), start_positions)
        np.save(os.path.join(temp_dir, "end_positions.npy"), end_positions)
        
        yield temp_dir, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "start": start_positions,
            "end": end_positions
        }

def test_bert_qa_loader_dto_structure(dummy_squad_data):
    """
    [핵심 검증 1] load_single이 반환하는 페이로드가 정확히
    {"input": {...}, "label": {...}} DTO 형식을 띠고 있는지 확인
    """
    temp_dir, _ = dummy_squad_data
    mock_spec = Mock(spec=Model_Spec)
    loader = BertQALoader(model_spec=mock_spec, dataset_path=temp_dir)
    
    payload = loader.load_single()
    
    # 1. 1차 Depth 검증
    assert "input" in payload
    assert "label" in payload
    
    # 2. 2차 Depth (Input) 검증
    assert "input_ids" in payload["input"]
    assert "attention_mask" in payload["input"]
    assert isinstance(payload["input"]["input_ids"], np.ndarray) # Zero-Latency 원칙 (텐서 추출)
    
    # 3. 2차 Depth (Label) 검증
    assert "start_positions" in payload["label"]
    assert "end_positions" in payload["label"]

def test_bert_qa_loader_metadata(dummy_squad_data):
    """
    [핵심 검증 2] BenchmarkRunner에게 콜레이션(Collation) 병합을 무시하라고 
    지시하는 O(1) 플래그(is_static_batched)가 켜져 있는지 확인
    """
    temp_dir, _ = dummy_squad_data
    mock_spec = Mock(spec=Model_Spec)
    loader = BertQALoader(model_spec=mock_spec, dataset_path=temp_dir)
    
    meta = loader.get_metadata()
    assert meta["is_static_batched"] is True
    assert meta["total_samples"] == 10

def test_bert_qa_loader_load_batch_slicing(dummy_squad_data):
    """
    [핵심 검증 3] load_batch 호출 시 리스트를 랩핑하지 않고, 
    잘려진 거대 Numpy 덩어리 자체가 통짜로 반환되는지 확인 (Fast-path)
    """
    temp_dir, arrays = dummy_squad_data
    mock_spec = Mock(spec=Model_Spec)
    loader = BertQALoader(model_spec=mock_spec, dataset_path=temp_dir)
    
    batch_payload = loader.load_batch(batch_size=4)
    
    # 4개짜리 통짜 덩어리인지 확인
    assert batch_payload["input"]["input_ids"].shape == (4, 384)
    # 실제 값과 일치하는지 무결성 검증
    np.testing.assert_array_equal(batch_payload["input"]["input_ids"], arrays["input_ids"][:4])
    np.testing.assert_array_equal(batch_payload["label"]["start_positions"], arrays["start"][:4])
    np.testing.assert_array_equal(batch_payload["label"]["end_positions"], arrays["end"][:4])
