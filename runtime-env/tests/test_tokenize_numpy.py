import os
import sys
import tempfile
import numpy as np
import pytest
from pathlib import Path

# datasets 패키지 모듈을 로드하기 위해 임시 경로 추가 (scripts/tokenize_to_numpy.py → datasets/prepare_text_numpy.py)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets")))
import prepare_text_numpy

def test_universal_tokenizer_with_mock_csv(tmp_path, monkeypatch):
    """
    네트워크(HuggingFace Hub) 트래픽을 유발하지 않는 완벽한 독립 테스트.
    임시 '.csv' 파일을 생성하여 전처리 시스템에 주입한 뒤,
    (N, SEQ_LEN) 크기의 3가지 Numpy 배열 블록들이 생성되는지 검증합니다.
    """
    
    # 1. 목업(Mock) CSV 데이터셋 생성
    mock_csv_path = tmp_path / "mock_reviews.csv"
    mock_csv_path.write_text(
        "review_text,sentiment\n"
        "I absolutely love this benchmark framework!,1\n"
        "Dynamic shape bug is quite disappointing system.,0\n",
        encoding="utf-8"
    )
    
    # 2. 전처리 텐서가 저장될 임시 디렉토리
    output_dir = tmp_path / "baked_numpy"
    
    # 3. argparse 파라미터 오버라이딩 (명령줄 입력 시뮬레이션)
    test_args = [
        "prepare_text_numpy.py",
        "--model-id", "bert-base-uncased",
        "--seq-len", "128",
        "--csv-file", str(mock_csv_path),
        "--text-column", "review_text",       
        "--label-column", "sentiment",        
        "--output-dir", str(output_dir)
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    
    # 4. 전처리 메인 파이프라인 실행
    prepare_text_numpy.main()
    
    # 5. [파일 생성 검증] 3종의 Numpy 배열이 디스크에 생성되었는가?
    id_path = output_dir / "input_ids.npy"
    mask_path = output_dir / "attention_mask.npy"
    label_path = output_dir / "labels.npy"
    
    assert id_path.exists(), "input_ids.npy 파일이 생성되지 않았습니다."
    assert mask_path.exists(), "attention_mask.npy 파일이 생성되지 않았습니다."
    assert label_path.exists(), "labels.npy 파일이 생성되지 않았습니다."
    
    # 6. [텐서 무결성 검증] 정적 패딩(Static Shape)이 강제되었는가?
    np_ids = np.load(id_path)
    np_mask = np.load(mask_path)
    np_labels = np.load(label_path)
    
    # 레코드는 2개, 지정한 시퀀스 길이(max_length)는 128로 설정됨
    assert np_ids.shape == (2, 128), f"예상 Shape: (2, 128), 실제: {np_ids.shape}"
    assert np_mask.shape == (2, 128), f"예상 Shape: (2, 128), 실제: {np_mask.shape}"
    assert np_labels.shape == (2,), f"예상 Shape: (2,), 실제: {np_labels.shape}"
    
    # 7. [논리 검증] 정답(Labels) 맵핑 무결성 검증
    assert np_labels[0] == 1, "첫 번째 샘플의 정답이 1로 매핑되지 않았습니다."
    assert np_labels[1] == 0, "두 번째 샘플의 정답이 0으로 매핑되지 않았습니다."
