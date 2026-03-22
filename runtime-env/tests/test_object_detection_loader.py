import os
import sys
import numpy as np

# 프로젝트 루트 경로를 sys.path에 추가 (src 패키지 인식 용이)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.core.model_spec import Model_Spec, Task
from src.dataloader import create_dataloader

def main():
    print("="*60)
    print(" Object Detection DataLoader Test ")
    print("="*60)
    
    # 1. 테스트용 Model_Spec 생성 (예: YOLOv8 입력 사이즈 640x640)
    # 파일이 명시되지 않은 Dummy 스펙입니다.
    dummy_spec = Model_Spec(
        name="dummy_yolov8",
        task=Task.OBJECT_DETECTION,
        input_shapes={"images": (1, 3, 640, 640)},  # Object detection 스탠다드
        input_dtype={"images": "float32"},
        output_shapes={"output0": (1, 84, 8400)},
        model_paths={}
    )
    
    # 2. 데이터셋 로컬 경로 설정 (이전 단계에서 다운로드 받은 datasets/coco)
    dataset_path = os.path.join(project_root, "datasets", "coco")
    
    if not os.path.exists(dataset_path):
        print(f"[!] COCO 데이터셋 폴더를 찾을 수 없습니다: {dataset_path}")
        print("[!] 먼저 datasets/load_coco.py 를 실행해주세요.")
        return

    print(f"[*] Dataset Path: {dataset_path}")
    print(f"[*] Target Input Shape (Resized): {dummy_spec.input_shapes['images']}")

    # 3. Factory 패턴을 통한 DataLoader 인스턴스화
    try:
        loader = create_dataloader(
            model_spec=dummy_spec,
            dataset_path=dataset_path
        )
        print("[+] DataLoader 생성 및 COCO JSON 파싱 성공!")
    except Exception as e:
        print(f"[!] DataLoader 생성 실패: {e}")
        return

    # 4. 메타데이터 파싱 확인
    meta = loader.get_metadata()
    print("\n[1. Metadata Parsing]")
    for k, v in meta.items():
        print(f"  - {k}: {v}")

    # 5. 단일 데이터 로드(load_single) 테스트
    print("\n[2. Testing load_single()]")
    try:
        single_data = loader.load_single()
        
        img_path = single_data['img_path']
        original_size = single_data['original_size']
        tensor_shape = single_data['input'].shape
        targets = single_data['targets']
        boxes = targets['boxes']
        labels = targets['labels']
        
        print(f"  - Image Path: {img_path}")
        print(f"  - Original Size (H, W): {original_size}")
        print(f"  - Preprocessed Tensor Shape: {tensor_shape} (Expected: (3, 640, 640))")
        print(f"  - Bounding Boxes count: {len(boxes)}")
        
        if len(boxes) > 0:
            print(f"  - First Box (Scaled coords [x, y, w, h]): {boxes[0]}")
            print(f"  - First Box Label (Category ID): {labels[0]}")
    except Exception as e:
        print(f"[!] 단일 데이터 로드(load_single) 에러 발생: {e}")

    # 6. 배치 데이터 로드(load_batch) 테스트
    print("\n[3. Testing load_batch(3)]")
    try:
        batch_data = loader.load_batch(batch_size=3)
        print(f"  - Batch size returned: {len(batch_data)}")
        for i, item in enumerate(batch_data):
            num_boxes = len(item['targets']['boxes'])
            base_name = os.path.basename(item['img_path'])
            orig_size = item['original_size']
            print(f"    [{i+1}] File: {base_name} | Original: {orig_size} | BBox Count: {num_boxes}")
    except Exception as e:
        print(f"[!] 배치 데이터 로드(load_batch) 에러 발생: {e}")

    print("\n" + "="*60)
    print(" Component Test Completed! ")
    print("="*60)

if __name__ == "__main__":
    main()
