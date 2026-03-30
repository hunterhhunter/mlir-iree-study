import os
import sys
from pathlib import Path

# 프로젝트 루트 경로를 sys.path에 추가 (src 패키지 인식 용이)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.core.model_spec import Model_Spec, Task
from src.core.compiled_model import CompiledModel
from src.core.benchmarkrunner import BenchmarkRunner
from src.dataloader import create_dataloader
from src.runtimes import create_runtime
from src.evaluators import create_evaluator

def main():
    print("="*60)
    print(" 🚀 실제 YOLOv5m(Ultralytics) E2E 벤치마크 테스트 ")
    print("="*60)
    
    onnx_model_path = os.path.join(project_root, 'models/yolov5m', 'yolov5m.onnx')
    dataset_path = os.path.join(project_root, 'datasets', 'coco')
    
    if not os.path.exists(onnx_model_path):
        print(f"[!] ONNX 모델을 찾을 수 없습니다: {onnx_model_path}")
        print("[!] 먼저 'models/download_yolov5m.py' 스크립트를 실행해주세요.")
        return

    if not os.path.exists(dataset_path):
        print(f"[!] COCO 데이터셋 폴더를 찾을 수 없습니다: {dataset_path}")
        print("[!] 먼저 'datasets/load_coco.py' 를 실행해주세요.")
        return
    
    # 1. 파이프라인 컴포넌트 규격 (Model_Spec) 정의
    print("\n---------- [1단계: Model_Spec 설정] ----------")
    # Ultralytics Export(v8 규격 호환) 모델은 'images' 입력을 받고 'output0' (1, 84, 8400) 형태를 출력합니다.
    det_spec = Model_Spec(
        name="yolov5m_ultra",
        task=Task.OBJECT_DETECTION,
        input_shapes={"images": (1, 3, 640, 640)},
        input_dtype={"images": "float32"},
        output_shapes={"output0": (1, 84, 8400)},
        model_paths={"onnx": onnx_model_path}
    )
    
    # 2. 객체 인스턴스화 (Factory 기반)
    print("\n---------- [2단계: 팩토리(Factory) 기반 컴포넌트 생성] ----------")
    from src.utils.dataset_resolver import resolve_dataset_paths
    image_dir, label_path = resolve_dataset_paths(det_spec.task, dataset_path, "", "")
    loader = create_dataloader(model_spec=det_spec, dataset_path=dataset_path, image_dir=image_dir, label_path=label_path)
    print(f"[+] DataLoader 생성 완료 (Loaded {loader.total_samples} samples)")
    
    compiled_model = CompiledModel(spec=det_spec, backend_name="onnx", artifact_path=Path(onnx_model_path))
    runtime = create_runtime(backend_name="onnx", device="cpu")
    runtime.load(compiled_model)
    print("[+] ONNX Runtime 인스턴스 로드 완료")
    
    # mAP 채점을 위한 평가기 생성
    evaluator = create_evaluator(model_spec=det_spec, conf_threshold=0.25, iou_threshold=0.45)
    print("[+] Evaluator (Object Detection) 생성 완료")
    
    # 3. 종합 오케스트레이션 (BenchmarkRunner) 구동
    print("\n---------- [3단계: E2E BenchmarkRunner 측정 파이프라인 구동] ----------")
    runner = BenchmarkRunner(dataloader=loader, runtime=runtime, evaluator=evaluator)
    
    # 본인 스펙에 맞춰 batch_size 조정 가능
    try:
        results = runner.run(warmup_runs=1, batch_size=4)
    except Exception as e:
        print(f"\n[!] 벤치마크 러너 실행 중 에러가 발생했습니다: {e}")
        return
        
    # 4. 채점 평가 결과 출력
    print("\n" + "="*40)
    print(" 🏆 Object Detection Final Metrics ")
    print("="*40)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("="*40)
    
    runtime.unload()

if __name__ == "__main__":
    main()
