import os
import sys
import numpy as np
from pathlib import Path

# 프로젝트 루트 경로를 sys.path에 추가 (src 패키지 인식 용이)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.core.model_spec import Model_Spec, Task
from src.core.compiled_model import CompiledModel
from src.core.benchmarkrunner import BenchmarkRunner
from src.dataloader.image_classification_loader import ImageClassificationLoader
from src.runtimes.onnx_rt import OnnxRuntime
from src.evaluators.resnet50_evaluator import ResNet50Evaluator

def prepare_dummy_dataset(base_path, num_samples=10):
    """임의의 ImageNet 형태 더미 데이터셋 20장을 만들어냅니다."""
    import json
    from PIL import Image
    
    img_dir = os.path.join(base_path, "images")
    if os.path.exists(img_dir) and len(os.listdir(img_dir)) >= num_samples:
        print(f"[*] Dummy dataset exists at {base_path}")
        return
        
    print(f"[*] Generating {num_samples} dummy ImageNet samples at {base_path}...")
    os.makedirs(img_dir, exist_ok=True)
    
    labels_map = {}
    for i in range(num_samples):
        img_name = f"dummy_{i:04d}.jpg"
        img_path = os.path.join(img_dir, img_name)
        
        # (H, W, C) 노이즈 이미지
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(img_path)
        
        # 0 ~ 999 
        labels_map[img_name] = int(np.random.randint(0, 1000))
        
    labels_path = os.path.join(base_path, "labels.json")
    with open(labels_path, "w") as f:
        json.dump(labels_map, f, indent=4)

def prepare_resnet50_onnx(onnx_path):
    """지정된 경로에 모델이 없다면 Torchvision에서 ONNX로 변환 후 저장합니다."""
    if os.path.exists(onnx_path):
        print(f"[*] ONNX model exists at {onnx_path}")
        return
        
    print(f"[*] Exporting ResNet50 to {onnx_path} using Torchvision...")
    try:
        import torch
        import torchvision.models as models
    except ImportError:
        print("[!] PyTorch and Torchvision are required for export. Install them or provide the ONNX model manually.")
        sys.exit(1)
        
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    # 0.13+ 부터 지원되는 최신 weights API 사용 고려
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    except Exception:
        model = models.resnet50(pretrained=True)
        
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"[*] Export completed.")

def main():
    print("="*60)
    print(" ResNet50 Benchmark Runner Test (ImageNet-1K) ")
    print("="*60)
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    onnx_model_path = os.path.join(project_root, 'models/resnet-50', 'resnet50.onnx')
    # 사용자가 'load_imagenet_1k.py'로 직접 받은 3000개 샘플 디렉토리를 가리키도록 설정합니다.
    dataset_path = os.path.join(project_root, 'datasets/imagenet_1k')
    image_dir = os.path.join(dataset_path, 'val')
    label_file = os.path.join(dataset_path, 'val_labels.txt')
    
    # 다운로드가 되어있지 않은 파일이 있다면 실행 환경 내에서 동적 생성합니다.
    prepare_resnet50_onnx(onnx_model_path)
    # 이미 load_imagenet_1k.py 로 데이터를 확보하였으므로 더미 생성을 주석 처리합니다.
    # prepare_dummy_dataset(dataset_path, num_samples=3000)
    
    # 1. 아키텍처 규격에 따른 스펙 명세서 정의
    # Hugging Face 허브에서 다운받은 모델의 경우 입력 이름이 보통 'pixel_values', 출력이 'logits'로 명명됩니다.
    resnet_spec = Model_Spec(
        name="resnet50",
        task=Task.IMAGE_CLASSIFICATION,
        input_shapes={"pixel_values": (1, 3, 224, 224)},
        input_dtype={"pixel_values": "float32"},
        output_shapes={"logits": (1, 1000)},
        model_paths={"onnx": onnx_model_path}
    )
    
    # 2. 컴파일러가 반환했다고 가정하는 컴파일 아티팩트 객체
    compiled_model = CompiledModel(
        spec=resnet_spec,
        backend_name="onnxruntime",
        artifact_path=Path(onnx_model_path)
    )
    
    # 3. 객체 주입 및 초기화
    loader = ImageClassificationLoader(
        model_spec=resnet_spec,
        dataset_path=dataset_path,
        image_dir=image_dir,
        label_file=label_file
    )
    
    runtime = OnnxRuntime(device="cpu")
    runtime.load(compiled_model)
    
    evaluator = ResNet50Evaluator(top_k=(1, 5))
    
    # 4. 오케스트레이터(BenchmarkRunner) 구동
    runner = BenchmarkRunner(dataloader=loader, runtime=runtime, evaluator=evaluator)
    results = runner.run(warmup_runs=2, batch_size=4)
    
    # 5. 채점 평가 결과 출력
    print("\n" + "="*40)
    print(" Benchmark Final Metrics ")
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
