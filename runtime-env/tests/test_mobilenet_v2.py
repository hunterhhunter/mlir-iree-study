import os
import sys
import numpy as np
import onnx
from pathlib import Path

# 프로젝트 루트 경로를 sys.path에 추가 (src 패키지 인식 용이)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.core.model_spec import Model_Spec, Task
from src.core.compiled_model import CompiledModel
from src.core.benchmarkrunner import BenchmarkRunner
from src.dataloader.image_classification_loader import ImageClassificationLoader
from src.runtimes.onnx_rt import OnnxRuntime
# ResNet50Evaluator는 사실상 보편적인 ImageNet-1K 분류 평가기이므로 그대로 재사용 가능합니다.
from src.evaluators.resnet50_evaluator import ResNet50Evaluator

def get_onnx_io_names(onnx_path):
    """ONNX 모델의 첫 번째 입력과 출력의 이름을 자동으로 추출합니다."""
    model = onnx.load(onnx_path)
    input_name = model.graph.input[0].name
    output_name = model.graph.output[0].name
    return input_name, output_name

def main():
    print("="*60)
    print(" MobileNet-V2 Benchmark Runner Test (ImageNet-1K) ")
    print("="*60)
    
    # 방금 다운로드하신 모델 경로를 지정합니다.
    onnx_model_path = os.path.join(project_root, 'models/Kalray_mobilenet-v2', 'mobilenetv2.onnx')
    
    # 사용자님이 확보하신 3000장 ImageNet 데이터셋 경로 설정
    dataset_path = os.path.join(project_root, 'datasets/imagenet_1k')
    image_dir = os.path.join(dataset_path, 'val')
    label_file = os.path.join(dataset_path, 'val_labels.txt')
    
    if not os.path.exists(onnx_model_path):
        print(f"[!] 모델 파일을 찾을 수 없습니다: {onnx_model_path}")
        return

    # ONNX 모델에서 직접 입력/출력 텐서 이름을 파싱하여 매핑 오류를 원천 차단
    input_name, output_name = get_onnx_io_names(onnx_model_path)
    print(f"[*] Detected ONNX Input Name: {input_name}")
    print(f"[*] Detected ONNX Output Name: {output_name}")
    
    # 1. 아키텍처 규격에 따른 스펙 명세서 정의
    mobilenet_spec = Model_Spec(
        name="mobilenet_v2",
        task=Task.IMAGE_CLASSIFICATION,
        input_shapes={input_name: (1, 3, 224, 224)},
        input_dtype={input_name: "float32"},
        output_shapes={output_name: (1, 1000)},
        model_paths={"onnx": onnx_model_path}
    )
    
    # 2. 컴파일러가 반환했다고 가정하는 컴파일 아티팩트 객체
    compiled_model = CompiledModel(
        spec=mobilenet_spec,
        backend_name="onnxruntime",
        artifact_path=Path(onnx_model_path)
    )
    
    # 3. 객체 주입 및 초기화
    loader = ImageClassificationLoader(
        model_spec=mobilenet_spec,
        dataset_path=dataset_path,
        image_dir=image_dir,
        label_file=label_file
    )
    
    runtime = OnnxRuntime(device="cpu")
    runtime.load(compiled_model)
    
    evaluator = ResNet50Evaluator(top_k=(1, 5))
    
    # 4. 오케스트레이터(BenchmarkRunner) 구동
    runner = BenchmarkRunner(dataloader=loader, runtime=runtime, evaluator=evaluator)
    # 허깅페이스에서 다운받은 Kalray/mobilenet-v2 ONNX 모델은 Batch Size가 1로 고정(정적)되어 있습니다.
    # 따라서 batch_size를 1로 지정하여 구동합니다.
    results = runner.run(warmup_runs=2, batch_size=1)
    
    # 5. 채점 평가 결과 출력
    print("\n" + "="*40)
    print(" MobileNet-V2 Final Metrics ")
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
