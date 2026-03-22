# Antigravity Benchmark Framework

이 디렉토리는 다양한 AI 모델(ResNet50, MobileNet 등)과 런타임 백엔드(ONNX, IREE)의 추론 성능을 일관성 있게 측정하기 위한 벤치마크 프레임워크 소스 코드를 담고 있습니다.

## 주요 패키지 구성 요소
- **`main.py`**: 터미널 커맨드라인(CLI) 파라미터를 파싱하여 프레임워크를 구동하는 **통합 진입점(Entrypoint)**.
- **`core/`**: 벤치마크 전체 루프를 총괄하는 엔진(`BenchmarkRunner`)과 메타데이터/DTO 클래스(`Model_Spec`, `InferenceResult`).
- **`dataloader/`**: 다양한 타겟 벤치마크 데이터셋에서 입력 데이터를 전처리하고 공급하는 모듈.
- **`runtimes/`**: 하드웨어 가속기 및 추론 엔진 백엔드(ONNX, IREE 등)를 통합된 규격(인터페이스)으로 제어하는 모듈.
- **`evaluators/`**: 추론된 결괏값의 레이블을 분석하고 검증하여 성능 메트릭(Top-1, Top-5, Latency 등)을 채점하는 모듈.

## CLI 실행 가이드 (`main.py`)
```bash
# ONNX 런타임을 이용한 MobileNet-V2 GPU 가속 테스트 예시
uv run src/main.py --model mobilenet_v2 \
                   --onnx models/google-mobilenet-v2/mobilenet.onnx \
                   --dataset datasets/imagenet_1k/ \
                   --batch-size 1 \
                   --device cuda \
                   --backend onnxruntime
```
