# Benchmark Framework

이 디렉토리는 다양한 AI 모델(ResNet50, MobileNet 등)과 런타임 백엔드(ONNX, IREE)의 추론 성능을 일관성 있게 측정하기 위한 벤치마크 프레임워크 소스 코드를 담고 있습니다.

## 주요 패키지 구성 요소
- **`main.py`**: 터미널 커맨드라인(CLI) 파라미터를 파싱하여 프레임워크를 구동하는 **통합 진입점(Entrypoint)**.
- **`core/`**: 벤치마크 전체 루프를 총괄하는 엔진(`BenchmarkRunner`)과 메타데이터/DTO 클래스(`Model_Spec`, `InferenceResult`).
- **`dataloader/`**: 다양한 타겟 벤치마크 데이터셋에서 입력 데이터를 전처리하고 공급하는 모듈.
- **`runtimes/`**: 하드웨어 가속기 및 추론 엔진 백엔드(ONNX, IREE 등)를 통합된 규격(인터페이스)으로 제어하는 모듈.
- **`evaluators/`**: 추론된 결괏값의 레이블을 분석하고 검증하여 성능 메트릭(Top-1, Top-5, Latency 등)을 채점하는 모듈.

## 🚀 CLI 실행 가이드 (Zero-Config)

본 벤치마크 프레임워크는 **Zero-Config Auto-Prepare** 아키텍처를 적용하여, 모델명만 지정하면 내장된 프로필 레지스트리가 평가용 모델 스펙, 데이터셋, 배치 경로 등을 자동 탐지하고 누락 시 알아서 다운로드 및 변환합니다.

### 🖼️ 1. 이미지 분류 (Image Classification)
```bash
# ResNet50 모델 평가 (ImageNet 1K 데이터셋 디렉토리 자동 스니핑)
uv run src/main.py --model resnet50
```

### 🎯 2. 객체 탐지 (Object Detection)
```bash
# YOLOv5m 성능 평가 (COCO128 데이터셋)
uv run src/main.py --model yolov5m
```

### 📊 3. 자연어 분류 (NLP Classification)
```bash
# BERT Base 모델을 이용한 SST-2 감성 분석
uv run src/main.py --model bert-base-uncased
```

### 🧠 4. 기계 독해 (Question Answering)
```bash
# SQuAD v1 정답 도출 성능 평가
uv run src/main.py --model bert-base-uncased-squad-v1
```

### 📈 5. 시계열 예측 (Time-Series Forecasting)
```bash
# PatchTST 모델을 활용한 ETTh1 시계열 벤치마크
uv run src/main.py --model patchtst-fm-r1
```

### 💬 6. 언어 생성 (LLM Generation)
Llama 등의 생성형 대형 언어 모델은 `onnxruntime` 대신 메모리 컨트롤이 뛰어난 전용 가속기인 `vLLM` 백엔드를 필수로 지정해야 합니다.
*(주의: Llama 계열은 접근이 제한된 Gated 모델이므로 사전에 터미널에서 `uv run huggingface-cli login` 인증을 완료해야 다운로드가 승인됩니다.)*
```bash
# Llama 3.2 3B 모델 가동 (SQuAD v2 데이터셋 기반)
uv run src/main.py --model llama-3.2-3b --backend vllm

# Llama 3.1 8B 모델 가동
uv run src/main.py --model llama-3.1-8b --backend vllm
```

> **💡 부가 설정 팁**: 
> * 전체 데이터셋 평가 시간이 너무 오래 걸릴 때는 `--max-steps 1` 과 같은 인자를 맨 뒤에 붙여 1사이클만 돌려볼 수 있습니다.
> * 하드웨어 장치를 변경하거나 커스텀 모델을 테스트하고 싶다면, `--device cuda` 또는 `--onnx custom.onnx` 처럼 원하는 인자만 수동으로 타이핑하세요. 사용자의 직접 입력이 항상 내부 관례 시스템보다 **최우선**으로 적용됩니다.
