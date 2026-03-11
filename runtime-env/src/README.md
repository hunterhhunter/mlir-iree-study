# Source Code & Implementation Details

본 디렉토리는 MobileNetV2 IREE 런타임 평가 프레임워크의 핵심 소스 코드를 포함하고 있습니다. 모듈화된 설계를 통해 데이터 로딩, 모델 실행, 결과 평가 로직이 분리되어 있습니다.

## 📂 폴더 및 파일 구성 (Directory Structure)

*   **`main.py`**: 프레임워크의 통합 진입점(Entry Point). 모델 컴파일, 데이터 로딩, 추론 루프 및 최종 평가 보고를 제어합니다.
*   **`data_loader.py`**: `transformers` 및 `PyTorch DataLoader` 기반의 데이터 파이프라인. ImageNet 규격에 맞는 표준 전처리 및 배치 로딩을 담당합니다.
*   **`runtimes/`**: 이기종 하드웨어 실행을 위한 런타임 래퍼 모음.
    *   `iree_rt.py`: ONNX 모델을 MLIR을 거쳐 VMFB로 컴파일하고, IREE 컨텍스트를 재사용하여 고속 추론을 수행합니다.
*   **`evaluator/`**: 모델 성능 분석 모듈.
    *   `MobileNet_evaluator.py`: Top-1/5 Accuracy 및 Macro-Precision, Recall, F1-Score를 산출하는 평가기입니다.

## 🚀 실행 가이드 (Execution Guide)

프레임워크를 처음 사용하시는 경우, 아래의 **3단계 워크플로우**를 순차적으로 진행하시면 됩니다.

### 1단계: 환경 변수 설정 (WSL2/CUDA 사용자 필수)
WSL2 환경에서 NVIDIA GPU 가속을 활용하기 위해서는 시스템 라이브러리 경로 설정이 반드시 선행되어야 합니다. 터미널에서 다음 명령어를 입력하여 경로를 추가해 주십시오.

```bash
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH}"
```

### 2단계: 데이터셋 구축 (최초 1회 수행)
Hugging Face 허브로부터 ImageNet-1K 검증 데이터셋 1,000장을 자동으로 확보하고 구조화하는 단계입니다. 스트리밍 방식을 사용하여 네트워크 부하를 최소화하면서 빠르게 준비하실 수 있습니다.

```bash
# 1. Hugging Face 인증 (최초 1회, Access Token 필요)
huggingface-cli login

# 2. 데이터셋 다운로드 및 레이블 매핑 자동화 실행
python3 datasets/load_imagenet_1k.py
```

### 3단계: 통합 정밀도 평가 실행
데이터셋 준비가 완료되었다면, 이제 MobileNetV2 모델에 대한 추론 및 성능 측정을 시작합니다. 타겟 장치(CPU 또는 CUDA)에 맞춰 명령어를 선택하여 실행하시면 됩니다.

```bash
# [추천] CUDA (NVIDIA GPU) 가속기를 사용하는 경우
python3 src/main.py --model models/mobilenetv2-10.onnx --eval --device cuda --batch_size 1

# LLVM-CPU 모드로 실행하는 경우
python3 src/main.py --model models/mobilenetv2-10.onnx --eval --device cpu --batch_size 1
```

## 💡 주요 실행 파라미터 설명
*   `--model`: 평가할 ONNX 모델 파일의 경로.
*   `--eval`: 데이터셋 루프를 통한 정밀도 평가 모드 활성화.
*   `--device`: 실행 장치 선택 (`cpu` 또는 `cuda`).
*   `--batch_size`: 한 번에 처리할 이미지 수 (기본값: 1).

## 🔮 향후 구현 계획 (Future Roadmap)

*   **Evaluator 모듈 고도화**: 현재 `MobileNetEvaluator`로 특화된 평가 로직을 일반적인 `BaseEvaluator` 인터페이스와 모델별 서브 클래스로 완전히 분리하여, 새로운 신경망 구조(Transformer 등)를 손쉽게 통합할 수 있도록 개선할 계획입니다.
*   **데이터셋 스크립트 일반화**: 현재 ImageNet-1K에 고정된 `load_imagenet_1k.py`를 파라미터화하여, COCO, CIFAR 등 다양한 데이터셋을 하나의 스크립트로 구축할 수 있도록 범용성을 확보할 예정입니다.
*   **모델 자동 다운로드 시스템**: 사용자가 모델 경로를 직접 지정하지 않아도, Hugging Face Hub나 ONNX Model Zoo로부터 최적화된 모델을 자동으로 검색하고 내려받는 `download_model.py` 기능을 추가할 예정입니다.
*   **이기종 런타임 통합**: `src/runtimes/` 내에 `pytorch_rt.py`, `tensorflow_rt.py`, `tvm_rt.py` 등을 순차적으로 구현하여, 동일 하드웨어 상의 엔진별 성능 비교 기능을 완성할 계획입니다.
