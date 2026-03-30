# Models Management & Export Utility

이 디렉토리는 Hugging Face 등 외부 소스에서 모델 가중치를 다운로드하고, 추론 환경(ONNX Runtime 등)에 맞게 백엔드 포맷(ONNX)으로 변환(Export) 및 양자화(Quantization)하는 스크립트들을 보관합니다.

---

## 🚀 1. 모델 다운로드 스크립트

### `download_model_from_huggingface.py`
Hugging Face 레포지토리의 전체 파일을 로컬 디렉토리로 스냅샷 다운로드합니다. 이 스크립트를 사용하면 매번 인터넷에서 불러오는 네트워크 오버헤드를 줄이고 오프라인 환경에서 테스트를 돌리기 편해집니다.

* **사용 예시**:
```bash
# 기본 사용 (전체 모델 로드)
uv run models/download_model_from_huggingface.py --name ibm-granite/granite-timeseries-patchtst

# 특정 포맷(onnx) 파일과 기본 JSON들만 필터링해서 다운로드 (용량 절약)
uv run models/download_model_from_huggingface.py --name meta-llama/Llama-3.1-8B-Instruct --format safetensors
```
* 다운로드된 파일은 기본적으로 `models/[저장소 구조]` 형태로 보관됩니다. (예: `models/ibm-granite_granite-timeseries-patchtst`)

### `download_yolov5m.py`
미리 지정된 YOLOv5m 등의 비전 모델 다운로드 전용 스크립트입니다.
```bash
uv run models/download_yolov5m.py
```

---

## 🛠 2. ONNX 변환 (Export) 스크립트

### `export_onnx_optimum.py`
Hugging Face의 `optimum-cli`의 파이썬 래퍼(Wrapper) 스크립트입니다. Hugging Face 생태계에 편입된 대부분의 모델(ResNet, LLaMA 등)을 손쉽게 ONNX 포맷으로 추출합니다.

* **매개변수**:
  - `--model`: Hugging Face repo ID 또는 로컬에 다운받은 경로
  - `--task`: 작업 종류 (예: `image-classification`, `text-generation`)
  - `--dtype` (선택): 변환할 Precision (예: `fp16`, `fp32`)

* **사용 예시** (ResNet-50을 이미지 분류 모델로 Export):
```bash
uv run models/export_onnx_optimum.py \
    --model microsoft/resnet-50 \
    --task image-classification \
    --output models/microsoft_resnet-50-ONNX \
    --no-post-process
```

### `export_onnx_patchtst.py`
Optimum CLI를 완벽히 지원하지 않는 **PatchTST 계열 시계열 모델 전용 수동 변환 스크립트**입니다. `torch.onnx.export`를 직접 호출하여 모델을 변환하고 더미 입력값을 통해 정합성을 검증합니다.

* **사용 예시**:
```bash
# 로컬에 다운로드 받은 patchtst-fm 모델 변환 (ETTh1 기준 셋업)
uv run models/export_onnx_patchtst.py \
    --model ibm-granite/granite-timeseries-patchtst \
    --output models/ibm-granite_granite-timeseries-patchtst-ONNX/model.onnx \
    --context-length 512 \
    --channels 7 \
    --prediction-length 96
```

---

## 🗜 3. 양자화 (Quantization) 스크립트

### `quantize_onnx_int8.py`
용량이 큰 모델(LLM 등)을 실행하기 위해 **ONNX 동적 양자화(Dynamic INT8 Quantization)**를 수행하여 메모리 사용량을 줄입니다. 외부 데이터 파일 형태(`.onnx_data`)로 분할된 대규모 가중치 연산도 자동으로 인식하고 처리합니다. 
`MatMulConstBOnly=True` 옵션을 적용해 가중치 Matrix Multiplication 노드만 양자화하여 성능을 보존합니다.

* **사용 예시** (LLaMA 모델을 INT8로 양자화):
```bash
uv run models/quantize_onnx_int8.py \
    --input models/meta-llama_Llama-3.1-8B-ONNX-fp16/model.onnx \
    --output models/meta-llama_Llama-3.1-8B-ONNX-int8
```
* 변환이 완료되면 지정한 폴더 안의 `.json` 설정 파일들도 자동으로 복사되며 메모리에 최적화된 `model.onnx` 형태가 생성됩니다.
