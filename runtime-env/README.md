# AI Benchmark Framework

ONNX/vLLM 백엔드에서 AI 모델의 추론 성능을 측정하는 통합 벤치마크 프레임워크입니다. 모델 이름 하나만으로 다운로드부터 추론까지 자동으로 실행되는 Zero-Config 방식을 지원합니다.

## 빠른 시작

```bash
# 환경 설정
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Zero-Config 실행 (모델/데이터셋 자동 다운로드 포함)
cd runtime-env
python src/main.py --model resnet50
python src/main.py --model yolov5m
python src/main.py --model bert-base-uncased
python src/main.py --model llama-3.2-3b --backend vllm
python src/main.py --model patchtst-fm-r1
```

## 지원 모델

| 모델 이름 | 태스크 | 백엔드 | 데이터셋 |
|---|---|---|---|
| `resnet50` | 이미지 분류 | onnxruntime | ImageNet-1K |
| `yolov5m` | 객체 탐지 | onnxruntime | COCO128 |
| `bert-base-uncased` | 텍스트 분류 (SST-2) | onnxruntime | SST-2 numpy |
| `bert-base-uncased-squad-v1` | 질문 답변 (SQuAD) | onnxruntime | SQuAD numpy |
| `llama-3.1-8b` | 텍스트 생성 | vllm | SQuAD 2.0 |
| `llama-3.2-3b` | 텍스트 생성 | vllm / onnxruntime | SQuAD 2.0 |
| `patchtst-fm-r1` | 시계열 예측 | onnxruntime | ETTh1 |

## CLI 옵션

```
python src/main.py --model <name> [options]

필수:
  --model           모델 프로필 이름 (위 표 참조)

선택 (생략 시 프로필 기본값 사용):
  --onnx            ONNX 모델 파일 경로
  --model-path      HuggingFace 모델 디렉토리 (vLLM 백엔드)
  --dataset         데이터셋 경로
  --backend         onnxruntime | vllm (기본: onnxruntime)
  --device          cpu | cuda (기본: cpu)
  --batch-size, -b  배치 크기 (기본: 1)
  --warmup, -w      웜업 횟수 (기본: 2)
  --max-new-tokens  LLM 최대 생성 토큰 수 (기본: 256)
  --max-model-len   vLLM KV 캐시 최대 컨텍스트 길이
  --debug           샘플별 예측/정답 로그 출력
```

## 아키텍처

```
src/main.py (CLI 오케스트레이터)
      |
      v
BenchmarkRunner (src/core/benchmarkrunner.py)
      |
      +-- DataLoader  (src/dataloader/)    ← 데이터 배치 공급
      |     |
      |     +-- Preprocessor (src/preprocessor/)  ← 모델별 전처리
      |
      +-- Runtime     (src/runtimes/)      ← 추론 실행 (ONNX / vLLM)
      |
      +-- Evaluator   (src/evaluators/)    ← 메트릭 계산
```

각 레이어는 팩토리 함수(`create_dataloader`, `create_runtime`, `create_evaluator`)를 통해 생성됩니다. 새 모델 지원은 `src/core/model_profiles.py`에 프로필을 추가하고 각 레이어에 구현체를 추가하면 됩니다.

## 모델/데이터셋 준비

Zero-Config 실행 시 모델과 데이터셋이 없으면 자동으로 `prepare_*.py` 스크립트가 실행됩니다. 수동으로 실행하려면:

```bash
# 모델 다운로드
python models/prepare_resnet50_kalray.py
python models/prepare_yolov5m.py
python models/prepare_bert_sst2.py
python models/prepare_llama_3_2_3b.py  # Hugging Face 토큰 필요
python models/prepare_patchtst.py

# 데이터셋 다운로드
python datasets/prepare_imagenet_1k.py
python datasets/prepare_coco128.py
python datasets/prepare_text_numpy.py  # BERT 텍스트 분류용
python datasets/prepare_squad2.py      # Llama / BERT QA용
python datasets/prepare_etth1.py       # PatchTST용
```

## 테스트

```bash
# 전체 테스트
python -m pytest tests/ -v

# 단위 테스트만 (모델 파일 불필요)
python -m pytest tests/test_factory_api.py tests/test_bert_qa_loader.py -v

# 전체 ONNX 벤치마크 일괄 실행
python tests/run_all_onnx_benchmarks.py
```
