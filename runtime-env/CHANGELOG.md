# Changelog

All notable changes to this project will be documented in this file.

## [0.0.3.0] - 2026-04-06

### Added
- `src/preprocessor/` 패키지 신설: 모든 태스크의 전처리 로직을 독립 모듈로 분리
  - `BasePreprocessor` 추상 베이스 클래스
  - `ImagePreprocessor`, `LlamaPreprocessor`, `BertClassificationPreprocessor`, `BertQAPreprocessor`
  - `ETTmPreprocessor`, `ObjectDetectionPreprocessor`
  - `PreprocessStrategy` 및 전략 구현체 (`MLPerfResNet50Preprocess`, `SQuADPreprocessStrategy` 등)
- BERT 전체 파이프라인: SST-2 분류(`bert-base-uncased`)와 SQuAD QA(`bert-base-uncased-squad-v1`) 지원
- Llama 3.1 8B / 3.2 3B 전체 파이프라인: vLLM 백엔드 + SQuAD 2.0 평가, TTFT/TPOT/Throughput 측정
- PatchTST 시계열 예측 파이프라인: ETTh1 데이터셋, 96-step 예측
- YOLOv5m 객체 탐지 파이프라인: COCO128 데이터셋, mAP 평가
- `src/adapters/loadgen_adapter.py`: MLPerf LoadGen 호환 어댑터
- `src/runtimes/vllm_rt.py`: vLLM 백엔드 런타임 (자기회귀 생성, TTFT/TPOT 분리 추적)
- `src/runtimes/onnx_rt.py`: KV 캐시 없는 자기회귀 `generate()` 경로 추가
- `src/utils/dataset_resolver.py`: 데이터셋 경로 추론 로직 일원화 (DI 패턴)
- `src/utils/cuda_preload.py`: CUDA 공급자 사전 로드 유틸리티
- 전체 테스트 스위트 15개 파일: 팩토리 API 100% 커버리지, 각 모델별 로더/평가기/E2E 회귀 테스트
- `tests/run_all_onnx_benchmarks.py`: 전체 ONNX 벤치마크 일괄 실행 스크립트
- `datasets/prepare_cifar10.py`, `prepare_coco128.py`, `prepare_etth1.py` 등 데이터셋 준비 스크립트 확장
- `models/prepare_bert_squad.py`, `prepare_bert_sst2.py`, `prepare_llama_3_1_8b.py` 등 모델 준비 스크립트 확장
- `models/quantize_onnx_int8.py`: ONNX 모델 INT8 정적 양자화 스크립트
- `models/inspect_onnx_model.py`: ONNX 그래프 I/O 이름 검사 유틸리티

### Changed
- `src/dataloader/` 각 로더에서 전처리 로직을 `preprocessor/` 패키지로 분리
- `LlamaPreprocessor.get_cache_path()`: `qa_id + cfg_hash(tokenizer_path:max_length)` 기반 캐시 경로 생성 — 설정 변경 시 stale 캐시 재사용 방지
- `main.py`: Zero-Config 지원 — `SUPPORTED_PROFILES` 레지스트리에서 태스크/경로를 자동 추론, 누락된 모델/데이터셋은 `prepare_*.py` 스크립트 자동 실행
- `main.py`: `--max-new-tokens`, `--max-model-len`, `--debug` CLI 옵션 추가
- 스크립트/데이터셋 명명 규칙 통일: 모든 준비 스크립트를 `prepare_` 접두사로 표준화

### Fixed
- `export_onnx_hf.py`: `subprocess.run(cmd, shell=True)` + f-string → list 기반 인자로 변경 (shell injection 방지)
- `onnx_rt.py`: 누락된 입력에 대해 명시적 `ValueError` 발생 (이전: silent drop)
- `strategies.py`: `MLPerfResNet50Preprocess` — resize 후 이미지가 crop 크기보다 작을 때 명시적 `ValueError` 발생
- `LlamaLoader`: 기본 `dataset_path` `SQuAD_2` → `squad2` (실제 디렉토리명 일치)
- `BertClassificationLoader`: 오류 메시지의 스크립트 이름 `tokenize_to_numpy.py` → `prepare_text_numpy.py` 갱신

## [0.0.2.0] - 2026-04-04

### Fixed
- `src/` 전체 21개 파일의 상대 임포트(`from ..X`) → 절대 임포트(`from X`)로 변환 — `python src/main.py` 실행 시 `ImportError: attempted relative import beyond top-level package` 오류 해결
- `LlamaLoader`: `--dataset datasets/squad2/val.json`처럼 `.json` 파일 경로를 직접 전달하면 경로 끝에 `/val.json`이 중복 추가되던 버그 수정 (`datasets/squad2/val.json/val.json` → 올바르게 인식)

### Added
- `main.py`: `--max-model-len` CLI 옵션 추가 — vLLM 백엔드의 KV 캐시 메모리 부족 시 컨텍스트 길이를 제한할 수 있음 (예: `--max-model-len 32768`)
- `models/download_llama_3_2_3b.py`, `models/download_llama_3_1_8b.py`, `models/download_resnet50_kalray.py`: 모델별 독립 다운로드 스크립트 추가

## [0.0.1.0] - 2026-04-04

### Changed
- 스크립트/데이터셋 경로 재구조화: `scripts/` 전처리 스크립트 → `datasets/`, 모델 스크립트 → `models/`로 이동, 파일명 언더스코어 통일 (`iree-cmd-*.py` → `iree_*.py`)
- `download_model_from_huggingface.py` → `download_hf_model.py` 간략화
- `export_onnx_optimum.py` → `export_onnx_hf.py` 이름 통일
- vLLM Runtime `eos_token_id` 파라미터 → `stop_token_ids`로 일관성 개선

### Fixed
- ONNX Runtime `warmup()`: LLM 패딩 trim을 `NLP_GENERATION` 태스크에서만 적용 (BERT 등 고정 shape 모델 shape mismatch 방지)
- `LlamaLoader`: 기본 dataset_path `SQuAD_2` → `squad2` (실제 디렉토리명 일치)
- `BertClassificationLoader`: 오류 메시지의 스크립트 이름 `tokenize_to_numpy.py` → `prepare_text_numpy.py` 갱신
- `test_tokenize_numpy.py`: sys.path `scripts/` → `datasets/` (이동된 `prepare_text_numpy.py` 위치 반영)

### Added
- `tests/run_all_onnx_benchmarks.py`: 전체 ONNX 벤치마크 일괄 실행 스크립트
