# Changelog

All notable changes to this project will be documented in this file.

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
