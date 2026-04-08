# Changelog

All notable changes to this project will be documented in this file.

## [0.0.1.0] - 2026-04-06

### Added
- `src/preprocessor/` 패키지 신설: 모델별 전처리 로직을 독립 모듈로 분리
  - `BasePreprocessor` 추상 베이스 클래스
  - `ImagePreprocessor`, `LlamaPreprocessor`, `BertClassificationPreprocessor`, `BertQAPreprocessor`
  - `ETTmPreprocessor`, `ObjectDetectionPreprocessor`
  - `PreprocessStrategy` 및 구체 전략들(`MLPerfResNet50Preprocess`, `SQuADPreprocessStrategy` 등)

### Changed
- `src/dataloader/` 각 로더에서 전처리 로직을 `preprocessor/` 패키지로 분리
- `LlamaPreprocessor.get_cache_path()`: `qa_id` 단독이 아닌 `qa_id + cfg_hash(tokenizer_path:max_length)`로 캐시 경로 생성 — 설정 변경 시 stale 캐시 재사용 방지

### Fixed
- `export_onnx_hf.py`: `subprocess.run(cmd, shell=True)` + f-string → list 기반 인자로 변경 (shell injection 방지)
- `onnx_rt.py`: 불필요한 입력 silent drop → 누락된 입력에 대해 명시적 `ValueError` 발생
- `strategies.py`: `MLPerfResNet50Preprocess` — resize 후 이미지가 crop 크기보다 작을 때 명시적 `ValueError` 발생 (이전: 음수 좌표로 잘못된 crop 결과 반환)
