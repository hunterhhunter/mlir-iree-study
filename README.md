# MLIR & IREE Compiler Research Study

이 레포지토리는 MLIR(Multi-Level Intermediate Representation)과 IREE 컴파일러 구조를 학습하고, 실제 모델을 컴파일하여 GPU 성능을 분석하기 위한 연구/학습 공간입니다.

## Project Goal (6 Weeks)
- **이론 학습:** LLVM/MLIR의 기본 구조(Dialect, Pass, Op) 및 IREE 컴파일 파이프라인 이해
- **실험 및 구현:** 딥러닝 모델(시각, 시계열, 언어모델)을 IREE로 컴파일하여 GPU(NVIDIA)에서 실행
- **성능 분석:** 컴파일 옵션별 성능 변화를 측정하고 시각화하여 최적화 포인트 분석

---

## 📂 Directory Structure

이 프로젝트는 다음과 같은 폴더 구조로 관리됩니다.

```text
mlir-iree-study/
├── 📁 src/         # 완성된 메인 코드 (Shared Codebase)
├── 📁 docs/        # 연구 보고서 및 학습 노트 (Documentation)
├── 📁 scripts/     # 자동화 및 유틸리티 스크립트 (Utilities)
└── 📁 personal/    # 개인별 실험 공간
└── 📁 runtime-env/ # 모델을 IREE, Pytorch 등 실행해볼 수 있는 환경
