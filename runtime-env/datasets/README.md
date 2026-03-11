# Evaluation Datasets Management

본 디렉토리는 MobileNetV2 및 기타 모델의 정밀도 평가에 사용되는 데이터셋과 관련 스크립트를 관리합니다.

## 📦 자동 구축 방법 (Automatic Setup)

`load_imagenet_1k.py` 스크립트를 사용하여 Hugging Face로부터 ImageNet-1K 검증 세트 1,000개를 자동으로 내려받고 구조화할 수 있습니다.

### 1. 사전 요구사항
*   Hugging Face 계정 및 Access Token이 필요합니다.
*   터미널에서 `huggingface-cli login` 명령어로 인증을 완료해야 합니다.

### 2. 다운로드 실행
가상환경이 활성화된 상태에서 프로젝트 루트 디렉토리 또는 본 디렉토리에서 다음 명령어를 실행합니다.

```bash
# 프로젝트 루트 기준
python datasets/load_imagenet_1k.py
```

## 📂 디렉토리 구조 (Directory Structure)

스크립트 실행 후 다음과 같은 구조로 데이터가 배치됩니다.

```text
datasets/
├── imagenet_1k/                # ImageNet-1K 전용 네임스페이스
│   ├── val/                    # JPEG 이미지 파일 (image_0000.jpg ~ image_0999.jpg)
│   └── val_labels.txt          # 정답 레이블 매핑 파일 (파일명 index)
├── load_imagenet_1k.py         # 데이터셋 구축 자동화 스크립트
└── README.md                   # 본 가이드 문서
```

## 📝 데이터 규격 및 레이블 매핑

*   **이미지 포맷**: 모든 이미지는 추론 성능 일관성을 위해 3채널 RGB JPEG 형식으로 저장됩니다.
*   **레이블 매핑**: `val_labels.txt`는 공백으로 구분된 `[파일명] [인덱스]` 형식을 따릅니다.
    *   예시: `image_0207.jpg 207` (Golden Retriever)
*   **인덱스 체계**: ImageNet-1K의 표준 1,000개 클래스 인덱스(0~999)를 따르며, 이는 `MobileNet_evaluator.py`에서 정확도 산출 시 활용됩니다.

## ⚠️ 주의사항

1.  **용량 관리**: 1,000개의 샘플은 약 150~200MB의 용량을 차지합니다.
2.  **데이터 무결성**: 다운로드 중 네트워크 오류가 발생할 경우 `datasets/imagenet_1k/val` 폴더를 삭제하고 다시 실행하는 것을 권장합니다.
