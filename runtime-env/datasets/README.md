# Evaluation Datasets Management

본 디렉토리는 MobileNetV2, PatchTST 및 기타 모델의 정밀도 평가에 사용되는 데이터셋과 관련 스크립트를 관리합니다.

## 📦 데이터셋 구축 방법

### 1. ImageNet-1K (Vision)
`load_imagenet_1k.py` 스크립트를 사용하여 Hugging Face로부터 ImageNet-1K 검증 세트 1,000개를 자동으로 내려받고 구조화합니다.
- **사전 요구사항**: Hugging Face 계정 / Access Token, `huggingface-cli login` 인증 완료
- **실행**:
```bash
python datasets/load_imagenet_1k.py
```

### 2. SQuAD 2.0 (NLP)
질의응답(Question Answering) 모델 평가에 사용되는 SQuAD 2.0 데이터셋의 검증 세트(`dev-v2.0.json`)를 자동으로 다운로드합니다.
- **실행**:
```bash
uv run datasets/download_squad2.py
```

### 3. ETTh1 / ETTm1 (Time Series Forecasting)
PatchTST 등 시계열 예측 모델 평가를 위해서는 논문에서 사용하는 ETDataset이 필요합니다.
직접 wget 명령어를 이용해 csv 원본 파일을 확보합니다.

- **ETTh1 다운로드**:
- [링크](https://github.com/zhouhaoyi/ETDataset/tree/main) 에 존재

## 📂 디렉토리 구조 (Directory Structure)

스크립트 실행 및 다운로드 후 다음과 같은 구조로 데이터가 배치됩니다.

```text
datasets/
├── imagenet_1k/                # ImageNet-1K 전용 네임스페이스
│   ├── val/                    # JPEG 이미지 파일
│   └── val_labels.txt          # 정답 레이블 매핑 파일 (파일명 index)
├── etth1/                      # ETTh1 전용 네임스페이스
│   ├── ETTh1.csv               # 7개 다변량 채널 시계열 데이터
│   └── .cache_npz/             # Dataloader npz 캐시 스토리지 (자동생성)
├── load_imagenet_1k.py         # 데이터셋 구축 자동화 스크립트
└── README.md                   # 본 가이드 문서
```

## 📝 데이터 규격 및 주의사항

* **ImageNet**: `val_labels.txt`는 공백 구분 형식(`이미지명 인덱스`)을 따릅니다.
* **ETTh1**: `date`를 제외한 `HUFL, HULL, MUFL, MULL, LUFL, LULL, OT` 7채널을 사용합니다.
* 데이터 파싱과 분할(Split)은 각 `Loader` 모듈 안에서 자동으로 최적화되어 처리됩니다.
