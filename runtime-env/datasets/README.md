# Evaluation Datasets Management

본 디렉토리는 AI Benchmark Framework의 정밀도 평가에 사용되는 데이터셋과 관련 스크립트를 관리합니다. 모든 준비 스크립트는 `prepare_` 접두사로 통일되어 있으며, Zero-Config 실행 시 `main.py`가 자동으로 호출합니다.

## 📦 데이터셋 구축 방법

### 1. ImageNet-1K (Vision — ResNet50)
`prepare_imagenet_1k.py` 스크립트를 사용하여 Hugging Face로부터 ImageNet-1K 검증 세트 1,000개를 자동으로 내려받고 구조화합니다.
- **사전 요구사항**: Hugging Face 계정 / Access Token, `huggingface-cli login` 인증 완료
- **실행**:
```bash
python datasets/prepare_imagenet_1k.py
```

### 2. COCO128 (Vision — YOLOv5m)
YOLO 객체 탐지 모델 평가에 사용되는 COCO128 데이터셋을 자동으로 다운로드합니다.
- **실행**:
```bash
python datasets/prepare_coco128.py
```

### 3. SQuAD 2.0 (NLP — BERT QA / Llama)
질의응답(Question Answering) 모델 평가에 사용되는 SQuAD 2.0 데이터셋의 검증 세트(`dev-v2.0.json`)를 자동으로 다운로드합니다.
- **실행**:
```bash
python datasets/prepare_squad2.py
```

### 4. SST-2 Numpy (NLP — BERT 분류)
BERT 텍스트 분류 평가용 SST-2 데이터셋을 사전 토크나이즈하여 numpy 형식으로 저장합니다.
- **실행**:
```bash
python datasets/prepare_text_numpy.py
```

### 5. SQuAD Numpy (NLP — BERT QA)
BERT QA 평가용 SQuAD 데이터를 numpy 형식으로 오프라인 분해합니다.
- **실행**:
```bash
python datasets/prepare_squad_numpy.py
```

### 6. ETTh1 (Time Series — PatchTST)
PatchTST 등 시계열 예측 모델 평가를 위해 ETDataset을 자동으로 다운로드합니다.
- **실행**:
```bash
python datasets/prepare_etth1.py
```

## 📂 디렉토리 구조 (Directory Structure)

스크립트 실행 및 다운로드 후 다음과 같은 구조로 데이터가 배치됩니다.

```text
datasets/
├── imagenet_1k/                # ImageNet-1K 전용 네임스페이스
│   ├── val/                    # JPEG 이미지 파일
│   └── val_labels.txt          # 정답 레이블 매핑 파일 (파일명 index)
├── coco128/                    # COCO128 전용 네임스페이스
│   ├── images/val2017/         # JPEG 이미지
│   └── labels/val2017/         # YOLO 형식 레이블 (.txt)
├── squad2/                     # SQuAD 2.0 전용 네임스페이스
│   └── dev-v2.0.json           # 검증 세트
├── text_numpy/                 # BERT 분류용 사전 토크나이즈 데이터
│   └── *.npz                   # 토큰 ID / 어텐션 마스크 / 레이블
├── etth1/                      # ETTh1 전용 네임스페이스
│   ├── ETTh1.csv               # 7개 다변량 채널 시계열 데이터
│   └── .cache_npz/             # Dataloader npz 캐시 스토리지 (자동생성)
├── prepare_imagenet_1k.py      # ImageNet-1K 다운로드 스크립트
├── prepare_coco128.py          # COCO128 다운로드 스크립트
├── prepare_squad2.py           # SQuAD 2.0 다운로드 스크립트
├── prepare_squad_numpy.py      # SQuAD numpy 변환 스크립트
├── prepare_text_numpy.py       # SST-2 numpy 변환 스크립트
├── prepare_etth1.py            # ETTh1 다운로드 스크립트
└── README.md                   # 본 가이드 문서
```

## 📝 데이터 규격 및 주의사항

* **ImageNet**: `val_labels.txt`는 공백 구분 형식(`이미지명 인덱스`)을 따릅니다.
* **COCO128**: YOLO 형식 레이블(`.txt`) 사용, 각 줄은 `class cx cy w h` 정규화 좌표입니다.
* **ETTh1**: `date`를 제외한 `HUFL, HULL, MUFL, MULL, LUFL, LULL, OT` 7채널을 사용합니다.
* 데이터 파싱과 분할(Split)은 각 `Loader` 모듈 안에서 자동으로 최적화되어 처리됩니다.
