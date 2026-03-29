# Scripts
컴파일, 벤치마크, 프로파일링을 위한 유틸리티 스크립트 모음입니다.
- iree-*.py: IREE 컴파일 및 실행 워크플로우
- preprocess_*.py: 데이터 전처리 로직

---

## 🚀 전처리 스크립트: `tokenize_to_numpy.py`
자연어 평가 모델을 타겟팅 하드웨어에서 벤치마크하기 전, 원시 텍스트 데이터셋을 **정적 차원의 순수 Numpy 배열** 형식으로 오프라인 추출해 두는 유틸리티.

### 📌 사용 예시

**1. 기본 HuggingFace 데이터셋 굽기 (SST-2)**
```bash
python scripts/tokenize_to_numpy.py \
  --model-id bert-base-uncased \
  --dataset-name glue \
  --dataset-config sst2 \
  --split validation \
  --text-column sentence \
  --label-column label \
  --seq-len 128 \
  --output-dir datasets/sst2_numpy
```

**2. 규격이 전혀 다른 타사 HuggingFace 데이터셋 굽기 (Rotten Tomatoes 등)**
(텍스트가 담긴 기둥 이름이 `sentence`가 아니라 `text`이거나, 시퀀스 길이를 `64`로 줄이고 싶을 때)
```bash
python scripts/tokenize_to_numpy.py \
  --dataset-name rotten_tomatoes \
  --dataset-config default \
  --split validation \
  --text-column text \
  --label-column label \
  --seq-len 64 \
  --output-dir datasets/tomato_numpy
```

**3. 로컬 보안 데이터 (CSV 파일) 굽기**
HuggingFace 접속 없이, 내 컴퓨터에 있는 `csv` 파일을 직접 구워야 할 때 사용.
```bash
python scripts/tokenize_to_numpy.py \
  --csv-file /절대경로/my_custom_review.csv \
  --text-column "고객리뷰내용" \
  --label-column "긍부정01정답" \
  --seq-len 256 \
  --output-dir datasets/custom_csv_numpy
```

### ⚙️ 지원 파라미터
| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--model-id` | `bert-base-uncased` | 사용할 토크나이저의 HF 모델 ID |
| `--seq-len` | `128` | NPU/IREE 최적화를 위한 정적 패딩 고정 길이 |
| `--csv-file` | `""` | 사용할 로컬 CSV 파일의 절대 경로 (이 값이 있으면 HF 다운로드 무시) |
| `--dataset-name` | `glue` | 허깅페이스 데이터셋 이름 |
| `--dataset-config` | `sst2` | 허깅페이스 데이터셋 하위 설정 이름 |
| `--split` | _(동적 할당)_ | 데이터셋 범주 (값을 안 주면 HF는 `validation`, CSV는 `train` 자동 할당) |
| `--text-column` | `sentence` | 문장(Text)이 들어있는 데이터 열의 명칭 |
| `--label-column` | `label` | 정답(Label)이 들어있는 데이터 열의 명칭 |
| `--output-dir` | `../datasets/baked_numpy` | 구워진 `input_ids.npy` 파일 등 3종이 저장될 디렉토리 경로 |
