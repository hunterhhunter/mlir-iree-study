# MLPerf Inference Architecture

> MLPerf Inference v4.x 기준. 공식 레포: [mlcommons/inference](https://github.com/mlcommons/inference)

---

## 1. 전체 계층 구조

```mermaid
graph TB
    subgraph "MLPerf Inference Stack"
        direction TB

        subgraph "Measurement Layer"
            LG["🔬 LoadGen (C++ Library)\n─────────────────\n• 쿼리 발행 스케줄러\n• 나노초 정밀 시간 측정\n• 통계 검증 (QPS, Latency)\n• 결과 로그 생성"]
        end

        subgraph "Interface Layer"
            SUT["📦 SUT\n(System Under Test)\n─────────────────\n• issue_queries() 콜백\n• flush_queries() 콜백\n• 모델별로 직접 구현"]
            QSL["📂 QSL\n(Query Sample Library)\n─────────────────\n• load_query_samples() 콜백\n• unload_query_samples() 콜백\n• 인덱스 기반 샘플 접근"]
        end

        subgraph "Implementation Layer"
            PP["⚙️ Pre/Post Processing\n─────────────────\n• Resize / Crop / Normalize\n• NMS / Decode / Softmax\n• 모델별 참조 구현 제공"]
            BE["🚀 Backend\n─────────────────\n• TensorRT\n• ONNX Runtime\n• TFLite\n• PyTorch\n• (Custom NPU)"]
        end

        subgraph "Data & Model Layer"
            DS["🗄️ Dataset\n─────────────────\n• ImageNet (분류)\n• COCO (탐지)\n• LibriSpeech (음성)\n• SQuAD (NLP)"]
            MDL["🧠 Model\n─────────────────\n• ResNet-50\n• SSD-MobileNet\n• BERT-Large\n• GPT-J 6B\n• Llama 2"]
        end
    end

    LG -->|"QuerySample 발행"| SUT
    LG -->|"샘플 인덱스 지정"| QSL
    SUT -->|"추론 요청"| BE
    QSL -->|"데이터 로드"| PP
    PP -->|"텐서 반환"| SUT
    BE -->|"결과 반환\nQuerySampleResponse"| LG
    DS -->|"raw 이미지/텍스트"| PP
    MDL -->|"가중치/그래프"| BE

    style LG fill:#1a237e,color:#fff
    style SUT fill:#1565c0,color:#fff
    style QSL fill:#1565c0,color:#fff
    style PP fill:#2e7d32,color:#fff
    style BE fill:#e65100,color:#fff
    style DS fill:#4a148c,color:#fff
    style MDL fill:#4a148c,color:#fff
```

---

## 2. LoadGen 동작 방식 (시나리오별)

```mermaid
sequenceDiagram
    participant LG as LoadGen (C++)
    participant QSL as Query Sample Library
    participant SUT as System Under Test
    participant BE as Backend

    Note over LG: TestSettings 로드<br/>(시나리오, target_qps, count...)

    LG->>QSL: LoadSamplesToRam(sample_list)
    QSL-->>LG: (완료)

    rect rgb(20, 60, 100)
        Note over LG,BE: ── 측정 구간 시작 ──

        loop 각 쿼리 발행
            LG->>SUT: IssueQuery(QuerySample[])
            SUT->>BE: forward(input_tensor)
            BE-->>SUT: output_tensor
            SUT->>LG: QuerySamplesComplete(Response[])
        end

        Note over LG,BE: ── 측정 구간 종료 ──
    end

    LG->>QSL: UnloadSamplesFromRam(sample_list)
    LG->>LG: 결과 로그 저장 (mlperf_log_*.txt)
```

---

## 3. 4가지 측정 시나리오

```mermaid
graph LR
    subgraph "SingleStream"
        direction TB
        SS1[쿼리 1] --> SS2[쿼리 2] --> SS3[쿼리 3] --> SS4[...]
        SSM["측정값: 90th percentile Latency\n기준: ≤ target_latency_ns"]
    end

    subgraph "MultiStream"
        direction TB
        MS1["쿼리 묶음 1\n(N samples 동시)"] --> MS2["쿼리 묶음 2"] --> MS3[...]
        MSM["측정값: 99th percentile Latency\n기준: N=8 samples 처리 시간"]
    end

    subgraph "Offline"
        direction TB
        OF1["전체 데이터셋\n한 번에 투입"] --> OF2["결과 수집"]
        OFM["측정값: Throughput (samples/sec)\n기준: ≥ target_qps"]
    end

    subgraph "Server"
        direction TB
        SV1["Poisson 분포\n쿼리 스트림"] --> SV2["비동기 처리"] --> SV3["응답 반환"]
        SVM["측정값: 99th percentile Latency\n기준: ≤ target_latency_ns @ target_qps"]
    end
```

| 시나리오 | 대상 환경 | 핵심 지표 |
|--|--|--|
| **SingleStream** | Edge 디바이스 (카메라, 로봇) | P90 Latency |
| **MultiStream** | 멀티카메라, 센서 융합 | P99 Latency |
| **Offline** | 배치 처리, 데이터센터 | Throughput (QPS) |
| **Server** | 실시간 API 서버 | P99 Latency @ QPS |

---

## 4. 디렉토리 구조 (공식 레포)

```
mlcommons/inference/
├── loadgen/                    # C++ LoadGen 코어 + Python 바인딩
│   ├── loadgen.cc              # 시나리오별 쿼리 스케줄러
│   ├── query_sample_library.h  # QSL 인터페이스 정의
│   └── system_under_test.h     # SUT 인터페이스 정의
│
├── vision/
│   └── classification_and_detection/
│       └── python/
│           ├── dataset.py      # Dataset ABC + Imagenet / COCO 구현
│           ├── main.py         # SUT + QSL 콜백 조립 진입점
│           └── backend_*.py    # TF / ONNX / TFLite 백엔드
│
├── language/                   # BERT, GPT-J, Llama 등 NLP 태스크
├── speech_recognition/         # RNN-T 등 음성 인식 태스크
│
└── compliance/                 # 공식 제출 검증 스크립트
    └── nvidia/
        └── TEST01/ ...         # 제출 규정 준수 테스트
```

---

## 5. Antigravity ↔ MLPerf 컴포넌트 매핑

```mermaid
graph TB
    subgraph "Antigravity Framework (자체)"
        A_DL["DataLoader ABC\n+ load_by_index()"]
        A_RT["Runtime ABC"]
        A_EV["Evaluator ABC"]
        A_BR["BenchmarkRunner\n(빠른 실험 모드)"]
        A_LGA["LoadGenAdapter\n(어댑터 계층)"]
        A_C["Compiler ABC\n(자체 고유 계층)"]
    end

    subgraph "MLPerf (공식 측정 모드)"
        M_QSL["QSL\nload_query_samples()"]
        M_SUT["SUT\nissue_queries()"]
        M_LG["LoadGen C++\n(시간 측정 엔진)"]
        M_ACC["accuracy.py\n(정확도 검증)"]
    end

    A_DL -->|"위임"| M_QSL
    A_RT -->|"위임"| M_SUT
    M_QSL --> A_LGA
    M_SUT --> A_LGA
    A_LGA --> M_LG
    A_EV -.->|"정확도 로직 참조"| M_ACC

    A_C -.->|"MLPerf에는\n없는 계층"| A_RT

    style A_LGA fill:#2196F3,color:#fff
    style A_C fill:#FF9800,color:#fff
    style M_LG fill:#1a237e,color:#fff
```

| Antigravity | MLPerf 대응 | 비고 |
|--|--|--|
| `DataLoader.load_by_index()` | `QSL.LoadSamplesToRam()` | 인덱스 기반 샘플 접근 |
| `Runtime.run()` | `SUT.IssueQuery()` 내부 | 추론 실행 |
| `BenchmarkRunner` | `LoadGen` + `main.py` | 단, LoadGen이 C++ 정밀도 |
| `Evaluator` | `accuracy.py` | 전/후처리 알고리즘 참조 |
| `Compiler` ABC | **(없음)** | Antigravity 고유 계층 |
| `LoadGenAdapter` | `SUT` + `QSL` 구현체 | 어댑터 브릿지 |

---

## 6. 공식 제출(Submission) 구조

```
submission/
├── systems/
│   └── {system_name}.json          # HW 스펙 (CPU, RAM, NPU 등)
├── measurements/
│   └── {system}/{model}/{scenario}/
│       ├── mlperf.conf             # LoadGen 설정
│       └── user.conf               # 사용자 오버라이드
└── results/
    └── {system}/{model}/{scenario}/
        ├── performance/
        │   └── run_1/
        │       └── mlperf_log_summary.txt   # ✅ 공식 측정 결과
        └── accuracy/
            └── mlperf_log_accuracy.json     # ✅ 정확도 검증 결과
```

> [!NOTE]
> Antigravity에서 공식 제출을 목표로 한다면 `LoadGenAdapter`가 위 `measurements/` + `results/` 디렉토리 구조를 자동 생성하도록 Phase 4에서 구현하면 됩니다.
