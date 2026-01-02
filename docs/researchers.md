# ML Compiler & NPU Researchers

머신러닝 컴파일러(ML Compiler), NPU 아키텍처, 그리고 시스템 소프트웨어를 연구하는 국내외 연구자 및 연구실을 정리한 리스트입니다.

## 🇰🇷 Domestic (South Korea)

### 1. POSTECH (포항공과대학교)
* **[Prof. Kwang-sun Kim (김광선 교수님)](https://cal.postech.ac.kr/)** - *Computer Architecture Laboratory (CAL)*
    * **Research Areas:** Computer Architecture, AI Accelerators, Simulation Frameworks.
    * **Key Works:**
        * **[`pytorch-sim`](https://github.com/PSAL-POSTECH/PyTorchSim):** PyTorch 기반의 아키텍처/시스템 시뮬레이션 프레임워크.
        * DRAM-PIM(Processing-in-Memory) 아키텍처 및 시스템 최적화.

### 2. KAIST (한국과학기술원)
* **[Prof. Jeehoon Kang (강지훈 교수님)](https://sf.snu.ac.kr/jeehoon.kang/)** - *Concurrent & Parallel Programming Lab*
    * **Research Areas:** 컴파일러 최적화, 병렬 프로그래밍, SW/HW 검증.
    * **Key Works:** 텐서 컴파일러의 수학적 검증 및 최적화 연구.

### 3. SNU (서울대학교)
* **[Prof. Bernhard Egger](https://csap.snu.ac.kr/)** - *Computer Systems and Platforms Lab (CSAP)*
    * **Research Areas:** Compilers, Runtime Systems.
    * **Key Works:** 이기종 컴퓨팅을 위한 런타임 스케줄링, 컴파일러 최적화.
* **[Prof. Jin-Soo Kim (김진수 교수님)](http://csl.snu.ac.kr/)** - *Computer Systems Lab*
    * **Research Areas:** Operating Systems, Storage Systems for AI.

### 4. UNIST (울산과학기술원)
* **[Prof. Woongki Baek (백웅기 교수님)]** - *DCS Lab*
    * **Research Areas:** 효율적인 런타임 시스템, 모바일/클라우드 컴퓨팅 최적화.

---

## International (Global)

### 1. Key Figures in ML Compilers
* **[Chris Lattner](https://nondot.org/sabre/)** (Modular AI, ex-Google/Apple)
    * **Contribution:** **LLVM**, Clang, Swift, **MLIR**의 창시자. 현재 Modular에서 `Mojo` 언어 및 차세대 AI 엔진 개발 중.
* **[Tianqi Chen](https://tqchen.com/)** (CMU / OctoML)
    * **Contribution:** **TVM (Apache TVM)**, XGBoost, MXNet 창시자. ML 컴파일러 자동 최적화(AutoTVM) 분야의 선구자.
* **[Jonathan Ragan-Kelley](https://people.csail.mit.edu/jrk/)** (MIT)
    * **Contribution:** **Halide** (이미지 처리용 DSL) 창시자. Exo 컴파일러 등 DSL 기반의 최적화 연구.

### 2. Leading Universities & Labs
* **University of Washington (SAMPL Group):** TVM이 탄생한 곳. 딥러닝 시스템과 컴파일러 연구의 성지.
* **UC Berkeley (RISE Lab / Sky Lab):** Ray, Clipper 등 분산 ML 시스템 연구.
* **UIUC (Vikram Adve):** LLVM 프로젝트가 시작된 곳.