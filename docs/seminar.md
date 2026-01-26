# MLIR & IREE Seminar: Toy Chapter 4 정리

## 📂 발표 자료
*   **Google Drive Link:** [발표 자료 링크 (수정 필요)](https://drive.google.com/...)
*   **주제:** MLIR Interfaces, Inlining, and Intraprocedural Shape Inference
*   **일시:** 2026년 1월 25일

---

## 📽️ 세션 핵심 내용

### 1. 인터페이스(Interface)의 도입 배경
*   **문제점:** Dialect마다 최적화 로직을 개별 구현할 경우 막대한 코드 중복 발생.
*   **해결책:** MLIR의 **Interfaces**를 사용하여 변환기(Transformation)가 Dialect의 내부 구현을 몰라도 필요한 정보를 얻을 수 있는 "불투명한 연결(Opaque Hooking)" 구조 설계.

### 2. 함수 인라이닝 (Function Inlining) 구현
*   **목표:** 전역적인 최적화 및 모양 추론을 위해 모든 함수 호출을 본문으로 대체.
*   **핵심 단계:**
    *   `DialectInlinerInterface` 구현: 인라이닝 허용 여부 및 터미네이터(`toy.return`) 처리 로직 정의.
    *   `toy.cast` 연산 정의: 인라이닝 시 발생하는 타입 불일치(Ranked vs Unranked) 해결을 위해 `materializeCallConversion` 훅 사용.
    *   함수 가시성 조정: `main`을 제외한 함수를 `private`으로 설정하여 인라이닝 후 불필요한 코드 자동 제거 유도.

### 3. 모양 추론 (Shape Inference) 알고리즘
*   **특징:** Intraprocedural(함수 내부) 모양 전파 수행.
*   **작업 목록(Worklist) 알고리즘:**
    1.  결과 모양이 확정되지 않은 모든 연산을 목록에 수집.
    2.  목록 내 연산 중 모든 입력 모양이 확정된 연산을 선택.
    3.  `ShapeInference` 인터페이스의 `inferShapes()`를 호출하여 출력 모양 확정.
    4.  목록이 비워질 때까지 반복 실행.

### 4. 패스 매니저(Pass Manager) 파이프라인
*   **구성 순서:** `Inliner` -> `ShapeInference` -> `Canonicalizer` -> `CSE`.
*   **최적화:** `pm.nest<toy::FuncOp>()`를 활용하여 함수 단위 패스들을 명시적으로 중첩시켜 멀티스레드 병렬 실행 유도.

---

## 🛠️ 주요 코드 위치 (Ch4)
*   `include/toy/Ops.td`: 연산 및 인터페이스 선언 (ODS)
*   `mlir/Dialect.cpp`: 인터페이스 실제 구현 및 Dialect 등록
*   `mlir/ShapeInferencePass.cpp`: 모양 추론 패스 알고리즘 구현
*   `toyc.cpp`: 전체 컴파일러 파이프라인 제어 및 진입점

---

## 🔗 참고 자료
*   [MLIR Official Tutorial: Chapter 4](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-4/)
*   [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)
