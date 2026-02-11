# MLIR Toy Tutorial Chapter 6: Lowering to LLVM and CodeGeneration

## 6장 목표
- 5장까지는 toy -> Affine 변환으로 MLIR 내부의 변환을 다뤘다면 6장에서는 toy -> LLVMIR, Affine -> LLVMIR 변환으로 toy -> LLVMIR의 변환을 다룸
- toy -> LLVMIR 변환을 toy.print -> printf 변환을 통해 학습
- MLIR의 LLVMIR -> LLVM으로의 변환을 배움

## Lower To LLVM 워크플로우 
### 1단계: 5장에서 Affine으로 변환하지 않은 toy.print를 LLVMIR로 변환하기 (LowerToLLVM.cpp)
1. OpConversionPattern를 상속받는 PrintOpLowering 객체 선언하기
2. LLVMIR로 printOp를 낮추기 위한 헬퍼 메서드 getPrintfType, getOrInsertPrintf, getOrCreateGlobalString 선언
3. PrintOpLowering의 matchAndRewrite 메서드 작성하기

### 2단계: LLVM Lowering Pass 등록하기 (LowerToLLVM.cpp)
1. Lowering Pass 등록을 위해 PassWrapper를 상속받는 ToyToLLVMLoweringPass 객체 선언
2. ToyToLLVMLoweringPass의 runOnOperation 메서드 구현
    - MLIR은 Affine -> ... -> LLVM으로의 lowering은 구현되어있어 toy -> Affine을 구현하면 나머지는 메서드 만으로 lowering이 가능
    - 반복문 변환
        - Affine -> SCF 변환 등록 (populateAffineToStdConversionPatterns(patterns))
        - SCF -> Control Flow 변환 등록 (populateSCFToControlFlowConversionPatterns(patterns))
        - CF -> LLVM 변환 등록 (cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns))
    - Arith 변환
        - Arith -> LLVM 변환 등록 (mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns))
    - MemRef 변환
        - Memref -> LLVM 변환 등록 (populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns))
    - Func 변환
        - Func -> LLVM 변환 등록 (populateFuncToLLVMConversionPatterns(typeConverter, patterns))
    - toy 변환
        - Toy.print -> LLVM 변환 등록 (patterns.add<PrintOpLowering>(&getContext()))
3. Pass 등록을 위해 ToyToLLVMLoweringPass의 unique_ptr로 반환하는 함수 mlir::toy::createLowerToLLVMPass 작성
4. include/toy/Passes.h에 위 함수를 등록

### 3단계: 실제 lowering 실행을 위한 toyc.cpp 수정
1. LLVMIR로 코드를 변환하는 dumpLLVMIR 함수 작성
2. JIT 실행을 위한 runJit 함수 작성
3. dumpMLIR 함수에 LLVMDialect 등록 및 LLVM 인라이닝 등록
4. LLVM이나 JIT를 하기 위해서 Affine lowering을 우선하는 로직을 작성
5. -emit= 실행 인자로 주기 위한 enum Action에 LLVMIR, JIT 추가
6. emitAction 함수에 인자에 걸맞는 인자-실행함수 등록

## 알아야 할 핵심 내용
### 1. PrintOpLowering의 핵심 역할
- Type Conversion: PrintOp는 단순히 연산만 바꾸는게 아니라 입력으로 들어오는 MemRef 타입을 LLVM이 이해할 수 있는 구조체로 해석해야함. - matchAndRewrite
- Side Effect: printf의 호출은 외부 시스템 함수를 부르는 것으로, LLVM::CallOp를 생성시 Symbol Table에서 해당 함수를 찾거나 생성하는 로직
### 2. Lowering Pass의 typeConverter
- MLIR의 고수준 타입(MemRef, index, F64)을 LLVM의 저수준 타입(Ptr, I64, Double)으로 1:1 매핑해주는 엔진
- 모든 populate 함수들이 이 type_converter를 인자로 받는 이유
### 3. toyc.cpp의 변환 파이프라인
- ModuleOp의 특수성: ConversionTarget을 설정할 때 대부분 Illegal로 설정하지만, ModuleOp와 LLVMDialect만은 Legal로 남겨두어야함. 그래야 모듈 안에 LLVM연산을 담을 수 있음.
- dumpLLVMIR 함수에서 LLVM IR로 넘어간 뒤에도 makeOptimizaingTransformer를 통해 LLVM 레벨의 최적화로 이어지도록 해야함.
---

## 6장의 MLIR 컴파일 레이어
1. Toy Dialect Layer: 고수준 연산 (toy.print, toy.mul)
2. Conversion Layer: PrintOpLowering, typeConverter
3. LLVM Dialect Layer(MLIR LLVM 방언): llvm.call, llvm.getElementptr
4. Translation Layer(CodeGen): mlir::translateModuleToLLVMIR
5. Execition Layer(JIT): mlir::ExecutionEngine -> **Machine Code**
---

## MemRef 중심의 코드 비교 (MLIR-LLVM Dialect / LLVM)
자세한 코드는 ch6-llvm_output.txt, ch6-mlir-llvm_output.txt에서 확인할 수 있습니다.
### 1. MemRef 구조체의 변화
| 항목 | MLIR (LLVM Dialect) | 진짜 LLVM IR |
| :--- | :--- | :--- |
| **타입 표현 (Struct)** | `!llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>` | `{ ptr, ptr, i64, [2 x i64], [2 x i64] }` |
| **초기화 (Poison)** | `llvm.mlir.poison` | `poison` |
| **상수 선언 (Constant)** | `llvm.mlir.constant(1.0 : f64) : f64` | `double 1.000000e+00` |
| **포인터 연산 (GEP)** | `llvm.getelementptr %ptr[%idx] : (...) -> !llvm.ptr, f64` | `getelementptr inbounds double, ptr %ptr, i64 %idx` |
| **제어 흐름 (Branch)** | `llvm.br ^bb1(%arg : i64)` (Block Arguments) | `phi i64 [ %val, %label ]` (Phi Nodes) |

### 2. 메모리 할당(malloc)의 계산 방식
- MLIR-LLVM Dialect
    1. llvm.mlir.zero( 0번지 포인터 생성)
    2. llvm.getelementptr %0[6] (0번지에서 f64타입을로 6칸 뒤의 주소 계산)
    3. llvm.ptrtoint (그 주소를 정수로 변환 = 48바이트)
    4. llvm.call @malloc(%size) 호출
    - 타입 크기를 수동으로 계산하는 과정이 연산으로 남아있음
- LLVM IR
    1. %1 = call ptr @malloc(i64 48)
    - 번역 과정에서 최적화가 일어나, 주소 계산이 사라지고 48이라는 상수로 합쳐짐