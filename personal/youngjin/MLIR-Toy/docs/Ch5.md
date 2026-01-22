# MLIR Toy Tutorial Chapter 5: Partial Lowering: Affine Dialect
## 5장 목표
- 추상 -> 구체: 지금까지는 toy -> MLIR 변환 + 최적화였다면, 이제는 MLIR -> LLVM을 위한 중간단계인 Affine Dialect를 도입
- Tensor -> MemRef: 값이 불변하는 추상 Tensor 타입을, 실제 메모리 주소를 가리키는 MemRef로 변환
- 구체화: toy.mul 연산을 실제 메모리 버퍼를 순회하는 affine.for 루프로 구체화

## Lowering 워크플로우
### 1단계: Pass 및 변환 타겟 설정 (LowerToAffineLoops.cpp, passes.h)
1. Pass 선언: PassWrapper를 상속받아 ToyToAffineLoweringPass 구조체를 선언
2. 타겟 정의: Legal(Affine, Arith, Func, MemRef) 방언, Illegal(Toy) 방언, Dynamically Legal(toy.print)(조건부 합법) 방언을 정의
3. 패턴 등록: RewritePatternset에 변환 패턴을 등록
4. ToyToAffineLoweringPass를 Passes.h에 등록(toyc.cpp에서 사용할 수 있도록 공개)

### 2단계: 메모리 관리 전략 (insertAllocAndDealloc) (LowerToAffineLoops.cpp)
1. 타입 변환: convertTensorToMemRef함수가 tensor<2x3xf64> -> memref<2x3xf64>로 변환
2. Alloc 삽입: 연산 결과값을 저장할 버퍼를 만들기 위해 memref.alloc를 블록 최상단에 삽입
3. Dealloc 삽입: 메모리 누수를 막기 위해 memref.dealloc을 블록 최하단(Terminator 직전)에 삽입

### 3단계: 패턴 재작성 (OpConversionPattern) (LowerToAffineLoops.cpp)
1. matchAndRewrite: 변환할 연산(op)과 이미 변환된 피연산자(adapter)를 인자로 받음.
2. Loop Nesting: affine.buildAffineLoopNest로 텐서의 차원만큼 루프를 돌게됨
3. Load/Store: adapter에서 값을 읽고(affine.load), 연산(arith.mulf) 후, 할당된 메모리에 쓰기(affine.store)

## 코드 비교: Raw vs Opt
-opt 옵션 추가하면 최적화가 일어나는데 다른 부분은 다 동일하나 transpose(b) -> b * b의 흐름이 통합되어 transpose 하는 동시에 그 인자를 곱하여 그 자리에 저장함.

원래 코드
```mlir
// transpose 루프
affine.for %arg0 = 0 to 3 {
    affine.for %arg1 = 0 to 2 {
    %0 = affine.load %alloc_6[%arg1, %arg0] : memref<2x3xf64>
    affine.store %0, %alloc_5[%arg0, %arg1] : memref<3x2xf64>
    }
}
// Mul 루프
affine.for %arg0 = 0 to 3 {
    affine.for %arg1 = 0 to 2 {
    %0 = affine.load %alloc_5[%arg0, %arg1] : memref<3x2xf64>
    %1 = arith.mulf %0, %0 : f64
    affine.store %1, %alloc[%arg0, %arg1] : memref<3x2xf64>
    }
}
```

-opt 코드
```mlir
    affine.for %arg0 = 0 to 3 {
      affine.for %arg1 = 0 to 2 {
        %0 = affine.load %alloc_6[%arg1, %arg0] : memref<2x3xf64>
        // Mul loop에서 %alloc_5을 load하는 과정이 사라짐
        affine.store %0, %alloc_5[%arg0, %arg1] : memref<3x2xf64>
        // 읽은 값으로 바로 곱셈 수행
        %1 = arith.mulf %0, %0 : f64
        affine.store %1, %alloc[%arg0, %arg1] : memref<3x2xf64>
      }
    }
```

## 인사이트
- 재귀 알고리즘이 실제 문제를 해결하는데 어떻게 쓰이는지를 깨달은 점 (생가고 못했음)
- 최적화 패스는 PassWrapper를 상속받아 구현한다는 점
- lowering은 OpConversionPattern을 상속받아 구현한다는 점
- lowering이 코드 재작성으로 3장 최적화때와 마찬가지로 matchAndRewrite 함수를 override한다는점
- rewrite할 때 중요한건 함수의 재작성에 있어 재작성된 함수에 입력을 잘 매핑시켜주는것
- 단순히 코드만 lowering만 한게 아니라 수학적 최적화가 가능한 형태(Affine)로 바꾼것 (tiling loop Fusion)
- 값(SSA)에서 메모리(MemRef)로의 변환: 원래 다루던 Tensor는 값이 한 번 정해지면 값이 바뀌지 않지만, 메모리는 store로 내용을 언제든 바꿀 수 있음. => 추상적인 수학 연산 -> 구체적인 컴퓨터 메모리 제어 명령으로 변환
- lowering 할 때 legal, illegal의 등록
- lowering 중 type Conversion 문제: toy.transpose -> MemRef로 바뀌었음에도 toy.print는 여전히 Tensor를 원했음. 그래서 서로 다른 타입을 연결하기 위해 cast를 끼워넣음. -> 그러지 말고 모든 연산이 lowering된 타입을 지원하도록 함.