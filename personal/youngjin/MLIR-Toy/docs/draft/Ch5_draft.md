
5장 목표
현재까지는 단순히 toy -> mlir로의 변환과 mlir 표현에서의 최적화를 했다면
5장에서는 mlir -> llvm 사이에 존재하는 Affine Dialect를 사용해 
텐서(Tensor)가 실제 메모리(MemRef)로 전환되는 과정을 학습

toy.mul 연산을 메모리 버퍼를 순회하는 for 루프로 구체화하는 과정이라고 보면됨.

MLIR의 DialectConversion 프레임워크가 허용X 연산 -> 허용 연산으로 변환할 수 있게 함.

우선 첫 단계는 mlir의 PassManager에 lowering을 등록해야하므로
LowerToAffineLopps.cpp 파일에 4장처럼 PassWrapper를 상속받는 ToyToAffineLoweringPass 구조체를 선언
그리고 ToyToAffineLoweringPass에 runOnOperation() 메서드를 구현
이 내부에는 최종 타겟의 등록, 어떤 방언이 쓰이는지, 쓰이는 불법(Toy) 방언은 뭔지
이걸 재작성해야하는지, 그중 어떤 패턴을 재작성 해야하는지(RewirtePatternSet)
그리고 applyPartialConversion으로 변환 시도
그리고 우선 Transpose만 Affine 변환 시도

두번째 단계는 메모리와 관련된 처리인데
convertTensorToMemRef 함수는 텐서 -> MemRef 로의 타입 변환을 도와줌.
insertAllocAndDealloc 함수는 런타임에서 메모리 할당이 필요한 시점(번수 선언 등)에
컴파일링하는 도중 이 함수를 호출해 IR 가장 앞에 메모리 할당하는 코드 삽입 + IR 가장 뒤에 메모리 해제하는 코드 삽입을 자동으로 해줌.

그리고 LowerToAffineLoops.cpp에 unique한 Pass 객체를 생성해 반환하는 createLowerToAffinePass 함수구현

Passes.h에 이 함수 등록

Cmake 수정 후 빌드완료했으나 나중에 사용하지 않아 자동으로 사라짐...

나머지 연산들도 변환하기로 함.

나머지 연산 변환 후 오타들 잡아내면 -emit=mlir-affine 옵션으로 아핀 변환이 가능함

이 때 Ops.td의 PrintOp에  let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input); 로 입력값의 허용 타입을 확장해주어야한다.



내가 5장을 하며 깨달은 점
- 재귀 알고리즘이 실제 문제를 해결하는데 어떻게 쓰이는지를 깨달은 점 (생가고 못했음)
- 최적화 패스는 PassWrapper를 상속받아 구현한다는 점
- lowering은 OpConversionPattern을 상속받아 구현한다는 점
- lowering이 코드 재작성으로 3장 최적화때와 마찬가지로 matchAndRewrite 함수를 override한다는점
- rewrite할 때 중요한건 함수의 재작성에 있어 재작성된 함수에 입력을 잘 매핑시켜주는것

- 단순히 코드만 lowering만 한게 아니라 수학적 최적화가 가능한 형태(Affine)로 바꾼것 (tiling loop Fusion)
- 값(SSA)에서 메모리(MemRef)로의 변환: 원래 다루던 Tensor는 값이 한 번 정해지면 값이 바뀌지 않지만, 메모리는 store로 내용을 언제든 바꿀 수 있음. => 추상적인 수학 연산 -> 구체적인 컴퓨터 메모리 제어 명령으로 변환
- lowering 할 때 legal, illegal의 등록
- lowering 중 type Conversion 문제: toy.transpose -> MemRef로 바뀌었음에도 toy.print는 여전히 Tensor를 원했음. 그래서 서로 다른 타입을 연결하기 위해 cast를 끼워넣음. -> 그러지 말고 모든 연산이 lowering된 타입을 지원하도록 함.