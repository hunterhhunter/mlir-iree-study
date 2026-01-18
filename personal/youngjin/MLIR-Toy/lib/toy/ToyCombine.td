

#ifndef TOY_COMBINE
#define TOY_COMBINE

include "mlir/IR/PatternBase.td"
include "toy/Ops.td"

// 패턴 매칭을 위한 부모 클래스
// class Pattern<
//      dag sourcePattern,                      // 입력
//      list<dag> resultPatterns,               // 출력
//      list<dag> additionalConstraints = [],   // 추가 조건
//      list<dag> supplementalPatterns = [],
//      dag nenefitsAdded = (addBenefit 0)      // 우선순위 점수
// >;

// Pat은 Pattern 상속받으면서 사용하기 쉽게 만든 래퍼 객체

// Basic Patter-Match and Rewrite
// Reshape pattern - reshape(reshape(a)) = a
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp %arg)), 
                                   (ReshapeOp $arg)>;


// MLIR의 getType은 객체명이 아니라 Element Type, Rank, Dimension을 반환
// NativeCodeCall은 C++의 함수를 호출하겠다는 의미
// 아래 함수는 $0을 $1번의 모양으로 컴파일 할 때 변경하겠다는 의미 - 아래 상수 폴딩에서 사용
def ReshapeConstant :
    NativeCodeCall<"$0.reshape(::llvm::cast<ShapedType>($1.getType()))">;

// 상수 폴딩 패턴인데 ReshapeOp:$res (ConstantOp $arg) = Reshape(constant) -> constant(constant(reshape완료))
def FoldConstantReshapeOptPattern : Pat<
    (ReshapeOp:$res (ConstantOp $arg)),
    (ConstantOp (ReshapeConstant $arg, $res))>;

// 변환시 제약 설정 여기서는 타입이 같을 때
// Cpred = C++ predicate (참/거짓 반환하는 코드 조각)
def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
// ReshapeOp의 출력: $res, 입력: $arg
// reshape(arg) -> replaceWithValue(arg)로 변환 (단 두 type(element type, Rank, dimentions)이 같을 때)
// 결국 2x2 -> 2x2로 바꾸는 연산을 제거하는 것
def RedundantReshapeOptPattern : Pat<
    (ReshapeOp:$res $arg), (replaceWithValue $arg),
    [(TypesAreIdentical $res, $arg)]>;

#endif