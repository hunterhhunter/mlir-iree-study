# MLIR Toy Dialect Chapter 4: Step-by-Step Implementation Guide

이 문서는 Toy 다이얼렉트의 인라이닝(Inlining)과 캐스트(Cast) 기능을 구현하기 위한 전체 과정을 10개 단계로 나열한 가이드입니다.

---

### Step 1: 인라이닝 인터페이스 정의 (`Dialect.cpp`)
범용 인라이너가 Toy 연산의 정책을 결정할 수 있도록 `DialectInlinerInterface`를 상속받아 구현합니다.

```cpp
// Dialect.cpp
struct ToyInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // 모든 Toy 연산은 인라이닝이 가능함
  bool isLegalToInline(Operation *call, Operation *callable, bool) const final { return true; }
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final { return true; }
  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final { return true; }

  // Terminator 처리: toy.return의 피연산자를 호출부 결과로 RAUW(Replace All Uses With)
  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
    auto returnOp = cast<ReturnOp>(op);
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }
};
```
---
### Step 2: 연산 인터페이스(Traits) 및 속성 추가 (`Ops.td`)
인라이너가 호출 구조를 파악할 수 있도록 표준 인터페이스를 연산에 등록합니다.
```TableGen
// Ops.td 상단
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// GenericCallOp: CallOpInterface 적용 및 표준 속성(arg_attrs, res_attrs) 추가
def GenericCallOp : Toy_Op<"generic_call", [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<Toy_Type>:$inputs,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  // ...
}
```
---
### Step 3: 다이얼렉트 초기화 시 인터페이스 등록 (`Dialect.cpp`)
생성한 인라이닝 인터페이스를 다이얼렉트에 주입합니다.

```cpp
// Dialect.cpp
void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "toy/Ops.cpp.inc"
  >();
  // 인터페이스 등록
  addInterfaces<ToyInlinerInterface>();
}
```
---
### Step 4: 호출 인터페이스 헬퍼 메서드 구현 (`Dialect.cpp`)
`DeclareOpInterfaceMethods`로 선언된 C++ 메서드들을 구체화합니다.
```cpp
// Dialect.cpp
// GenericCallOp 메서드
CallInterfaceCallable GenericCallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}
void GenericCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  (*this)->setAttr("callee", cast<SymbolRefAttr>(callee));
}
Operation::operand_range GenericCallOp::getArgOperands() { return getInputs(); }
MutableOperandRange GenericCallOp::getArgOperandsMutable() { return getInputsMutable(); }

// FuncOp 메서드 - 저는 구현을 안 했는데 실행이 되었습니다.
Region *FuncOp::getCallableRegion() { return &getBody(); }
```
---
### Step 5: 함수 심볼 가시성(Visibility) 제어 (`MLIRGen.cpp`)
`main`을 제외한 함수를 `private`으로 설정하여 사후 삭제(DCE)가 가능하게 합니다.
```cpp
// MLIRGen.cpp
mlir::toy::FuncOp mlirGen(FunctionAST &funcAST) {
  // ... (IR 생성 로직)
  if (funcAST.getProto()->getName() != "main")
    function.setPrivate();
  return function;
}
```
### Step 6: 형변환 연산 CastOp 정의 (`Ops.td`)
타입 불일치 해결을 위한 `CastOp`를 정의합니다.
```TableGen
// Ops.td
include "mlir/Interfaces/CastInterfaces.td"

def CastOp : Toy_Op<"cast", [
    DeclareOpInterfaceMethods<CastOpInterface>,
    Pure,
    SameOperandsAndResultShape
  ]> {
  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}
```
---
### Step 7: 캐스트 적합성 검사 구현 (`Dialect.cpp`)
`CastOp`가 유효한 타입 변환인지 검증하는 로직을 추가합니다.
```cpp
// Dialect.cpp
bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  auto input = llvm::dyn_cast<TensorType>(inputs.front());
  auto output = llvm::dyn_cast<TensorType>(outputs.front());
  if (!input || !output || input.getElementType() != output.getElementType())
    return false;
  return !input.hasRank() || !output.hasRank() || input == output;
}
```
---
### Step 8: 타입 구체화(Materialization) 훅 추가 (`Dialect.cpp`)
인라이닝 중 타입이 다를 경우 `CastOp`를 자동 생성하도록 인터페이스를 보완합니다.
```cpp
// Dialect.cpp 내 ToyInlinerInterface 클래스 내부
Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                     Type resultType, Location loc) const final {
  return CastOp::create(builder, loc, resultType, input);
}
```
---
### Step 9: 패스 파이프라인 구성 및 사후 정리 (`toyc.cpp`)
인라이닝 실행 및 지저분한 IR 정리를 위한 패스를 등록합니다.
```cpp
// toyc.cpp
mlir::PassManager pm(module.getContext());
pm.addPass(mlir::createInlinerPass());           // 인라이닝 및 Cast 삽입
pm.addPass(mlir::createCanonicalizerPass());     // 불필요한 Cast 제거
pm.addPass(mlir::createSymbolDCEPass());         // 미사용 private 함수 삭제
```
