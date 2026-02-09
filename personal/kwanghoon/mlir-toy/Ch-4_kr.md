# 4장: 인터페이스로 일반 변환 활성화하기

[TOC]

## 배경: 확장 가능한 IR 다루기

MLIR은 방언을 통해 다양한 추상화 수준을 표현할 수 있습니다. 앞에서 정의한 토이 방언이 한 예입니다. 방언마다 표현하는 추상화는 다르지만, 공통적으로 수행하고 싶은 변환과 분석이 존재합니다. 이를 무턱대고 각 방언마다 구현하면 내부 알고리즘이 거의 동일한 탓에 많은 코드가 중복됩니다. 대신 변환이 토이 같은 방언에 불투명하게 연결해 필요한 정보를 얻을 수 있으면 좋겠습니다.

MLIR은 [앞 장](Ch-3_kr.md)에서 본 것처럼, 연산에 훅(`getCanonicalizationPatterns`)을 등록해 항상 사용할 수 있는 핵심 변환들을 제공합니다. 하지만 이런 훅 방식은 확장성이 떨어집니다. 그래서 표현만큼이나 인프라도 확장 가능하도록 [인터페이스](../../Interfaces.md)라는 일반적 해결책이 설계되었습니다. 인터페이스는 방언과 연산이 변환·분석에 정보를 제공하는 범용 메커니즘입니다.

## 코드 생성을 위한 형태 추론 준비

현재 토이 IR은 일반 텐서를 다루며, 상수를 초기화할 때를 제외하면 텐서 형태를 알 수 없습니다. 이는 최적화와 코드 생성을 복잡하게 만듭니다. 다행히 계산을 따라 형태를 전파하면 결국 모두 알 수 있습니다. 문제는 사용자 정의 제네릭 함수 호출을 어떻게 처리하느냐입니다. 호출 지점마다 다른 형태가 도출될 수 있습니다. 인자 타입에 기반한 기호 추론을 할 수도 있지만, 언어에 제어 흐름이 더 들어가면 일반화하기 어렵습니다. 또 다른 방법은 함수 특수화로, 새로운 인자 형태가 나오면 호출된 함수를 복제해 특수화하는 방식입니다. 토이에서는 모든 함수 호출을 인라인한 뒤, 함수 내부에서 형태를 전파하는 접근을 택합니다.

### 인라이닝

토이 전용 인라이너를 작성할 수도 있지만, 원하는 복잡도에 따라 꽤 까다로워집니다. 비용 모델을 무시하더라도, 구조적 변환만으로도 처음부터 구현하긴 어렵습니다. 다행히 MLIR은 방언이 끼어들 수 있는 일반 인라이너를 제공합니다. 토이에서 해야 할 일은 인라이너가 연결할 수 있도록 [인터페이스](../../Interfaces.md)를 제공하는 것입니다.

먼저 토이 방언에서 어떤 연산이 인라이닝 가능한지 제약을 정의해야 합니다. 이 정보는 [방언 인터페이스](../../Interfaces.md/#dialect-interfaces)를 통해 제공합니다. 이는 방언이 재정의할 가상 훅 집합을 담은 클래스입니다. 여기서는 `DialectInlinerInterface`를 사용합니다.

```c++
/// 토이 연산 인라이닝을 처리하는 인터페이스입니다.
/// 기본 인터페이스를 상속한 뒤 필요한 메서드를 재정의합니다.
struct ToyInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// 주어진 호출 연산이 특정 호출 가능한 연산에 인라인될 수 있는지 검사합니다.
  /// 토이에서는 언제나 인라인 가능하므로 true를 반환합니다.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// 주어진 연산이 특정 리전에 인라인 가능한지 검사합니다. 토이에서는 모두 가능.
  bool isLegalToInline(Operation *, Region *, bool,
                       IRMapping &) const final {
    return true;
  }

  /// 'src' 리전이 'dest' 리전에 인라인될 수 있는지 검사합니다.
  /// 토이는 어떤 함수든 인라인 가능하므로 true를 반환합니다.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }

  /// 종료 연산이 인라인되었을 때 호출됩니다. 토이에서 종료 연산은 toy.return뿐입니다.
  /// 반환 인자를 호출 결과 값에 대체합니다.
  void handleTerminator(Operation *op,
                        ValueRange valuesToRepl) const final {
    // toy.return만 처리하면 됩니다.
    auto returnOp = cast<ReturnOp>(op);

    // 반환 피연산자로 값들을 바로 대체합니다.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }
};
```

또한 인라이너는 private 가시성의 사용되지 않는 함수 정의만 제거합니다. 따라서 MLIR 생성기에서 메인 함수를 제외한 함수의 가시성을 설정해야 합니다.

```c++
/// 새 함수를 생성해 MLIR 모듈에 추가합니다.
mlir::toy::FuncOp mlirGen(FunctionAST &funcAST) {
  ...
  // main이 아니라면 private으로 설정합니다.
  if (funcAST.getProto()->getName() != "main")
    function.setPrivate();

  return function;
}
```

그다음 연산을 등록했던 것처럼 방언에 인터페이스를 등록합니다.

```c++
void ToyDialect::initialize() {
  addInterfaces<ToyInlinerInterface>();
}
```

이제 인라이너가 `toy.generic_call`을 호출 연산으로, `toy.func`를 함수로 인식할 수 있게 해야 합니다. MLIR은 연산을 "call-like" 또는 "callable-like"로 표시할 수 있는 [연산 인터페이스](../../Interfaces.md/#attributeoperationtype-interfaces)를 제공합니다. 방언 인터페이스와 달리 연산 인터페이스는 특정 연산에 핵심적인 보다 세밀한 정보를 제공합니다. 여기서는 `CallOpInterface`와 `CallableOpInterface`를 추가합니다.

연산 정의 파일(`Ops.td`)에 인터페이스 정의를 포함합니다.

```tablegen
include "mlir/Interfaces/CallInterfaces.td"
```

그리고 `GenericCallOp`의 트레이트 목록에 추가합니다.

```tablegen
def FuncOp : Toy_Op<"func",
    [FunctionOpInterface, IsolatedFromAbove]> {
  ...
}

def GenericCallOp : Toy_Op<"generic_call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  ...
}
```

위에서는 `DeclareOpInterfaceMethods` 지시어를 사용해 `GenericCallOp` 클래스 선언에 인터페이스 메서드 선언을 자동으로 추가했습니다. 다만 `CallOpInterface`에는 인자·결과 속성을 처리하는 메서드가 포함되므로, `GenericCallOp` 정의에 해당 이름의 속성을 넣어야 합니다.

```tablegen
let arguments = (ins
  ...
  OptionalAttr<DictArrayAttr>:$arg_attrs,
  OptionalAttr<DictArrayAttr>:$res_attrs
);
```

`FuncOp` 클래스의 `extraClassDeclaration`에는 이미 다음 정의가 들어 있습니다.

```c++
/// callable한 함수 연산의 리전을 반환합니다.
Region *FuncOp::getCallableRegion() { return &getBody(); }

// ....

/// generic_call 연산의 피호출자를 반환합니다.
CallInterfaceCallable GenericCallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

/// generic_call 연산의 피호출자를 설정합니다.
void GenericCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
}

/// 호출 대상 함수의 인자 피연산자를 반환합니다.
Operation::operand_range GenericCallOp::getArgOperands() { return getInputs(); }

/// 호출 대상 함수의 인자 피연산자를 변경 가능한 범위로 반환합니다.
MutableOperandRange GenericCallOp::getArgOperandsMutable() {
  return getInputsMutable();
}
```

이제 인라이너가 토이 방언을 인지했으므로, 패스 매니저에 인라이너 패스를 추가할 수 있습니다.

```c++
  pm.addPass(mlir::createInlinerPass());
```

작동 예제를 살펴보겠습니다.

```mlir
toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
  %2 = toy.mul %0, %1 : tensor<*xf64>
  toy.return %2 : tensor<*xf64>
}
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>
  %2 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>
  %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64>
  %4 = toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
  %5 = toy.generic_call @multiply_transpose(%3, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
  toy.print %5 : tensor<*xf64>
  toy.return
}
```

`multiply_transpose`를 두 번 호출했으니 main에 인라인되길 기대하지만, 결과가 바뀌지 않습니다. 마지막 조각이 빠졌기 때문입니다. 호출 가장자리에는 숨겨진 타입 변환이 있습니다. 위 예에서 generic_call의 인자는 `tensor<2x3xf64>`인데, 함수 입력은 `tensor<*xf64>`를 기대합니다. 이 차이를 해결하려면 인라이너가 명시적 변환 연산을 삽입해야 합니다. 이를 위해 서로 다른 형태 사이의 변환을 표현하는 `toy.cast` 연산(`ToyCastOp`)을 방언에 추가합니다.

```tablegen
def CastOp : Toy_Op<"cast", [
    DeclareOpInterfaceMethods<CastOpInterface>,
    Pure,
    SameOperandsAndResultShape]
  > {
  let summary = "shape cast operation";
  let description = [{
    "cast" 연산은 데이터를 변경하지 않고 한 타입의 텐서를 동등한 타입으로 변환합니다.
    입력과 출력 타입은 동일한 원소 타입의 텐서 타입이어야 합니다. 둘 다 랭크가 있다면
    형태가 같아야 하며, 상수 차원이 서로 다르면 연산이 잘못된 것입니다.
  }];

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}
```

이 캐스트 연산 정의에는 `CastOpInterface`가 트레이트로 추가되어 있습니다. 이 인터페이스는 항등 캐스트 접기, 검증 등 캐스트 연산에 필요한 유틸리티를 제공합니다. `areCastCompatible` 메서드를 정의해 인터페이스에 연결합니다.

```c++
/// 입력과 출력 타입이 이 캐스트 연산과 호환되는지 검사합니다.
bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  // 입력은 같은 원소 타입의 텐서여야 합니다.
  TensorType input = llvm::dyn_cast<TensorType>(inputs.front());
  TensorType output = llvm::dyn_cast<TensorType>(outputs.front());
  if (!input || !output || input.getElementType() != output.getElementType())
    return false;
  // 둘 다 랭크가 있다면 형태가 같아야 합니다.
  return !input.hasRank() || !output.hasRank() || input == output;
}
```

적절한 캐스트 연산을 갖추었으니, 필요할 때 삽입하도록 `ToyInlinerInterface`의 훅을 재정의합니다.

```c++
struct ToyInlinerInterface : public DialectInlinerInterface {
  ...

  /// 호출과 호출 대상 리전 사이의 타입 불일치를 해결하기 위한 변환을 재질화합니다.
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type resultType,
                                       Location conversionLoc) const final {
    return CastOp::create(builder, conversionLoc, resultType, input);
  }
};
```

이제 파이프라인을 다시 실행하면 다음처럼 기대한 결과를 얻습니다.

```mlir
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %1 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %2 = toy.cast %1 : tensor<2x3xf64> to tensor<*xf64>
  %3 = toy.cast %0 : tensor<2x3xf64> to tensor<*xf64>
  %4 = toy.transpose(%2 : tensor<*xf64>) to tensor<*xf64>
  %5 = toy.transpose(%3 : tensor<*xf64>) to tensor<*xf64>
  %6 = toy.mul %4, %5 : tensor<*xf64>
  toy.print %6 : tensor<*xf64>
  toy.return
}
```

참고: 일반 인라이너는 단순화도 수행하므로 출력이 더 깔끔할 수 있습니다.

### 함수 내부 형태 추론

모든 함수를 인라인하면, 정적·동적 형태 연산이 섞인 main 함수만 남습니다. 이제 단일 함수 내부에서 형태를 전파하는 간단한 패스를 작성할 수 있습니다. 토이 방언 연산의 제약을 직접 인코딩할 수도 있지만, 다른 방언에도 적용할 수 있는 일반적 변환으로 작성하는 편이 좋습니다. 가능한 한 일반적으로 표현해 두면 다른 방언에서도 유용하게 확장할 수 있습니다.

형태 추론 문제를 핵심으로 단순화하면, 연산이 정적으로 알려진 입력에 대해 어떤 출력 형태를 기대하는지 알려달라는 것입니다. 필요 이상으로 복잡하게 만들 수도 있지만, 지금은 단순하게 유지하겠습니다. 이 속성은 특정 연산에 핵심적이므로, 결과 형태를 추론해야 하는 연산에 지정할 수 있는 연산 인터페이스를 정의합니다.

연산과 마찬가지로, 연산 정의 명세(ODS) 프레임워크를 이용해 [연산 인터페이스](../../Interfaces.md/#attributeoperationtype-interfaces)를 정의할 수 있습니다.

인터페이스는 `OpInterface`를 상속해 정의하며, 생성될 C++ 인터페이스 클래스의 이름을 템플릿 인자로 받습니다. 우리는 `ShapeInference`라고 이름 붙입니다. 설명도 함께 제공합니다.

```tablegen
def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let description = [{
    타입 추론 중에 사용할 수 있도록 연산의 반환 타입을 추론하는 등록 메서드에 접근하는 인터페이스입니다.
  }];
}
```

다음으로 연산이 제공해야 하는 인터페이스 메서드를 정의합니다. 메서드는 설명, 문자열 형태의 C++ 반환 타입, 문자열 메서드 이름, 그리고 필요한 경우 선택 요소로 구성됩니다. 자세한 내용은 [ODS 문서](../../Interfaces.md/#attributeoperationtype-interfaces)를 참고하세요.

```tablegen
def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  ...

  let methods = [
    InterfaceMethod<"현재 연산의 출력 형태를 추론해 설정합니다.",
                    "void", "inferShapes">
  ];
}
```

인터페이스를 정의했으니, `CallOpInterface`를 `GenericCallOp`에 추가했던 것과 유사하게 필요한 토이 연산에 붙입니다.

```tablegen
def MulOp : Toy_Op<"mul",
    [..., DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  ...
}
```

이제 각 연산은 `inferShapes()` 메서드 정의를 제공해야 합니다. 예를 들어 `mul` 연산은 입력과 동일한 형태를 결과 형태로 추론합니다.

```c++
/// shape inference 인터페이스에서 요구하는 MulOp의 출력 형태 추론입니다.
void MulOp::inferShapes() { getResult().setType(getLhs().getType()); }
```

이 시점에서 필요한 토이 연산마다 출력 형태를 추론하는 메커니즘이 마련되었습니다. ShapeInferencePass는 함수 단위로 작동하며, 각 함수를 독립적으로 실행합니다. MLIR은 모든 독립 연산에 실행할 수 있는 일반 [OperationPass](../../PassManagement.md/#operation-pass)도 지원하지만, 현재 모듈에는 함수만 있으므로 일반화할 필요가 없습니다.

패스를 구현하려면 `mlir::OperationPass<FuncOp>`를 상속한 클래스를 만들고 `runOnOperation()`을 재정의하면 됩니다.

```c++
class ShapeInferencePass
    : public mlir::PassWrapper<ShapeInferencePass, OperationPass<FuncOp>> {
  void runOnOperation() override {
    FuncOp function = getOperation();
    ...
  }
};
```

동시에 패스 인스턴스를 만드는 헬퍼도 정의합니다.

```c++
std::unique_ptr<mlir::Pass> mlir::toy::createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}
```

형태 추론 알고리즘은 다음과 같이 동작합니다.

1. 동적 형태 텐서를 반환하는 모든 연산을 워크리스트에 담습니다. 이 연산들이 형태 추론 대상입니다.
2. 워크리스트를 반복합니다.
    - 인자를 모두 구체적으로 가진 다음 처리할 연산을 찾습니다.
    - 없다면 반복을 종료합니다.
    - 연산을 워크리스트에서 제거합니다.
    - 인자 타입으로 결과 형태를 추론합니다.
3. 워크리스트가 비어 있으면 알고리즘이 성공한 것입니다.

연산을 처리할 때는 아래 코드처럼 `ShapeInference` 인터페이스가 등록되어 있는지 확인합니다.

```c++
  // 연산에 출력 형태 추론을 요청합니다.
  LDBG() << "Inferring shape for: " << *op;

  /// 인터페이스를 갖고 있는지는 캐스팅으로 확인합니다.
  if (ShapeInference shapeOp = dyn_cast<ShapeInference>(op)) {
    shapeOp.inferShapes();
  } else {
    op->emitError("shape inference 인터페이스 없이 형태를 추론할 수 없습니다");
    return signalPassFailure();
  }
```

이제 패스를 패스 매니저에 추가합니다.

```c++
  pm.addPass(mlir::createShapeInferencePass());
```

원래 예제를 다시 실행하면 다음과 같은 결과를 얻습니다.

```mlir
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %2 = toy.mul %1, %1 : tensor<3x2xf64>
  toy.print %2 : tensor<3x2xf64>
  toy.return
}
```

`toyc-ch4`를 빌드해 직접 실행해보세요. `toyc-ch4 test/Examples/Toy/Ch4/codegen.toy -emit=mlir -opt` 명령을 사용할 수 있습니다.

[다음 장](Ch-5_kr.md)에서는 더 계산 집약적인 토이 연산을 최적화하기 위해 낮은 수준 방언을 대상으로 코드 생성 과정을 시작합니다.
