# 2장: 기본 MLIR 방출하기

[TOC]

이제 언어와 AST에 익숙해졌으니 MLIR이 토이 언어를 어떻게 컴파일하는 데 도움을 주는지 살펴보겠습니다.

## 소개: 다단계 중간 표현

LLVM([Kaleidoscope 튜토리얼](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/index.html) 참조) 같은 다른 컴파일러는 고정된 타입과 (보통 *저수준*/RISC 유사한) 명령 집합을 제공합니다. 따라서 특정 언어의 프런트엔드는 언어 고유의 타입 검사, 분석, 변환을 LLVM IR을 방출하기 전에 모두 수행해야 합니다. 예를 들어 Clang은 AST를 이용해 정적 분석뿐 아니라 C++ 템플릿 인스턴스화(클론 및 재작성) 같은 변환까지 실행합니다. C/C++보다 고수준 구성 요소를 가진 언어는 AST에서 LLVM IR을 생성하기까지 복잡한 하향 과정을 거칠 수 있습니다.

이 결과 여러 프런트엔드가 분석과 변환에 필요한 인프라의 상당 부분을 재구현하게 됩니다. MLIR은 확장성을 염두에 두고 설계되어 이 문제를 해결합니다. MLIR에는 미리 정의된 명령(*operation*)이나 타입이 거의 없습니다.

## MLIR과 상호작용

[언어 참조](../../LangRef.md)

MLIR은 완전히 확장 가능한 인프라로 설계되었습니다. 속성(상수 메타데이터), 연산, 타입에 대한 닫힌 집합이 존재하지 않습니다. MLIR은 [방언(dialect)](../../LangRef.md/#dialects) 개념으로 이러한 확장성을 지원합니다. 방언은 고유한 `namespace` 아래 추상화를 묶는 메커니즘을 제공합니다.

MLIR에서 [연산](../../LangRef.md/#operations)은 추상화와 계산의 핵심 단위로, LLVM 명령과 유사합니다. 연산은 응용 프로그램별 의미를 가질 수 있으며, LLVM의 핵심 IR 구조(명령, 전역, 모듈 등)를 표현할 수 있습니다.

다음은 토이 언어의 `transpose` 연산에 대한 MLIR 어셈블리 예입니다.

```mlir
%t_tensor = "toy.transpose"(%tensor) {inplace = true} : (tensor<2x3xf64>) -> tensor<3x2xf64> loc("example/file/path":12:1)
```

이 MLIR 연산을 구성 요소별로 살펴보겠습니다.

-   `%t_tensor`

    *   이 연산이 정의하는 결과에 부여된 이름입니다([충돌 방지를 위한 접두 부호 포함](../../LangRef.md/#identifiers-and-keywords)). 연산은 0개 이상의 결과(SSA 값)를 정의할 수 있으며, 여기서는 단일 결과만 다루겠습니다. 이름은 파싱 중에만 사용되고 메모리 표현에는 저장되지 않습니다.

-   `"toy.transpose"`

    *   연산의 이름입니다. 고유한 문자열이어야 하며, 방언의 네임스페이스가 `.` 앞에 붙습니다. 즉 `toy` 방언의 `transpose` 연산을 의미합니다.

-   `(%tensor)`

    *   입력 피연산자(인자)의 목록입니다. 다른 연산이 정의한 SSA 값이나 블록 인자를 참조합니다.

-   `{ inplace = true }`

    *   0개 이상의 속성 딕셔너리입니다. 속성은 항상 상수인 특별한 피연산자입니다. 여기서는 `inplace`라는 불리언 속성이 true로 설정되어 있습니다.

-   `(tensor<2x3xf64>) -> tensor<3x2xf64>`

    *   함수형 표기법으로 인자와 반환값의 타입을 명시합니다.

-   `loc("example/file/path":12:1)`

    *   이 연산이 소스 코드의 어느 위치에서 유래했는지 나타냅니다.

위 예에서 일반적인 연산의 형태를 볼 수 있습니다. MLIR의 연산 집합은 확장 가능합니다. 연산은 다음 개념을 사용해 모델링되며, 이를 통해 연산을 일반적으로 다루고 조작할 수 있습니다.

-   연산 이름
-   SSA 피연산자 목록
-   [속성](../../LangRef.md/#attributes) 목록
-   결과 값들의 [타입](../../LangRef.md/#type-system) 목록
-   디버깅을 위한 [소스 위치](../../Diagnostics.md/#source-locations)
-   [후속 블록](../../LangRef.md/#blocks) 목록(주로 분기용)
-   [리전](../../LangRef.md/#regions) 목록(함수 같은 구조적 연산용)

MLIR의 모든 연산에는 반드시 소스 위치가 붙습니다. LLVM에서는 디버그 위치가 메타데이터라 쉽게 버려질 수 있지만, MLIR에서는 위치가 핵심 요구 사항이며 API가 이를 의존하고 조작합니다. 따라서 위치를 생략하려면 명시적으로 선택해야 하며, 실수로 제거되는 일은 없습니다.

예를 들어 변환이 한 연산을 다른 연산으로 대체할 때, 새 연산도 위치 정보를 반드시 가져야 합니다. 이렇게 해야 연산이 어디에서 왔는지 추적할 수 있습니다.

`mlir-opt`는 컴파일러 패스를 테스트하는 도구로, 기본적으로 위치 정보를 출력하지 않습니다. `-mlir-print-debuginfo` 플래그를 사용하면 위치 정보를 포함시킬 수 있습니다(`mlir-opt --help`로 더 많은 옵션을 확인하세요).

### 불투명 API

MLIR은 속성, 연산, 타입 등 모든 IR 요소를 사용자 정의할 수 있게 설계되었습니다. 동시에 IR 요소는 위에서 설명한 기본 개념으로 항상 축약할 수 있습니다. 이를 통해 MLIR은 어떤 연산이라도 파싱, 표현, [왕복(round-trip)](../../../getting_started/Glossary.md/#round-trip)할 수 있습니다. 예를 들어 위 토이 연산을 `.mlir` 파일에 넣고 `toy` 방언을 등록하지 않아도 `mlir-opt`로 왕복할 수 있습니다.

```mlir
func.func @toy_func(%tensor: tensor<2x3xf64>) -> tensor<3x2xf64> {
  %t_tensor = "toy.transpose"(%tensor) { inplace = true } : (tensor<2x3xf64>) -> tensor<3x2xf64>
  return %t_tensor : tensor<3x2xf64>
}
```

등록되지 않은 속성, 연산, 타입의 경우 MLIR은 지배 영역(dominance) 같은 구조적 제약만 강제하며 나머지는 완전히 불투명하게 취급합니다. 예를 들어 MLIR은 등록되지 않은 연산이 어떤 데이터 타입을 처리할 수 있는지, 몇 개의 피연산자와 결과를 가지는지 모릅니다. 초기 부팅 단계에서는 유용할 수 있지만, 성숙한 시스템에서는 권장되지 않습니다. 등록되지 않은 연산은 변환과 분석에서 보수적으로 다뤄야 하고, 구성과 조작도 훨씬 어렵습니다.

이러한 동작은 토이에서 잘못된 IR을 만들어도 검증기에서 잡히지 않는 모습을 통해 확인할 수 있습니다.

```mlir
func.func @main() {
  %0 = "toy.print"() : () -> tensor<2x3xf64>
}
```

여기엔 여러 문제가 있습니다. `toy.print`는 종료 연산이 아니고, 피연산자를 받아야 하며 값을 반환해서는 안 됩니다. 다음 절에서는 방언과 연산을 MLIR에 등록하고 검증기에 연결하며, 연산을 다루기 위한 더 나은 API를 추가할 것입니다.

## 토이 방언 정의하기

MLIR과 제대로 상호작용하려면 새 토이 방언을 정의해야 합니다. 이 방언은 토이 언어 구조를 모델링하면서 고수준 분석과 변환을 위한 경로를 제공합니다.

```c++
/// 토이 방언 정의입니다. 방언은 mlir::Dialect를 상속받아 사용자 정의 속성, 연산, 타입을 등록합니다.
/// 이후 장에서 일반 동작을 변경하기 위해 가상 메서드를 재정의하는 예시도 다룹니다.
class ToyDialect : public mlir::Dialect {
public:
  explicit ToyDialect(mlir::MLIRContext *ctx);

  /// 방언 네임스페이스 접근자입니다.
  static llvm::StringRef getDialectNamespace() { return "toy"; }

  /// 속성, 연산, 타입 등을 등록하는 초기화 함수입니다.
  void initialize();
};
```

이것은 C++에서 방언을 정의하는 방법이지만, MLIR은 [TableGen](https://llvm.org/docs/TableGen/ProgRef.html)을 사용한 선언적 정의도 지원합니다. 선언적 정의는 새 방언을 만들 때 필요한 보일러플레이트를 크게 줄이고, 방언과 함께 문서를 생성하기도 쉽습니다. 선언형으로 토이 방언을 정의하면 다음과 같습니다.

```tablegen
// ODS 프레임워크에서 'toy' 방언을 정의합니다.
def Toy_Dialect : Dialect {
  let name = "toy";
  let summary = "토이 언어 분석과 최적화를 위한 고수준 방언";
  let description = [{
    토이 언어는 함수 정의, 수학 계산, 결과 출력이 가능한 텐서 기반 언어입니다.
    이 방언은 분석과 최적화에 적합한 표현을 제공합니다.
  }];
  let cppNamespace = "toy";
}
```

생성물을 확인하려면 다음과 같이 `mlir-tblgen` 명령을 실행합니다.

```shell
${build_root}/bin/mlir-tblgen -gen-dialect-decls ${mlir_src_root}/examples/toy/Ch2/include/toy/Ops.td -I ${mlir_src_root}/include/
```

방언을 정의했으면 이제 MLIRContext에 로드할 수 있습니다.

```c++
  context.loadDialect<ToyDialect>();
```

기본적으로 `MLIRContext`는 [Builtin 방언](../../Dialects/Builtin.md)만 로드하므로, 토이처럼 다른 방언은 명시적으로 로드해야 합니다.

## 토이 연산 정의하기

이제 `Toy` 방언이 있으므로 연산을 정의할 수 있습니다. 이를 통해 시스템이 의미 정보를 제공받고 활용할 수 있습니다. 예로 `toy.constant` 연산을 만들어봅니다. 이 연산은 토이 언어의 상수를 표현합니다.

```mlir
 %4 = "toy.constant"() {value = dense<1.0> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
```

이 연산은 피연산자가 없고 `value`라는 [DenseElements](../../Dialects/Builtin.md/#denseintorfpelementsattr) 속성을 사용해 상수 값을 나타내며, [RankedTensorType](../../Dialects/Builtin.md/#rankedtensortype) 하나를 결과로 반환합니다. 연산 클래스는 [CRTP](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern) 기반의 `mlir::Op`를 상속하며 선택적으로 [traits](../../Traits)를 지정해 동작을 조정합니다. 트레이트는 접근자, 검증 등을 추가하는 메커니즘입니다.

```c++
class ConstantOp : public mlir::Op<
                     ConstantOp,
                     mlir::OpTrait::ZeroOperands,
                     mlir::OpTrait::OneResult,
                     mlir::OpTrait::OneTypedResult<TensorType>::Impl> {
 public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "toy.constant"; }

  mlir::DenseElementsAttr getValue();

  LogicalResult verifyInvariants();

  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Type result, mlir::DenseElementsAttr value);
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::DenseElementsAttr value);
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    double value);
};
```

연산은 `ToyDialect::initialize`에서 등록합니다.

```c++
void ToyDialect::initialize() {
  addOperations<ConstantOp>();
}
```

### Op와 Operation 비교

연산을 정의했다면 접근하고 변환해야 합니다. MLIR에는 연산과 관련된 두 가지 주요 클래스가 있습니다: `Operation`과 `Op`. `Operation`은 모든 연산을 일반적으로 모델링하며, 특정 연산의 속성을 묘사하지 않는 불투명한 API를 제공합니다. 반면, 각 구체적인 연산은 `Op` 파생 클래스로 표현됩니다. 예를 들어 `ConstantOp`는 입력이 없고 항상 동일한 값을 반환하는 연산을 나타냅니다. `Op` 파생 클래스는 `Operation*`에 대한 스마트 포인터 래퍼로, 연산 특화 접근자와 타입 안전한 속성을 제공합니다. 즉 토이 연산을 정의할 때 `Operation`을 다루기 위한 의미 있는 인터페이스를 정의하는 셈입니다. `ConstantOp` 자체는 멤버를 가지지 않으며, 데이터는 참조된 `Operation` 객체에 저장됩니다. 이런 설계 때문에 연산은 값으로 전달하는 것이 일반적입니다. 일반적인 `Operation*`에서 LLVM의 캐스팅 인프라를 이용해 구체적인 `Op` 인스턴스를 얻을 수 있습니다.

```c++
void processConstantOp(mlir::Operation *operation) {
  ConstantOp op = llvm::dyn_cast<ConstantOp>(operation);
  if (!op)
    return;
  mlir::Operation *internalOperation = op.getOperation();
  assert(internalOperation == operation);
}
```

### ODS(Operation Definition Specification) 활용

C++ 템플릿을 직접 특수화하는 것 외에도 MLIR은 [ODS](../../DefiningDialects/Operations.md) 프레임워크를 통한 선언적 연산 정의를 지원합니다. 연산과 관련된 사실을 TableGen 레코드에 간결하게 기술하면, 컴파일 시 동등한 `mlir::Op` 특수화가 생성됩니다. 간결하고 C++ API 변경에도 안정적이므로 MLIR에서는 ODS 사용을 권장합니다.

연산 정의를 단순화하기 위해 토이 방언 연산의 기본 클래스를 정의해봅니다.

```tablegen
class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;
```

이제 `ConstantOp`를 정의합니다. C++ 정의에서의 `ZeroOperands`, `OneResult` 트레이트는 이후에 정의할 `arguments`와 `results`에서 자동 추론됩니다.

```tablegen
def ConstantOp : Toy_Op<"constant"> {
}
```

`mlir-tblgen -gen-op-defs` 같은 명령으로 TableGen이 생성하는 C++ 코드를 확인해보면, 수동 구현과 비교하는 데 도움이 됩니다.

#### 인자와 결과 정의

연산의 인자(속성 또는 SSA 피연산자 타입)와 결과(생성되는 SSA 값의 타입)를 정의합니다.

```tablegen
def ConstantOp : Toy_Op<"constant"> {
  let summary = "constant operation";
  let description = [{
    리터럴을 SSA 값으로 변환합니다. 데이터는 속성으로 붙습니다.

      %0 = "toy.constant"()
         { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
        : () -> tensor<2x3xf64>
  }];

  let arguments = (ins F64ElementsAttr:$value);
  let results = (outs F64Tensor);
}
```

이름을 붙이면(`$value`) 자동으로 접근자(`ConstantOp::value()`)가 생성됩니다.

#### 연산 문서화

위에서 summary와 description을 추가해 문서를 자동 생성할 수 있게 했습니다.

#### 연산 의미 검증

ODS는 우리가 정의한 제약을 바탕으로 대부분의 검증 로직을 생성하므로, 추가 검증이 필요 없는 경우가 많습니다. 추가 검증을 넣고 싶다면 [`verifier`](../../DefiningDialects/Operations.md/#custom-verifier-code) 필드에 C++ 코드를 작성하면 됩니다. ODS가 모든 구조적 불변식을 확인한 후 실행됩니다.

```tablegen
let hasVerifier = 1;
```

#### `build` 메서드 연결

ODS는 단순한 빌더는 자동 생성합니다. 나머지 빌더는 [`builders`](../../DefiningDialects/Operations.md/#custom-builder-methods) 필드로 정의합니다.

```tablegen
let builders = [
  OpBuilder<(ins "DenseElementsAttr":$value), [{
    build(builder, result, value.getType(), value);
  }]>,
  OpBuilder<(ins "double":$value)>
];
```

#### 맞춤 어셈블리 형식 지정

현재 토이 연산은 모두 일반(generic) 어셈블리 형식으로 출력됩니다. MLIR은 [선언적](../../DefiningDialects/Operations.md/#declarative-assembly-format) 또는 C++ 방식으로 맞춤 어셈블리 형식을 정의할 수 있어, 불필요한 요소를 제거하고 가독성을 높일 수 있습니다.

##### `toy.print`

`toy.print`의 기본 출력이 다소 장황하므로 더 간결한 형식을 원하는 경우, 다음처럼 정의할 수 있습니다.

```mlir
toy.print %5 : tensor<*xf64> loc(...)
```

C++ 방식으로는 `hasCustomAssemblyFormat = 1`을 설정하고 `.cpp` 파일에 파서와 프린터를 구현합니다.

```c++
void PrintOp::print(mlir::OpAsmPrinter &printer) {
  printer << "toy.print " << getInput();
  printer.printOptionalAttrDict(getOperation()->getAttrs());
  printer << " : " << getInput().getType();
}

mlir::ParseResult PrintOp::parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand inputOperand;
  mlir::Type inputType;
  if (parser.parseOperand(inputOperand) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(inputType))
    return mlir::failure();
  if (parser.resolveOperand(inputOperand, inputType, result.operands))
    return mlir::failure();
  return mlir::success();
}
```

선언형 형식으로는 다음과 같이 매핑됩니다.

```tablegen
let assemblyFormat = "$input attr-dict `:` type($input)";
```

선언형 형식은 더 많은 기능을 제공하므로 C++ 구현 전에 꼭 확인해보세요. 몇몇 연산 형식을 다듬으면 다음과 같이 훨씬 읽기 쉬운 IR을 얻을 수 있습니다.

```mlir
module {
  toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:10)
    %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:25)
    %2 = toy.mul %0, %1 : tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:25)
    toy.return %2 : tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:3)
  } loc("test/Examples/Toy/Ch2/codegen.toy":4:1)
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":9:17)
    %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":9:3)
    %2 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64> loc("test/Examples/Toy/Ch2/codegen.toy":10:17)
    %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":10:3)
    %4 = toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":11:11)
    %5 = toy.generic_call @multiply_transpose(%3, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":12:11)
    toy.print %5 : tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":13:3)
    toy.return loc("test/Examples/Toy/Ch2/codegen.toy":8:1)
  } loc("test/Examples/Toy/Ch2/codegen.toy":8:1)
} loc(unknown)
```

## 토이 전체 예제

이제 "토이 IR"을 생성할 수 있습니다. `toyc-ch2`를 빌드한 뒤 위 예제에 대해 `toyc-ch2 test/Examples/Toy/Ch2/codegen.toy -emit=mlir -mlir-print-debuginfo`를 실행해보세요. 왕복 테스트도 할 수 있습니다: `toyc-ch2 test/Examples/Toy/Ch2/codegen.toy -emit=mlir -mlir-print-debuginfo 2> codegen.mlir` 후 `toyc-ch2 codegen.mlir -emit=mlir`. 마지막 정의 파일에 대해 `mlir-tblgen`을 실행하고 생성된 C++ 코드도 살펴보세요.

이제 MLIR은 토이 방언과 연산을 인지합니다. [다음 장](Ch-3_kr.md)에서는 새 방언을 활용해 토이 언어에 특화된 고수준 분석과 변환을 구현합니다.
