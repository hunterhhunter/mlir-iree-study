//===- Ops.td - Toy 방언 연산 정의 --------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Toy 방언의 연산들을 정의합니다.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_OPS
#define TOY_OPS

include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "toy/ShapeInferenceInterface.td"

// ODS 프레임워크 내에서 'toy' 방언의 정의를 제공하여 연산들을 정의할 수 있게 합니다.
def Toy_Dialect : Dialect {
  let name = "toy";
  let cppNamespace = "::mlir::toy";
}

// Toy 방언 연산들을 위한 기본 클래스입니다. 이 연산은 OpBase.td의 기본 `Op` 클래스를 
// 상속받으며 다음을 제공합니다:
//   * 연산의 부모 방언.
//   * 연산의 니모닉(mnemonic), 즉 방언 접두사가 없는 이름.
//   * 연산의 트레이트(trait) 목록.
class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Toy 연산들
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

// 위의 기본 'Toy_Op' 클래스를 상속받아 toy 연산을 정의합니다.
// 여기서는 연산의 니모닉과 트레이트 목록을 제공합니다. 
// 상수 연산은 순수(pure) 연산이므로 'Pure'로 표시하며, 사용되지 않으면 제거될 수 있습니다.
def ConstantOp : Toy_Op<"constant", [Pure]> {
  // 이 연산에 대한 요약과 설명을 제공합니다. 이는 방언 내 연산들의 문서를 
  // 자동으로 생성하는 데 사용될 수 있습니다.
  let summary = "constant";
  let description = [{
    상수 연산은 리터럴을 SSA 값으로 변환합니다. 데이터는 속성(attribute)으로서 
    연산에 부착됩니다. 예시:

    ```mlir
      %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                        : tensor<2x3xf64>
    ```
  }];

  // 상수 연산은 유일한 입력으로 속성을 받습니다.
  let arguments = (ins F64ElementsAttr:$value);

  // 상수 연산은 TensorType의 단일 값을 반환합니다.
  let results = (outs F64Tensor);

  // 연산이 커스텀 파서와 프린터 메서드를 가짐을 나타냅니다.
  let hasCustomAssemblyFormat = 1;

  // 상수 연산을 위한 커스텀 빌드 메서드를 추가합니다. 이 메서드는 MLIR이 
  // 연산을 생성하는 데 사용하는 `state`를 채웁니다. 즉, 
  // `ConstantOp::create(builder, ...)`를 사용할 때 호출됩니다.
  let builders = [
    // 주어진 상수 텐서 값으로 상수를 빌드합니다.
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,

    // 주어진 상수 부동 소수점 값으로 상수를 빌드합니다.
    OpBuilder<(ins "double":$value)>
  ];

  // 이 연산에 대한 추가 검증이 필요함을 나타냅니다.
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def AddOp : Toy_Op<"add",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise addition operation";
  let description = [{
    "add" 연산은 두 텐서 간의 요소별(element-wise) 덧셈을 수행합니다.
    두 텐서 피연산자의 모양(shape)은 일치해야 합니다.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  // 연산이 커스텀 파서와 프린터 메서드를 가짐을 나타냅니다.
  let hasCustomAssemblyFormat = 1;

  // 두 입력 피연산자로부터 AddOp를 빌드할 수 있게 합니다.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

def CastOp : Toy_Op<"cast", [
     DeclareOpInterfaceMethods<CastOpInterface>,
     DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
     Pure,
     SameOperandsAndResultShape
  ]> {
  let summary = "shape cast operation";
  let description = [{
    "cast" 연산은 데이터 요소를 변경하지 않고 텐서를 한 타입에서 동등한 다른 타입으로 
    변환합니다. 소스와 목적지 타입은 모두 동일한 요소 타입을 가진 텐서 타입이어야 합니다. 
    만약 둘 다 랭크(차원)가 있다면, 모양(shape)이 반드시 일치해야 합니다. 
    일치하지 않는 상수 차원으로 변환하는 경우 이 연산은 유효하지 않습니다.
  }];

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor:$output);

  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def FuncOp : Toy_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove
  ]> {
  let summary = "user defined function operation";
  let description = [{
    "toy.func" 연산은 사용자 정의 함수를 나타냅니다. 이들은 toy 계산을 
    포함하는 호출 가능한(callable) SSA-영역 연산입니다.

    예시:

    ```mlir
    toy.func @main() {
      %0 = toy.constant dense<5.500000e+00> : tensor<f64>
      %1 = toy.reshape(%0 : tensor<f64>) to tensor<2x2xf64>
      toy.print %1 : tensor<2x2xf64>
      toy.return
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
  >];

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface 메서드들
    //===------------------------------------------------------------------===//

    /// 이 함수의 인자 타입들을 반환합니다.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// 이 함수의 결과 타입들을 반환합니다.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    Region *getCallableRegion() { return &getBody(); }
  }];

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// GenericCallOp
//===----------------------------------------------------------------------===//

def GenericCallOp : Toy_Op<"generic_call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "generic call operation";
  let description = [{
    제네릭 호출은 인자의 모양에 대해 특수화(specialized)되어야 하는 사용자 정의 
    함수에 대한 호출을 나타냅니다. 피호출자(callee) 이름은 속성을 통해 심볼 
    참조로 부착됩니다. 인자 목록은 피호출자가 기대하는 인자들과 일치해야 합니다. 
    예시:

    ```mlir
     %4 = toy.generic_call @my_func(%1, %3)
           : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    ```

    이는 "my_func"라는 이름의 함수가 존재하고 두 개의 인자를 받을 때만 유효합니다.
  }];

  // 제네릭 호출 연산은 피호출자를 위한 심볼 참조 속성과 호출을 위한 입력들을 받습니다.
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<F64Tensor>:$inputs,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  // 제네릭 호출 연산은 TensorType의 단일 값을 반환합니다.
  let results = (outs F64Tensor);

  // 선언적 포맷을 사용하여 어셈블리 출력 및 파싱을 특수화합니다.
  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  // 제네릭 호출 연산을 위한 커스텀 빌드 메서드를 추가합니다.
  let builders = [
    OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
  ];
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

def MulOp : Toy_Op<"mul",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise multiplication operation";
  let description = [{
    "mul" 연산은 두 텐서 간의 요소별(element-wise) 곱셈을 수행합니다.
    두 텐서 피연산자의 모양은 일치해야 합니다.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  // 연산이 커스텀 파서와 프린터 메서드를 가짐을 나타냅니다.
  let hasCustomAssemblyFormat = 1;

  // 두 입력 피연산자로부터 MulOp를 빌드할 수 있게 합니다.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//

def PrintOp : Toy_Op<"print"> {
  let summary = "print operation";
  let description = [{
    "print" 내장 연산은 주어진 입력 텐서를 출력하며, 결과를 생성하지 않습니다.
  }];

  // print 연산은 출력할 입력 텐서를 하나 받습니다.
  let arguments = (ins F64Tensor:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

def ReshapeOp : Toy_Op<"reshape", [Pure]> {
  let summary = "tensor reshape operation";
  let description = [{
    Reshape 연산은 입력 텐서를 동일한 수의 요소를 가지지만 다른 모양을 가진 
    새로운 텐서로 변환합니다. 예시:

    ```mlir
       %0 = toy.reshape (%arg1 : tensor<10xf64>) to tensor<5x2xf64>
    ```
  }];

  let arguments = (ins F64Tensor:$input);

  // reshape 연산은 정적으로 모양이 결정된(statically shaped) 텐서를 반환할 것으로 기대합니다.
  let results = (outs StaticShapeTensorOf<[F64]>);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];

  // 이 연산에 정준화(canonicalization) 패턴을 등록할 수 있게 합니다.
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp : Toy_Op<"return", [Pure, HasParent<"FuncOp">,
                                 Terminator]> {
  let summary = "return operation";
  let description = [{
    "return" 연산은 함수 내에서의 반환 연산을 나타냅니다.
    이 연산은 선택적으로 텐서 피연산자를 하나 받으며 결과를 생성하지 않습니다.
    피연산자의 타입은 이 연산을 포함하는 함수의 시그니처와 일치해야 합니다. 
    예시:

    ```mlir
      toy.func @foo() -> tensor<2xf64> {
        ...
        toy.return %0 : tensor<2xf64>
      }
    ```
  }];

  // return 연산은 반환할 선택적인 입력 피연산자를 하나 받습니다. 
  // 이 값은 이를 둘러싼 함수의 반환 타입과 일치해야 합니다.
  let arguments = (ins Variadic<F64Tensor>:$input);

  // return 연산은 입력이 존재할 때만 포맷에 출력합니다.
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  // 반환 피연산자가 없는 ReturnOp를 빌드할 수 있게 합니다.
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>
  ];

  // C++ 연산 클래스 정의에 추가적인 유틸리티 정의를 제공합니다.
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  // 이 연산에 대한 추가 검증이 필요함을 나타냅니다.
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

def TransposeOp : Toy_Op<"transpose",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "transpose operation";

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];

  // 이 연산에 정준화 패턴을 등록할 수 있게 합니다.
  let hasCanonicalizer = 1;

  // 입력 피연산자로부터 TransposeOp를 빌드할 수 있게 합니다.
  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];

  // 이 연산에 대한 추가 검증이 필요함을 나타냅니다.
  let hasVerifier = 1;
}

#endif // TOY_OPS