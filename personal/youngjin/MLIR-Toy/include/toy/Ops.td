include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

// ODS 프레임워크를 이용한 방언 선언
def Toy_Dialect : Dialect {
  // MLIR 시스템 내부에서 식별할 Unique한 ID
  let name = "toy";

  // 방언에 대한 요약 설명
  let summary = "A high-level dialect for analyzing and optimizing the "
                "Toy language";

  // 방언에 대한 구체적인 설명
  let description = [{
    The Toy language is a tensor-based language that allows you to define
    functions, perform some math computation, and print results.
  }];

  // C++의 Namespace를 설정하려 mlir::toy::..로 접근 가능
  let cppNamespace = "::mlir::toy";
}

// 토이 연산자 기본 클래스 생성
class Toy_Op<string mnemonic, list<Trait> traits = []> :
  Op<Toy_Dialect, mnemonic, traits>;

// 연산지 기본 구조 및 입력, 출력 제공
def ConstantOp : Toy_Op<"constant"> {
  let summary = "constant operation";
  let description = [{
  Constant operation turns a literal into an SSA value. The data is attached
  to the operation as an attribute. For example:

    %0 = "toy.constant"()
        { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
      : () -> tensor<2x3xf64>
  }];
  // 상수 연산은 속성을 유일한 입력으로 받음
  let arguments = (ins F64ElementsAttr:$value);
  // 상수 연산은 텐서타입 단일값을 반환
  let results = (outs F64Tensor);

  // 추가적인 검증로직을 직접 작성하겠다는 의미 (C++ 파일에 verify() 함수 선언됨)
  let hasVerifier = 1;

  // C++에서 이 연산을 쉽게 정의할 수 있게 만들어진 builders
  let builders = [
    OpBuilder<(ins "mlir::DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,
    OpBuilder<(ins "double":$value)>
  ];

  // 이 Op가 출력하는 포맷 변경
  let assemblyFormat = "$value attr-dict `:` type(results)";
}

// === AddOp (덧셈) ===
def AddOp : Toy_Op<"add", [SameOperandsAndResultType]> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  // 입력: 텐서 2개(lhs, rhs)
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  // // C++ 에서 만들기 위한 빌더 추가
  // let builders = [
  //   OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
  // ];

  let assemblyFormat = "$lhs, $rhs attr-dict : type($lhs), type($rhs)";
}

def PrintOp : Toy_Op<"print"> {
  let arguments = (ins F64Tensor:$input);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}