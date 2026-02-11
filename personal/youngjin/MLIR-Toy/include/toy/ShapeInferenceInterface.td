//===----------------------------------------------------------------------===//
//
// Defines the operations of the Shape Inference Op Interface.
//
//===----------------------------------------------------------------------===//

#ifndef SHAPE_INFERENCE_INTERFACE
#define SHAPE_INFERENCE_INTERFACE

include "mlir/IR/OpBase.td"

// OpInterface는 MLIR TableGen에서 제공하는 템플릿 클래스
// 이를 상속받는 것은 새로운 OpInterface 정의를 의미
// 특정 연산에 종속되는 형태 추론이 아닌 하나의 인터페이스로 정의해
// 연산들이 이 인터페이스를 가짐으로써 형태추론을 할 수 있도록
def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let cppNamespace = "::mlir::toy";
  let description = [{
    Interface to access a registered method to infer the return types for an
    operation that can be used during type inference.
  }];

  // 구현 필요 함수 요구
  // 260119 ch4 - shapeinference
  let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.",
                    "void", "inferShapes">
  ];
}

#endif // SHAPE_INFERENCE_INTERFACE