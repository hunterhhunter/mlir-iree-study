// lib/toy/Ops.cpp
#include "toy/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

// [핵심] 네임스페이스 정리
using namespace mlir;
using namespace mlir::toy;

// === ConstantOp 구현 ===

void ConstantOp::build(OpBuilder &builder, OperationState &state, double value) {
  auto dataType = RankedTensorType::get({}, builder.getF64Type());
  auto dataAttr = DenseElementsAttr::get(dataType, value);
  
  // [체크] 여기서 state에 재료를 다 넣어주고 있나요?
  ConstantOp::build(builder, state, dataType, dataAttr);
}

LogicalResult ConstantOp::verify() {
  return success();
}

// === AddOp 구현 ===

// void AddOp::build(OpBuilder &builder, OperationState &state, Value lhs, Value rhs) {
//   state.addTypes(lhs.getType());
//   state.addOperands({lhs, rhs});
// }

// [핵심] TableGen이 만든 연산 구현 코드 포함
#define GET_OP_CLASSES
#include "toy/Ops.cpp.inc"