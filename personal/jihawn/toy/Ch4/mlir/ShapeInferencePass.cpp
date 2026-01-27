//===- ShapeInferencePass.cpp - Shape Inference ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// 이 파일은 함수 특수화를 통해 배열 모양의 프로시저 간(interprocedural) 전파를 
// 수행하는 Function 레벨 패스를 구현합니다.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "toy/Dialect.h"
#include "toy/Passes.h"
#include "toy/ShapeInferenceInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

#define DEBUG_TYPE "shape-inference"

using namespace mlir;
using namespace toy;

/// 자동 생성된 모양 추론 인터페이스 정의를 포함합니다.
#include "toy/ShapeInferenceOpInterfaces.cpp.inc"

namespace {
/// ShapeInferencePass는 프로시저 내(intra-procedural) 모양 추론을 
/// 수행하는 패스입니다.
///
///    알고리즘:
///
///   1) 동적으로 형성된 텐서를 반환하는 모든 연산을 포함하는 작업 목록(worklist)을 
///      작성합니다. 이들은 모양 추론이 필요한 연산들입니다.
///   2) 작업 목록을 반복합니다:
///     a) 처리할 연산을 찾습니다: 작업 목록에서 모든 인자가 제네릭이 아닌(non-generic)
///        준비된 다음 연산을 찾습니다.
///     b) 연산을 찾지 못하면 루프를 중단합니다.
///     c) 작업 목록에서 연산을 제거합니다.
///     d) 인자 타입으로부터 출력 모양을 추론합니다.
///   3) 작업 목록이 비어 있으면 알고리즘이 성공한 것입니다.
///
struct ShapeInferencePass
    : public mlir::PassWrapper<ShapeInferencePass, OperationPass<toy::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeInferencePass)
  StringRef getArgument() const override { return "toy-shape-inference"; }

  void runOnOperation() override {
    auto f = getOperation();

    // 모양 추론이 필요한 연산들로 작업 목록을 채웁니다:
    // 이들은 동적 모양을 반환하는 연산들입니다.
    llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
    f.walk([&](mlir::Operation *op) {
      if (returnsDynamicShape(op))
        opWorklist.insert(op);
    });

    // 모든 연산이 추론되거나 변경이 없을 때까지(고정점, fix point) 작업 목록의 
    // 연산들을 반복합니다.
    while (!opWorklist.empty()) {
      // 추론 준비가 된 다음 연산, 즉 모든 피연산자가 이미 해결된(제네릭이 아닌) 
      // 연산을 찾습니다.
      auto nextop = llvm::find_if(opWorklist, allOperandsInferred);
      if (nextop == opWorklist.end())
        break;

      Operation *op = *nextop;
      opWorklist.erase(op);

      // 연산에게 출력 모양을 추론하도록 요청합니다.
      LDBG() << "Inferring shape for: " << *op;
      if (auto shapeOp = dyn_cast<ShapeInference>(op)) {
        shapeOp.inferShapes();
      } else {
        op->emitError("unable to infer shape of operation without shape "
                      "inference interface");
        return signalPassFailure();
      }
    }

    // 작업 목록이 비어있지 않다면, 이는 실패를 나타냅니다.
    if (!opWorklist.empty()) {
      f.emitError("Shape inference failed, ")
          << opWorklist.size() << " operations couldn't be inferred\n";
      signalPassFailure();
    }
  }

  /// 주어진 연산의 모든 피연산자가 추론되었는지 반환하는 유틸리티 메서드입니다.
  static bool allOperandsInferred(Operation *op) {
    return llvm::all_of(op->getOperandTypes(), [](Type operandType) {
      return llvm::isa<RankedTensorType>(operandType);
    });
  }

  /// 주어진 연산이 동적으로 형성된 결과를 가지는지 반환하는 유틸리티 메서드입니다.
  static bool returnsDynamicShape(Operation *op) {
    return llvm::any_of(op->getResultTypes(), [](Type resultType) {
      return !llvm::isa<RankedTensorType>(resultType);
    });
  }
};
} // namespace

/// Shape Inference 패스를 생성합니다.
std::unique_ptr<mlir::Pass> mlir::toy::createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}