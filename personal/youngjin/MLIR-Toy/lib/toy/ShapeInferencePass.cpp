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

// !! using namespace ... 쓰면 프로젝트 커질수록 의존성 지옥....
// 그냥 mlir::.. 이렇게 쓰는게 귀찮더라도 좋은듯
namespace mlir {
namespace toy {

namespace {
/// ShapeInferencePass 구현
// 260119 ch4 - shapeinference
struct ShapeInferencePass
    : public ::mlir::PassWrapper<ShapeInferencePass, OperationPass<toy::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeInferencePass)
  
  ::llvm::StringRef getArgument() const override { return "toy-shape-inference"; }

  //
  void runOnOperation() override {
    auto f = getOperation();

    ::llvm::SmallPtrSet<::mlir::Operation *, 16> opWorklist;
    // 1. 함수 전체를 순회하면서 작업 목록을 만듦
    f.walk([&](::mlir::Operation *op) {
      if (returnsDynamicShape(op))
        // 동적 타입 반환시 insert <*xf64>
        opWorklist.insert(op);
    });

    // 2. 작업 목록이 빌 때까지 반복 - 순서가 매우 중요
    while (!opWorklist.empty()) {
      // 입력값의 모양을 다 아는 연산을 찾기 (등록+구현 되어있어야함.)
      auto nextop = ::llvm::find_if(opWorklist, allOperandsInferred);

      if (nextop == opWorklist.end())
        break;

      ::mlir::Operation *op = *nextop;
      opWorklist.erase(op);

      LDBG() << "Inferring shape for: " << *op;
      
      // 3. ShapeInference 인터페이스로 캐스팅하여 inferShapes 호출
      // 이 때 dynamic_casting인 것이 중요 -> shapeInference 기능을 가지는지 확인
      // 인터페이스 캐스팅 시 명시적 네임스페이스 사용
      if (auto shapeOp = ::llvm::dyn_cast<::mlir::toy::ShapeInference>(op)) {
        shapeOp.inferShapes();
      } else {
        op->emitError("unable to infer shape of operation without shape inference interface");
        return signalPassFailure();
      }
    }

    if (!opWorklist.empty()) {
      f.emitError("Shape inference failed, ")
          << opWorklist.size() << " operations couldn't be inferred\n";
      signalPassFailure();
    }
  }

  // 모든 입력이 RankedTensorType인지 bool로 return 하는 함수
  static bool allOperandsInferred(::mlir::Operation *op) {
    return ::llvm::all_of(op->getOperandTypes(), [](::mlir::Type t) {
      return ::llvm::isa<::mlir::RankedTensorType>(t);
    });
  }

  // 반환형이 <*xf64> 인지 bool로 return하는 함수
  static bool returnsDynamicShape(::mlir::Operation *op) {
    // op->getResultTypes: 반환하는 결과를 가져와 모양이 확정되지 않았는지 확인
    return ::llvm::any_of(op->getResultTypes(), [](::mlir::Type t) {
      return !::llvm::isa<::mlir::RankedTensorType>(t);
    });
  }
};
} // namespace

/// Pass 생성 함수 (mlir::toy 네임스페이스 내부)
std::unique_ptr<::mlir::Pass> createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}

} // namespace toy
} // namespace mlir