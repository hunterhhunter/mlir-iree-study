# MLIR Toy Dialect Chapter 4: Shape Inference Implementation Guide

이 문서는 인라이닝 이후 불완전한 타입 정보(`tensor<*xf64>`)를 가진 연산들의 형태를 결정하는 **Shape Inference** 인터페이스와 패스 구현 과정을 나열합니다.

---

### Step 1: 형태 추론 연산 인터페이스 정의 (`ShapeInferenceInterface.td`)
모든 연산이 각자의 로직으로 결과 형태를 추론할 수 있도록 공통 규격인 `OpInterface`를 정의합니다.

```TableGen
#ifndef SHAPE_INFERENCE_INTERFACE
#define SHAPE_INFERENCE_INTERFACE

include "mlir/IR/OpBase.td"

def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let cppNamespace = "::mlir::toy";
  let description = [{
    Interface to access a registered method to infer the return types for an
    operation that can be used during type inference.
  }];

  let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.",
                    "void", "inferShapes">
  ];
}

#endif // SHAPE_INFERENCE_INTERFACE
```
---
### Step 2: 생성되는 인터페이스 include (`ShapeInferenceInterface.h`)
위 `ShapeInferenceInterface.td`와 같은 위치에 파일을 만들고 아래 내용을 채워넣습니다.
```TableGen
#ifndef MLIR_TUTORIAL_TOY_SHAPEINFERENCEINTERFACE_H_
#define MLIR_TUTORIAL_TOY_SHAPEINFERENCEINTERFACE_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace toy {
/// Include the auto-generated declarations.
#include "toy/ShapeInferenceOpInterfaces.h.inc"
} // namespace toy
} // namespace mlir

#endif // MLIR_TUTORIAL_TOY_SHAPEINFERENCEINTERFACE_H_
```
---
### Step 3: 개별 연산에 인터페이스 적용 (`Ops.td`)
형태 추론이 필요한 연산(Add, Mul, Cast, generic_call, Transpose)의 트레이트 리스트에 위에서 정의한 인터페이스를 추가합니다.
```TableGen
def AddOp : Toy_Op<"add",
  [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {...}
  
def MulOp : Toy_Op<"mul",
  [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {...}
  
def CastOp : Toy_Op<"cast", [
     DeclareOpInterfaceMethods<CastOpInterface>,
     DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
     Pure,
     SameOperandsAndResultShape]> {...}

def GenericCallOp : Toy_Op<"generic_call",
    [DeclareOpInterfaceMethods<CallOpInterface>,
     DeclareOpInterfaceMethods<ShapeInferenceOpInterface>] > {...}

def TransposeOp : Toy_Op<"transpose",
  [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {...}
```
---
### Step 4: 인터페이스 메서드 구현 (`Dialect.cpp`)
각 연산별로 실제 형태를 계산하는 C++ 로직을 작성합니다. (예: 곱셈 연산은 입력의 형태를 그대로 결과에 전파)
```cpp
void AddOp::inferShapes() { getResult().setType(getLhs().getType()); }

// Dialect.cpp
void MulOp::inferShapes() {
  // LHS(Left Hand Side)의 타입을 결과 타입으로 설정
  getResult().setType(getLhs().getType());
}

void CastOp::inferShapes() {
  // Cast의 경우 입력 형태를 결과에 그대로 전파
  getResult().setType(getInput().getType());
}

void GenericCallOp::inferShapes() {
  auto endpoints = getOperation()->getParentOfType<mlir::ModuleOp>().lookupSymbol<FuncOp>(getCallee());
  if (!endpoints)
    return;
    
  auto resultType = endpoints.getFunctionType().getResult(0);
  
  if (llvm::isa<UnrankedTensorType>(resultType))
    return;
    
  getResult().setType(cast<TensorType>(resultType));
}

void TransposeOp::inferShapes() {
  auto arrayTy = llvm::cast<RankedTensorType>(getOperand().getType());
  SmallVector<int64_t, 2> dims(llvm::reverse(arrayTy.getShape()));
  getResult().setType(RankedTensorType::get(dims, arrayTy.getElementType()));
}
```
---
### Step 4: 형태 추론 패스(Pass) 골격 생성 (`ShapeInferencePass.cpp`)
`FuncOp`를 대상으로 작동하며, 함수 내의 모든 연산을 순회하며 타입을 전파하는 패스 클래스를 정의합니다.
```cpp
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
struct ShapeInferencePass
    : public ::mlir::PassWrapper<ShapeInferencePass, OperationPass<toy::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeInferencePass)
  ::llvm::StringRef getArgument() const override { return "toy-shape-inference"; }
  
  void runOnOperation() override {
    auto f = getOperation();
    ::llvm::SmallPtrSet<::mlir::Operation *, 16> opWorklist;
    
    // 1. 함수 전체를 순회하면서 작업 목록을 만듦
    f.walk([&](::mlir::Operation *op) {
      if (returnsDynamicShape(op))
        // 동적 타입 반환시 insert
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

  static bool allOperandsInferred(::mlir::Operation *op) {
    return ::llvm::all_of(op->getOperandTypes(), [](::mlir::Type t) {
      return ::llvm::isa<::mlir::RankedTensorType>(t);
    });
  }

  static bool returnsDynamicShape(::mlir::Operation *op) {
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
```
---
### Step 5: 생성한 최적화 Pass의 진입점(함수) 노출 (`Passes.h`)
`include/toy/Passes.h`에 작성한 최적화 pass의 진입점(함수)을 노출해줍니다.
```cpp
#ifndef TOY_PASSES_H
#define TOY_PASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace toy {
std::unique_ptr<Pass> createShapeInferencePass();
} // namespace toy
} // namespace mlir

#endif // TOY_PASSES_H
```
---
### Step 6: 각종 include 해결
```
// Ops.td 상단
include "toy/ShapeInferenceInterface.td"

// Dialect.h 상단
#inlcude "toy/ShapeInferenceOpInterfaces.h.inc"

// Dialect.cpp 하단
#inclde "toy/ShapeInferenceOpInterfaces.cpp.inc"
```
---
### Step 7: MLIR에 최적화 Pass 등록 (toyc.cpp)
```cpp
static int dumpMLIR() {
	if (enableOpt) {
		// ....
		// uncOp는 외부 환경과 격리된(IsolateFromAbove) 특성을 가져
	    // 함수 내부의 최적화가 다른 함수의 IR구조에 직접적인 영향을 주지 않음
	    // 따라서 pm.nest를 사용하면 개별 func연산에 집중해 병렬 최적화 + 패스 로컬리티가 가능함.
		mlir::OpPassManager &optPM = pm.nest<mlir::toy::FuncOp>();
	    optPM.addPass(mlir::toy::createShapeInferencePass());
	    optPM.addPass(mlir::createCanonicalizerPass());
	    optPM.addPass(mlir::createCSEPass());
	}
    module->dump();
    return 0;
}
```