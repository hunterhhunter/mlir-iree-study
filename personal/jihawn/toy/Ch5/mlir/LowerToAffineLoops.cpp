//====- LowerToAffineLoops.cpp - Toy에서 Affine+Std로 부분적으로 낮추기 --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// 이 파일은 Toy 연산들을 아핀 루프, memref 연산 및 표준 연산들의 조합으로 부분적으로 낮추는 것을 구현합니다.
// 이 낮추기는 모든 호출이 인라인화되었고, 모든 모양이 해결되었음을 가정합니다.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "toy/Dialect.h"
#include "toy/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

using namespace mlir;

//===----------------------------------------------------------------------===//
// ToyToAffine 변환 패턴
//===----------------------------------------------------------------------===//

/// 주어진 RankedTensorType을 해당하는 MemRefType으로 변환합니다.
static MemRefType convertTensorToMemRef(RankedTensorType type) {
  return MemRefType::get(type.getShape(), type.getElementType());
}

/// 주어진 MemRefType에 대한 할당 및 해제를 삽입합니다.
static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
  auto alloc = memref::AllocOp::create(rewriter, loc, type);

  // 블록의 시작 부분에 할당하도록 합니다.
  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  // 이 할당을 블록의 끝에서 해제하도록 합니다. Toy 함수들은 제어 흐름이 없으므로 괜찮습니다.
  auto dealloc = memref::DeallocOp::create(rewriter, loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}

/// 이것은 낮춰진 루프의 반복을 처리하는 데 사용되는 함수 타입을 정의합니다.
/// 입력으로 OpBuilder와 반복을 위한 루프 유도 변수의 범위를 받습니다.
/// 현재 반복 인덱스에 저장할 값을 반환합니다.
using LoopIterationFn =
    function_ref<Value(OpBuilder &rewriter, ValueRange loopIvs)>;

static void lowerOpToLoops(Operation *op, PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {
  auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
  auto loc = op->getLoc();

  // 이 연산의 결과에 대한 할당 및 해제를 삽입합니다.
  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  // 모양의 차원당 하나의 루프를 갖는 아핀 루프의 중첩을 생성합니다.
  // buildAffineLoopNest 함수는 빌더, 위치 및 루프 유도 변수의 범위가 주어졌을 때
  // 가장 안쪽 루프의 본문을 구성하는 데 사용되는 콜백을 받습니다.
  SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
  SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);
  affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, tensorType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        // 리라이터와 루프 유도 변수들로 처리 함수를 호출합니다. 이 함수는
        // 현재 인덱스에 저장할 값을 반환할 것입니다.
        Value valueToStore = processIteration(nestedBuilder, ivs);
        affine::AffineStoreOp::create(nestedBuilder, loc, valueToStore, alloc,
                                      ivs);
      });

  // 이 연산을 생성된 할당으로 대체합니다.
  rewriter.replaceOp(op, alloc);
}

namespace {
//===----------------------------------------------------------------------===//
// ToyToAffine 변환 패턴: 이진 연산
//===----------------------------------------------------------------------===//

template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public OpConversionPattern<BinaryOp> {
  using OpConversionPattern<BinaryOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<BinaryOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(BinaryOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, rewriter, [&](OpBuilder &builder, ValueRange loopIvs) {
      // 내부 루프에서 'lhs'와 'rhs'의 요소에 대한 로드를 생성합니다.
      auto loadedLhs =
          affine::AffineLoadOp::create(builder, loc, adaptor.getLhs(), loopIvs);
      auto loadedRhs =
          affine::AffineLoadOp::create(builder, loc, adaptor.getRhs(), loopIvs);

      // 로드된 값들에 대해 수행되는 이진 연산을 생성합니다.
      return LoweredBinaryOp::create(builder, loc, loadedLhs, loadedRhs);
    });
    return success();
  }
};
using AddOpLowering = BinaryOpLowering<toy::AddOp, arith::AddFOp>;
using MulOpLowering = BinaryOpLowering<toy::MulOp, arith::MulFOp>;

//===----------------------------------------------------------------------===//
// ToyToAffine 변환 패턴: 상수 연산
//===----------------------------------------------------------------------===//

struct ConstantOpLowering : public OpConversionPattern<toy::ConstantOp> {
  using OpConversionPattern<toy::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    DenseElementsAttr constantValue = op.getValue();
    Location loc = op.getLoc();

    // 상수 연산을 낮출 때, 우리는 상수 값들을 해당하는 memref 할당에 할당하고 배정합니다.
    auto tensorType = llvm::cast<RankedTensorType>(op.getType());
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // 우리는 가장 큰 차원까지 상수 인덱스들을 생성할 것입니다.
    // 많은 양의 중복 연산을 피하기 위해 이러한 상수들을 미리 생성합니다.
    auto valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;

    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(0, *llvm::max_element(valueShape)))
        constantIndices.push_back(
            arith::ConstantIndexOp::create(rewriter, loc, i));
    } else {
      // 이것은 랭크가 0인 텐서의 경우입니다.
      constantIndices.push_back(
          arith::ConstantIndexOp::create(rewriter, loc, 0));
    }

    // 상수 연산은 다차원 상수를 나타내므로, 각 요소에 대해 스토어를 생성해야 합니다.
    // 다음 펑터는 상수 모양의 차원을 재귀적으로 순회하며,
    // 재귀가 기본 사례에 도달할 때 스토어를 생성합니다.
    SmallVector<Value, 2> indices;
    auto valueIt = constantValue.value_begin<FloatAttr>();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      // 마지막 차원은 재귀의 기본 사례이며, 이 시점에서 주어진 인덱스에 요소를 저장합니다.
      if (dimension == valueShape.size()) {
        affine::AffineStoreOp::create(
            rewriter, loc, arith::ConstantOp::create(rewriter, loc, *valueIt++),
            alloc, llvm::ArrayRef(indices));
        return;
      }

      // 그렇지 않으면, 현재 차원을 반복하고 인덱스들을 리스트에 추가합니다.
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    // 첫 번째 차원부터 요소 저장 재귀를 시작합니다.
    storeElements(/*dimension=*/0);

    // 이 연산을 생성된 할당으로 대체합니다.
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine 변환 패턴: Func 연산
//===----------------------------------------------------------------------===//

struct FuncOpLowering : public OpConversionPattern<toy::FuncOp> {
  using OpConversionPattern<toy::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // 우리는 다른 모든 함수들이 인라인화되었다고 예상하므로 main 함수만 낮춥니다.
    if (op.getName() != "main")
      return failure();

    // 주어진 main이 입력과 결과가 없는지 확인합니다.
    if (op.getNumArguments() || op.getFunctionType().getNumResults()) {
      return rewriter.notifyMatchFailure(op, [](Diagnostic &diag) {
        diag << "expected 'main' to have 0 inputs and 0 results";
      });
    }

    // 동일한 영역을 가진 새로운 non-toy 함수를 생성합니다.
    auto func = mlir::func::FuncOp::create(rewriter, op.getLoc(), op.getName(),
                                           op.getFunctionType());
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine 변환 패턴: Print 연산
//===----------------------------------------------------------------------===//

struct PrintOpLowering : public OpConversionPattern<toy::PrintOp> {
  using OpConversionPattern<toy::PrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // 우리는 이 패스에서 "toy.print"를 낮추지 않지만, 그것의 피연산자들을 업데이트해야 합니다.
    rewriter.modifyOpInPlace(op,
                             [&] { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine 변환 패턴: Return 연산
//===----------------------------------------------------------------------===//

struct ReturnOpLowering : public OpConversionPattern<toy::ReturnOp> {
  using OpConversionPattern<toy::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // 이 낮추기 동안, 우리는 모든 함수 호출이 인라인화되었다고 예상합니다.
    if (op.hasOperand())
      return failure();

    // 우리는 "toy.return"을 "func.return"으로 직접 낮춥니다.
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine 변환 패턴: Transpose 연산
//===----------------------------------------------------------------------===//

struct TransposeOpLowering : public OpConversionPattern<toy::TransposeOp> {
  using OpConversionPattern<toy::TransposeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, rewriter, [&](OpBuilder &builder, ValueRange loopIvs) {
      Value input = adaptor.getInput();

      // 역순 인덱스로부터 로드를 생성하여 요소들을 전치합니다.
      SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
      return affine::AffineLoadOp::create(builder, loc, input, reverseIvs);
    });
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// ToyToAffineLoweringPass
//===----------------------------------------------------------------------===//

/// 이것은 나머지 코드를 Toy 다열렉트에 유지하면서 계산 집약적인 Toy 연산들(예를 들어 matmul 같은...)을
/// 아핀 루프들로 부분적으로 낮추는 것입니다.
namespace {
struct ToyToAffineLoweringPass //해당 익명 네임스페이스 안에 실제 패스 클래스를 정의(struct vs class: class는 기본 접근 지정자가 private인 반면, struct는 public임)
    : public PassWrapper<ToyToAffineLoweringPass, OperationPass<ModuleOp>> { 
      //MLIR의 PassWrapper를 상속받아 패스를 정의(CRTP 패턴 사용: PassWrapper<Derived, Base>)
      //첫번째 인자로 자기 자신을 전달, 두번째 인자로 패스가 작동할 기본 단위(OperationPass<ModuleOp>)를 전달

  //MLIR의 타입 시스템 (TypeID)에 이 패스를 등록하는 매크로
  //왜 필요할까? MLIR은 런타임에 타입 정보를 추적하고, 패스 관리 및 최적화에 사용
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToAffineLoweringPass)

  //커맨드 라인에서 이 패스를 식별하는 데 사용되는 문자열 인자를 반환하는 메서드
  StringRef getArgument() const override { return "toy-to-affine"; }

  //이 패스를 실행할 예정인데 결과물로 이런 다열렉트들을 만들어낼거라고 미리 신고하는 함수
  //왜 필요할까? MLIR은 패스 실행 전에 필요한 다열렉트들을 미리 로드하여 메모리 효율성을 증대시킴. 미리 신고하지 않고 affine.for를 생성하려고 하면,
  //컴파일러가 죽을 가능성이 있음
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, func::FuncDialect,
                    memref::MemRefDialect>();
  }

  //실제 변환작업은 여기서 하겠다라고 선언하는 함수, final인 이유는 이 함수가 더 이상 오버라이드 되지 않도록 하기 위함
  void runOnOperation() final;
};
} // namespace

//위의 runOnOperation 메서드의 실제 구현
void ToyToAffineLoweringPass::runOnOperation() {
  // 정의해야 할 첫 번째는 변환 대상입니다. 이것은 이 낮추기의 최종 목표를 정의할 것입니다.
  //즉, 변환의 목표를 설정하는 객체를 만들고 있음. 
  //ConversionTarget의 생성자는 getContext()를 통한 전역적인 문맥을 받아 현재 MLIR 컨텍스트와 연결된 정보를 사용
  ConversionTarget target(getContext());

  // 우리는 이 낮추기에 대해 합법적인 대상인 특정 연산들 또는 다열렉트들을 정의합니다.
  // 우리의 경우, 'Affine', 'Arith', 'Func', 및 'MemRef' 다열렉트들의 조합으로 낮추고 있습니다.
  target.addLegalDialect<affine::AffineDialect, BuiltinDialect, //affine: 루프 최적화용 연산들, Builtin: 기본 MLIR 연산들(예: 함수, 블록 등)
                         arith::ArithDialect, func::FuncDialect, //arith: 산술 연산들, func: 함수 정의 및 호출 연산들
                         memref::MemRefDialect>(); //memref: 메모리 참조 연산들

  // 또한 Toy 다열렉트를 불법(Illegal)으로 정의하여 이러한 연산들 중 하나라도 변환되지 않으면
  // 변환이 실패하도록 합니다. 사실 우리는 부분적인 낮추기를 원하므로,
  // 낮추기를 원하지 않는 Toy 연산인 'toy.print'를 명시적으로 '합법(legal)'으로 표시합니다.
  // 하지만 'toy.print'는 (TensorType에서 MemRefType으로 변환함에 따라) 피연산자들이
  // 업데이트되어야 하므로, 피연산자들이 합법일 경우에만 그것을 '합법'으로 취급합니다.
  // 이 줄 덕분에 컴파일러는 어떻게든 Toy 연산을 찾아서 다른 걸로 바꾸려고 시도하게 됨
  target.addIllegalDialect<toy::ToyDialect>();

  //동적 합법: 람다 함수 [](toy::PrintOp op) { ... }가 true를 반화하면 합법, false면 불법
  target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
    return llvm::none_of(op->getOperandTypes(), //print 연산에 들어오는 피연산자 타입들 중에
                         [](Type type) { return llvm::isa<TensorType>(type); }); //TensorType이 하나라도 있으면 false 반환(불법), 없으면 true 반환(합법)
  });

  // 이제 변환 대상이 정의되었으므로, Toy 연산들을 낮출 패턴 집합을 제공하기만 하면 됩니다.
  // 불법 연산들을 합법 연산으로 어떻게 바꿀지를 담은 패턴 묶음을 만들기
  RewritePatternSet patterns(&getContext());
  patterns.add<AddOpLowering, ConstantOpLowering, FuncOpLowering, MulOpLowering,
               PrintOpLowering, ReturnOpLowering, TransposeOpLowering>(
      &getContext());

  // 대상과 재작성 패턴들이 정의되었으므로, 이제 변환을 시도할 수 있습니다.
  // 변환은 우리의 '불법' 연산들 중 하나라도 성공적으로 변환되지 않으면
  // 실패 신호를 보낼 것입니다.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns)))) //부분변환을 수행하는 함수(완전 변환 applyFullConversion도 있지만, DynamincLegal을 쓸 땐 보통 이걸 씀)
    signalPassFailure(); //만약 위의 applyPartialConversion이 실패한다면, 이 패스가 실패했음을 신호로 보냄(컴파일 중단)
}

/// Toy IR의 부분집합(예: matmul)에 대해 'Affine' 및 'Std' 다열렉트의 연산들을 낮추기 위한 패스를 생성합니다.
std::unique_ptr<Pass> mlir::toy::createLowerToAffinePass() {
  return std::make_unique<ToyToAffineLoweringPass>();
}