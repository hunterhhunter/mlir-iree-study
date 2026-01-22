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
#include <llvm/Support/LogicalResult.h>
#include <memory>
#include <mlir/IR/Builders.h>
#include <utility>

using namespace mlir;

//===----------------------------------------------------------------------===//
// ToyToAffine Conversion Patterns
//===----------------------------------------------------------------------===//

/// Convert the given RankedTensorType into the corresponding MemRefType
/// 텐서->MemRef로 타입만 바꾸는 함수임.
struct MemRefType convertTensorToMemRef(RankedTensorType type) {
    return MemRefType::get(type.getShape(), type.getElementType());
}

// Insert an allocation and deallocation for the given MemrefType.
static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
    auto alloc = memref::AllocOp::create(rewriter, loc, type);

    // 블록의 시작에 할당
    auto *parentBlock = alloc->getBlock();
    alloc->moveBefore(&parentBlock->front());

    // 블록의 끝에 alloc을 할당 해제
    // toy 함수들이 제어 흐름을 가지고 있지 않아서 괜찮음.
    auto dealloc = memref::DeallocOp::create(rewriter, loc, alloc);
    dealloc->moveBefore(&parentBlock->back());
    return alloc;
}

using LoopIterationFn = 
    function_ref<Value(OpBuilder &rewriter, ValueRange loopIvs)>;

static void lowerOpToLoops(Operation *op, PatternRewriter &rewriter,
                           LoopIterationFn precessIteration) {
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
    auto loc = op->getLoc();

    // Insert an Allocation and Deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);
    affine::buildAffineLoopNest(
        rewriter, loc, lowerBounds, tensorType.getShape(), steps,
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
            Value valueToStore = precessIteration(nestedBuilder, ivs);
            affine::AffineStoreOp::create(nestedBuilder, loc, valueToStore, alloc,
                                          ivs);
        });

    rewriter.replaceOp(op, alloc);
}

//===----------------------------------------------------------------------===//
// ToyToAffine Conversion Patterns: Binary operations
//===----------------------------------------------------------------------===//

template<typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public OpConversionPattern<BinaryOp> {
    using OpConversionPattern<BinaryOp>::OpConversionPattern;
    using OpAdaptor = typename OpConversionPattern<BinaryOp>::OpAdaptor;

    llvm::LogicalResult
    matchAndRewrite(BinaryOp op, OpAdaptor adaptor,
                   ConversionPatternRewriter &rewriter) const final {
        auto loc = op->getLoc();
        lowerOpToLoops(op, rewriter, [&](OpBuilder &builder, ValueRange loopIvs) {
            auto loadedLhs = affine::AffineLoadOp::create(builder, loc, adaptor.getLhs(), loopIvs);
            auto loadedRhs = affine::AffineLoadOp::create(builder, loc, adaptor.getRhs(), loopIvs);

            return LoweredBinaryOp::create(builder, loc, loadedLhs, loadedRhs);
        });
        return success();
    }
};
using AddOpLowering = BinaryOpLowering<toy::AddOp, arith::AddFOp>;
using MulOpLowering = BinaryOpLowering<toy::MulOp, arith::MulFOp>;

//===----------------------------------------------------------------------===//
// ToyToAffine Conversion Patterns: Constant operations
//===----------------------------------------------------------------------===//

struct ConstantOpLowering : public OpConversionPattern<toy::ConstantOp> {
  using OpConversionPattern<toy::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    DenseElementsAttr constantValue = op.getValue();
    Location loc = op.getLoc();

    // When lowering the constant operation, we allocate and assign the constant
    // values to a corresponding memref allocation.
    auto tensorType = llvm::cast<RankedTensorType>(op.getType());
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // We will be generating constant indices up-to the largest dimension.
    // Create these constants up-front to avoid large amounts of redundant
    // operations.
    auto valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;

    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(0, *llvm::max_element(valueShape)))
        constantIndices.push_back(
            arith::ConstantIndexOp::create(rewriter, loc, i));
    } else {
      // This is the case of a tensor of rank 0.
      constantIndices.push_back(
          arith::ConstantIndexOp::create(rewriter, loc, 0));
    }

    // The constant operation represents a multi-dimensional constant, so we
    // will need to generate a store for each of the elements. The following
    // functor recursively walks the dimensions of the constant shape,
    // generating a store when the recursion hits the base case.
    SmallVector<Value, 2> indices;
    auto valueIt = constantValue.value_begin<FloatAttr>();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      // The last dimension is the base case of the recursion, at this point
      // we store the element at the given index.
      if (dimension == valueShape.size()) {
        affine::AffineStoreOp::create(
            rewriter, loc, arith::ConstantOp::create(rewriter, loc, *valueIt++),
            alloc, llvm::ArrayRef(indices));
        return;
      }

      // Otherwise, iterate over the current dimension and add the indices to
      // the list.
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    // Start the element storing recursion from the first dimension.
    storeElements(/*dimension=*/0);

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

// struct ConstantOpLowering : public OpConversionPattern<toy::ConstantOp> {
//     using OpConversionPattern<toy::ConstantOp>::OpConversionPattern;

//     LogicalResult
//     matchAndRewrite(toy::ConstantOp op, OpAdaptor adaptor,
//                     ConversionPatternRewriter &rewriter) const final {
//         DenseElementsAttr constantValue = op.getValue();
//         Location loc = op.getLoc();

//         auto tensorType = llvm::cast<RankedTensorType>(op.getType());
//         auto memRefType = convertTensorToMemRef(tensorType);
//         auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

//         auto valueShape = memRefType.getShape(); // <2x3>이면 [2, 3]
//         // SmallVector<T, N>은 원소가 N개 이하면 스택 메모리를, 초과면 힙을 할당하도록 하는 자료구조
//         // 그래서 인덱스 숫자가 8개 초과시 힙을 사용할 수 있게해서
//         // 숫자가 커지더라도 힙을 사용해 문제가 없도록 함.
//         SmallVector<Value, 8> constantIndices; // 인덱스 값들의 캐시 저장소

//         if (!valueShape.empty()) {
//             // 모양 중 가장 큰 숫자를 찾음. <2x3>  이면 3
//             // 0 부터 2까지(0, 1, 2) 미리 만듦
//             // <2x3>이면 행: 0, 1 / 열: 0, 1, 2 이므로 결국 필요한 숫자는 0, 1, 2
//             // 그래서 텐서 순회에 필요한 숫자를 미리 저장
//             // 인덱스도 IR 상의 연산 결과여야함.
//             for (auto i : llvm::seq<int64_t>(0, *llvm::max_element(valueShape)))
//                 constantIndices.push_back(
//                     arith::ConstantIndexOp::create(rewriter, loc, i));
//         } else {
//             constantIndices.push_back(
//                 arith::ConstantIndexOp::create(rewriter, loc, 0));
//         }

//         // 실제 memRef에 상수값을 저장하는 로직
//         // 2차원을 넘어가면 heap에 할당
//         SmallVector<Value, 2> indices;
//         auto valueIt = constantValue.value_begin<FloatAttr>();
//         // DFS로 indices에 좌표를 push, pop해가며 값을 저장하는 로직
//         // 비효율적이니 상용 컴파일러에서는 배열을 글로벌 데이터 섹션 + memcpy로 한방에 복사함.
//         // 또 IR 자체에 affine.for 루프를 생성해줌.
//         // 재귀를 쓰는 이유는 동적으로 for 구문을 생성할 수 있는 문법이 없기 때문
//         std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
//             if (dimension == valueShape.size()) {
//                 affine::AffineStoreOp::create(
//                     rewriter, loc, arith::ConstantOp::create(rewriter, loc, *valueIt++),
//                     alloc, llvm::ArrayRef(indices));
//             }

//             for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
//                 indices.push_back(constantIndices[i]);
//                 storeElements(dimension + 1);
//                 indices.pop_back();
//             }
//         };

//         storeElements(/*dimension=*/0);
        
//         rewriter.replaceOp(op, alloc);

//         return success();
//     }
// };

//===----------------------------------------------------------------------===//
// ToyToAffine Conversion Patterns: Func operations
//===----------------------------------------------------------------------===//

struct FuncOpLowering : public OpConversionPattern<toy::FuncOp> {
    using OpConversionPattern<toy::FuncOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(toy::FuncOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const final {
        // 메인만 한정해서 재작성하는 함수
        if (op.getName() != "main")
            return failure();

        if (op.getNumArguments() || op.getFunctionType().getNumResults()) {
            return rewriter.notifyMatchFailure(op, [](Diagnostic &diag) {
                diag << "expected 'main' to have 0 inputs and 0 results";
            });
        }

        auto func = mlir::func::FuncOp::create(rewriter, op.getLoc(), op.getName(),
                                               op.getFunctionType());

        // 기존 toy.func가 가지던 Region, body를 func.func로 옮기는 작업
        rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
        rewriter.eraseOp(op);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// ToyToAffine Conversion Patterns: Print operations
//===----------------------------------------------------------------------===//

struct PrintOpLowering : public OpConversionPattern<toy::PrintOp> {
    using OpConversionPattern<toy::PrintOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(toy::PrintOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const final {
        // 기존의 toy.print 껍데기를 그대로 유지하겠다는 의미로 피연산자만 Tensor -> MemRef로 변한걸 적용
        rewriter.modifyOpInPlace(op, 
                                 [&] { op->setOperands(adaptor.getOperands()); });
        return success();
   }
};

//===----------------------------------------------------------------------===//
// ToyToAffine Conversion Patterns: Return operations
//===----------------------------------------------------------------------===//

struct ReturnOpLowering : public OpConversionPattern<toy::ReturnOp> {
    using OpConversionPattern<toy::ReturnOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(toy::ReturnOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const final {
    if (op.hasOperand())
        return failure();

    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return success();
    }
};

//===----------------------------------------------------------------------===//
// ToyToAffine Conversion Patterns: Transpose operations
//===----------------------------------------------------------------------===//

struct TransposeOpLowering : public OpConversionPattern<toy::TransposeOp> {
    using OpConversionPattern<toy::TransposeOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(toy::TransposeOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const final {
        auto loc = op->getLoc();
        lowerOpToLoops(op, rewriter, [&](OpBuilder &builder, ValueRange loopIvs) {
            Value input = adaptor.getInput();

            SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
            return affine::AffineLoadOp::create(builder, loc, input, reverseIvs);
        });
        return success();
    }
};

//===----------------------------------------------------------------------===//
// ToyToAffineLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine lopops of the toy operations that are
/// computationally intensive (like matmul for exampl...) while keeping the
/// rest of the code in the Toy dialect.
namespace {
struct ToyToAffineLoweringPass 
    : public PassWrapper<ToyToAffineLoweringPass, OperationPass<ModuleOp>> {
    
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToAffineLoweringPass)
    StringRef getArgument() const override { return "toy-to-affine"; }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<affine::AffineDialect, func::FuncDialect,
                        memref::MemRefDialect>();
    }

    void runOnOperation() final;
};
} // namespace

void ToyToAffineLoweringPass::runOnOperation() {
    // 첫 단계는 변환 대상을 정의하는 것. 이는 최종 lowering 대상을 정의하는 것
    ConversionTarget target(getContext());

    // lowering에 적합한 특정 연산, 방언을 정의한다.
    // Toy에서는 Affine, Arith, Func, MemRef Dialect의 조합으로 lowering 할 것이므로
    target.addLegalDialect<affine::AffineDialect, BuiltinDialect, 
                           arith::ArithDialect, func::FuncDialect,
                           memref::MemRefDialect>();

    target.addIllegalDialect<toy::ToyDialect>();

    target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
        return llvm::none_of(op->getOperandTypes(),
                             [](Type type) { return llvm::isa<TensorType>(type); });
    });

    RewritePatternSet patterns(&getContext());
    patterns.add<AddOpLowering, ConstantOpLowering, FuncOpLowering, MulOpLowering,
                PrintOpLowering, ReturnOpLowering, TransposeOpLowering>(
        &getContext());
    
    if (failed(
            applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
}

/// create a pass for lowering operations in the Affine and std dialects
/// for a subset of the toy ir
std::unique_ptr<Pass> mlir::toy::createLowerToAffinePass() {
    return std::make_unique<ToyToAffineLoweringPass>();
}