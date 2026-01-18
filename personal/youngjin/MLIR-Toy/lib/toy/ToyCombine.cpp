#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "toy/Dialect.h"
using namespace mlir;
using namespace toy;

// namespace {
// /// 선언적 재정의 패턴의 내용을 포함
// #include "ToyCombine.inc"
// } // namespace

/// TransposeOp를 C++ 스타일로 재작성패턴 정의
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
    // 여기 적는 패턴은 모든 toy 언어에 적용됨
    // benefit에 1을 적으면 기존 패턴보다 1만큼의 수익성이 있다는 것을 알림.
    SimplifyRedundantTranspose(mlir::MLIRContext *context)
        : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

    
    llvm::LogicalResult
    matchAndRewrite(TransposeOp op,
                    mlir::PatternRewriter &rewriter) const override {
        // Transpose의 피연산자(Operand)를 불러옴 (= transpose(x))
        mlir::Value transposeInput = op.getOperand();
        // 피연산자를 만든 Operation이 TransposeOp와 같은지 확인
        TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();
        
        if (!transposeInputOp)
            return failure();

        // 기존의 transpose(transpose(x))를 x로 변환
        rewriter.replaceOp(op, {transposeInputOp.getOperand()});
        return success();
    }
};

void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
    results.add<SimplifyRedundantTranspose>(context);
}

// ToyCombine.td에서 자동 생성되는 패턴을 사용
void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
    results.add<ReshapeReshapeOptPattern, RedundantReshapeOptPattern,
                FoldConstantReshapeOptPattern>(context);
}