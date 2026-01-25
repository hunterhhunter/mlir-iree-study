#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "toy/Dialect.h"
#include "toy/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"
#include <llvm/Support/LogicalResult.h>
#include <memory>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <utility>

using namespace mlir;

namespace {
    class PrintOpLowering : public OpConversionPattern<toy::PrintOp> {
    public:
        using OpConversionPattern<toy::PrintOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(toy::PrintOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const override {
            auto *context = rewriter.getContext();
            auto memRefType = llvm::cast<MemRefType>((*op->operand_type_begin()));
            auto memRefShape = memRefType.getShape();
            auto loc = op->getLoc();

            ModuleOp parentModule = op->getParentOfType<ModuleOp>();

            auto printfRef = getOrInsertPrintf(rewriter, parentModule);
            Value formatSpecifierCst = getOrCreateGlobalString(
                loc, rewriter, "frmt_spec", StringRef("%f \0", 4), parentModule);
            Value newLineCst = getOrCreateGlobalString(
                loc, rewriter, "nl", StringRef("\n\0", 2), parentModule);

            SmallVector<Value, 4> loopIvs;
            for (unsigned i = 0, e = memRefShape.size(); i != e; ++i) {
                auto lowerBound = arith::ConstantIndexOp::create(rewriter, loc, 0);
                auto upperBound = 
                    arith::ConstantIndexOp::create(rewriter, loc, memRefShape[i]);
                auto step = arith::ConstantIndexOp::create(rewriter, loc, 1);
                auto loop = 
                    scf::ForOp::create(rewriter, loc, lowerBound, upperBound, step);
                for (Operation &nested : make_early_inc_range(*loop.getBody()))
                    rewriter.eraseOp(&nested);
                loopIvs.push_back(loop.getInductionVar());

                rewriter.setInsertionPointToEnd(loop.getBody());

                if (i != e -1)
                    LLVM::CallOp::create(rewriter, loc, getPrintfType(context), printfRef,
                                            newLineCst);
                scf::YieldOp::create(rewriter, loc);
                rewriter.setInsertionPointToStart(loop.getBody());
            }

            auto elementLoad = 
                memref::LoadOp::create(rewriter, loc, op.getInput(), loopIvs);
            LLVM::CallOp::create(rewriter, loc, getPrintfType(context), printfRef,
                                 ArrayRef<Value>({formatSpecifierCst, elementLoad}));
            
            rewriter.eraseOp(op);
            return success();
        }

    
    private:
        /// printf의 함수 원형을 선언 = C스타일의 코드 -> MLIR 스타일 코드로 변환한 것
        /// 리턴타입, 인자타입을 정의함
        static LLVM::LLVMFunctionType getPrintfType(MLIRContext *context) {
            // printf는 출력 성공시 출력된 문자 수 반환(int) -> 그래서 i32를 리턴타입으로 잡음
            auto llvmI32Ty = IntegerType::get(context, 32);
            // printf의 1번 인자는 포맷 문자열 = 메모리 주소 -> ptr타입
            auto llvmPtrTy = LLVM::LLVMPointerType::get(context);
            // printf인자가 여러개일 수 있음 -> isVarArg로 가변인자 허용
            auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmPtrTy,
                                                          /*isVarArg=*/true);
            return llvmFnType;
        }

        // Print 함수가 있으면 symbol을 가져오고, 없으면 등록하는 함수
        static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                                   ModuleOp module) {
            auto *context = module.getContext();
            // printf 함수 먼저 검색 있으면 return
            if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
                return SymbolRefAttr::get(context, "printf");

            // 없으면 커서의 현재 위치를 저장 - insertGuard 객체 생성시 현재 커서 위치를 저장
            // 객체가 소멸될 때 커서가 생성시 저장된 커서 위치로 이동됨
            PatternRewriter::InsertionGuard insertGuard(rewriter);
            // 커서를 가장 위로 이동
            rewriter.setInsertionPointToStart(module.getBody());
            // getPrintfType에서 정의한 printf의 타입 정보들을 가져와
            // LLVMFuncOp 함수 껍데기 만들기 
            LLVM::LLVMFuncOp::create(rewriter, module.getLoc(), "printf",
                                     getPrintfType(context));
            // 등록 후 심볼 레퍼런스 반환
            return SymbolRefAttr::get(context, "printf");
        }

        // 문자열 리터럴 관리
        static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                             StringRef name, StringRef value,
                                             ModuleOp module) {
            LLVM::GlobalOp global;
            // 전역 Op 공간에 name을 가진 Op가 있나 검색 
            // 없으면 새로 만들기
            if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
                OpBuilder::InsertionGuard insertGuard(builder);
                builder.setInsertionPointToStart(module.getBody());
                auto type = LLVM::LLVMArrayType::get(
                    IntegerType::get(builder.getContext(), 8), value.size());
                global = LLVM::GlobalOp::create(builder, loc, type, /*isConstant=*/true,
                                                LLVM::Linkage::Internal, name,
                                                builder.getStringAttr(value),
                                                /*alignment=*/0);
            }

            Value globalPtr = LLVM::AddressOfOp::create(builder, loc, global);
            // 숫자 0 생성
            Value cst0 = LLVM::ConstantOp::create(builder, loc, builder.getI64Type(),
                                                  builder.getIndexAttr(0));
            // GEP (GetElementPtr) 연산: 주소 계산기
            // builder, loc, 결과 타입 (ptr), 입력 타입, 입력 주소(배열 전체 주소), 인덱스{0, 0}
            // 인덱스 2개인 이유: 배열의 0번째 + 배열내부의 0번째 원소
            return LLVM::GEPOp::create(
                builder, loc, LLVM::LLVMPointerType::get(builder.getContext()),
                global.getType(), globalPtr, ArrayRef<Value>({cst0, cst0}));
        }
    };
} // namespace

//===----------------------------------------------------------------------===//
// ToyToLLVMLoweringPass
//===----------------------------------------------------------------------===//

namespace {
    struct ToyToLLVMLoweringPass
            : public PassWrapper<ToyToLLVMLoweringPass, OperationPass<ModuleOp>> {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToLLVMLoweringPass)
        StringRef getArgument() const override { return "toy-to-llvm"; }

        void getDependentDialects(DialectRegistry &registry) const override {
            registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
        }
        void runOnOperation() final;
    };
}

void ToyToLLVMLoweringPass::runOnOperation() {
    LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();

    // LLVM 표현으로 변환하기 위해 TypeConverter를 사용
    // 이 변환기는 타입->타입 매핑을 지정
    LLVMTypeConverter typeConverter(&getContext());

    RewritePatternSet patterns(&getContext());
    // Affine -> SCF로 변환
    populateAffineToStdConversionPatterns(patterns);
    // SCF -> Control Flow로 변환 
    populateSCFToControlFlowConversionPatterns(patterns);
    // CF -> LLVM로 변환
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    // Arith 연산 -> LLVM로 변환
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    // MemRef -> LLVM로 변환
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    // Func -> LLVM로 변환
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    // Toy -> LLVM 변환 등록
    patterns.add<PrintOpLowering>(&getContext());

    auto module = getOperation();
    if (failed(applyFullConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::toy::createLowerToLLVMPass() {
  return std::make_unique<ToyToLLVMLoweringPass>();
}