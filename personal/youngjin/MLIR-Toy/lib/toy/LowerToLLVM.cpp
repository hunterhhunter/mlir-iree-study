#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
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
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"
#include <cstddef>
#include <cstdint>
#include <llvm/Support/LogicalResult.h>
#include <memory>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <string>
#include <utility>
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

// 텐서 -> MemRef 타입 변환 헬퍼
static MemRefType convertTensorToMemRef(RankedTensorType type) {
    return MemRefType::get(type.getShape(), type.getElementType());
}

// Global MemRef 생성 헬퍼
static Value createGlobalMemRefFromDenseAttr(Location loc, ConversionPatternRewriter &rewriter,
                                             DenseElementsAttr denseAttr, Operation *op) {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto tensorType = llvm::cast<RankedTensorType>(denseAttr.getType());
    auto memRefType = convertTensorToMemRef(tensorType);
    
    static int globalCount = 0;
    std::string name = "__global_const_" + std::to_string(globalCount++) + "_" + std::to_string(reinterpret_cast<uintptr_t>(op));

    {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        rewriter.create<memref::GlobalOp>(
            loc, name, rewriter.getStringAttr("private"), memRefType,
            denseAttr, true, IntegerAttr()
        );
    }
    return rewriter.create<memref::GetGlobalOp>(loc, memRefType, name);
}

namespace {

// 1. PrintOpLowering
class PrintOpLowering : public OpConversionPattern<toy::PrintOp> {
public:
    using OpConversionPattern<toy::PrintOp>::OpConversionPattern;

    PrintOpLowering(TypeConverter &typeConverter, MLIRContext *context)
    : OpConversionPattern(typeConverter, context, 10) {}

    LogicalResult matchAndRewrite(toy::PrintOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        Value input = adaptor.getInput();
        printRecursive(op.getLoc(), input, op.getOperand().getType(), 
                       rewriter, op->getParentOfType<ModuleOp>());
        rewriter.eraseOp(op);
        return success();
    }

private:
    static LLVM::LLVMFunctionType getPrintfType(MLIRContext *context) {
        auto llvmI32Ty = IntegerType::get(context, 32);
        auto llvmPtrTy = LLVM::LLVMPointerType::get(context);
        return LLVM::LLVMFunctionType::get(llvmI32Ty, llvmPtrTy, true);
    }

    static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter, ModuleOp module) {
        auto *context = module.getContext();
        if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
            return SymbolRefAttr::get(context, "printf");

        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        LLVM::LLVMFuncOp::create(rewriter, module.getLoc(), "printf", getPrintfType(context));
        return SymbolRefAttr::get(context, "printf");
    }

    static Value getOrCreateGlobalString(Location loc, OpBuilder &builder, StringRef name, StringRef value, ModuleOp module) {
        LLVM::GlobalOp global;
        if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
            OpBuilder::InsertionGuard insertGuard(builder);
            builder.setInsertionPointToStart(module.getBody());
            auto type = LLVM::LLVMArrayType::get(IntegerType::get(builder.getContext(), 8), value.size());
            global = LLVM::GlobalOp::create(builder, loc, type, true, LLVM::Linkage::Internal, name, builder.getStringAttr(value), 0);
        }
        Value globalPtr = LLVM::AddressOfOp::create(builder, loc, global);
        Value cst0 = LLVM::ConstantOp::create(builder, loc, builder.getI64Type(), builder.getIndexAttr(0));
        return LLVM::GEPOp::create(builder, loc, LLVM::LLVMPointerType::get(builder.getContext()), global.getType(), globalPtr, ArrayRef<Value>({cst0, cst0}));
    }

    void printRecursive(Location loc, Value loweredValue, Type toyType, ConversionPatternRewriter &rewriter, ModuleOp parentModule) const {
        if (auto structType = llvm::dyn_cast<toy::StructType>(toyType)) {
            for (size_t i = 0; i < structType.getElementTypes().size(); ++i) {
                Value member = rewriter.create<LLVM::ExtractValueOp>(loc, loweredValue, ArrayRef<int64_t>{static_cast<int64_t>(i)});
                printRecursive(loc, member, structType.getElementTypes()[i], rewriter, parentModule);
            }
        } else if (auto tensorType = llvm::dyn_cast<RankedTensorType>(toyType)) {
            auto memRefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());
            printMemRef(loc, loweredValue, memRefType, rewriter, parentModule);
        } else if (auto memRefType = llvm::dyn_cast<MemRefType>(toyType)) {
            printMemRef(loc, loweredValue, memRefType, rewriter, parentModule);
        }
    }

    void printMemRef(Location loc, Value memRef, MemRefType memRefType, ConversionPatternRewriter &rewriter, ModuleOp parentModule) const {
        auto *context = rewriter.getContext();
        auto printfRef = getOrInsertPrintf(rewriter, parentModule);
        Value formatSpec = getOrCreateGlobalString(loc, rewriter, "frmt_spec", StringRef("%f \0", 4), parentModule);
        Value newLine = getOrCreateGlobalString(loc, rewriter, "nl", StringRef("\n\0", 2), parentModule);

        // Data Pointer 추출
        Value dataPtr = rewriter.create<LLVM::ExtractValueOp>(loc, memRef, ArrayRef<int64_t>{1});

        Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value numElements = rewriter.create<arith::ConstantIndexOp>(loc, memRefType.getNumElements());
        Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

        auto loop = rewriter.create<scf::ForOp>(loc, zero, numElements, step);
        
        // Loop Body
        rewriter.setInsertionPointToStart(loop.getBody());
        Value iv = loop.getInductionVar();
        Value ivI64 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), iv);
        Value ptrToElem = rewriter.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(context), rewriter.getF64Type(), dataPtr, ArrayRef<Value>{ivI64});
        Value elemVal = rewriter.create<LLVM::LoadOp>(loc, rewriter.getF64Type(), ptrToElem);
        rewriter.create<LLVM::CallOp>(loc, getPrintfType(context), printfRef, ArrayRef<Value>{formatSpec, elemVal});
        
        // [CRITICAL FIX] YieldOp 필수!
        rewriter.create<scf::YieldOp>(loc);

        // Loop End
        rewriter.setInsertionPointAfter(loop);
        rewriter.create<LLVM::CallOp>(loc, getPrintfType(context), printfRef, ArrayRef<Value>{newLine});
    }
};

// 2. StructConstantOpLowering
struct StructConstantOpLowering : public OpConversionPattern<toy::StructConstantOp> {
    using OpConversionPattern<toy::StructConstantOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(toy::StructConstantOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const final {
        auto loc = op->getLoc();
        Type destType = typeConverter->convertType(op.getType());
        if (!destType || !llvm::isa<LLVM::LLVMStructType>(destType)) 
            return rewriter.notifyMatchFailure(op, "expected convert type to be LLVM struct");

        ArrayAttr valuesAttr = op.getValue();
        Value currentStruct = rewriter.create<LLVM::UndefOp>(loc, destType);
        
        for (auto [index, attr] : llvm::enumerate(valuesAttr)) {
            auto denseAttr = llvm::cast<DenseElementsAttr>(attr);
            Value memRefVal = createGlobalMemRefFromDenseAttr(loc, rewriter, denseAttr, op);
            
            // Cast 생성
            Type llvmElemType = typeConverter->convertType(memRefVal.getType());
            auto castOp = rewriter.create<UnrealizedConversionCastOp>(loc, llvmElemType, memRefVal);
            
            currentStruct = rewriter.create<LLVM::InsertValueOp>(loc, currentStruct, castOp.getResult(0), ArrayRef<int64_t>{(int64_t)index});
        }
        rewriter.replaceOp(op, currentStruct);
        return success();
    }
};

// 3. StructAccessOpLowering
class StructAccessOpLowering : public OpConversionPattern<toy::StructAccessOp> {
    using OpConversionPattern<toy::StructAccessOp>::OpConversionPattern;
public:
    LogicalResult matchAndRewrite(toy::StructAccessOp op, OpAdaptor adapter,
                                  ConversionPatternRewriter &rewriter) const override {
        Type resultType = typeConverter->convertType(op.getType());
        if (!resultType) return failure();

        Value loweredInput = adapter.getInput();
        uint64_t index = op.getIndex();

        rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(op, resultType, loweredInput, ArrayRef<int64_t>{static_cast<int64_t>(index)});
        return success();
    }
};

} // namespace

// 4. ToyToLLVMTypeConverter
class ToyToLLVMTypeConverter : public LLVMTypeConverter {
public:
  ToyToLLVMTypeConverter(MLIRContext *ctx) : LLVMTypeConverter(ctx) {
    addConversion([&](toy::StructType type) -> Type {
      llvm::SmallVector<Type, 8> elementTypes;
      for (auto t : type.getElementTypes()) {
        Type converted = convertType(t);
        if (!converted) return {};
        elementTypes.push_back(converted);
      }
      return LLVM::LLVMStructType::getLiteral(type.getContext(), elementTypes);
    });

    addConversion([&](RankedTensorType type) -> Type {
      return convertType(MemRefType::get(type.getShape(), type.getElementType()));
    });
  }
};

// 5. ToyToLLVMLoweringPass (The Mega Pass)
namespace {
    struct ToyToLLVMLoweringPass : public PassWrapper<ToyToLLVMLoweringPass, OperationPass<ModuleOp>> {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToLLVMLoweringPass)
        StringRef getArgument() const override { return "toy-to-llvm"; }

        void getDependentDialects(DialectRegistry &registry) const override {
            registry.insert<LLVM::LLVMDialect, scf::SCFDialect, cf::ControlFlowDialect, mlir::arith::ArithDialect, mlir::memref::MemRefDialect, mlir::func::FuncDialect>();
        }
        void runOnOperation() final;
    };
}

void ToyToLLVMLoweringPass::runOnOperation() {
    LLVMConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalOp<ModuleOp>();
    
    // [CRITICAL FIXED] Cast Op를 합법으로 인정 (이게 없어서 터졌던 겁니다!)
    target.addLegalOp<UnrealizedConversionCastOp>(); 

    ToyToLLVMTypeConverter typeConverter(&getContext());
    RewritePatternSet patterns(&getContext());

    patterns.add<StructAccessOpLowering, StructConstantOpLowering, PrintOpLowering>(typeConverter, &getContext());
    populateAffineToStdConversionPatterns(patterns);
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    populateSCFToControlFlowConversionPatterns(patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);

    auto module = getOperation();
    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<mlir::Pass> mlir::toy::createLowerToLLVMPass() {
  return std::make_unique<ToyToLLVMLoweringPass>();
}