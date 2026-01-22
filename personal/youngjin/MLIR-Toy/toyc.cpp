//===- toyc.cpp - The Toy Compiler ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "toy/AST.h"
#include "toy/Lexer.h"
#include "toy/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>
#include <system_error>
#include <utility>

// [변경] Chapter 1에서는 MLIR 관련 헤더가 필요 없습니다.
#ifndef CH1
#include "mlir/IR/Diagnostics.h"
#include "toy/Dialect.h"
#include "toy/MLIRGen.h"
#include "toy/Passes.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/OwningOpRef.h"

// [CH5 추가] 하강 및 변환 관련 헤더
#ifdef CH5
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
// #include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#endif

#endif

using namespace toy;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
enum InputType { Toy, MLIR };
} // namespace
static cl::opt<enum InputType> inputType(
    "x", cl::init(Toy), cl::desc("Decided the kind of output desired"),
    cl::values(clEnumValN(Toy, "toy", "load the input file as a Toy source.")),
    cl::values(clEnumValN(MLIR, "mlir",
                          "load the input file as an MLIR file")));

namespace {
enum Action { 
    None, 
    DumpAST, 
    DumpMLIR,
// [CH5 추가] Affine Lowering 결과 출력 옵션
#ifdef CH5
    DumpMLIRAffine 
#endif
};
} // namespace

static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump"))
#ifdef CH5
    , cl::values(clEnumValN(DumpMLIRAffine, "mlir-affine", 
                            "output the MLIR dump after affine lowering"))
#endif
);

// 최적화 옵션(-opt)은 Chapter 3 이상에서 활성화됩니다.
#if defined (CH3) || defined (CH4) || defined (CH5)
static cl::opt<bool> enableOpt("opt", cl::desc("Enable Optimizations"));
#endif

/// Returns a Toy AST resulting from parsing the file or a nullptr on error.
static std::unique_ptr<toy::ModuleAST>
parseInputFile(llvm::StringRef filename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }
  auto buffer = fileOrErr.get()->getBuffer();
  LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
  Parser parser(lexer);
  return parser.parseModule();
}

// [변경] loadMLIR과 dumpMLIR 함수는 Chapter 1에서 아예 컴파일되지 않게 막습니다.
#ifndef CH1
static int loadMLIR(llvm::SourceMgr &sourceMgr, mlir::MLIRContext &context,
                    mlir::OwningOpRef<mlir::ModuleOp> &module) {
  // Handle '.toy' input to the compiler.
  if (inputType != InputType::MLIR &&
      !llvm::StringRef(inputFilename).ends_with(".mlir")) {
    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST)
      return 6;
    module = mlirGen(context, *moduleAST);
    return !module ? 1 : 0;
  }

  // Otherwise, the input is '.mlir'.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }

  // Parse the input mlir.
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }
  return 0;
}

static int dumpMLIR() {
  // [CH5 변경] Dialect Registry를 사용하여 필요한 모든 Dialect 등록
  mlir::DialectRegistry registry;
  registry.insert<mlir::toy::ToyDialect>();

#ifdef CH5
  // CH5에서는 Affine, Arith, Func, MemRef 다이얼렉트가 필요함
  // mlir::func::registerAllExtensions(registry);
  registry.insert<mlir::affine::AffineDialect, 
                  mlir::memref::MemRefDialect, 
                  mlir::arith::ArithDialect, 
                  mlir::func::FuncDialect>();
#endif

  mlir::MLIRContext context(registry);

  context.getOrLoadDialect<mlir::toy::ToyDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module;
  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  if (int error = loadMLIR(sourceMgr, context, module))
    return error;

  // [변경] 최적화 및 하강 로직 수행 (CH3, CH4, CH5)
#if defined (CH3) || defined (CH4) || defined (CH5)
  mlir::PassManager pm(module.get()->getName());
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
    return 4;

  // [CH5] Affine Lowering 여부 확인
  bool isLoweringToAffine = false;
#ifdef CH5
  isLoweringToAffine = emitAction >= Action::DumpMLIRAffine;
#endif

  // [중요] 최적화 옵션이 켜져있거나(enableOpt), Affine으로 하강(isLoweringToAffine)하려면
  // 먼저 Toy 레벨의 최적화(Inlining, Shape Inference)가 선행되어야 합니다.
  // Shape Inference가 되어야 텐서 크기를 알고 메모리를 할당할 수 있기 때문입니다.
  if (enableOpt || isLoweringToAffine) {
    #if defined(CH4) || defined(CH5)
    // 1. 인라이닝 수행
    pm.addPass(mlir::createInlinerPass());
    
    // 2. Toy 함수 내부 최적화 (Shape Inference 포함)
    // 아직 toy.func 상태이므로 mlir::toy::FuncOp에 대해 수행
    mlir::OpPassManager &optPM = pm.nest<mlir::toy::FuncOp>();
    optPM.addPass(mlir::toy::createShapeInferencePass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
    #endif
  }

#ifdef CH5
  if (isLoweringToAffine) {
    // 3. Toy -> Affine 부분 하강 수행
    pm.addPass(mlir::toy::createLowerToAffinePass());

    // 4. 하강 후 정리
    // *중요*: Lowering이 끝나면 toy.func가 표준 func.func로 변환됩니다.
    // 따라서 이후 패스는 mlir::func::FuncOp를 대상으로 해야 합니다.
    mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());

    // 5. 최적화 옵션이 켜져있다면 Affine 전용 고수준 최적화 추가 수행
    if (enableOpt) {
      optPM.addPass(mlir::affine::createLoopFusionPass());
      optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
    }
  }
#endif

  // 설정된 파이프라인 실행
  if (mlir::failed(pm.run(*module)))
    return 4;

#endif // End of Optimization Block

  module->dump();
  return 0;
}
#endif // End of #ifndef CH1

static int dumpAST() {
  if (inputType == InputType::MLIR) {
    llvm::errs() << "Can't dump a Toy AST when the input is MLIR\n";
    return 5;
  }

  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST)
    return 1;

  dump(*moduleAST);
  return 0;
}

int main(int argc, char **argv) {
#ifndef CH1
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
#if defined(CH3) || defined(CH4) || defined(CH5)
  mlir::registerPassManagerCLOptions(); // PassManager 옵션
#endif
#endif

  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

  switch (emitAction) {
  case Action::DumpAST:
    return dumpAST();
  case Action::DumpMLIR:
#ifndef CH1
    return dumpMLIR();
#else
    llvm::errs() << "MLIR dump is not supported in Chapter 1.\n";
    return 1;
#endif

#ifdef CH5
  case Action::DumpMLIRAffine:
    return dumpMLIR();
#endif
  default:
    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  }

  return 0;
}