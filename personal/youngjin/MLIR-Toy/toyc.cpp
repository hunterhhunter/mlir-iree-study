//===- toyc.cpp - The Toy Compiler ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the entry point for the Toy compiler.
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
#include <mlir/IR/OwningOpRef.h>
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
enum Action { None, DumpAST, DumpMLIR };
} // namespace
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")));

// 최적화 옵션(-opt)은 Chapter 3에서만 활성화됩니다.
#if defined (CH3) || defined (CH4)
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
  mlir::MLIRContext context;
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::toy::ToyDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module;
  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  if (int error = loadMLIR(sourceMgr, context, module))
    return error;

  // [변경] 실제 최적화 수행 로직은 Chapter 3일 때만 컴파일합니다.
#if defined (CH3) || defined (CH4)
  if (enableOpt) {
    mlir::PassManager pm(module.get()->getName());
    // Apply any generic pass manager command line options and run the pipeline.
    if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
      return 4;
    
    #ifdef CH4
    // ch4-inliner에서 추가
    pm.addPass(mlir::createInlinerPass());
    // Now that there is only one function, we can infer the shapes of each of
    // the operations.
    // uncOp는 외부 환경과 격리된(IsolateFromAbove) 특성을 가져
    // 함수 내부의 최적화가 다른 함수의 IR구조에 직접적인 영향을 주지 않음
    // 따라서 pm.nest를 사용하면 개별 func연산에 집중해 병렬 최적화 + 패스 로컬리티가 가능함.
    mlir::OpPassManager &optPM = pm.nest<mlir::toy::FuncOp>();
    optPM.addPass(mlir::toy::createShapeInferencePass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());

    #endif
    
    // Add a run of the canonicalizer to optimize the mlir module.
    pm.addNestedPass<mlir::toy::FuncOp>(mlir::createCanonicalizerPass());
    if (mlir::failed(pm.run(*module)))
      return 4;
  }
#endif

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
#if defined(CH3) || defined(CH4)
  mlir::registerPassManagerCLOptions(); // PassManager 옵션은 CH3만
#endif
#endif

  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

  switch (emitAction) {
  case Action::DumpAST:
    return dumpAST();
  case Action::DumpMLIR:
// [변경] dumpMLIR 호출도 Chapter 1에서는 불가능하게 막습니다.
#ifndef CH1
    return dumpMLIR();
#else
    llvm::errs() << "MLIR dump is not supported in Chapter 1.\n";
    return 1;
#endif
  default:
    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  }

  return 0;
}