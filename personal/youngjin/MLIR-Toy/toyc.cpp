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

// 기본 LLVM Support 헤더 (모든 챕터 공통)
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

// JIT 타겟 머신 설정 시 필요 (DataLayout 등)
#include "llvm/IR/DataLayout.h"

#include <cassert> // Input 1에 있던 assert 포함
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

// [Chapter 1] 이후부터 필요한 MLIR 공통 헤더
#ifndef CH1
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "toy/Dialect.h"
#include "toy/MLIRGen.h"
#include "toy/Passes.h"
#endif

// [Chapter 5] 최적화 및 부분 하강 (Partial Lowering)
#ifdef CH5
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#endif

// [Chapter 6] LLVM 코드 생성 및 JIT 컴파일 (Code Generation)
#ifdef CH6
// 1. LLVM Dialect 관련
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"

// 2. Target Translation (MLIR -> LLVM IR 변환 인터페이스)
// 이 헤더들이 없으면 "translation interface not registered" 에러가 발생합니다.
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

// 3. JIT Execution Engine (실행기)
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"

// 4. LLVM Backend Support (타겟 머신 초기화)
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetSelect.h" 
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
    DumpMLIRAffine,
#endif
#ifdef CH6
    DumpMLIRLLVM,
    DumpLLVMIR,
    RunJIT
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

#ifdef CH6
    , cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm", "output the MLIR dump after llvm lowering"))
    , cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")),
      cl::values(clEnumValN(RunJIT, "jit", "run the JIT execution engine"))
#endif
);

// 최적화 옵션(-opt)은 Chapter 3 이상에서 활성화됩니다.
#if defined (CH3) || defined (CH4) || defined (CH5) || defined(CH6)
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

#ifdef CH6
static int dumpLLVMIR(mlir::ModuleOp module) {
  // Register the translation to LLVM IR with the MLIR context.
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // Convert the module to LLVM IR in a new LLVM IR context.
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Configure the LLVM Module
  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    llvm::errs() << "Could not create JITTargetMachineBuilder\n";
    return -1;
  }

  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    llvm::errs() << "Could not create TargetMachine\n";
    return -1;
  }
  mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(),
                                                        tmOrError.get().get());

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  llvm::errs() << *llvmModule << "\n";
  return 0;
}

static int runJit(mlir::ModuleOp module) {
  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Register the translation from MLIR to LLVM IR, which must happen before we
  // can JIT-compile.
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // An optimization pipeline to use within the execution engine.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.transformer = optPipeline;
  auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  // Invoke the JIT-compiled function.
  auto invocationResult = engine->invokePacked("main");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }

  return 0;
}
#endif

static int dumpMLIR() {
  // [CH5 변경] Dialect Registry를 사용하여 필요한 모든 Dialect 등록
  mlir::DialectRegistry registry;
  registry.insert<mlir::toy::ToyDialect>();

#if defined(CH5) || defined(CH6) || defined(CH7)
  // CH5에서는 Affine, Arith, Func, MemRef 다이얼렉트가 필요함
  // 인라이닝 동작을 위한 등록 
  mlir::func::registerAllExtensions(registry);

  registry.insert<mlir::affine::AffineDialect, 
                  mlir::memref::MemRefDialect, 
                  mlir::arith::ArithDialect, 
                  mlir::func::FuncDialect>();
#endif

#if defined(CH6) || defined(CH7)
  registry.insert<mlir::LLVM::LLVMDialect>();
  mlir::LLVM::registerInlinerInterface(registry);
#endif

  mlir::MLIRContext context(registry);

  context.getOrLoadDialect<mlir::toy::ToyDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module;
  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  if (int error = loadMLIR(sourceMgr, context, module))
    return error;

  // [변경] 최적화 및 하강 로직 수행 (CH3, CH4, CH5)
#if defined(CH3) || defined(CH4) || defined(CH5) || defined(CH6) || defined(CH7)
  mlir::PassManager pm(module.get()->getName());
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
    return 4;

  // [CH5] Affine Lowering 여부 확인
  bool isLoweringToAffine = false;
#if defined(CH5) || defined(CH6) || defined(CH7)
  isLoweringToAffine = emitAction >= Action::DumpMLIRAffine;
#endif

// Ch6 LLVM 
#if defined(CH6) || defined(CH7)
  // LLVM으로 가거나 JIT를 하려면 당연히 Affine 하강이 선행되어야 함
  bool isLoweringToLLVM = emitAction >= Action::DumpMLIRLLVM;
  isLoweringToAffine |= isLoweringToLLVM;
#endif

  // [중요] 최적화 옵션이 켜져있거나(enableOpt), Affine으로 하강(isLoweringToAffine)하려면
  // 먼저 Toy 레벨의 최적화(Inlining, Shape Inference)가 선행되어야 합니다.
  // Shape Inference가 되어야 텐서 크기를 알고 메모리를 할당할 수 있기 때문입니다.
  if (enableOpt || isLoweringToAffine) {
    #if defined(CH4) || defined(CH5) || defined(CH6) || defined(CH7)
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

#if defined(CH5) || defined(CH6) || defined(CH7)
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

// CH6 LLVm Lowering Pass 추가
#if defined(CH6) || defined(CH7)
  if (isLoweringToLLVM) {
    pm.addPass(mlir::toy::createLowerToLLVMPass());

    // [Chapter 7 추가] 디버그 정보(Line Table) 생성을 위한 DIScope 패스
    // 이 패스는 LLVM Dialect 내의 함수들에 디버그 범위를 지정하여
    // 나중에 JIT나 디버거에서 소스 위치를 추적할 수 있게 돕습니다.
    #ifdef CH7
    pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
    #endif
  }
#endif

  // 설정된 파이프라인 실행
  if (mlir::failed(pm.run(*module)))
    return 4;
#endif // End of Optimization Block

#if defined(CH6) || defined(CH7)
  if (emitAction == Action::DumpLLVMIR)
    return dumpLLVMIR(*module);
  if (emitAction == Action::RunJIT)
    return runJit(*module);
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

// CH6 액션 연결
#ifdef CH6
    case Action::DumpMLIRLLVM:
    case Action::DumpLLVMIR:
    case Action::RunJIT:
      return dumpMLIR();
#endif
  default:
    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  }

  return 0;
}