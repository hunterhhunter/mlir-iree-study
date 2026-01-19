# 6장: LLVM 하향과 코드 생성

[TOC]

[이전 장](Ch-5_kr.md)에서 [방언 변환](../../DialectConversion.md) 프레임워크를 도입해 많은 `Toy` 연산을 최적화를 위해 어파인 루프 중첩으로 부분 하향했습니다. 이번 장에서는 마침내 LLVM으로 하향해 코드 생성을 진행합니다.

## LLVM으로 하향하기

이번에도 방언 변환 프레임워크를 활용해 대부분의 작업을 처리하되, 이번에는 [LLVM 방언](../../Dialects/LLVM.md)으로 완전히 변환합니다. 다행히 `toy` 연산 중 남은 것은 `toy.print` 하나뿐입니다. LLVM으로 변환을 살펴보기 전에, 우선 `toy.print`를 하향하겠습니다. 이 연산은 각 요소마다 `printf`를 호출하는 비어파인(non-affine) 루프 중첩으로 낮출 것입니다. 방언 변환 프레임워크가 [추이적 하향](../../../getting_started/Glossary.md/#transitive-lowering)을 지원하므로, 직접 LLVM 방언 연산을 생성할 필요는 없습니다. 추이적 하향이란, 프레임워크가 하나의 연산을 완전히 합법화할 때 여러 패턴을 연달아 적용할 수 있다는 뜻입니다. 여기서는 LLVM 방언의 분기 형태 대신 구조화된 루프 중첩을 생성합니다. 이후 루프 연산을 LLVM으로 낮추는 단계가 존재하기만 하면 전체 하향은 성공합니다.

하향 중 `printf` 선언은 다음처럼 가져오거나 생성할 수 있습니다.

```c++
/// 필요하다면 모듈에 printf 선언을 삽입하고, 그 심벌 참조를 반환합니다.
static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                           ModuleOp module,
                                           LLVM::LLVMDialect *llvmDialect) {
  auto *context = module.getContext();
  if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
    return SymbolRefAttr::get(context, "printf");

  // printf 함수 선언을 생성합니다. 시그니처는 i32 (i8*, ...)입니다.
  auto llvmI32Ty = IntegerType::get(context, 32);
  auto llvmI8PtrTy =
      LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
  auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy,
                                                /*isVarArg=*/true);

  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  LLVM::LLVMFuncOp::create(rewriter, module.getLoc(), "printf", llvmFnType);
  return SymbolRefAttr::get(context, "printf");
}
```

이제 printf 하향이 정의되었으므로, 하향에 필요한 요소를 지정합니다. 구성 요소는 [이전 장](Ch-5_kr.md)에서 다룬 것과 거의 같습니다.

### 변환 대상

이번에는 최상위 모듈을 제외하면 모두 LLVM 방언으로 낮춥니다.

```c++
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<mlir::LLVMDialect>();
  target.addLegalOp<mlir::ModuleOp>();
```

### 타입 변환기

이번 하향에서는 현재 사용 중인 MemRef 타입을 LLVM 표현으로 변환해야 합니다. 이를 위해 타입 변환기를 사용합니다. 변환기는 한 타입을 다른 타입으로 매핑하는 방법을 지정하며, 블록 인자를 포함한 복잡한 하향에서 필요합니다. 토이 방언 전용 타입이 없다면 기본 변환기로 충분합니다.

```c++
  LLVMTypeConverter typeConverter(&getContext());
```

### 변환 패턴

변환 대상을 정의했으니, 하향에 사용할 패턴을 제공해야 합니다. 현재 컴파일 단계에서 우리는 `toy`, `affine`, `arith`, `std` 연산이 혼합된 상태입니다. 다행히 `affine`, `arith`, `std` 방언에는 LLVM 방언으로 변환하는 패턴이 이미 제공됩니다. 이 패턴들은 [추이적 하향](../../../getting_started/Glossary.md/#transitive-lowering)에 의존해 여러 단계에 걸쳐 IR을 낮춥니다.

```c++
  mlir::RewritePatternSet patterns(&getContext());
  mlir::populateAffineToStdConversionPatterns(patterns, &getContext());
  mlir::cf::populateSCFToControlFlowConversionPatterns(patterns, &getContext());
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter,
                                                          patterns);
  mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(patterns, &getContext());

  // 남은 토이 연산은 PrintOp 하나입니다.
  patterns.add<PrintOpLowering>(&getContext());
```

### 전체 하향

이번에는 LLVM으로 완전히 하향해야 하므로 `FullConversion`을 사용합니다. 이렇게 하면 변환이 끝난 뒤 합법 연산만 남습니다.

```c++
  mlir::ModuleOp module = getOperation();
  if (mlir::failed(mlir::applyFullConversion(module, target, patterns)))
    signalPassFailure();
```

현재 예제를 다시 살펴보면 다음과 같습니다.

```mlir
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %3 = toy.mul %2, %2 : tensor<3x2xf64>
  toy.print %3 : tensor<3x2xf64>
  toy.return
}
```

이제 LLVM 방언으로 하향해 다음과 같은 코드를 얻습니다.

```mlir
llvm.func @free(!llvm<"i8*">)
llvm.func @printf(!llvm<"i8*">, ...) -> i32
llvm.func @malloc(i64) -> !llvm<"i8*">
llvm.func @main() {
  %0 = llvm.mlir.constant(1.000000e+00 : f64) : f64
  %1 = llvm.mlir.constant(2.000000e+00 : f64) : f64

  ...

^bb16:
  %221 = llvm.extractvalue %25[0] : !llvm<"{ double*, i64, [2 x i64], [2 x i64] }">
  %222 = llvm.mlir.constant(0 : index) : i64
  %223 = llvm.mlir.constant(2 : index) : i64
  %224 = llvm.mul %214, %223 : i64
  %225 = llvm.add %222, %224 : i64
  %226 = llvm.mlir.constant(1 : index) : i64
  %227 = llvm.mul %219, %226 : i64
  %228 = llvm.add %225, %227 : i64
  %229 = llvm.getelementptr %221[%228] : (!llvm."double*">, i64) -> !llvm<"f64*">
  %230 = llvm.load %229 : !llvm<"double*">
  %231 = llvm.call @printf(%207, %230) : (!llvm<"i8*">, f64) -> i32
  %232 = llvm.add %219, %218 : i64
  llvm.br ^bb15(%232 : i64)

  ...

^bb18:
  %235 = llvm.extractvalue %65[0] : !llvm<"{ double*, i64, [2 x i64], [2 x i64] }">
  %236 = llvm.bitcast %235 : !llvm<"double*"> to !llvm<"i8*">
  llvm.call @free(%236) : (!llvm<"i8*">) -> ()
  %237 = llvm.extractvalue %45[0] : !llvm<"{ double*, i64, [2 x i64], [2 x i64] }">
  %238 = llvm.bitcast %237 : !llvm<"double*"> to !llvm<"i8*">
  llvm.call @free(%238) : (!llvm<"i8*">) -> ()
  %239 = llvm.extractvalue %25[0] : !llvm<"{ double*, i64, [2 x i64], [2 x i64] }">
  %240 = llvm.bitcast %239 : !llvm<"double*"> to !llvm<"i8*">
  llvm.call @free(%240) : (!llvm<"i8*">) -> ()
  llvm.return
}
```

LLVM 방언 하향에 대한 자세한 내용은 [LLVM IR 대상](../../TargetLLVMIR.md)을 참고하세요.

## 코드 생성: MLIR 밖으로 나가기

이제 코드 생성 직전 단계에 있습니다. LLVM 방언으로 코드를 생성했으니, LLVM IR로 내보내고 JIT를 설정해 실행하면 됩니다.

### LLVM IR 내보내기

모듈이 LLVM 방언 연산만 포함하므로 LLVM IR로 내보낼 수 있습니다. 프로그래밍 방식으로는 다음 유틸리티를 호출합니다.

```c++
  std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(module);
  if (!llvmModule)
    /* ... 에러 처리 ... */
```

모듈을 LLVM IR로 내보내면 다음과 같은 코드가 생성됩니다.

```llvm
define void @main() {
  ...

102:
  %103 = extractvalue { double*, i64, [2 x i64], [2 x i64] } %8, 0
  %104 = mul i64 %96, 2
  %105 = add i64 0, %104
  %106 = mul i64 %100, 1
  %107 = add i64 %105, %106
  %108 = getelementptr double, double* %103, i64 %107
  %109 = memref.load double, double* %108
  %110 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double %109)
  %111 = add i64 %100, 1
  cf.br label %99

  ...

115:
  %116 = extractvalue { double*, i64, [2 x i64], [2 x i64] } %24, 0
  %117 = bitcast double* %116 to i8*
  call void @free(i8* %117)
  %118 = extractvalue { double*, i64, [2 x i64], [2 x i64] } %16, 0
  %119 = bitcast double* %118 to i8*
  call void @free(i8* %119)
  %120 = extractvalue { double*, i64, [2 x i64], [2 x i64] } %8, 0
  %121 = bitcast double* %120 to i8*
  call void @free(i8* %121)
  ret void
}
```

생성된 LLVM IR에 최적화를 적용하면 상당히 간결해집니다.

```llvm
define void @main()
  %0 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double 1.000000e+00)
  %1 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double 1.600000e+01)
  %putchar = tail call i32 @putchar(i32 10)
  %2 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double 4.000000e+00)
  %3 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double 2.500000e+01)
  %putchar.1 = tail call i32 @putchar(i32 10)
  %4 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double 9.000000e+00)
  %5 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double 3.600000e+01)
  %putchar.2 = tail call i32 @putchar(i32 10)
  ret void
}
```

LLVM IR을 덤프하는 전체 코드는 `examples/toy/Ch6/toyc.cpp`의 `dumpLLVMIR()` 함수에 있습니다.

```c++
int dumpLLVMIR(mlir::ModuleOp module) {
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/EnableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  llvm::errs() << *llvmModule << "\n";
  return 0;
}
```

### JIT 설정하기

LLVM 방언이 담긴 모듈을 실행하려면 `mlir::ExecutionEngine` 인프라를 사용합니다. 이는 LLVM JIT을 감싼 유틸리티로 `.mlir`을 입력으로 받습니다. JIT 설정 전체 코드는 `Ch6/toyc.cpp`의 `runJit()` 함수에서 확인할 수 있습니다.

```c++
int runJit(mlir::ModuleOp module) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/EnableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  auto maybeEngine = mlir::ExecutionEngine::create(module,
      /*llvmModuleBuilder=*/nullptr, optPipeline);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  auto invocationResult = engine->invoke("main");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }

  return 0;
}
```

빌드 디렉터리에서 직접 실행해 볼 수 있습니다.

```shell
$ echo 'def main() { print([[1, 2], [3, 4]]); }' | ./bin/toyc-ch6 -emit=jit
1.000000 2.000000
3.000000 4.000000
```

`-emit=mlir`, `-emit=mlir-affine`, `-emit=mlir-llvm`, `-emit=llvm` 옵션으로 여러 IR 단계도 비교해 보세요. [`--mlir-print-ir-after-all`](../../PassManagement.md/#ir-printing) 같은 옵션으로 파이프라인 전반의 IR 변화를 추적할 수도 있습니다.

이번 절에서 사용한 예제 코드는 test/Examples/Toy/Ch6/llvm-lowering.mlir에 있습니다.

지금까지는 원시 타입만 다뤘습니다. [다음 장](Ch-7_kr.md)에서는 합성 `struct` 타입을 추가합니다.
