# 5장: 최적화를 위한 부분 하향

[TOC]

이제 실제 코드를 생성해 토이 언어를 실행해 보고 싶어집니다. LLVM을 사용해 코드를 생성할 예정이지만, 단순히 LLVM 빌더 인터페이스만 보여주는 것은 흥미롭지 않습니다. 대신 하나의 함수 안에서 여러 방언이 공존하도록 단계적으로 하향하는 방법을 살펴보겠습니다.

이번 장에서는 어파인 변환을 최적화하는 방언인 `Affine`에서 구현된 기존 최적화를 재사용하고자 합니다. 이 방언은 계산이 많은 부분을 겨냥하며, `toy.print` 같은 내장 함수는 표현할 수 없습니다. 오히려 그래야 합니다! 따라서 토이의 계산 집약 부분은 `Affine`으로 하향하고, [다음 장](Ch-6_kr.md)에서는 `print`를 낮추기 위해 직접 `LLVM IR` 방언을 대상으로 합니다. 이 하향 과정에서 토이가 사용하는 [TensorType](../../Dialects/Builtin.md/#rankedtensortype)에서 어파인 루프 중첩으로 인덱싱되는 [MemRefType](../../Dialects/Builtin.md/#memreftype)으로 바꿀 것입니다. 텐서는 메모리에 존재하지 않는 값 타입의 데이터 시퀀스를 추상적으로 표현합니다. 반면 MemRef는 메모리 영역을 가리키는 구체적인 참조로, 더 낮은 수준의 버퍼 접근을 의미합니다.

# 방언 변환

MLIR에는 다양한 방언이 존재하므로, 방언 사이를 [변환](../../../getting_started/Glossary.md/#conversion)하는 통합된 프레임워크가 필요합니다. 이를 위해 `DialectConversion` 프레임워크가 있습니다. 이 프레임워크는 *불법(illegal)* 연산을 *합법(legal)* 연산 집합으로 변환합니다. 사용하려면 두 가지(선택적으로 세 번째 요소)를 제공해야 합니다.

*   [변환 대상](../../DialectConversion.md/#conversion-target)

    -   어떤 연산이나 방언이 합법인지 공식적으로 정의합니다. 합법이 아닌 연산은 [합법화](../../../getting_started/Glossary.md/#legalization)를 위해 재작성 패턴이 필요합니다.

*   [재작성 패턴](../../DialectConversion.md/#rewrite-pattern-specification) 집합

    -   [패턴](../QuickstartRewrites.md)으로 구성되며, *불법* 연산을 0개 이상의 *합법* 연산으로 변환합니다.

*   (선택) [타입 변환기](../../DialectConversion.md/#type-conversion)

    -   제공하면 블록 인자의 타입을 변환할 때 사용합니다. 이번 변환에서는 필요하지 않습니다.

## 변환 대상 정의

우리는 계산이 많은 토이 연산을 `Affine`, `Arith`, `Func`, `MemRef` 방언의 연산 조합으로 변환해 이후 최적화를 진행하고자 합니다. 하향의 첫 단계로 변환 대상을 정의합니다.

```c++
void ToyToAffineLoweringPass::runOnOperation() {
  // 하향의 최종 목표를 정의합니다.
  mlir::ConversionTarget target(getContext());

  // 이번 하향에서 합법으로 간주할 방언을 지정합니다.
  target.addLegalDialect<affine::AffineDialect, arith::ArithDialect,
                         func::FuncDialect, memref::MemRefDialect>();

  // 토이 방언은 불법으로 지정해 변환되지 않은 연산이 남으면 실패하도록 합니다.
  // 다만 부분 하향을 원하므로 `toy.print`는 합법으로 표시하되,
  // 피연산자 타입이 합법적인 경우에만 허용합니다.
  target.addIllegalDialect<ToyDialect>();
  target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
    return llvm::none_of(op->getOperandTypes(), llvm::IsaPred<TensorType>);
  });
  ...
}
```

위에서 토이 방언을 불법으로 지정하고, 이어서 `toy.print`를 합법으로 설정했습니다. 이 순서를 바꾸어도 괜찮습니다. 개별 연산 정의가 방언 정의보다 항상 우선하므로 순서는 중요하지 않습니다. 자세한 내용은 `ConversionTarget::getOpInfo`를 참고하세요.

## 변환 패턴

변환 대상을 정의했다면, 이제 *불법* 연산을 *합법* 연산으로 바꾸는 방법을 정의합니다. [3장](Ch-3_kr.md)에서 소개한 정규화 프레임워크처럼, [`DialectConversion` 프레임워크](../../DialectConversion.md)는 변환 로직을 수행하기 위해 특별한 `ConversionPattern`을 사용합니다. `ConversionPattern`은 기존 `RewritePattern`과 달리 재맵된 피연산자를 담은 `operands`(또는 `adaptor`) 매개변수를 추가로 받습니다. 이는 타입 변환 시 새로운 타입의 값으로 동작하면서 기존 타입을 매치해야 하기 때문입니다. 이번 하향에서는 토이가 사용하던 [TensorType](../../Dialects/Builtin.md/#rankedtensortype)을 [MemRefType](../../Dialects/Builtin.md/#memreftype)으로 바꾸는 데 이 성질이 유용합니다. 예로 `toy.transpose` 연산을 하향하는 일부를 살펴보겠습니다.

```c++
/// `toy.transpose`를 어파인 루프 중첩으로 낮춥니다.
struct TransposeOpLowering : public OpConversionPattern<toy::TransposeOp> {
  using OpConversionPattern<toy::TransposeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, rewriter,
                   [&](OpBuilder &builder, ValueRange loopIvs) {
                     Value input = adaptor.getInput();

                     // 역 인덱스로 로드해 요소를 전치합니다.
                     SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
                     return affine::AffineLoadOp::create(builder, loc, input,
                                                         reverseIvs);
                   });
    return success();
  }
};
```

이제 하향 과정에서 사용할 패턴 목록을 준비합니다.

```c++
void ToyToAffineLoweringPass::runOnOperation() {
  ...

  // 하향에 사용할 패턴 집합을 제공합니다.
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<..., TransposeOpLowering>(&getContext());

  ...
```

## 부분 하향 수행

패턴을 정의했으니 실제 하향을 수행할 수 있습니다. `DialectConversion` 프레임워크는 다양한 모드를 제공하지만, 이번에는 `toy.print`를 아직 낮추지 않으므로 부분 하향을 사용합니다.

```c++
void ToyToAffineLoweringPass::runOnOperation() {
  ...

  // 대상과 패턴이 정의되었으니 변환을 시도합니다.
  // 불법 연산이 남아 있으면 실패합니다.
  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, patterns)))
    signalPassFailure();
}
```

### 부분 하향 시 고려 사항

하향 결과를 보기 전에 부분 하향 설계 시 고려해야 할 점을 짚어보겠습니다. 이번 하향에서는 값 타입인 TensorType을 메모리에 할당된 MemRefType으로 바꿉니다. 하지만 `toy.print`는 낮추지 않았으므로, 두 세계를 임시로 연결해야 합니다. 접근 방식마다 장단점이 있습니다.

*   버퍼에서 `load`를 생성해 값 타입 인스턴스를 만듦

    `toy.print` 정의를 그대로 유지할 수 있지만, 최적화 이후에야 보이는 전체 복사가 발생해 `Affine` 방언에서의 최적화 폭이 줄어듭니다.

*   낮은 타입을 다루는 새로운 `toy.print` 버전 생성

    숨겨진 복사가 없어 최적화에 유리하지만, 기존 정의와 중복되는 새로운 연산 정의가 필요합니다. [ODS](../../DefiningDialects/Operations.md)에서 베이스 클래스를 정의해 중복을 줄일 수 있지만 운영은 별도로 해야 합니다.

*   기존 `toy.print`가 낮은 타입도 처리하도록 업데이트

    구현이 단순하고 추가 복사도 없으며 새로운 연산 정의도 만들 필요 없습니다. 다만 토이 방언에서 추상화 레벨이 섞이는 단점이 있습니다.

간단하게 세 번째 방법을 택하겠습니다. 작동 정의 파일에서 PrintOp의 타입 제약을 업데이트하면 됩니다.

```tablegen
def PrintOp : Toy_Op<"print"> {
  ...

  // 출력할 입력 텐서를 받습니다.
  // 부분 하향 중 상호 운용을 위해 F64MemRef도 허용합니다.
  let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input);
}
```

## 토이 전체 예제

구체적인 예제를 살펴보겠습니다.

```mlir
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %3 = toy.mul %2, %2 : tensor<3x2xf64>
  toy.print %3 : tensor<3x2xf64>
  toy.return
}
```

어파인 하향을 파이프라인에 추가하면 다음과 같은 IR을 생성합니다.

```mlir
func.func @main() {
  %cst = arith.constant 1.000000e+00 : f64
  %cst_0 = arith.constant 2.000000e+00 : f64
  %cst_1 = arith.constant 3.000000e+00 : f64
  %cst_2 = arith.constant 4.000000e+00 : f64
  %cst_3 = arith.constant 5.000000e+00 : f64
  %cst_4 = arith.constant 6.000000e+00 : f64

  // 입력과 출력을 위한 버퍼를 할당합니다.
  %0 = memref.alloc() : memref<3x2xf64>
  %1 = memref.alloc() : memref<3x2xf64>
  %2 = memref.alloc() : memref<2x3xf64>

  // 입력 버퍼를 상수 값으로 초기화합니다.
  affine.store %cst, %2[0, 0] : memref<2x3xf64>
  affine.store %cst_0, %2[0, 1] : memref<2x3xf64>
  affine.store %cst_1, %2[0, 2] : memref<2x3xf64>
  affine.store %cst_2, %2[1, 0] : memref<2x3xf64>
  affine.store %cst_3, %2[1, 1] : memref<2x3xf64>
  affine.store %cst_4, %2[1, 2] : memref<2x3xf64>

  // 입력 버퍼에서 전치 값을 읽어 다음 버퍼에 저장합니다.
  affine.for %arg0 = 0 to 3 {
    affine.for %arg1 = 0 to 2 {
      %3 = affine.load %2[%arg1, %arg0] : memref<2x3xf64>
      affine.store %3, %1[%arg0, %arg1] : memref<3x2xf64>
    }
  }

  // 곱셈 후 출력 버퍼에 저장합니다.
  affine.for %arg0 = 0 to 3 {
    affine.for %arg1 = 0 to 2 {
      %3 = affine.load %1[%arg0, %arg1] : memref<3x2xf64>
      %4 = affine.load %1[%arg0, %arg1] : memref<3x2xf64>
      %5 = arith.mulf %3, %4 : f64
      affine.store %5, %0[%arg0, %arg1] : memref<3x2xf64>
    }
  }

  // 버퍼의 값을 출력합니다.
  toy.print %0 : memref<3x2xf64>
  memref.dealloc %2 : memref<2x3xf64>
  memref.dealloc %1 : memref<3x2xf64>
  memref.dealloc %0 : memref<3x2xf64>
  return
}
```

## 어파인 최적화 활용

단순한 하향은 정확하지만 효율성 면에서는 미흡합니다. 예를 들어 `toy.mul` 하향 결과에는 불필요한 로드가 있습니다. `LoopFusion`과 `AffineScalarReplacement` 패스를 추가하면 다음과 같이 개선됩니다.

```mlir
func.func @main() {
  %cst = arith.constant 1.000000e+00 : f64
  %cst_0 = arith.constant 2.000000e+00 : f64
  %cst_1 = arith.constant 3.000000e+00 : f64
  %cst_2 = arith.constant 4.000000e+00 : f64
  %cst_3 = arith.constant 5.000000e+00 : f64
  %cst_4 = arith.constant 6.000000e+00 : f64

  // 입력과 출력을 위한 버퍼를 할당합니다.
  %0 = memref.alloc() : memref<3x2xf64>
  %1 = memref.alloc() : memref<2x3xf64>

  // 입력 버퍼를 상수 값으로 초기화합니다.
  affine.store %cst, %1[0, 0] : memref<2x3xf64>
  affine.store %cst_0, %1[0, 1] : memref<2x3xf64>
  affine.store %cst_1, %1[0, 2] : memref<2x3xf64>
  affine.store %cst_2, %1[1, 0] : memref<2x3xf64>
  affine.store %cst_3, %1[1, 1] : memref<2x3xf64>
  affine.store %cst_4, %1[1, 2] : memref<2x3xf64>

  affine.for %arg0 = 0 to 3 {
    affine.for %arg1 = 0 to 2 {
      // 입력 버퍼에서 전치 값을 읽습니다.
      %2 = affine.load %1[%arg1, %arg0] : memref<2x3xf64>

      // 곱셈 후 출력 버퍼에 저장합니다.
      %3 = arith.mulf %2, %2 : f64
      affine.store %3, %0[%arg0, %arg1] : memref<3x2xf64>
    }
  }

  // 버퍼의 값을 출력합니다.
  toy.print %0 : memref<3x2xf64>
  memref.dealloc %1 : memref<2x3xf64>
  memref.dealloc %0 : memref<3x2xf64>
  return
}
```

불필요한 할당이 제거되고, 두 루프가 융합되며, 중복 `load`가 없어진 것을 확인할 수 있습니다. `toyc-ch5`를 빌드해 `toyc-ch5 test/Examples/Toy/Ch5/affine-lowering.mlir -emit=mlir-affine` 명령으로 직접 실행해 보세요. `-opt`를 추가해 최적화도 확인할 수 있습니다.

이번 장에서는 최적화를 목적으로 부분 하향의 일부 측면을 살펴보았습니다. [다음 장](Ch-6_kr.md)에서는 LLVM을 대상으로 코드 생성을 진행하며 방언 변환을 계속 다룹니다.
