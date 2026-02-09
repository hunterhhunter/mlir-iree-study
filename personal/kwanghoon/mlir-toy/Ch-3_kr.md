# 3장: 고수준 언어 특화 분석과 변환

[TOC]

입력 언어의 의미를 밀접하게 반영한 방언을 만들면, MLIR에서 언어 AST에서 주로 수행되던 고수준 분석·변환·최적화를 실행할 수 있습니다. 예를 들어 `clang`은 C++ 템플릿 인스턴스화를 위해 상당히 [무거운 메커니즘](https://clang.llvm.org/doxygen/classclang_1_1TreeTransform.html)을 사용합니다.

컴파일러 변환은 지역적(local)과 전역적(global)으로 나눌 수 있습니다. 이 장에서는 토이 방언과 그 고수준 의미를 활용해 LLVM에서는 구현이 어려운 지역 패턴 매치 기반 변환을 수행하는 방법을 살펴봅니다. 이를 위해 MLIR의 [Generic DAG Rewriter](../../PatternRewriter.md)를 사용합니다.

패턴 매치 기반 변환을 구현하는 방법은 두 가지입니다.

1. 명령형 C++ 패턴 매치와 재작성
2. 선언적 규칙 기반 테이블 주도 [Declarative Rewrite Rules](../../DeclarativeRewrites.md)(DRR)

DRR을 사용하려면 [2장](Ch-2_kr.md)에서 설명한 것처럼 연산이 ODS로 정의되어 있어야 합니다.

## C++ 스타일 패턴 매치와 재작성으로 Transpose 최적화하기

간단한 패턴부터 시작해 보겠습니다. 서로 상쇄되는 전치 두 번을 제거하는 `transpose(transpose(X)) -> X` 변환입니다. 이에 해당하는 토이 예시는 다음과 같습니다.

```toy
def transpose_transpose(x) {
  return transpose(transpose(x));
}
```

이는 다음 IR로 표현됩니다.

```mlir
toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  %1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
  toy.return %1 : tensor<*xf64>
}
```

이 변환은 토이 IR에서는 간단히 매치되지만 LLVM에서는 어렵습니다. 예를 들어 현재 Clang은 임시 배열을 제거하지 못하며, 순진한 전치 연산은 다음과 같은 루프로 표현됩니다.

```c++
#define N 100
#define M 100

void sink(void *);
void double_transpose(int A[N][M]) {
  int B[M][N];
  for(int i = 0; i < N; ++i) {
    for(int j = 0; j < M; ++j) {
       B[j][i] = A[i][j];
    }
  }
  for(int i = 0; i < N; ++i) {
    for(int j = 0; j < M; ++j) {
       A[i][j] = B[j][i];
    }
  }
  sink(A);
}
```

IR에서 트리 형태 패턴을 매치해 다른 연산 집합으로 대체하는 간단한 C++ 재작성 기법을 사용하려면, MLIR `Canonicalizer` 패스에 `RewritePattern`을 구현해 연결할 수 있습니다.

```c++
/// transpose(transpose(x)) -> x 접기
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
  /// 모든 toy.transpose에 이 패턴을 적용하도록 등록합니다.
  /// "benefit"은 패턴의 우선순위를 정하는 데 사용됩니다.
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

  /// 패턴을 매치하고 재작성하는 메서드입니다. rewriter는 재작성 순서를
  /// 조정하며, 이 안에서 IR을 변경할 때 사용해야 합니다.
  llvm::LogicalResult
  matchAndRewrite(TransposeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // 현재 transpose의 입력을 살펴봅니다.
    mlir::Value transposeInput = op.getOperand();
    TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

    // 입력이 다른 transpose가 아니면 매치 실패입니다.
    if (!transposeInputOp)
      return failure();

    // 중복 transpose를 발견했습니다. rewriter를 사용해 교체합니다.
    rewriter.replaceOp(op, {transposeInputOp.getOperand()});
    return success();
  }
};
```

이 재작성 구현은 `ToyCombine.cpp`에 있습니다. [정규화 패스](../../Canonicalization.md)는 등록된 변환을 탐욕적·반복적으로 적용합니다. 새로운 변환을 적용하려면 `hasCanonicalizer = 1`을 설정하고 패턴을 정규화 프레임워크에 등록해야 합니다.

```c++
// 정규화 프레임워크에 패턴을 등록합니다.
void TransposeOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<SimplifyRedundantTranspose>(context);
}
```

`toyc.cpp`의 메인도 갱신해 최적화 파이프라인을 추가해야 합니다. MLIR은 LLVM과 유사하게 `PassManager`를 통해 최적화를 실행합니다.

```c++
  mlir::PassManager pm(module->getName());
  pm.addNestedPass<mlir::toy::FuncOp>(mlir::createCanonicalizerPass());
```

이제 `toyc-ch3 test/Examples/Toy/Ch3/transpose_transpose.toy -emit=mlir -opt`를 실행하면 패턴이 동작하는 모습을 볼 수 있습니다.

```mlir
toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  toy.return %arg0 : tensor<*xf64>
}
```

예상대로 이제 함수 인자를 직접 반환해 transpose 연산을 우회합니다. 하지만 여전히 transpose가 하나 남아 있습니다. 이는 이상적이지 않습니다. 패턴이 마지막 transpose를 함수 입력으로 대체한 뒤, 이제는 사용되지 않는 transpose 입력이 남았기 때문입니다. 정규화기는 죽은 연산을 정리할 줄 알지만, MLIR은 연산이 부작용을 갖는다고 보수적으로 가정합니다. `TransposeOp`에 `Pure` 트레이트를 추가해 해결할 수 있습니다.

```tablegen
def TransposeOp : Toy_Op<"transpose", [Pure]> {...}
```

다시 `toyc-ch3 test/transpose_transpose.toy -emit=mlir -opt`를 실행합니다.

```mlir
toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
  toy.return %arg0 : tensor<*xf64>
}
```

완벽합니다! `transpose` 연산이 하나도 남지 않아 최적의 코드가 되었습니다.

다음 절에서는 Reshape 연산과 관련된 패턴 매치 최적화를 DRR로 구현합니다.

## DRR로 Reshape 최적화하기

선언적 규칙 기반 패턴 매치와 재작성(DRR)은 연산 DAG 기반 선언형 재작성기로, 패턴 매치와 재작성 규칙을 테이블 형태 문법으로 제공합니다.

```tablegen
class Pattern<
    dag sourcePattern, list<dag> resultPatterns,
    list<dag> additionalConstraints = [],
    dag benefitsAdded = (addBenefit 0)>;
```

`SimplifyRedundantTranspose`와 비슷한 중복 reshape 최적화는 DRR로 더 간단하게 표현할 수 있습니다.

```tablegen
// Reshape(Reshape(x)) = Reshape(x)
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)),
                                   (ReshapeOp $arg)>;
```

각 DRR 패턴에 대응하는 자동 생성 C++ 코드는 `path/to/BUILD/tools/mlir/examples/toy/Ch3/ToyCombine.inc`에서 확인할 수 있습니다.

DRR은 변환이 인자나 결과의 특정 속성에 의존할 때 인자 제약을 추가하는 방법도 제공합니다. 예를 들어 입력과 출력 형태가 동일할 때 reshape를 제거하는 변환이 이에 해당합니다.

```tablegen
def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
def RedundantReshapeOptPattern : Pat<
  (ReshapeOp:$res $arg), (replaceWithValue $arg),
  [(TypesAreIdentical $res, $arg)]>;
```

일부 최적화는 명령 인자에 대한 추가 변환이 필요합니다. 이때 NativeCodeCall을 사용하면 C++ 헬퍼 함수를 호출하거나 인라인 C++ 코드를 실행해 더 복잡한 변환을 적용할 수 있습니다. 예를 들어 상수를 입력으로 받는 reshape를 상수 자체를 재형성해 제거하는 FoldConstantReshape 최적화를 들 수 있습니다.

```tablegen
def ReshapeConstant : NativeCodeCall<"$0.reshape(($1.getType()).cast<ShapedType>())">;
def FoldConstantReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantOp $arg)),
  (ConstantOp (ReshapeConstant $arg, $res))>;
```

다음 `trivial_reshape.toy` 프로그램으로 이러한 reshape 최적화를 시연합니다.

```c++
def main() {
  var a<2,1> = [1, 2];
  var b<2,1> = a;
  var c<2,1> = b;
  print(c);
}
```

```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
    %1 = toy.reshape(%0 : tensor<2xf64>) to tensor<2x1xf64>
    %2 = toy.reshape(%1 : tensor<2x1xf64>) to tensor<2x1xf64>
    %3 = toy.reshape(%2 : tensor<2x1xf64>) to tensor<2x1xf64>
    toy.print %3 : tensor<2x1xf64>
    toy.return
  }
}
```

`toyc-ch3 test/Examples/Toy/Ch3/trivial_reshape.toy -emit=mlir -opt`를 실행하면 패턴이 작동하는 모습을 확인할 수 있습니다.

```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00], [2.000000e+00]]> : tensor<2x1xf64>
    toy.print %0 : tensor<2x1xf64>
    toy.return
  }
}
```

예상대로 정규화 후 reshape 연산이 모두 사라집니다.

선언적 재작성 방법에 대한 자세한 내용은 [Table-driven Declarative Rewrite Rule (DRR)](../../DeclarativeRewrites.md)을 참조하세요.

이 장에서는 항상 사용할 수 있는 훅을 통해 핵심 변환을 활용하는 방법을 살펴보았습니다. [다음 장](Ch-4_kr.md)에서는 인터페이스를 활용해 확장성이 더 좋은 일반 해법을 알아봅니다.
