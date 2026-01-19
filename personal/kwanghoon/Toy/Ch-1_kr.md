# 1장: 토이 언어와 AST

[TOC]

## 언어 소개

이 튜토리얼은 “토이(Toy)”라고 부를 간단한 언어를 예시로 사용합니다(네이밍은 늘 어렵죠...). 토이는 텐서를 기반으로 하며, 함수 정의, 수학 계산 수행, 결과 출력이 가능합니다.

내용을 단순하게 유지하기 위해 코드 생성은 랭크가 2 이하인 텐서로 제한하며, 토이에서 허용되는 유일한 자료형은 64비트 부동소수점(즉 C 언어의 double)입니다. 따라서 모든 값은 묵시적으로 배정밀도이며, `Value`는 불변(각 연산은 새로 할당한 값을 반환)이고, 메모리 해제는 자동으로 관리됩니다. 설명만으로는 이해가 어려우니 예시를 통해 살펴보겠습니다.

```toy
def main() {
  # <2, 3> 형태의 변수 `a`를 리터럴로 초기화합니다.
  # 형태는 제공된 리터럴에서 추론됩니다.
  var a = [[1, 2, 3], [4, 5, 6]];

  # b는 a와 동일하며, 리터럴 텐서는 묵시적으로 재구성됩니다.
  # 새로운 변수를 정의하는 방식으로 텐서의 형태를 바꿀 수 있습니다(요소 수는 동일해야 합니다).
  var b<2, 3> = [1, 2, 3, 4, 5, 6];

  # transpose()와 print()는 내장 함수입니다.
  # 아래 코드는 a와 b를 전치한 뒤 요소별 곱을 수행하고 결과를 출력합니다.
  print(transpose(a) * transpose(b));
}
```

타입 검사는 타입 추론을 통해 정적으로 수행되며, 텐서 형태가 필요할 때만 명시적으로 선언하면 됩니다. 함수는 제네릭으로, 매개변수는 비순위 텐서(즉 텐서라는 사실만 알 뿐 차원은 모름)입니다. 함수는 호출 지점에서 발견되는 새로운 시그니처마다 특수화됩니다. 사용자 정의 함수를 추가해 앞선 예제를 다시 살펴보겠습니다.

```toy
# 미지의 형태를 가진 인자를 처리하는 사용자 정의 제네릭 함수입니다.
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  # <2, 3> 형태의 변수 `a`를 리터럴로 초기화합니다.
  var a = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];

  # 이 호출은 두 인수 모두 <2, 3>으로 multiply_transpose를 특수화하고,
  # 변수 `c` 초기화 시 반환 형태를 <3, 2>로 추론합니다.
  var c = multiply_transpose(a, b);

  # 동일한 형태 <2, 3>으로 다시 호출하면 이전에 특수화된 버전을 재사용해 <3, 2>를 반환합니다.
  var d = multiply_transpose(b, a);

  # 두 차원 모두 <3, 2>인 새로운 호출은 multiply_transpose의 또 다른 특수화를 트리거합니다.
  var e = multiply_transpose(c, d);

  # 마지막으로, 호환되지 않는 형태(<2, 3>과 <3, 2>)로 호출하면 형상 추론 오류가 발생합니다.
  var f = multiply_transpose(a, c);
}
```

## AST

위 코드의 AST는 매우 직접적입니다. 출력 예시는 다음과 같습니다.

```
Module:
  Function 
    Proto 'multiply_transpose' @test/Examples/Toy/Ch1/ast.toy:4:1
    Params: [a, b]
    Block {
      Return
        BinOp: * @test/Examples/Toy/Ch1/ast.toy:5:25
          Call 'transpose' [ @test/Examples/Toy/Ch1/ast.toy:5:10
            var: a @test/Examples/Toy/Ch1/ast.toy:5:20
          ]
          Call 'transpose' [ @test/Examples/Toy/Ch1/ast.toy:5:25
            var: b @test/Examples/Toy/Ch1/ast.toy:5:35
          ]
    } // Block
  Function 
    Proto 'main' @test/Examples/Toy/Ch1/ast.toy:8:1
    Params: []
    Block {
      VarDecl a<> @test/Examples/Toy/Ch1/ast.toy:11:3
        Literal: <2, 3>[ <3>[ 1.000000e+00, 2.000000e+00, 3.000000e+00], <3>[ 4.000000e+00, 5.000000e+00, 6.000000e+00]] @test/Examples/Toy/Ch1/ast.toy:11:11
      VarDecl b<2, 3> @test/Examples/Toy/Ch1/ast.toy:15:3
        Literal: <6>[ 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00] @test/Examples/Toy/Ch1/ast.toy:15:17
      VarDecl c<> @test/Examples/Toy/Ch1/ast.toy:19:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:19:11
          var: a @test/Examples/Toy/Ch1/ast.toy:19:30
          var: b @test/Examples/Toy/Ch1/ast.toy:19:33
        ]
      VarDecl d<> @test/Examples/Toy/Ch1/ast.toy:22:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:22:11
          var: b @test/Examples/Toy/Ch1/ast.toy:22:30
          var: a @test/Examples/Toy/Ch1/ast.toy:22:33
        ]
      VarDecl e<> @test/Examples/Toy/Ch1/ast.toy:25:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:25:11
          var: c @test/Examples/Toy/Ch1/ast.toy:25:30
          var: d @test/Examples/Toy/Ch1/ast.toy:25:33
        ]
      VarDecl f<> @test/Examples/Toy/Ch1/ast.toy:28:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:28:11
          var: a @test/Examples/Toy/Ch1/ast.toy:28:30
          var: c @test/Examples/Toy/Ch1/ast.toy:28:33
        ]
    } // Block
```

`examples/toy/Ch1/` 디렉터리에서 동일한 결과를 재현하고 예제를 직접 실행해 볼 수 있습니다. `path/to/BUILD/bin/toyc-ch1 test/Examples/Toy/Ch1/ast.toy -emit=ast`를 실행해 보세요.

렉서 코드는 전부 하나의 헤더(`examples/toy/Ch1/include/toy/Lexer.h`)에 있으며, 파서는 `examples/toy/Ch1/include/toy/Parser.h`에 있는 재귀 하향 파서입니다. 이런 형태의 렉서/파서에 익숙하지 않다면, LLVM 칼레이도스코프 튜토리얼 1~2장에서 자세히 설명하는 구현과 매우 유사합니다.

[다음 장](Ch-2_kr.md)에서는 이 AST를 MLIR로 변환하는 방법을 살펴봅니다.
