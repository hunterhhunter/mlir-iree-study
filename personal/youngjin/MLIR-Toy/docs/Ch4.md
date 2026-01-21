# 인라이닝 + 형태추론 워크플로우

## 1. 인라이닝 패스
1. generic_call 발견: InlinerPass가 IR을 스캔하다 generic_call 연산을 찾음
2. ToyInlinerInterface의 isLegalToInline을 호출해 인라이닝 가능 여부 확인
3. 호출 대상 함수 내부 로직(Region)을 그대로 복사해옴. 이 때 타입은 여전히 tensor<*xf64>
4. 인자 매핑: 함수의 입력 파라미터(%arg0)를 호출 시점에 전달된 실제 값(%main_value)으로 연결
5. SSA 연결(Rewriting): handleterminator가 실행되어, 함수의 return값이 가리키던 SSA 값을 generic_call의 결과값(%n - main_value) 자리로 대치
    - 이를 RAUW(Replace All Uses With) 매커니즘이라 부름.
    - 코드로 보면 이해가 쉽습니다.

## 2. 형태 추론 패스 (타입 확정)
인라이닝이 끝나면 main 함수 안에 구체적인 형태를 가진 상수와 형태를 모르는 연산이 뒤섞여있음.
6. 전파 시작: main 입구에 정의된 toy.constant(tensor<2x3xf64>) 정보를 확인
7. 도미노 전파: 인라이닝을 들어온 연산들을 하나씩 방문하며 inferShapes()를 호출
    - transpose(%1(tensor<2x3xf64>)) -> tensor<3x2xf64>
8. 최종확정: 모든 연산의 결과 타입이 구체적인 shape으로 업데이트

---

## 코드로 보는 인라이닝과 형태 추론
### Phase 1: 초기 상태
```mlir
// [Caller]
toy.func @main() {
  // 상수 정의 (Ranked)
  %0 = toy.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>
  
  // 함수 호출: 인자는 Ranked이지만, 함수 정의가 Unranked를 받으므로 타입 불일치 발생
  %m = toy.generic_call @my_func(%0) : (tensor<2x2xf64>) -> tensor<*xf64>
  
  toy.print %m : tensor<*xf64>
  toy.return
}

// [Callee]
toy.func @my_func(%arg1: tensor<*xf64>) -> tensor<*xf64> {
  %f = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
  toy.return %f : tensor<*xf64>
}
```

### Phase 2: 인라이닝 실행 (Cloning & SSA Rewriting)
InlinerPass가 작동하여 callee 함수를 clone해옴.
이 때 인자 매핑과 RAUW가 동작
```mlir
toy.func @main() {
  %0 = toy.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>

  // 1. [Cast 삽입]: 인라이너가 타입 차이를 해결하기 위해 toy.cast 자동 삽입
  %0_cast = toy.cast %0 : tensor<2x2xf64> to tensor<*xf64>

  // 2. [Cloning & Mapping]: @my_func 본문 복제, %arg1은 %0_cast로 매핑됨 - 인라이닝 4번
  %f_cloned = toy.transpose(%0_cast : tensor<*xf64>) to tensor<*xf64>

  // 3. [Rewiring]: 원래 %m을 쓰던 toy.print가 %f_cloned를 보도록 RAUW 수행 - 인라이닝 5번
  toy.print %f_cloned : tensor<*xf64>
  toy.return
}
```

### Phase 3: 형태 추론 
```mlir
toy.func @main() {
  // 1. 출발점: tensor<2x2xf64>
  %0 = toy.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>

  // 2. inferShapes() 호출: toy.cast 최적화
  // 입력이 2x2이므로 캐스트 결과도 사실상 2x2임을 인지함
  %0_cast = toy.cast %0 : tensor<2x2xf64> to tensor<2x2xf64>

  // 3. inferShapes() 호출: toy.transpose 차원 전파
  // 입력이 2x2이므로 결과도 2x2로 업데이트됨
  %f_cloned = toy.transpose(%0_cast : tensor<2x2xf64>) to tensor<2x2xf64>

  toy.print %f_cloned : tensor<2x2xf64>
  toy.return
}
```

### Phase 4: 최종 정규화 (Canonicalization) 
```mlir
toy.func @main() {
  // 최적화된 최종 결과
  %0 = toy.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>
  
  // 캐스트는 사라지고, transpose는 구체적인 타입을 가짐
  %1 = toy.transpose(%0 : tensor<2x2xf64>) to tensor<2x2xf64>
  
  toy.print %1 : tensor<2x2xf64>
  toy.return
}
```

## 중요한 개념
- 스코프 기반 심볼 테이블 (Scoped Symbol Table), SSA: **컴파일러를 위한 개념들.md**를 참고.
