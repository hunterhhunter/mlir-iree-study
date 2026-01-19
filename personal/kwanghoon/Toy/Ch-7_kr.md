# 7장: 토이에 합성 타입 추가하기

[TOC]

[이전 장](Ch-6_kr.md)에서는 토이 프런트엔드에서 LLVM IR까지 이어지는 엔드 투 엔드 컴파일 흐름을 살펴보았습니다. 이번 장에서는 토이 언어에 새로운 합성 `struct` 타입을 추가해 확장합니다.

## 토이에서 `struct` 정의하기

우선 `toy` 소스 언어에서 이 타입의 인터페이스를 정의해야 합니다. 토이에서 `struct` 타입의 일반 문법은 다음과 같습니다.

```toy
# struct 키워드 뒤에 이름을 붙여 정의합니다.
struct MyStruct {
  # struct 내부에는 초기값이나 형태 없이 변수 선언을 나열합니다.
  # 이전에 정의한 struct를 사용할 수도 있습니다.
  var a;
  var b;
}
```

이제 struct는 변수나 함수 매개변수로 사용할 수 있으며, `var` 대신 struct 이름을 사용합니다. struct의 멤버는 `.` 연산자로 접근합니다. `struct` 타입 값은 합성 초기화(composite initializer)로 초기화하며, 이는 `{}`로 둘러싼 다른 초기값의 콤마 구분 목록입니다. 예시는 다음과 같습니다.

```toy
struct Struct {
  var a;
  var b;
}

# 사용자 정의 제네릭 함수도 struct 타입을 다룰 수 있습니다.
def multiply_transpose(Struct value) {
  # '.' 연산자로 멤버에 접근합니다.
  return transpose(value.a) * transpose(value.b);
}

def main() {
  # 합성 초기화로 struct 값을 초기화합니다.
  Struct value = {[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]};

  # 변수를 넘기듯 함수에 인자로 전달합니다.
  var c = multiply_transpose(value);
  print(c);
}
```

## MLIR에서 `struct` 정의하기

MLIR에서도 struct 타입을 표현해야 합니다. MLIR은 우리가 필요한 타입을 제공하지 않으므로 직접 정의합니다. struct를 익명 컨테이너로 정의해 요소 타입 모음을 담도록 합니다. struct와 멤버 이름은 토이 컴파일러의 AST에만 필요하므로 MLIR 표현에는 인코딩하지 않습니다.

### 타입 클래스 정의

#### 타입 클래스 정의

[2장](Ch-2_kr.md)에서 언급했듯, MLIR의 [`Type`](../../LangRef.md/#type-system) 객체는 값 타입이며 내부 데이터를 저장하는 스토리지 객체를 사용합니다. `Type` 클래스는 내부 `TypeStorage` 객체를 감싸며, 이는 `MLIRContext` 인스턴스 안에서 유니크하게 관리됩니다. 새로운 `Type`을 만들면 스토리지 클래스를 생성해 유니크화합니다.

추가 데이터가 필요한 파라메트릭 타입(예: 요소 타입을 저장해야 하는 `struct`)을 정의하려면 파생 스토리지 클래스를 제공해야 합니다. [`index` 타입](../../Dialects/Builtin.md/#indextype) 같은 단일톤 타입은 추가 데이터가 필요 없으므로 기본 `TypeStorage`를 그대로 사용합니다.

##### 스토리지 클래스 정의

타입 스토리지는 타입 인스턴스를 생성하고 유니크화하는 데 필요한 모든 데이터를 담습니다. 파생 스토리지 클래스는 `mlir::TypeStorage`를 상속받아 `MLIRContext`가 유니크화에 사용하는 별칭과 훅을 제공해야 합니다. 아래는 struct 타입 스토리지 정의입니다.

```c++
/// 토이 StructType의 내부 스토리지입니다.
struct StructTypeStorage : public mlir::TypeStorage {
  /// KeyTy는 스토리지 인스턴스 인터페이스를 제공하는 필수 타입입니다.
  /// struct는 자신이 포함한 요소 타입으로 구조적으로 유니크화합니다.
  using KeyTy = llvm::ArrayRef<mlir::Type>;

  StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
      : elementTypes(elementTypes) {}

  /// 주어진 키와 현재 스토리지가 같은지 비교합니다.
  bool operator==(const KeyTy &key) const { return key == elementTypes; }

  /// 키 타입의 해시 함수를 정의합니다.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  /// 키 타입을 파라미터에서 구성합니다.
  static KeyTy getKey(llvm::ArrayRef<mlir::Type> elementTypes) {
    return KeyTy(elementTypes);
  }

  /// 새 스토리지 인스턴스를 생성합니다.
  static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);
    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(elementTypes);
  }

  /// struct의 요소 타입을 보관합니다.
  llvm::ArrayRef<mlir::Type> elementTypes;
};
```

##### 타입 클래스 정의

스토리지 클래스를 정의했으니 실제로 사용할 `StructType` 클래스를 정의합니다.

```c++
/// 토이 struct 타입으로, 요소 타입 모음을 나타냅니다.
class StructType : public mlir::Type::TypeBase<StructType, mlir::Type,
                                               StructTypeStorage> {
public:
  using Base::Base;

  /// 주어진 요소 타입으로 StructType 인스턴스를 만듭니다.
  static StructType get(llvm::ArrayRef<mlir::Type> elementTypes) {
    assert(!elementTypes.empty() && "expected at least 1 element type");
    mlir::MLIRContext *ctx = elementTypes.front().getContext();
    return Base::get(ctx, elementTypes);
  }

  /// 요소 타입을 반환합니다.
  llvm::ArrayRef<mlir::Type> getElementTypes() {
    return getImpl()->elementTypes;
  }

  /// 요소 타입 개수를 반환합니다.
  size_t getNumElementTypes() { return getElementTypes().size(); }
};
```

연산을 등록했던 것과 유사하게 `ToyDialect` 초기화에서 타입을 등록합니다.

```c++
void ToyDialect::initialize() {
  addTypes<StructType>();
}
```

(타입을 등록할 때는 스토리지 클래스 정의가 반드시 보이는 위치에 있어야 합니다.)

이제 토이에서 MLIR을 생성할 때 `StructType`을 사용할 수 있습니다. 자세한 내용은 examples/toy/Ch7/mlir/MLIRGen.cpp를 참고하세요.

### ODS에 노출하기

새 타입을 정의했다면 ODS(Operation Definition Specification) 프레임워크에 알려 연산 정의와 다이얼렉트 유틸리티 생성에 활용할 수 있어야 합니다. 간단한 예시는 다음과 같습니다.

```tablegen
// ODS에서 StructType을 사용할 수 있도록 정의합니다.
def Toy_StructType :
    DialectType<Toy_Dialect, CPred<"isa<StructType>($_self)">,
                "Toy struct type">;

// 토이 방언에서 사용할 타입 집합을 정의합니다.
def Toy_Type : AnyTypeOf<[F64Tensor, Toy_StructType]>;
```

### 파싱과 프린트

이제 `StructType`을 MLIR 생성과 변환에서 사용할 수 있지만, `.mlir`로 출력하거나 읽어오지는 못합니다. 이를 위해 `ToyDialect`에서 `parseType`과 `printType`을 재정의해야 합니다. 앞 절에서 ODS에 노출하면 해당 메서드 선언이 자동으로 제공됩니다.

```c++
class ToyDialect : public mlir::Dialect {
public:
  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
  void printType(mlir::Type type,
                 mlir::DialectAsmPrinter &printer) const override;
};
```

이 메서드는 필요한 기능을 쉽게 구현하도록 고수준 파서와 프린터를 제공합니다. 구현 전 출력 IR에서 struct 타입 문법을 정의합니다. [MLIR 언어 참조](../../LangRef.md/#dialect-types)에 따르면 방언 타입은 일반적으로 `!dialect-namespace<type-data>` 형식이며, 경우에 따라 보기 좋은 형태가 있습니다. 토이 파서와 프린터는 여기서 `type-data`를 담당합니다. 우리는 `StructType`을 다음 형태로 정의합니다.

```
  struct-type ::= `struct` `<` type (`,` type)* `>`
```

#### 파싱

파서 구현은 다음과 같습니다.

```c++
mlir::Type ToyDialect::parseType(mlir::DialectAsmParser &parser) const {
  // struct-type ::= `struct` `<` type (`,` type)* `>`
  if (parser.parseKeyword("struct") || parser.parseLess())
    return Type();

  SmallVector<mlir::Type, 1> elementTypes;
  do {
    SMLoc typeLoc = parser.getCurrentLocation();
    mlir::Type elementType;
    if (parser.parseType(elementType))
      return nullptr;

    if (!isa<mlir::TensorType, StructType>(elementType)) {
      parser.emitError(typeLoc, "element type for a struct must either "
                                "be a TensorType or a StructType, got: ")
          << elementType;
      return Type();
    }
    elementTypes.push_back(elementType);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseGreater())
    return Type();
  return StructType::get(elementTypes);
}
```

#### 프린트

프린터 구현은 다음과 같습니다.

```c++
void ToyDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const {
  StructType structType = type.cast<StructType>();
  printer << "struct<";
  llvm::interleaveComma(structType.getElementTypes(), printer);
  printer << '>';
}
```

이제 다음 예시를 확인해 봅니다.

```toy
struct Struct {
  var a;
  var b;
}

def multiply_transpose(Struct value) {
}
```

이는 다음과 같이 출력됩니다.

```mlir
module {
  toy.func @multiply_transpose(%arg0: !toy.struct<tensor<*xf64>, tensor<*xf64>>) {
    toy.return
  }
}
```

### `StructType` 연산하기

이제 struct 타입을 정의했고 IR에서 왕복(round-trip)할 수 있습니다. 다음 단계는 연산에서 struct를 사용할 수 있도록 지원하는 것입니다.

#### 기존 연산 업데이트

`ReturnOp` 같은 일부 기존 연산은 `Toy_StructType`을 처리하도록 업데이트해야 합니다.

```tablegen
def ReturnOp : Toy_Op<"return", [Terminator, HasParent<"FuncOp">]> {
  ...
  let arguments = (ins Variadic<Toy_Type>:$input);
  ...
}
```

#### 새로운 토이 연산 추가

기존 연산 외에도 struct를 다루는 새로운 연산을 몇 개 추가합니다.

##### `toy.struct_constant`

struct의 상수 값을 물질화합니다. 여기서는 각 struct 요소에 대한 상수를 담은 [array attribute](../../Dialects/Builtin.md/#arrayattr)를 사용합니다.

```mlir
  %0 = toy.struct_constant [
    dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
  ] : !toy.struct<tensor<*xf64>>
```

##### `toy.struct_access`

struct 값의 N번째 요소를 물질화합니다.

```mlir
  %1 = toy.struct_access %0[0] : !toy.struct<tensor<*xf64>> -> tensor<*xf64>
```

이 연산들을 사용하면 앞선 예제를 다시 살펴볼 수 있습니다.

```toy
struct Struct {
  var a;
  var b;
}

def multiply_transpose(Struct value) {
  return transpose(value.a) * transpose(value.b);
}

def main() {
  Struct value = {[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]};
  var c = multiply_transpose(value);
  print(c);
}
```

위 코드는 다음과 같은 MLIR 모듈을 생성합니다.

```mlir
module {
  toy.func @multiply_transpose(%arg0: !toy.struct<tensor<*xf64>, tensor<*xf64>>) -> tensor<*xf64> {
    %0 = toy.struct_access %arg0[0] : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
    %1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
    %2 = toy.struct_access %arg0[1] : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
    %3 = toy.transpose(%2 : tensor<*xf64>) to tensor<*xf64>
    %4 = toy.mul %1, %3 : tensor<*xf64>
    toy.return %4 : tensor<*xf64>
  }
  toy.func @main() {
    %0 = toy.struct_constant [
      dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>,
      dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    ] : !toy.struct<tensor<*xf64>, tensor<*xf64>>
    %1 = toy.generic_call @multiply_transpose(%0) : (!toy.struct<tensor<*xf64>, tensor<*xf64>>) -> tensor<*xf64>
    toy.print %1 : tensor<*xf64>
    toy.return
  }
}
```

#### `StructType` 연산 최적화

이제 `StructType`을 사용하는 연산이 생겼으니 새로운 상수 접기 기회도 많습니다.

인라이닝 이후 모듈은 다음과 같습니다.

```mlir
module {
  toy.func @main() {
    %0 = toy.struct_constant [
      dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>,
      dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    ] : !toy.struct<tensor<*xf64>, tensor<*xf64>>
    %1 = toy.struct_access %0[0] : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
    %2 = toy.transpose(%1 : tensor<*xf64>) to tensor<*xf64>
    %3 = toy.struct_access %0[1] : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
    %4 = toy.transpose(%3 : tensor<*xf64>) to tensor<*xf64>
    %5 = toy.mul %2, %4 : tensor<*xf64>
    toy.print %5 : tensor<*xf64>
    toy.return
  }
}
```

`toy.struct_constant`에 접근하는 `toy.struct_access`가 여러 개 있습니다. [3장](Ch-3_kr.md)의 FoldConstantReshape와 마찬가지로 연산 정의에서 `hasFolder`를 설정하고 `*Op::fold`를 구현해 접기를 추가할 수 있습니다.

```c++
/// 상수 접기.
OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return value(); }

/// struct 상수 접기.
OpFoldResult StructConstantOp::fold(FoldAdaptor adaptor) {
  return value();
}

/// struct 상수에 대한 단순 접근 접기.
OpFoldResult StructAccessOp::fold(FoldAdaptor adaptor) {
  auto structAttr = dyn_cast_or_null<mlir::ArrayAttr>(adaptor.getInput());
  if (!structAttr)
    return nullptr;

  size_t elementIndex = index().getZExtValue();
  return structAttr[elementIndex];
}
```

MLIR이 `StructType`에는 `StructConstant`, `TensorType`에는 `ConstantOp`를 생성하도록 하려면 다이얼렉트 훅 `materializeConstant`를 재정의해야 합니다. 이를 통해 일반 MLIR 연산이 토이 상수를 필요로 할 때 적절한 연산을 생성합니다.

```c++
mlir::Operation *ToyDialect::materializeConstant(mlir::OpBuilder &builder,
                                                 mlir::Attribute value,
                                                 mlir::Type type,
                                                 mlir::Location loc) {
  if (isa<StructType>(type))
    return StructConstantOp::create(builder, loc, type,
                                            cast<mlir::ArrayAttr>(value));
  return ConstantOp::create(builder, loc, type,
                                    cast<mlir::DenseElementsAttr>(value));
}
```

이제 파이프라인을 바꾸지 않고도 LLVM까지 하향할 수 있는 코드를 생성할 수 있습니다.

```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
    %2 = toy.mul %1, %1 : tensor<3x2xf64>
    toy.print %2 : tensor<3x2xf64>
    toy.return
  }
}
```

`toyc-ch7`을 빌드한 뒤 `toyc-ch7 test/Examples/Toy/Ch7/struct-codegen.toy -emit=mlir`로 직접 실행해 보세요. 사용자 정의 타입 정의에 대한 더 자세한 내용은 [DefiningAttributesAndTypes](../../DefiningDialects/AttributesAndTypes.md)를 참고하세요.
