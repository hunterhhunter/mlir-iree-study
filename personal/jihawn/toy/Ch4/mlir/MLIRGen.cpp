//===- MLIRGen.cpp - MLIR Generation from a Toy AST -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// 이 파일은 Toy 언어의 모듈 AST로부터 MLIR을 생성하는 간단한 IR 생성을 구현합니다.
//
//===----------------------------------------------------------------------===//

#include "toy/MLIRGen.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"
#include "toy/AST.h"
#include "toy/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "toy/Lexer.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include <cassert>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <vector>

using namespace mlir::toy;
using namespace toy;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace {

/// Toy AST로부터 간단한 MLIR 생성을 구현합니다.
///
/// 이것은 Toy 언어에 특화된 연산을 방출하여 언어의 의미(semantics)를 보존하고
/// (바라건대) 이러한 고수준 의미를 기반으로 정확한 분석 및 변환을 수행할 수 있게 합니다.
class MLIRGenImpl {
public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

  /// 공개 API: Toy 모듈(소스 파일) AST를 MLIR 모듈 연산으로 변환합니다.
  mlir::ModuleOp mlirGen(ModuleAST &moduleAST) {
    // 빈 MLIR 모듈을 생성하고 함수를 하나씩 코드 생성하여 모듈에 추가합니다.
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    for (FunctionAST &f : moduleAST)
      mlirGen(f);

    // 구성을 마친 후 모듈을 검증합니다. 이는 IR의 구조적 속성을 확인하고
    // Toy 연산에 대해 우리가 정의한 특정 검증기들을 호출합니다.
    if (failed(mlir::verify(theModule))) {
      theModule.emitError("module verification error");
      return nullptr;
    }

    return theModule;
  }

private:
  /// "모듈"은 Toy 소스 파일에 해당하며, 함수 목록을 포함합니다.
  mlir::ModuleOp theModule;

  /// 빌더는 함수 내부에 IR을 생성하는 것을 돕는 클래스입니다.
  /// 빌더는 상태를 가지며, 특히 "삽입 지점(insertion point)"을 유지합니다.
  /// 다음 연산들은 이 지점에 도입됩니다.
  mlir::OpBuilder builder;

  /// 심볼 테이블은 현재 스코프에서 변수 이름을 값에 매핑합니다.
  /// 함수에 진입하면 새로운 스코프가 생성되고 함수 인자들이 매핑에 추가됩니다.
  /// 함수 처리가 종료되면 스코프가 파괴되고 이 스코프에서 생성된 매핑들도 삭제됩니다.
  llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;

  /// Toy AST의 Location 정보를 MLIR Location 정보로 변환하는 도우미 함수입니다.
  mlir::Location loc(const Location &loc) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(*loc.file), loc.line,
                                     loc.col);
  }

  /// 현재 스코프에 변수를 선언합니다. 변수가 아직 선언되지 않았다면 성공을 반환합니다.
  llvm::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
    if (symbolTable.count(var))
      return mlir::failure();
    symbolTable.insert(var, value);
    return mlir::success();
  }

  /// 제공된 Toy AST 프로토타입만큼의 인자를 가진 MLIR 함수 프로토타입을 생성합니다.
  mlir::toy::FuncOp mlirGen(PrototypeAST &proto) {
    auto location = loc(proto.loc());

    // 이것은 제네릭 함수이며, 반환 타입은 나중에 추론됩니다.
    // 인자 타입은 일관되게 unranked 텐서입니다.
    llvm::SmallVector<mlir::Type, 4> argTypes(proto.getArgs().size(),
                                              getType(VarType{}));
    auto funcType = builder.getFunctionType(argTypes, /*results=*/{});
    return mlir::toy::FuncOp::create(builder, location, proto.getName(),
                                     funcType);
  }

  /// 새로운 함수를 생성하고 MLIR 모듈에 추가합니다.
  mlir::toy::FuncOp mlirGen(FunctionAST &funcAST) {
    // 변수 선언을 담을 스코프를 심볼 테이블에 생성합니다.
    ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);

    // 주어진 프로토타입에 대한 MLIR 함수를 생성합니다.
    builder.setInsertionPointToEnd(theModule.getBody());
    mlir::toy::FuncOp function = mlirGen(*funcAST.getProto());
    if (!function)
      return nullptr;

    // 이제 함수 본문을 시작합시다!
    mlir::Block &entryBlock = function.front();
    auto protoArgs = funcAST.getProto()->getArgs();

    // 모든 함수 인자를 심볼 테이블에 선언합니다.
    for (const auto nameValue :
         llvm::zip(protoArgs, entryBlock.getArguments())) {
      if (failed(declare(std::get<0>(nameValue)->getName(),
                         std::get<1>(nameValue))))
        return nullptr;
    }

    // 빌더의 삽입 지점을 함수 본문의 시작 부분으로 설정합니다.
    // 이 함수 내에서 연산을 생성하기 위해 코드 생성 내내 사용될 것입니다.
    builder.setInsertionPointToStart(&entryBlock);

    // 함수 본문을 방출(emit)합니다.
    if (mlir::failed(mlirGen(*funcAST.getBody()))) {
      function.erase();
      return nullptr;
    }

    // 반환 문이 방출되지 않은 경우 암시적으로 void를 반환합니다.
    // FIXME: 파서를 수정하여 항상 마지막 표현식을 반환하도록 할 수 있습니다.
    // (이는 나중에 REPL 케이스에 도움이 될 수 있습니다)
    ReturnOp returnOp;
    if (!entryBlock.empty())
      returnOp = dyn_cast<ReturnOp>(entryBlock.back());
    if (!returnOp) {
      ReturnOp::create(builder, loc(funcAST.getProto()->loc()));
    } else if (returnOp.hasOperand()) {
      // 그렇지 않고 이 반환 연산이 피연산자를 가지고 있다면 함수에 결과를 추가합니다.
      function.setType(builder.getFunctionType(
          function.getFunctionType().getInputs(), getType(VarType{})));
    }

    // 이 함수가 main이 아니라면, 가시성을 private으로 설정합니다.
    if (funcAST.getProto()->getName() != "main")
      function.setPrivate();

    return function;
  }

  /// 이항 연산을 방출합니다.
  mlir::Value mlirGen(BinaryExprAST &binop) {
    // 연산 자체를 방출하기 전에 연산의 양쪽 측면에 대한 연산을 먼저 방출합니다.
    // 예를 들어 표현식이 `a + foo(a)`라면:
    // 1) 먼저 LHS를 방문하면 `a`를 담고 있는 값에 대한 참조를 반환합니다.
    //    이 값은 선언 시점에 방출되어 심볼 테이블에 등록되었어야 하므로,
    //    아무것도 코드 생성되지 않습니다. 값이 심볼 테이블에 없다면 에러가
    //    방출되고 nullptr가 반환됩니다.
    // 2) 그 다음 RHS를 (재귀적으로) 방문하여 `foo`에 대한 호출이 방출되고
    //    결과 값이 반환됩니다. 에러가 발생하면 nullptr를 얻고 전파합니다.
    //
    mlir::Value lhs = mlirGen(*binop.getLHS());
    if (!lhs)
      return nullptr;
    mlir::Value rhs = mlirGen(*binop.getRHS());
    if (!rhs)
      return nullptr;
    auto location = loc(binop.loc());

    // 이항 연산자로부터 연산 이름을 유도합니다. 현재는 '+'와 '*'만 지원합니다.
    switch (binop.getOp()) {
    case '+':
      return AddOp::create(builder, location, lhs, rhs);
    case '*':
      return MulOp::create(builder, location, lhs, rhs);
    }

    emitError(location, "invalid binary operator '") << binop.getOp() << "'";
    return nullptr;
  }

  /// 이것은 표현식 내의 변수에 대한 참조입니다. 변수는 이미 선언되어 심볼 테이블에
  /// 값이 있어야 하며, 그렇지 않으면 에러를 방출하고 nullptr를 반환합니다.
  mlir::Value mlirGen(VariableExprAST &expr) {
    if (auto variable = symbolTable.lookup(expr.getName()))
      return variable;

    emitError(loc(expr.loc()), "error: unknown variable '")
        << expr.getName() << "'";
    return nullptr;
  }

  /// 반환 연산을 방출합니다. 생성 실패 시 failure를 반환합니다.
  llvm::LogicalResult mlirGen(ReturnExprAST &ret) {
    auto location = loc(ret.loc());

    // 'return'은 선택적 표현식을 취하므로, 여기서 그 케이스를 처리합니다.
    mlir::Value expr = nullptr;
    if (ret.getExpr().has_value()) {
      if (!(expr = mlirGen(**ret.getExpr())))
        return mlir::failure();
    }

    // 그렇지 않으면 이 반환 연산은 0개의 피연산자를 가집니다.
    ReturnOp::create(builder, location,
                     expr ? ArrayRef(expr) : ArrayRef<mlir::Value>());
    return mlir::success();
  }

  /// 리터럴/상수 배열을 방출합니다. 이는 `toy.constant` 연산에 첨부된 속성(Attribute)
  /// 내의 평탄화된 데이터 배열로 방출됩니다.
  /// 자세한 내용은 [Attributes](LangRef.md#attributes) 문서를 참조하십시오.
  /// 발췌:
  ///
  ///   속성은 변수가 허용되지 않는 곳에서 MLIR의 상수 데이터를 지정하는 메커니즘입니다 [...].
  ///   이들은 이름과 구체적인 속성 값으로 구성됩니다. 예상되는 속성들의 집합, 구조,
  ///   그리고 해석은 모두 그것들이 무엇에 첨부되는지에 따라 문맥적으로 달라집니다.
  ///
  /// 예시, 소스 레벨 문장:
  ///   var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  /// 은 다음과 같이 변환됩니다:
  ///   %0 = "toy.constant"() {value: dense<tensor<2x3xf64>,
  ///     [[1.000000e+00, 2.000000e+00, 3.000000e+00],
  ///      [4.000000e+00, 5.000000e+00, 6.000000e+00]]>} : () -> tensor<2x3xf64>
  ///
  mlir::Value mlirGen(LiteralExprAST &lit) {
    auto type = getType(lit.getDims());

    // 속성은 배열 내의 요소(숫자) 당 하나의 부동 소수점 값을 가진 벡터입니다.
    // 자세한 내용은 아래 `collectData()`를 참조하십시오.
    std::vector<double> data;
    data.reserve(llvm::product_of(lit.getDims()));
    collectData(lit, data);

    // 이 속성의 타입은 리터럴의 모양을 가진 64비트 부동 소수점 텐서입니다.
    mlir::Type elementType = builder.getF64Type();
    auto dataType = mlir::RankedTensorType::get(lit.getDims(), elementType);

    // 이것은 이 텐서 리터럴에 대한 값 목록을 담고 있는 실제 속성입니다.
    auto dataAttribute =
        mlir::DenseElementsAttr::get(dataType, llvm::ArrayRef(data));

    // MLIR op `toy.constant`를 빌드합니다. 이는 `ConstantOp::build` 메서드를 호출합니다.
    return ConstantOp::create(builder, loc(lit.loc()), type, dataAttribute);
  }

  /// 배열 리터럴을 구성하는 데이터를 축적하기 위한 재귀적 도우미 함수입니다.
  /// 제공된 벡터에 중첩된 구조를 평탄화합니다. 예를 들어 다음 배열의 경우:
  ///  [[1, 2], [3, 4]]
  /// 다음을 생성합니다:
  ///  [ 1, 2, 3, 4 ]
  /// 개별 숫자는 double로 표현됩니다.
  /// 속성은 MLIR이 연산에 상수를 첨부하는 방식입니다.
  void collectData(ExprAST &expr, std::vector<double> &data) {
    if (auto *lit = dyn_cast<LiteralExprAST>(&expr)) {
      for (auto &value : lit->getValues())
        collectData(*value, data);
      return;
    }

    assert(isa<NumberExprAST>(expr) && "expected literal or number expr");
    data.push_back(cast<NumberExprAST>(expr).getValue());
  }

  /// 호출 표현식을 방출합니다. `transpose` 내장 함수에 대해 특정 연산을 방출합니다.
  /// 다른 식별자들은 사용자 정의 함수로 간주됩니다.
  mlir::Value mlirGen(CallExprAST &call) {
    llvm::StringRef callee = call.getCallee();
    auto location = loc(call.loc());

    // 피연산자들을 먼저 코드 생성합니다.
    SmallVector<mlir::Value, 4> operands;
    for (auto &expr : call.getArgs()) {
      auto arg = mlirGen(*expr);
      if (!arg)
        return nullptr;
      operands.push_back(arg);
    }

    // 내장 호출은 커스텀 연산을 가지고 있으며, 이는 직관적인 방출을 의미합니다.
    if (callee == "transpose") {
      if (call.getArgs().size() != 1) {
        emitError(location, "MLIR codegen encountered an error: toy.transpose "
                            "does not accept multiple arguments");
        return nullptr;
      }
      return TransposeOp::create(builder, location, operands[0]);
    }

    // 그렇지 않으면 이것은 사용자 정의 함수에 대한 호출입니다.
    // 사용자 정의 함수에 대한 호출은 피호출자 이름을 속성으로 받는 
    // 커스텀 호출(custom call)로 매핑됩니다.
    return GenericCallOp::create(builder, location, callee, operands);
  }

  /// print 표현식을 방출합니다. 두 가지 내장 함수에 대해 특정 연산을 방출합니다:
  /// transpose(x)와 print(x).
  llvm::LogicalResult mlirGen(PrintExprAST &call) {
    auto arg = mlirGen(*call.getArg());
    if (!arg)
      return mlir::failure();

    PrintOp::create(builder, loc(call.loc()), arg);
    return mlir::success();
  }

  /// 단일 숫자에 대한 상수를 방출합니다 (FIXME: semantic? broadcast?)
  mlir::Value mlirGen(NumberExprAST &num) {
    return ConstantOp::create(builder, loc(num.loc()), num.getValue());
  }

  /// RTTI를 사용하여 올바른 표현식 서브클래스에 대해 코드 생성을 디스패치합니다.
  mlir::Value mlirGen(ExprAST &expr) {
    switch (expr.getKind()) {
    case toy::ExprAST::Expr_BinOp:
      return mlirGen(cast<BinaryExprAST>(expr));
    case toy::ExprAST::Expr_Var:
      return mlirGen(cast<VariableExprAST>(expr));
    case toy::ExprAST::Expr_Literal:
      return mlirGen(cast<LiteralExprAST>(expr));
    case toy::ExprAST::Expr_Call:
      return mlirGen(cast<CallExprAST>(expr));
    case toy::ExprAST::Expr_Num:
      return mlirGen(cast<NumberExprAST>(expr));
    default:
      emitError(loc(expr.loc()))
          << "MLIR codegen encountered an unhandled expr kind '"
          << Twine(expr.getKind()) << "'";
      return nullptr;
    }
  }

  /// 변수 선언을 처리합니다. 초기화(initializer)를 형성하는 표현식을 코드 생성하고
  /// 반환하기 전에 값을 심볼 테이블에 기록합니다.
  /// 이후의 표현식들은 심볼 테이블 조회를 통해 이 변수를 참조할 수 있습니다.
  mlir::Value mlirGen(VarDeclExprAST &vardecl) {
    auto *init = vardecl.getInitVal();
    if (!init) {
      emitError(loc(vardecl.loc()),
                "missing initializer in variable declaration");
      return nullptr;
    }

    mlir::Value value = mlirGen(*init);
    if (!value)
      return nullptr;

    // 초기화 값을 가지고 있지만, 변수가 특정 모양으로 선언된 경우
    // "reshape" 연산을 방출합니다. 이는 나중에 필요에 따라 최적화되어 사라질 것입니다.
    if (!vardecl.getType().shape.empty()) {
      value = ReshapeOp::create(builder, loc(vardecl.loc()),
                                getType(vardecl.getType()), value);
    }

    // 값을 심볼 테이블에 등록합니다.
    if (failed(declare(vardecl.getName(), value)))
      return nullptr;
    return value;
  }

  /// 표현식 목록을 코드 생성합니다. 그 중 하나라도 에러가 발생하면 실패를 반환합니다.
  llvm::LogicalResult mlirGen(ExprASTList &blockAST) {
    ScopedHashTableScope<StringRef, mlir::Value> varScope(symbolTable);
    for (auto &expr : blockAST) {
      // 변수 선언, 반환 문, 그리고 print에 대한 특정 처리입니다.
      // 이들은 블록 목록에만 나타날 수 있으며 중첩된 표현식에는 나타날 수 없습니다.
      if (auto *vardecl = dyn_cast<VarDeclExprAST>(expr.get())) {
        if (!mlirGen(*vardecl))
          return mlir::failure();
        continue;
      }
      if (auto *ret = dyn_cast<ReturnExprAST>(expr.get()))
        return mlirGen(*ret);
      if (auto *print = dyn_cast<PrintExprAST>(expr.get())) {
        if (mlir::failed(mlirGen(*print)))
          return mlir::success();
        continue;
      }

      // 일반 표현식 디스패치 코드 생성입니다.
      if (!mlirGen(*expr))
        return mlir::failure();
    }
    return mlir::success();
  }

  /// 모양 차원(shape dimensions) 목록으로부터 텐서 타입을 빌드합니다.
  mlir::Type getType(ArrayRef<int64_t> shape) {
    // 모양이 비어있다면, 이것은 unranked 타입입니다.
    if (shape.empty())
      return mlir::UnrankedTensorType::get(builder.getF64Type());

    // 그렇지 않으면 주어진 모양을 사용합니다.
    return mlir::RankedTensorType::get(shape, builder.getF64Type());
  }

  /// Toy AST 변수 타입으로부터 MLIR 타입을 빌드합니다 (위의 제네릭 getType으로 전달).
  mlir::Type getType(const VarType &type) { return getType(type.shape); }
};

} // namespace

namespace toy {

// 코드 생성을 위한 공개 API.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          ModuleAST &moduleAST) {
  return MLIRGenImpl(context).mlirGen(moduleAST);
}

} // namespace toy