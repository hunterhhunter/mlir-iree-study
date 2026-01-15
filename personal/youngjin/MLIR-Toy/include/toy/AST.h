#ifndef TOY_AST_H
#define TOY_AST_H

#include "toy/Lexer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <utility>
#include <vector>
#include <optional>

namespace toy {

/// A variable type with shape information.
struct VarType {
  std::vector<int64_t> shape;
};

/// Base class for all expression nodes.
class ExprAST {
public:
  enum ExprASTKind {
    Expr_VarDecl,
    Expr_Return,
    Expr_Num,
    Expr_Literal,
    Expr_Var,
    Expr_BinOp,
    Expr_Call,
    Expr_Print,
  };

  ExprAST(ExprASTKind kind, Location location)
      : kind(kind), location(std::move(location)) {}
  virtual ~ExprAST() = default;

  ExprASTKind getKind() const { return kind; }

  const Location &loc() { return location; }

private:
  const ExprASTKind kind;
  Location location;
};

/// A block-list of expressions.
// Type Alias typedef와 같은 문법
// 모든 type 별 ExprAST를 설계할 때 자원 관리를 하는 무거운 객체
// (내부적으로 힙 할당하는 애들, 복사가 불가능한 애들, 참조 카운트 오버헤드를 줄이고 싶을 때)
// move를 사용해 부모 객체에게 소유권을 이전해 더 편한 관리가 가능하도록 디자인 패턴을 사용함
// 여기서는 Location 객체를 부모 ExprAST에 이전시켜 효율적인 관리가 가능 (구현 상속의 정석적인 패턴)
using ExprASTList = std::vector<std::unique_ptr<ExprAST>>;

/// Expression class for numeric literals like "1.0".
class NumberExprAST : public ExprAST {
  double val;

public:
  NumberExprAST(Location loc, double val)
      : ExprAST(Expr_Num, std::move(loc)), val(val) {}

  double getValue() { return val; }

  /// LLVM style RTTI
  // LLVM을 위한 포인터 검증 메서드
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Num; }
};

/// Expression class for a literal value.
class LiteralExprAST : public ExprAST {
  std::vector<std::unique_ptr<ExprAST>> values;
  std::vector<int64_t> dims;

public:
  LiteralExprAST(Location loc, std::vector<std::unique_ptr<ExprAST>> values,
                 std::vector<int64_t> dims)
      : ExprAST(Expr_Literal, std::move(loc)), values(std::move(values)),
        dims(std::move(dims)) {}

  llvm::ArrayRef<std::unique_ptr<ExprAST>> getValues() { return values; }
  llvm::ArrayRef<int64_t> getDims() { return dims; }

  /// LLVM style RTTI
  // LLVM을 위한 포인터 검증 메서드
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Literal; }
};

/// Expression class for referencing a variable, like "a".
class VariableExprAST : public ExprAST {
  std::string name;

public:
  VariableExprAST(Location loc, llvm::StringRef name)
      : ExprAST(Expr_Var, std::move(loc)), name(name) {}

  // string <-> llvm::StringRef간 암시적 변환(자동)
  // llvm::StringRef 내부에 string을 인자로 받는 생성자가 있어 컴파일러가 자동으로
  // llvm::StringRef를 생성해서 return 함.
  // 그 과정에서 name을 가리키는 레퍼런스를 반환
  llvm::StringRef getName() { return name; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Var; }
};

/// Expression class for defining a variable.
class VarDeclExprAST : public ExprAST {
  std::string name;
  VarType type;
  std::unique_ptr<ExprAST> initVal;

public:
  VarDeclExprAST(Location loc, llvm::StringRef name, VarType type,
                 std::unique_ptr<ExprAST> initVal)
      : ExprAST(Expr_VarDecl, std::move(loc)), name(name),
        type(std::move(type)), initVal(std::move(initVal)) {}

  llvm::StringRef getName() { return name; }

  // *와 &의 차이점
  // 포인터의 경우 ->로 직접 넘어가서 봐야함. (역참조)
  // 참조자의 경우 .으로 바로 접근이 가능함 (원본 그 자체)

  // 함수에 붙은 *는 주소만 넘기고 소유권을 넘겨주지 않음. (unique_ptr과 연계되어 동작)
  // 객체를 반환하면 기본적으로 복사가 일어남. 그래서 객체 포인터로 넘겨 복사를 방지
  // + 복사될 때 객체 절단(ExprAST(부모클래스) 만큼만 복사해서 자식 클래스(VarDeclExprAST)의 내용은 지워짐) 방지
  ExprAST *getInitVal() { return initVal.get(); }

  // type에 대한 읽기 전용 참조자를 return (복사 방지)
  const VarType &getType() { return type; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_VarDecl; }
};

/// Expression class for a return operator.
class ReturnExprAST : public ExprAST {
  std::optional<std::unique_ptr<ExprAST>> expr;

public:
  ReturnExprAST(Location loc, std::optional<std::unique_ptr<ExprAST>> expr)
      : ExprAST(Expr_Return, std::move(loc)), expr(std::move(expr)) {}

  std::optional<ExprAST *> getExpr() {
    // optional 자체의 메서드 (. 사용)
    if (expr.has_value())
      // 내부 unique_ptr의 메서드 (-> 사용, optional 투과)
      return expr->get();
    return std::nullopt;
  }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Return; }
};

/// Expression class for a binary operator.
class BinaryExprAST : public ExprAST {
  char op;
  std::unique_ptr<ExprAST> lhs, rhs;

public:
  char getOp() { return op; }
  ExprAST *getLHS() { return lhs.get(); }
  ExprAST *getRHS() { return rhs.get(); }

  BinaryExprAST(Location loc, char op, std::unique_ptr<ExprAST> lhs,
                std::unique_ptr<ExprAST> rhs)
      : ExprAST(Expr_BinOp, std::move(loc)), op(op), lhs(std::move(lhs)),
        rhs(std::move(rhs)) {}

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_BinOp; }
};

/// Expression class for function calls.
class CallExprAST : public ExprAST {
  std::string callee;
  std::vector<std::unique_ptr<ExprAST>> args;

public:
  CallExprAST(Location loc, const std::string &callee,
              std::vector<std::unique_ptr<ExprAST>> args)
      : ExprAST(Expr_Call, std::move(loc)), callee(callee),
        args(std::move(args)) {}

  llvm::StringRef getCallee() { return callee; }
  llvm::ArrayRef<std::unique_ptr<ExprAST>> getArgs() { return args; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Call; }
};

/// Expression class for builtin print calls.
class PrintExprAST : public ExprAST {
  std::unique_ptr<ExprAST> arg;

public:
  PrintExprAST(Location loc, std::unique_ptr<ExprAST> arg)
      : ExprAST(Expr_Print, std::move(loc)), arg(std::move(arg)) {}

  ExprAST *getArg() { return arg.get(); }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Print; }
};

/// This class represents the "prototype" for a function, which captures its
/// name, and its argument names (thus implicitly the number of arguments the
/// function takes).
// 함수의 프로토타입으로 컴파일러가 외부 구현된 코드를 이해하기 위해서 존재
// 또 선언만 할 때나 정의 할 때나 함수의 이름과 인자는 필요하기 때문에 묶어둠
class PrototypeAST {
  Location location;
  std::string name;
  std::vector<std::unique_ptr<VariableExprAST>> args;

public:
  PrototypeAST(Location location, const std::string &name,
               std::vector<std::unique_ptr<VariableExprAST>> args)
      : location(std::move(location)), name(name), args(std::move(args)) {}

  const Location &loc() { return location; }
  llvm::StringRef getName() const { return name; }
  llvm::ArrayRef<std::unique_ptr<VariableExprAST>> getArgs() { return args; }
};

/// This class represents a function definition itself.
class FunctionAST {
  std::unique_ptr<PrototypeAST> proto;
  std::unique_ptr<ExprASTList> body;

public:
  FunctionAST(std::unique_ptr<PrototypeAST> proto,
              std::unique_ptr<ExprASTList> body)
      : proto(std::move(proto)), body(std::move(body)) {}
  PrototypeAST *getProto() { return proto.get(); }
  ExprASTList *getBody() { return body.get(); }
};

/// This class represents a list of functions to be processed together
class ModuleAST {
  std::vector<FunctionAST> functions;

public:
  ModuleAST(std::vector<FunctionAST> functions)
      : functions(std::move(functions)) {}

  auto begin() { return functions.begin(); }
  auto end() { return functions.end(); }
};

void dump(ModuleAST &);

} // namespace toy

#endif // TOY_AST_H