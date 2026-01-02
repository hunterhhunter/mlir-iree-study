/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Ops.td                                                               *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

// 발표 - Namespace
namespace mlir::toy {

// 발표 - summary
/// A high-level dialect for analyzing and optimizing the Toy language
/// The Toy language is a tensor-based language that allows you to define
///     functions, perform some math computation, and print results. This dialect
///     provides a representation of the language that is amenable to analysis and
///     optimization.
// 발표 - Dialect 상속
class ToyDialect : public ::mlir::Dialect {
  explicit ToyDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~ToyDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("toy");
  }
};

} // namespace mlir::toy
// 발표 - MLIR 컴파일러에 ID 등록
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::toy::ToyDialect)
