/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Ops.td                                                               *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::toy::ToyDialect)
namespace mlir::toy {


ToyDialect::ToyDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<ToyDialect>())
    
     {
  
  initialize();
}

ToyDialect::~ToyDialect() = default;


} // namespace mlir::toy
