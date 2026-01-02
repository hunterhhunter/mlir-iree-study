// include/toy/Dialect.h
#ifndef TOY_DIALECT_H
#define TOY_DIALECT_H

// [핵심] MLIR 기본 뼈대 헤더들 (이게 없으면 에러남!)
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

// 1. TableGen이 생성한 방언(Dialect) 선언 포함
//    (namespace mlir { namespace toy { ... } } 로 감싸져 있음)
#include "toy/Dialect.h.inc"

// 2. TableGen이 생성한 연산(Op) 선언 포함
#define GET_OP_CLASSES
#include "toy/Ops.h.inc"

#endif // TOY_DIALECT_H