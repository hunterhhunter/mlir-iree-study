//===- Dialect.h - Dialect definition for the Toy IR ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the IR Dialect for the Toy language.
// See docs/Tutorials/Toy/Ch-2.md for more information.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TUTORIAL_TOY_DIALECT_H_
#define MLIR_TUTORIAL_TOY_DIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"         // ch4-inliner에서 추가 CastOp를 위해
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "toy/ShapeInferenceOpInterfaces.h.inc"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace toy {
namespace detail {
struct StructTypeStorage;
}
}
}


/// Include the auto-generated header file containing the declaration of the toy
/// dialect.
#include "toy/Dialect.h.inc"

/// Include the auto-generated header file containing the declarations of the
/// toy operations.
#define GET_OP_CLASSES
#include "toy/Ops.h.inc"

//===----------------------------------------------------------------------===//
// Toy Types
//===----------------------------------------------------------------------===//
namespace mlir {
namespace toy {
    class StructType : public mlir::Type::TypeBase<StructType, mlir::Type,
                                                   detail::StructTypeStorage> {
    public:
        using Base::Base;

        static StructType get(llvm::ArrayRef<mlir::Type> elementType);

        llvm::ArrayRef<mlir::Type> getElementTypes();

        size_t getNumElementTypes() { return getElementTypes().size(); }

        static constexpr StringLiteral name = "toy.struct";
    };
}
}

#endif // MLIR_TUTORIAL_TOY_DIALECT_H_