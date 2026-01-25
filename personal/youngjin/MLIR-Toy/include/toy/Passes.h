//===----------------------------------------------------------------------===//
//
// This file exposes the entry points to create compiler passes for Toy.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_PASSES_H
#define TOY_PASSES_H

#include <memory>
#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>

namespace mlir {
class Pass;

namespace toy {
    std::unique_ptr<Pass> createShapeInferencePass();
    std::unique_ptr<Pass> createLowerToAffinePass();
    std::unique_ptr<Pass> createLowerToLLVMPass();
    } // namespace toy
} // namespace mlir

#endif // TOY_PASSES_H