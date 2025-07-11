#ifndef STANDALONE_PRINTFIRRTLOPSPASS_H
#define STANDALONE_PRINTFIRRTLOPSPASS_H

#include <memory>
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace standalone {

std::unique_ptr<mlir::Pass> createPrintFirrtlOpsPass();

/// Registers the pass for plugin loading
void registerPrintFirrtlOpsPass();

} // namespace standalone
} // namespace mlir

#endif // STANDALONE_PRINTFIRRTLOPSPASS_H
