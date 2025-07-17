#ifndef STANDALONE_LINECOVERAGEASS_H
#define STANDALONE_LINECOVERAGEASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace standalone {

std::unique_ptr<mlir::Pass> createLineCoveragePass();

void registerLineCoveragePass();

} // namespace standalone
} // namespace mlir

#endif // STANDALONE_LINECOVERAGEASS_H
