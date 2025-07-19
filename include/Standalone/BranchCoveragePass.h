#ifndef STANDALONE_BRANCHCOVERAGEASS_H
#define STANDALONE_BRANCHCOVERAGEASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace standalone {

std::unique_ptr<mlir::Pass> createBranchCoveragePass();

void registerBranchCoveragePass();

} // namespace standalone
} // namespace mlir

#endif // STANDALONE_BRANCHCOVERAGEASS_H
