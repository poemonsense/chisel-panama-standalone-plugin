#ifndef STANDALONE_COVERPOINTPASS_H
#define STANDALONE_COVERPOINTPASS_H

#include <memory>
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace standalone {

std::unique_ptr<mlir::Pass> createCoverPointPass();

void registerCoverPointPass();

} // namespace standalone
} // namespace mlir

#endif // STANDALONE_COVERPOINTPASS_H
