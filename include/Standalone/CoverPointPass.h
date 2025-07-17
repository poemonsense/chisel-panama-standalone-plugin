#ifndef STANDALONE_COVERPOINTPASS_H
#define STANDALONE_COVERPOINTPASS_H

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace standalone {

std::unique_ptr<mlir::Pass> createCoverPointPass();

void registerCoverPointPass();

void annotateCoverPoint(
  Operation *op,
  const std::string &name,
  const std::string &groupName,
  circt::firrtl::CircuitOp &circuit
);

void annotateCoverPoint(
  Operation *op,
  const std::string &groupName,
  circt::firrtl::CircuitOp &circuit
);

} // namespace standalone
} // namespace mlir

#endif // STANDALONE_COVERPOINTPASS_H
