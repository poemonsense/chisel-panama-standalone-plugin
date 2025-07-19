#include "Standalone/BranchCoveragePass.h"
#include "Standalone/CoverPointPass.h"

namespace mlir::standalone {
using namespace circt::firrtl;

#define GEN_PASS_DEF_BRANCHCOVERAGEPASS
#include "Standalone/BranchCoveragePass.h.inc"

class BranchCoveragePass
  : public impl::BranchCoveragePassBase<BranchCoveragePass> {
public:
  void runOnOperation() final;

private:
  Value getNonConstBranchCondition(Operation *op);
};

std::unique_ptr<mlir::Pass> createBranchCoveragePass() {
  return std::make_unique<BranchCoveragePass>();
}

void registerBranchCoveragePass() {
  PassRegistration<BranchCoveragePass>();
}

void BranchCoveragePass::runOnOperation() {
  auto circuit = getOperation();

  circuit.walk([&](Operation *op) {
    auto condVal = getNonConstBranchCondition(op);
    if (!condVal)
      return;
    annotateCoverPoint(findDefOp(condVal, op), "branch", circuit);
  });
}

Value BranchCoveragePass::getNonConstBranchCondition(Operation *op) {
  // when (cond) { ... }
  if (auto whenOp = dyn_cast<WhenOp>(op)) {
    auto cond = whenOp.getCondition();
    if (cond && !cond.getType().isConst())
      return cond;
  }
  // Mux(cond, ..., ...)
  else if (auto mux = dyn_cast<MuxPrimOp>(op)) {
    auto cond = mux.getSel();
    if (cond && !cond.getType().cast<FIRRTLType>().isConst())
      return cond;
  }
  return nullptr;
}

} // namespace mlir::standalone
