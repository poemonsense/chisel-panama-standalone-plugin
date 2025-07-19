#include "Standalone/LineCoveragePass.h"
#include "Standalone/CoverPointPass.h"

namespace mlir::standalone {
using namespace circt::firrtl;

#define GEN_PASS_DEF_LINECOVERAGEPASS
#include "Standalone/LineCoveragePass.h.inc"

class LineCoveragePass
  : public impl::LineCoveragePassBase<LineCoveragePass> {
public:
  void runOnOperation() final;

private:
  Value getNonConstBranchCondition(Operation *op);
};

std::unique_ptr<mlir::Pass> createLineCoveragePass() {
  return std::make_unique<LineCoveragePass>();
}

void registerLineCoveragePass() {
  PassRegistration<LineCoveragePass>();
}

void LineCoveragePass::runOnOperation() {
  auto circuit = getOperation();

  circuit.walk([&](Operation *op) {
    auto condVal = getNonConstBranchCondition(op);
    if (!condVal)
      return;

    auto condDefOp = condVal.getDefiningOp();
    if (!condDefOp) {
      OpBuilder builder(op);
      builder.setInsertionPoint(op);
      auto defName = builder.getStringAttr("line_cover_dummy");
      condDefOp = builder.create<WireOp>(op->getLoc(), condVal.getType(), defName);
      builder.create<ConnectOp>(condDefOp->getLoc(), condDefOp->getResult(0), condVal);
    }

    annotateCoverPoint(condDefOp, "line", circuit);
  });
}

Value LineCoveragePass::getNonConstBranchCondition(Operation *op) {
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
