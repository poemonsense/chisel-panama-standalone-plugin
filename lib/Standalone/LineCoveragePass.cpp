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
};

std::unique_ptr<mlir::Pass> createLineCoveragePass() {
  return std::make_unique<LineCoveragePass>();
}

void registerLineCoveragePass() {
  PassRegistration<LineCoveragePass>();
}

void LineCoveragePass::runOnOperation() {
  auto circuit = getOperation();

  circuit.walk([&](WhenOp whenOp) {
    auto condVal = whenOp.getCondition();
    if (!condVal || condVal.getType().isConst())
      return;

    auto condDefOp = condVal.getDefiningOp();
    if (!condDefOp) {
      OpBuilder builder(whenOp);
      builder.setInsertionPoint(whenOp);
      auto defName = builder.getStringAttr("cover_dummy_node");
      condDefOp = builder.create<WireOp>(whenOp.getLoc(), condVal.getType(), defName);
      builder.create<ConnectOp>(condDefOp->getLoc(), condDefOp->getResult(0), condVal);
    }

    annotateCoverPoint(condDefOp, "line", circuit);
  });
}


} // namespace mlir::standalone
