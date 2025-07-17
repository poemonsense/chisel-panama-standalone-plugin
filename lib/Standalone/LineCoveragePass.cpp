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

  circuit.walk([&](WhenOp when) {
    auto condVal = when.getCondition();
    if (!condVal || condVal.getType().isConst())
      return;

    auto condDefOp = condVal.getDefiningOp();

    annotateCoverPoint(condDefOp, "line", circuit);
  });
}


} // namespace mlir::standalone
