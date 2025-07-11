#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"

#include "Standalone/PrintFirrtlOpsPass.h"

namespace mlir::standalone {
#define GEN_PASS_DEF_PRINTFIRRTLOPSPASS
#include "Standalone/PrintFirrtlOpsPass.h.inc"

namespace {

class PrintFirrtlOpsPass
    : public impl::PrintFirrtlOpsPassBase<PrintFirrtlOpsPass> {
public:
  void runOnOperation() final {
    using namespace circt::firrtl;
    CircuitOp circuitOp = getOperation();
    circuitOp->walk([](Operation *op) {
      llvm::outs() << "Op: " << op->getName().getStringRef();
      if (auto attr = op->getAttrOfType<StringAttr>("sym_name"))
        llvm::outs() << ", Name: @" << attr.getValue();
      llvm::outs() << "\n";
    });
  }
};
} // namespace
} // namespace mlir::standalone

std::unique_ptr<mlir::Pass> mlir::standalone::createPrintFirrtlOpsPass() {
  return std::make_unique<PrintFirrtlOpsPass>();
}

void mlir::standalone::registerPrintFirrtlOpsPass() {
  PassRegistration<PrintFirrtlOpsPass>();
}