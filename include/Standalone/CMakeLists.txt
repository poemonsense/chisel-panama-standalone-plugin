add_mlir_dialect(StandaloneOps standalone)
add_mlir_doc(StandaloneDialect StandaloneDialect Standalone/ -gen-dialect-doc)
add_mlir_doc(StandaloneOps StandaloneOps Standalone/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS StandalonePasses.td)
mlir_tablegen(StandalonePasses.h.inc --gen-pass-decls)
add_public_tablegen_target(CIRCTStandalonePassesIncGen)

set(LLVM_TARGET_DEFINITIONS PrintFirrtlOpsPass.td)
mlir_tablegen(PrintFirrtlOpsPass.h.inc --gen-pass-decls)
add_public_tablegen_target(StandalonePrintFirrtlOpsPassIncGen)
