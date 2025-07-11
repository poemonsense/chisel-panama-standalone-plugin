add_mlir_dialect_library(CIRCTStandalone
        StandaloneTypes.cpp
        StandaloneDialect.cpp
        StandaloneOps.cpp
        StandalonePasses.cpp
        PrintFirrtlOpsPass.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Standalone

        DEPENDS
        MLIRStandaloneOpsIncGen
        CIRCTStandalonePassesIncGen
        StandalonePrintFirrtlOpsPassIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        CIRCTFIRRTL
        )
