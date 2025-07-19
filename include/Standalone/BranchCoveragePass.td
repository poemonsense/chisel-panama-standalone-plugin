//===- BranchCoveragePass.td - Print pass ---------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BRANCHCOVERAGE_PASS
#define BRANCHCOVERAGE_PASS

include "mlir/Pass/PassBase.td"

def BranchCoveragePass : Pass<"annotate-branch-coverage", "::circt::firrtl::CircuitOp"> {
  let summary = "Annotate branch coverage points";
  let description = [{
    Annotate branch coverage points.
  }];
  let constructor = "mlir::standalone::createBranchCoveragePass";
}

#endif // BRANCHCOVERAGE_PASS
