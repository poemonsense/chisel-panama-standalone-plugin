//===- LineCoveragePass.td - Print pass ---------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LINECOVERAGE_PASS
#define LINECOVERAGE_PASS

include "mlir/Pass/PassBase.td"

def LineCoveragePass : Pass<"annotate-line-coverage", "::circt::firrtl::CircuitOp"> {
  let summary = "Annotate line coverage points";
  let description = [{
    Annotate line coverage points.
  }];
  let constructor = "mlir::standalone::createLineCoveragePass";
}

#endif // LINECOVERAGE_PASS
