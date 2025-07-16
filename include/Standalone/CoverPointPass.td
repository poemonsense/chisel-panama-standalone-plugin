//===- CoverPointPass.td - Print pass ---------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COVERPOINT_PASS
#define COVERPOINT_PASS

include "mlir/Pass/PassBase.td"

def CoverPointPass : Pass<"extract-cover-point", "::circt::firrtl::CircuitOp"> {
  let summary = "Extract all customized cover points";
  let description = [{
    Extract all customized cover points to DiffTest.
  }];
  let constructor = "mlir::standalone::createCoverPointPass";
}

#endif // COVERPOINT_PASS
