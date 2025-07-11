//===- PrintFirrtlOpsPass.td - Print pass ---------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PRINTFIRRTLOPS_PASSES
#define PRINTFIRRTLOPS_PASSES

include "mlir/Pass/PassBase.td"

def PrintFirrtlOpsPass : Pass<"print-firrtl-ops", "::circt::firrtl::CircuitOp"> {
  let summary = "Print all FIRRTL ops";
  let description = [{
    Walks all FIRRTL operations and prints their name and optional symbol.
  }];
  let constructor = "mlir::standalone::createPrintFirrtlOpsPass";
}

#endif // PRINTFIRRTLOPS_PASSES
