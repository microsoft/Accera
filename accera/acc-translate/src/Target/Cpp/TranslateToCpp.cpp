//===- TranslateToCpp.cpp - Translating MLIR to C++ -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// #include "mlir/Target/TranslateToCpp.h"
#include "TranslateToCpp.h"
#include "CppPrinter.h"

using namespace llvm;

namespace mlir {

LogicalResult translateModuleToCpp(Operation *m, raw_ostream &os, bool isCuda) {
  cpp_printer::CppPrinter printer(os, isCuda);
  printer.registerAllDialectPrinters();
  RETURN_IF_FAILED(printer.runPrePrintingPasses(m));
  return printer.printModuleOp(cast<ModuleOp>(m));
}

} // namespace mlir
