//===- ScfDialectCppPrinter.h - SCF Dialect Printer -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef SCF_DIALECT_CPP_PRINTER_H_
#define SCF_DIALECT_CPP_PRINTER_H_

#include <mlir/Dialect/SCF/SCF.h>

#include "CppPrinter.h"

namespace mlir {
namespace cpp_printer {

struct ScfDialectCppPrinter : public DialectCppPrinter {
  ScfDialectCppPrinter(CppPrinter *printer) : DialectCppPrinter(printer) {}

  /// print Operation from StandardOps Dialect
  LogicalResult printDialectOperation(Operation *op, bool *skipped,
                                      bool *consumed) override;

  /// print scf::ForOp
  LogicalResult printForOp(scf::ForOp forOp);

  /// print scf::IfOp
  LogicalResult printIfOp(scf::IfOp ifOp);

  /// print scf::YieldOp. Assign each yielded value to the one at
  /// the same position from retValues.
  template <typename RangeT>
  LogicalResult printYieldOp(scf::YieldOp yieldOp, RangeT retValues);
};

} // namespace cpp_printer
} // namespace mlir

#endif
