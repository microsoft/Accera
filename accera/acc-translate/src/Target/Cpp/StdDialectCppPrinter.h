//===- StdDialectCppPrinter.h - StandardOps Dialect Printer ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef STD_DIALECT_CPP_PRINTER_H_
#define STD_DIALECT_CPP_PRINTER_H_

// #include "CppPrinter.h"
// #include "mlir/Dialect/StandardOps/IR/Ops.h"

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/Math/IR/Math.h>

#include "CppPrinter.h"

namespace mlir {
namespace cpp_printer {

struct StdDialectCppPrinter : public DialectCppPrinter {
  StdDialectCppPrinter(CppPrinter *printer) : DialectCppPrinter(printer) {}

  /// print Operation from StandardOps Dialect
  LogicalResult printDialectOperation(Operation *op, bool *skipped,
                                      bool *consumed) override;

  /// print standard binary ops such as '+', '-', '*', etc
  LogicalResult printStandardBinaryOp(Operation *binOp);

  /// print a CastOp where the dst type is an Integer whose signed-ness
  /// is determined by the argument isSigned
  LogicalResult printCastToIntegerOp(Operation *op, bool isSigned);

  /// print a ``simple'' CastOp such as IndexCastOp and TruncateIOp
  /// that can be converted into a cast expression without worrying
  /// about the signed-ness of the operands
  LogicalResult printSimpleCastOp(Operation *op);

  /// print AllocOp
  LogicalResult printAllocOp(memref::AllocOp allocOp);

  /// print AllocaOp
  LogicalResult printAllocaOp(memref::AllocaOp allocaOp);

  /// print CallOp
  LogicalResult printCallOp(CallOp constOp);

  /// print ConstantOp
  LogicalResult printConstantOp(ConstantOp constOp);

  /// print DeallocOp
  LogicalResult printDeallocOp(memref::DeallocOp deallocOp, bool *skipped);

  /// print DimOp
  LogicalResult printDimOp(memref::DimOp dimOp);

  /// print ExpOp
  LogicalResult printExpOp(math::ExpOp expOp);

  /// print LoadOp
  LogicalResult printLoadOp(memref::LoadOp loadOp);

  /// print MemRefCastOp
  LogicalResult printMemRefCastOp(memref::CastOp memRefCastOp);

  /// print ReturnOp
  LogicalResult printReturnOp(ReturnOp returnOp);

  /// print SelectOp as ternary operator
  LogicalResult printSelectOp(SelectOp selectOp);

  /// print StoreOp
  LogicalResult printStoreOp(memref::StoreOp storeOp);
};

} // namespace cpp_printer
} // namespace mlir

#endif
