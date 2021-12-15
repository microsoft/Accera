//===- GpuDialectCppPrinter.h - GPU Dialect Printer -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef GPU_DIALECT_CPP_PRINTER_H_
#define GPU_DIALECT_CPP_PRINTER_H_

// #include "CppPrinter.h"
// #include "mlir/Dialect/GPU/GPUDialect.h"

#include <mlir/Dialect/GPU/GPUDialect.h>

#include "CppPrinter.h"

namespace mlir {
namespace cpp_printer {

struct GpuDialectCppPrinter : public DialectCppPrinter {
  GpuDialectCppPrinter(CppPrinter *printer) : DialectCppPrinter(printer) {}

  /// print Operation from GPU Dialect
  LogicalResult printDialectOperation(Operation *op, bool *skipped,
                                      bool *consumed) override;

  LogicalResult printBarrierOp(gpu::BarrierOp barrierOp);

  LogicalResult printGridDimOp(gpu::GridDimOp gdimOp);

  LogicalResult printBlockDimOp(gpu::BlockDimOp bdimOp);

  LogicalResult printBlockIdOp(gpu::BlockIdOp bidOp);

  LogicalResult printThreadIdOp(gpu::ThreadIdOp tidOp);

  LogicalResult printVectorTypeArrayDecl(VectorType vecType,
                                         StringRef vecVar) override;

  LogicalResult printGpuFP16VectorType(VectorType vecType, StringRef vecVar);
};

} // namespace cpp_printer
} // namespace mlir

#endif
