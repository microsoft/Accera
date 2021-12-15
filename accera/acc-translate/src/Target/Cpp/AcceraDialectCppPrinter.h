//===- AcceraDialectCppPrinter.h - Argo Dialect Printer ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ARGO_DIALECT_CPP_PRINTER_H_
#define ARGO_DIALECT_CPP_PRINTER_H_

// #include "CppPrinter.h"
// #include "mlir/Dialect/Argo/IR/ArgoOps.h"

#include <ir/include/argo/ArgoOps.h>

#include "CppPrinter.h"

namespace mlir {
namespace cpp_printer {

struct AcceraDialectCppPrinter : public DialectCppPrinter {
  enum class MMAKernelKind { m8n8k4RowColfp32, InvalidKernel };

  AcceraDialectCppPrinter(CppPrinter *printer) : DialectCppPrinter(printer) {}

  LogicalResult printDialectOperation(Operation *op, bool *skipped,
                                      bool *consumed) override;

  LogicalResult addMMAKernel(FuncOp funcOp);

  void printMMAm8n8k4RowColFP32Def(StringRef kernelName);

  LogicalResult printMMAm8n8k4RowColFP32Call(Operation *callOp, FuncOp funcOp,
                                             StringRef kernelName);

  LogicalResult printIntrinsicCallOp(Operation *callOp, Operation *defFuncOp,
                                     bool *consumed) override;

  /// print out host function that launches the kernel
  LogicalResult printHostLaunchFunc();

  LogicalResult printPrologue() override;

  LogicalResult printEpilogue() override;

  LogicalResult runPrePrintingPasses(Operation *op) override;

  StringRef getMMAKernelName(MMAKernelKind kind) {
    assert(kind < MMAKernelKind::InvalidKernel);
    return MMAKernelNames[static_cast<unsigned>(kind)];
  }

  void addMMAKernelName(MMAKernelKind kind, StringRef name) {
    MMAKernelNames[static_cast<unsigned>(kind)] = name;
  }

  llvm::DenseMap<FuncOp, MMAKernelKind> FuncOpToMMAKernel;

  llvm::SmallVector<StringRef, 0> MMAKernelNames;

  llvm::SmallVector<FuncOp, 1> CudaKernels;
};

} // namespace cpp_printer
} // namespace mlir

#endif
