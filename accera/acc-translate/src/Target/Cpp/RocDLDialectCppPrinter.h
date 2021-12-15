//===- RocDLDialectCppPrinter.h - RocDL Dialect Printer -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ROCDL_DIALECT_CPP_PRINTER_H_
#define ROCDL_DIALECT_CPP_PRINTER_H_

#include "CppPrinter.h"

#include <mlir/Dialect/LLVMIR/ROCDLDialect.h>

namespace mlir
{
namespace cpp_printer
{

    struct RocDLDialectCppPrinter : public DialectCppPrinter
    {
        RocDLDialectCppPrinter(CppPrinter* printer) :
            DialectCppPrinter(printer) {}

        /// print Operation from GPU Dialect
        LogicalResult printDialectOperation(Operation* op, bool* skipped, bool* consumed) override;
        LogicalResult printMFMAOp(Operation* mfmaOp);
        LogicalResult printBarrierOp(ROCDL::BarrierOp barrierOp);
        LogicalResult printBlockDimXOp(ROCDL::BlockDimXOp op);
        LogicalResult printBlockDimYOp(ROCDL::BlockDimYOp op);
        LogicalResult printBlockDimZOp(ROCDL::BlockDimZOp op);
        LogicalResult printBlockIdXOp(ROCDL::BlockIdXOp op);
        LogicalResult printBlockIdYOp(ROCDL::BlockIdYOp op);
        LogicalResult printBlockIdZOp(ROCDL::BlockIdZOp op);
        LogicalResult printGridDimXOp(ROCDL::GridDimXOp op);
        LogicalResult printGridDimYOp(ROCDL::GridDimYOp op);
        LogicalResult printGridDimZOp(ROCDL::GridDimZOp op);
        LogicalResult printThreadIdXOp(ROCDL::ThreadIdXOp op);
        LogicalResult printThreadIdYOp(ROCDL::ThreadIdYOp op);
        LogicalResult printThreadIdZOp(ROCDL::ThreadIdZOp op);
        LogicalResult printMubufLoadOp(ROCDL::MubufLoadOp op);
        LogicalResult printMubufStoreOp(ROCDL::MubufStoreOp op);
    };

} // namespace cpp_printer
} // namespace mlir

#endif
